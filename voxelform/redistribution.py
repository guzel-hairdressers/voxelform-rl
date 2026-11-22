"""Experimental constant-volume neural exchanges, independent of FAR episodes.

Both masks are decoded from one actor pass. Invalid pairs are rejected whole;
neither a geometric candidate bank nor synthesized bridging voxels are used.
"""

from dataclasses import dataclass
import math, time
import numpy as np
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F
from torch.distributions import Normal
from .actions import STRUCTURE, exterior_air
from .patch_policy import gaussian_field


@dataclass
class Exchange:
    remove: np.ndarray
    add: np.ndarray
    version: int

    @property
    def count(self):
        return int(self.remove.sum())


def growth_mask(env):
    """One frozen roof layer plus ordinary facade growth, inside height ceiling."""
    solid = env.frontier.occupied
    roof = np.zeros_like(solid)
    roof[1:] = solid[:-1] & ~solid[1:]
    return (
        (env.frontier.add_skin | roof)
        & env.frontier.allowed
        & env.frontier.exterior_air
    )


def validate(env, move, maximum_buildings=2):
    f = env.frontier
    r, a = move.remove, move.add
    if move.version != f.version or r.shape != env.shape or a.shape != env.shape:
        return False
    if not r.any() or not a.any() or (r & a).any():
        return False
    low, high = env.acceptable_counts
    changed_count = env.count - int(r.sum()) + int(a.sum())
    if changed_count != env.count and not low <= changed_count <= high:
        return False
    if (r & ~f.eligible(-1)).any() or (a & ~growth_mask(env)).any():
        return False
    for mask in (r, a):
        lab, npatch = ndi.label(mask, np.ones((3, 3, 3), bool))
        sizes = np.bincount(lab.ravel())[1:]
        if f.config.maximum_patches and npatch > f.config.maximum_patches:
            return False
        if (sizes < min(f.config.minimum_patch, int(mask.sum()))).any():
            return False
    changed = (f.occupied & ~r) | a
    labels, n = ndi.label(changed, STRUCTURE)
    grounded = np.unique(labels[0])
    grounded = grounded[grounded != 0]
    if not 0 < n <= maximum_buildings or len(grounded) != n:
        return False
    air = exterior_air(changed)
    if (r & ~air).any() or (f.exterior_air & ~a & ~air).any():
        return False
    # Check final combined courtyard footprints; never validate two artificial
    # intermediate buildings or grow an enclosed room accidentally.
    if f.old_voids is not None:
        for z in range(env.nz):
            lab, ids, sizes = f._floor_voids(changed[z], z)
            oldlab, oldids, oldsizes = f.old_voids[z]
            if f.config.maximum_voids and len(ids) > max(
                f.config.maximum_voids, len(oldids)
            ):
                return False
            for i in ids:
                overlap = np.unique(oldlab[lab == i])
                prior = [oldsizes[j] for j in overlap if j in oldids]
                minimum = min([f.config.minimum_void_area, *prior])
                if sizes[i] < minimum:
                    return False
    return True


def commit(env, move, maximum_buildings=2, audit=True):
    if not validate(env, move, maximum_buildings):
        raise ValueError("Invalid or stale volume exchange")
    before = env.count
    f = env.frontier
    f.occupied[move.remove] = False
    f.occupied[move.add] = True
    f.version += 1
    f.refresh()
    env.occupied.masked_fill_(torch.as_tensor(move.remove, device=env.device), False)
    env.occupied.masked_fill_(torch.as_tensor(move.add, device=env.device), True)
    env.count = before - int(move.remove.sum()) + int(move.add.sum())
    env.edits += 1
    env.voxels_edited += int(move.remove.sum() + move.add.sum())
    env.last_changed_voxels = int(move.remove.sum() + move.add.sum())
    env.last_operation = "exchange"
    env.update_depth()
    assert int(env.occupied.sum().cpu()) == env.count
    if audit:
        env.evaluate()


def duplicate_top(env, component=1):
    """Optional addition primitive; copy a chosen building's highest storey."""
    labels, n = ndi.label(env.frontier.occupied, STRUCTURE)
    if component < 1 or component > n:
        raise ValueError("Unknown building component")
    cells = labels == component
    z = int(np.flatnonzero(cells.any((1, 2)))[-1])
    add = np.zeros(env.shape, bool)
    if z + 1 < env.nz:
        add[z + 1] = (
            cells[z] & env.frontier.allowed[z + 1] & ~env.frontier.occupied[z + 1]
        )
    return add


def project_pair(env, masks, scores, whole_storey=False, maximum_buildings=2):
    """Prune selected bits, then rebalance; never invent connectors or cells."""
    r, a = masks.copy()
    f = env.frontier
    for _ in range(4):
        r &= exterior_air(f.occupied & ~r)
        r = f._patch_filter(r, int(r.sum())) if r.any() else r
        a = f._patch_filter(a, int(a.sum())) if a.any() else a
        k = min(int(r.sum()), int(a.sum()))
        if k < min(f.config.minimum_patch, int(masks[0].sum())) or (
            whole_storey and k != int(masks[1].sum())
        ):
            return None
        for mask, confidence in ((r, scores[0]), (a, scores[1])):
            if int(mask.sum()) > k:
                ids = np.flatnonzero(mask)
                keep = ids[np.argsort(-confidence.ravel()[ids], kind="stable")[:k]]
                mask[:] = False
                mask.ravel()[keep] = True
        move = Exchange(r.copy(), a.copy(), f.version)
        if validate(env, move, maximum_buildings):
            return move
    return None


def choose(
    policy,
    env,
    count,
    training=False,
    maximum_buildings=2,
    retries=8,
    duplicate_component=None,
):
    """One CNN pass, joint Gaussian score for consumed paired mask draws."""
    if count < 1:
        raise ValueError("Exchange size must be positive")
    if retries < 1:
        raise ValueError("At least one draw is required")
    if policy.config.mode not in ("combined", "correlated"):
        raise ValueError("Paired edits require a correlated mask policy")
    start = time.perf_counter()
    policy._env = env
    remove = env.frontier.eligible(-1)
    add = growth_mask(env)
    if duplicate_component is not None:
        add = duplicate_top(env, duplicate_component)
        count = int(add.sum())
        if count > int(remove.sum()):
            return None
    count = min(int(count), int(remove.sum()), int(add.sum()))
    if not count:
        return None
    device = policy.device
    weights = torch.stack(
        (
            env.frontier.weights_tensor(-1, device),
            torch.as_tensor(add, device=device, dtype=torch.float32),
        )
    )
    legal = weights > 0
    logs = []
    move = None
    with torch.set_grad_enabled(training):
        fields, value = policy.fields(env)
        kernel = tuple(min(n, s) for n, s in zip(env.shape, policy.config.stride))
        mean = F.avg_pool3d(
            fields[:, None],
            kernel,
            stride=policy.config.stride,
            ceil_mode=True,
            count_include_pad=False,
        )[:, 0]
        std = policy.model.patch_log_std.clamp(-2, 1).exp()[:, None, None, None]
        dist = Normal(mean, std, validate_args=False)
        for attempt in range(retries):
            latent = dist.sample()
            decoded = F.interpolate(
                latent[:, None], size=env.shape, mode="trilinear", align_corners=False
            )[:, 0]
            if policy.config.mode == "combined":
                decoded = gaussian_field(decoded, policy.config.sigma)
            if policy.config.vertical_interior:
                interior = torch.as_tensor(
                    remove & ~env.frontier.facade_band, device=device
                )
                decoded = torch.stack(
                    (
                        torch.where(
                            interior,
                            decoded[0].mean(0, keepdim=True).expand_as(decoded[0]),
                            decoded[0],
                        ),
                        decoded[1],
                    )
                )
            scale = (
                (1.0, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0625)[min(attempt, 7)]
                if duplicate_component is None
                else 1.0
            )
            size = min(
                count,
                max(min(policy.config.minimum_patch, count), round(count * scale)),
            )
            scores = decoded.sigmoid() * weights
            ids = scores.masked_fill(~legal, -1).flatten(1).topk(size, dim=1).indices
            masks = (
                torch.zeros_like(scores.flatten(1), dtype=torch.bool)
                .scatter_(1, ids, True)
                .reshape(2, *env.shape)
                .cpu()
                .numpy()
            )
            if training:
                logs.append(dist.log_prob(latent).sum() / math.sqrt(latent.numel()))
            trial = project_pair(
                env,
                masks,
                scores.detach().cpu().numpy(),
                duplicate_component is not None,
                maximum_buildings,
            )
            if trial is not None and validate(env, trial, maximum_buildings):
                move = trial
                break
        if training:
            policy.tape.append(
                [torch.stack(logs).sum(), value, dist.entropy().mean(), None]
            )
    policy.generation_seconds = time.perf_counter() - start
    return move
