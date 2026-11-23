"""Unified remove/grow/exchange/finish policy, available from step zero.

The active preview enables linked whole-floor copying and net FAR guidance.
The default floor_copy=False preserves the original roof-patch research decoder.
Masks use frozen eligibility and whole-edit guards, without a phase gate.
"""

from dataclasses import asdict
import math, time
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal
from .policy import Learner, Network
from .patch_policy import PatchConfig, gaussian_field
from .redistribution import Exchange, growth_mask
from .actions import STRUCTURE, exterior_air
from .rollout import generalized_advantages

OPERATIONS = ("remove", "grow", "exchange", "finish")


def min_room_width(env):
    """Minimum horizontal structural/room width in metres on urban-scale plots."""
    return 3 if env.plot_area >= 400 else 1


def clean_floor_tensor(occ_bool, min_width=4):
    """Vectorized 2D k x k morphological opening across all floors (async, zero GPU-CPU syncs)."""
    is_2d = occ_bool.ndim == 2
    x = occ_bool[None] if is_2d else occ_bool
    k = int(min_width)
    if k <= 1:
        return occ_bool.clone()
    ny, nx = x.shape[-2], x.shape[-1]
    if ny < k or nx < k:
        out = torch.zeros_like(x, dtype=torch.bool)
        return out[0] if is_2d else out
    xf = x.float()[:, None]
    eroded = (F.avg_pool2d(xf, k, stride=1) > 1.0 - 1e-4).float()
    opened = (
        F.max_pool2d(F.pad(eroded, (k - 1, k - 1, k - 1, k - 1)), k, stride=1) > 0.5
    )
    out = opened[:, 0]
    return out[0] if is_2d else out


def clean_floor_numpy(occ_bool, min_width=4):
    """Exact k x k 2D morphological opening via separable bitwise NumPy slices (~0.15 ms)."""
    arr = np.asarray(occ_bool, dtype=bool)
    k = int(min_width)
    if k <= 1 or not arr.any():
        return arr.copy()
    is_2d = arr.ndim == 2
    x = arr[None] if is_2d else arr
    nz, ny, nx = x.shape
    if ny < k or nx < k:
        out = np.zeros_like(x)
        return out[0] if is_2d else out
    eh = x[:, :, : nx - k + 1].copy()
    for i in range(1, k):
        eh &= x[:, :, i : nx - k + 1 + i]
    er = eh[:, : ny - k + 1, :].copy()
    for i in range(1, k):
        er &= eh[:, i : ny - k + 1 + i, :]
    dh = np.zeros((nz, ny - k + 1, nx), dtype=bool)
    for i in range(k):
        dh[:, :, i : nx - k + 1 + i] |= er
    out = np.zeros_like(x)
    for i in range(k):
        out[:, i : ny - k + 1 + i, :] = out[:, i : ny - k + 1 + i, :] | dh
    return out[0] if is_2d else out


def long_narrow_mask(occ_bool, width=6, stub=2):
    """Vectorized detector for strips narrower than `width` m extending >`stub` m past a width x width core (async)."""
    is_2d = occ_bool.ndim == 2
    x = occ_bool[None] if is_2d else occ_bool
    k = int(width)
    if k <= 1:
        return torch.zeros_like(occ_bool, dtype=torch.bool)
    ny, nx = x.shape[-2], x.shape[-1]
    if ny < k or nx < k:
        out = x.clone()
        return out[0] if is_2d else out
    xf = x.float()[:, None]
    eroded = (F.avg_pool2d(xf, k, stride=1) > 1.0 - 1e-4).float()
    core = F.max_pool2d(F.pad(eroded, (k - 1, k - 1, k - 1, k - 1)), k, stride=1) > 0.5
    s = int(stub)
    near = (
        (F.max_pool2d(core.float(), 2 * s + 1, stride=1, padding=s) > 0.5)
        if s > 0
        else core
    )
    out = x & ~near[:, 0]
    return out[0] if is_2d else out


def long_narrow_numpy(occ_bool, width=6, stub=2):
    """CPU/NumPy wrapper delegating to the vectorized PyTorch long-strip detector."""
    arr = np.asarray(occ_bool, dtype=bool)
    if width <= 1 or not arr.any():
        return np.zeros_like(arr, dtype=bool)
    return long_narrow_mask(torch.from_numpy(arr), width=width, stub=stub).numpy()


def _allowed_long_narrow(env, width=6, stub=2):
    cache = getattr(env, "_allowed_ln_cache", None)
    key = (env.shape, int(width), int(stub))
    if cache is None or cache[0] != key:
        ln = long_narrow_mask(env.allowed, width=width, stub=stub)
        env._allowed_ln_cache = (key, ln)
        return ln
    return cache[1]


def _allowed_slim_2d(env, z):
    mw = min_room_width(env)
    if mw <= 1:
        return None
    cache = getattr(env, "_allowed_slim_cache", None)
    if cache is None or cache[0] != env.shape:
        a = env.frontier.allowed
        clean_a = clean_floor_numpy(a, mw)
        slim_a = a & ~clean_a
        env._allowed_slim_cache = (env.shape, slim_a)
        cache = env._allowed_slim_cache
    return cache[1][z]


def min_plate_area(env, z=None):
    """Minimum coherent floor-plate area for vertical extrusion."""
    base = max(4, env.frontier.config.minimum_patch)
    if env.plot_area >= 400 and (z is None or z >= 4):
        return max(base, min(96, max(16, int(round(0.06 * env.plot_area)))))
    return base


def coherent_plate(mask_2d, min_area, min_width=1, allowed_slim=None):
    """Require minimum plate area, a non-fin 2x2 interior core, and minimum room width."""
    if int(mask_2d.sum()) < min_area:
        return False
    if min_area >= 16:
        if int(ndi.binary_erosion(mask_2d, np.ones((2, 2), bool)).sum()) < 4:
            return False
    if min_width > 1:
        clean = clean_floor_numpy(mask_2d, min_width)
        if int(clean.sum()) < min_area:
            return False
        if allowed_slim is not None:
            if (mask_2d & ~clean & ~allowed_slim).any():
                return False
        elif not np.array_equal(mask_2d, clean):
            return False
    return True


def copy_plates(env):
    """Complete highest floors per mass plus exposed horizontal roof terraces.

    Plates are explicit growth primitives derived from the current building,
    filtering tiny upper slivers while letting stepped podium and tower roofs
    grow independently.
    """
    f = env.frontier
    labels, n = ndi.label(f.occupied, STRUCTURE)
    plates = []
    mw = min_room_width(env)

    def _add_plate(z, plate_2d):
        if not (0 < z < env.nz):
            return
        if not coherent_plate(
            plate_2d, min_plate_area(env, z), mw, _allowed_slim_2d(env, z)
        ):
            return
        if (
            f.config.maximum_edit_voxels
            and int(plate_2d.sum()) > f.config.maximum_edit_voxels
        ):
            return
        if (plate_2d & (~f.allowed[z] | f.occupied[z] | ~f.exterior_air[z])).any():
            return
        if any(p[z].any() and np.array_equal(p[z], plate_2d) for p in plates):
            return
        mask = np.zeros(env.shape, bool)
        mask[z] = plate_2d
        plates.append(mask)

    for i in range(1, n + 1):
        cells = labels == i
        levels = np.flatnonzero(cells.any((1, 2)))
        if not len(levels):
            continue
        z = int(levels[-1]) + 1
        if z < env.nz:
            _add_plate(z, cells[z - 1])
    roof_struct = ndi.generate_binary_structure(2, 1)
    for z in range(1, env.nz):
        roof_2d = f.occupied[z - 1] & ~f.occupied[z] & f.allowed[z] & f.exterior_air[z]
        if not roof_2d.any():
            continue
        rlab, rn = ndi.label(roof_2d, roof_struct)
        for j in range(1, rn + 1):
            _add_plate(z, rlab == j)
    return plates


def copy_eligibility(env):
    plates = copy_plates(env)
    return np.logical_or.reduce(plates) if plates else np.zeros(env.shape, bool)


def extract_tower_plate(env, plate, add_field, remove_mask=None, desired_area=None):
    """Extract a coherent, well-exposed tower sub-plate from a wide roof plate."""
    levels = np.flatnonzero(plate.any((1, 2)))
    if not len(levels):
        return plate
    z = int(levels[0])
    min_area = min_plate_area(env, z)
    mw = min_room_width(env)
    allowed_slim = _allowed_slim_2d(env, z)
    supported = plate[z].copy()
    if remove_mask is not None and z > 0:
        supported &= ~remove_mask[z - 1]
    avail = int(supported.sum())
    out = np.zeros(env.shape, bool)
    if avail < min_area:
        return out
    if (
        env.plot_area < 400
        or desired_area is None
        or avail <= max(min_area, int(desired_area))
    ):
        if coherent_plate(supported, min_area, mw, allowed_slim):
            out[z] = supported
        return out
    field_2d = (
        add_field.detach().cpu().numpy()
        if torch.is_tensor(add_field)
        else np.asarray(add_field, dtype=np.float32)
    )
    if field_2d.ndim == 3:
        field_2d = field_2d[z]
    k = max(min_area, min(avail, int(desired_area)))
    ids = np.flatnonzero(supported)
    chosen = ids[np.argsort(-field_2d.ravel()[ids], kind="stable")[:k]]
    sub = np.zeros_like(supported)
    sub.ravel()[chosen] = True
    roof_struct = ndi.generate_binary_structure(2, 1)
    slab, sn = ndi.label(sub, roof_struct)
    if sn > 1:
        sizes = np.bincount(slab.ravel())[1:]
        sub = slab == (int(np.argmax(sizes)) + 1)
    sub = ndi.binary_fill_holes(sub) & supported
    opened = (
        clean_floor_numpy(sub, mw)
        if mw > 1
        else ndi.binary_opening(sub, np.ones((2, 2), bool))
    )
    if int(opened.sum()) >= min_area and coherent_plate(opened, min_area, mw):
        olab, on = ndi.label(opened, roof_struct)
        if on >= 1:
            sizes = np.bincount(olab.ravel())[1:]
            cand = olab == (int(np.argmax(sizes)) + 1)
            if coherent_plate(cand, min_area, mw):
                sub = cand
    if coherent_plate(sub, min_area, mw, allowed_slim):
        out[z] = sub
        return out
    if coherent_plate(supported, min_area, mw, allowed_slim):
        out[z] = supported
    return out


def unstack_cells(env, r):
    """Identify complete upper 2D floor tiers selected for reversible unstacking."""
    solid = env.frontier.occupied
    unstacked = np.zeros(env.shape, bool)
    roof_struct = ndi.generate_binary_structure(2, 1)
    for z in range(env.nz - 1, 0, -1):
        if not solid[z].any():
            continue
        lab, n = ndi.label(solid[z], roof_struct)
        for i in range(1, n + 1):
            comp = lab == i
            clear_above = (z == env.nz - 1) or not bool(
                ((solid[z + 1] & ~(r[z + 1] & unstacked[z + 1])) & comp).any()
            )
            if clear_above and bool(r[z][comp].all()):
                unstacked[z, comp] = True
    return unstacked


def supported_removal(env, remove):
    """Prune partial cuts; preserve top-down support, whole-tier unstacking, and floor width."""
    r = remove.copy()
    solid = env.frontier.occupied
    labels = getattr(env, "copy_labels", np.zeros(env.shape, dtype=np.int64))
    mw = min_room_width(env)
    clean_solid = clean_floor_numpy(solid, mw) if mw > 1 else None
    max_lab = int(labels.max()) + 1 if labels.any() else 0

    # Copied floors remain coupled for partial cuts unless an entire top floor
    # tier is unstacked cleanly with nothing remaining above it.
    def _vertical_and_labels(r_cur):
        for _ in range(env.nz + 1):
            before = int(r_cur.sum())
            for z in range(env.nz - 2, -1, -1):
                r_cur[z] &= ~solid[z + 1] | r_cur[z + 1]
            if labels.any():
                unstacked = unstack_cells(env, r_cur)
                total = np.bincount(labels[solid & ~unstacked], minlength=max_lab)
                selected = np.bincount(labels[r_cur & ~unstacked], minlength=max_lab)
                complete = selected == total
                complete[0] = True
                r_cur &= complete[labels] | unstacked
            if int(r_cur.sum()) == before:
                break
        return r_cur

    r = _vertical_and_labels(r)
    if mw > 1 and r.any():
        for _ in range(mw):
            after = solid & ~r
            slim = (after & ~clean_floor_numpy(after, mw)) & clean_solid
            if not slim.any():
                break
            slim_t = torch.from_numpy(slim).float()[:, None]
            zone = (F.max_pool2d(slim_t, 3, stride=1, padding=1)[:, 0] > 0.5).numpy()
            if not (r & zone).any():
                break
            r &= ~zone
            for z in range(env.nz - 2, -1, -1):
                r[z] &= ~solid[z + 1] | r[z + 1]
        r = _vertical_and_labels(r)
    return r


def balance_copy(env, remove, addition, scores):
    """Prune a neural cut prefix with the exact source/copy area cost.

    Cutting a source cell removes both that cell and its prospective copy.
    Linked stacks add further atomic costs. Search a quota on the ONE generated
    ranking, never sample or score a library of alternative geometric masks.
    """
    mw = min_room_width(env)

    def result(r):
        a = addition.copy()
        a[1:] &= ~r[:-1]
        if mw > 1 and a.any():
            levels = np.flatnonzero(a.any((1, 2)))
            if len(levels) == 1:
                z = int(levels[0])
                a[z] = clean_floor_numpy(a[z], mw)
        return a, env.count - int(r.sum()) + int(a.sum())

    a, after = result(remove)
    low, high = env.acceptable_counts
    lower = low if env.count >= low else env.count + 1
    upper = high if env.count <= high else env.count - 1
    if lower <= after <= upper:
        return remove, a
    if after > upper:
        return remove, a  # Not enough selected cuts; do not invent any.
    target = env.goal_count
    if env.count < low:
        target = min(target, env.count + max(1, int(addition.sum()) // 2))
    target = max(lower, min(upper, target))
    ids = np.flatnonzero(remove)
    ids = ids[np.argsort(-scores.ravel()[ids], kind="stable")]
    lo, hi = 0, len(ids)
    best = np.zeros(env.shape, bool)
    while lo <= hi:
        n = (lo + hi) // 2
        r = np.zeros(env.shape, bool)
        r.ravel()[ids[:n]] = True
        r = supported_removal(env, r)
        _, count = result(r)
        if count >= target:
            best = r
            lo = n + 1
        else:
            hi = n - 1
    return best, result(best)[0]


def project_floor_copy(env, masks, scores, operation, net_guided):
    f = env.frontier
    r, a = masks.copy()
    mw = min_room_width(env)
    if r.any():
        r = supported_removal(env, r)
        unstacked = unstack_cells(env, r)
        r_lat = r & ~unstacked
        if r_lat.any():
            previous = f.components
            try:
                f.components = max(2, previous)
                action = f.project_mask(r_lat, scores[0], -1, int(r_lat.sum()))
            finally:
                f.components = previous
            if action is None and not unstacked.any():
                return None
            r = (action.mask if action is not None else np.zeros_like(r)) | unstacked
            r = supported_removal(env, r)
        elif unstacked.any():
            r = supported_removal(env, unstacked)
        else:
            r[:] = False
    if operation == "remove" and not r.any():
        return None
    if net_guided and operation == "exchange":
        if not r.any():
            return None
        r, a = balance_copy(env, r, a, scores[0])
    # Preserve the resulting supported plate or coherent tower sub-plate.
    if a.any():
        for z in range(1, env.nz):
            a[z] &= ~r[z - 1]
        if env.plot_area >= 400:
            levels = np.flatnonzero(a.any((1, 2)))
            if len(levels) == 1:
                z = int(levels[0])
                if mw > 1:
                    a[z] = clean_floor_numpy(a[z], mw)
                alab, an = ndi.label(a[z], ndi.generate_binary_structure(2, 1))
                if an > 1:
                    sizes = np.bincount(alab.ravel())[1:]
                    a[z] = alab == (int(np.argmax(sizes)) + 1)
                if not coherent_plate(
                    a[z], min_plate_area(env, z), mw, _allowed_slim_2d(env, z)
                ):
                    a[:] = False
            else:
                a[:] = False
    if operation == "remove" and not r.any() or operation == "grow" and not a.any():
        return None
    if operation == "exchange" and (not r.any() or not a.any()):
        return None
    after = env.count - int(r.sum()) + int(a.sum())
    if net_guided:
        low, high = env.acceptable_counts
        if env.count > high and not low <= after < env.count:
            return None
        if env.count < low and not env.count < after <= high:
            return None
        if low <= env.count <= high and not low <= after <= high:
            return None
    move = Exchange(r, a, f.version)
    return move if valid_move(env, move) else None


def valid_move(env, move, maximum_buildings=2):
    f = env.frontier
    r, a = move.remove, move.add
    if move.version != f.version or r.shape != env.shape or a.shape != env.shape:
        return False
    if not (r.any() or a.any()) or (r & a).any():
        return False
    floor_copy = getattr(env, "floor_copy_growth", False)
    organic_3d = getattr(env, "organic_3d_growth", False)
    unstacked = (
        unstack_cells(env, r) if (floor_copy and r.any()) else np.zeros(env.shape, bool)
    )
    if (r & ~(f.eligible(-1) | unstacked)).any() or (a & ~growth_mask(env)).any():
        return False
    for mask in (r, a):
        if not mask.any():
            continue
        labels, n = ndi.label(mask, np.ones((3, 3, 3), bool))
        if f.config.maximum_patches and n > f.config.maximum_patches:
            return False
        minimum = (
            1
            if f.config.allocation == "feedback"
            and int(mask.sum())
            <= f.config.minimum_patch * max(1, f.config.maximum_patches)
            else f.config.minimum_patch
        )
        if (np.bincount(labels.ravel())[1:] < minimum).any():
            return False
    changed = (f.occupied & ~r) | a
    if floor_copy:
        if (
            f.config.maximum_edit_voxels
            and int(r.sum() + a.sum()) > f.config.maximum_edit_voxels
        ):
            return False
        if (changed[1:] & ~changed[:-1]).any():
            return False
        if not np.array_equal(r, supported_removal(env, r)):
            return False
        mw = min_room_width(env)
        if (
            mw > 1
            and (
                (changed & ~clean_floor_numpy(changed, mw))
                & clean_floor_numpy(f.occupied, mw)
            ).any()
        ):
            return False
        if a.any():
            levels = np.flatnonzero(a.any((1, 2)))
            if len(levels) != 1 or levels[0] < 1:
                return False
            z = int(levels[0])
            if not coherent_plate(
                a[z], min_plate_area(env, z), mw, _allowed_slim_2d(env, z)
            ):
                return False
            plates = copy_plates(env)
            r_prev = r[z - 1]

            def _matches(plate):
                sup = plate[z] & ~r_prev
                if env.plot_area < 400:
                    return np.array_equal(a[z], sup)
                return plate[z].any() and not bool((a[z] & ~sup).any())

            if not any(_matches(plate) for plate in plates):
                return False
    elif organic_3d:
        if (
            f.config.maximum_edit_voxels
            and int(r.sum() + a.sum()) > f.config.maximum_edit_voxels
        ):
            return False
        mw = min_room_width(env)
        if mw > 1:
            clean_changed = clean_floor_numpy(changed, mw)
            clean_old = clean_floor_numpy(f.occupied, mw)
            if ((changed & ~clean_changed) & (clean_old | a)).any():
                return False
    labels, n = ndi.label(changed, STRUCTURE)
    grounded = np.unique(labels[0])
    grounded = grounded[grounded != 0]
    if not 0 < n <= maximum_buildings or len(grounded) != n:
        return False
    air = exterior_air(changed)
    if (r & ~air).any() or (f.exterior_air & ~a & ~air).any():
        return False
    if f.old_voids is not None:
        for z in range(env.nz):
            lab, ids, sizes = f._floor_voids(changed[z], z)
            oldlab, oldids, oldsizes = f.old_voids[z]
            if f.config.maximum_voids and len(ids) > max(
                f.config.maximum_voids, len(oldids)
            ):
                return False
            for i in ids:
                prior = [
                    oldsizes[j] for j in np.unique(oldlab[lab == i]) if j in oldids
                ]
                if sizes[i] < min([f.config.minimum_void_area, *prior]):
                    return False
    return True


def _remaining_organic_valid(
    env, f, r, a, changed, air_after, mw, clean_old, rem_elig=None, g_mask=None
):
    """Verify remaining non-duplicate organic 3D checks when connectivity and exterior air are known."""
    zs = changed.shape[0]
    if not (r.any() or a.any()) or (r & a).any():
        return False
    rem_s = f.eligible(-1)[:zs] if rem_elig is None else rem_elig
    g_s = growth_mask(env)[:zs] if g_mask is None else g_mask
    if (r & ~rem_s).any() or (a & ~g_s).any():
        return False
    if (
        f.config.maximum_edit_voxels
        and int(r.sum() + a.sum()) > f.config.maximum_edit_voxels
    ):
        return False
    for mask in (r, a):
        if not mask.any():
            continue
        labels, n = ndi.label(mask, np.ones((3, 3, 3), bool))
        if f.config.maximum_patches and n > f.config.maximum_patches:
            return False
        minimum = (
            1
            if f.config.allocation == "feedback"
            and int(mask.sum())
            <= f.config.minimum_patch * max(1, f.config.maximum_patches)
            else f.config.minimum_patch
        )
        if (np.bincount(labels.ravel())[1:] < minimum).any():
            return False
    if mw > 1:
        clean_changed = clean_floor_numpy(changed, mw)
        if ((changed & ~clean_changed) & (clean_old | a)).any():
            return False
    if (r & ~air_after).any() or (f.exterior_air[:zs] & ~a & ~air_after).any():
        return False
    if f.old_voids is not None:
        for z in range(zs):
            if not (r[z].any() or a[z].any()):
                continue
            lab, ids, sizes = f._floor_voids(changed[z], z)
            oldlab, oldids, oldsizes = f.old_voids[z]
            if f.config.maximum_voids and len(ids) > max(
                f.config.maximum_voids, len(oldids)
            ):
                return False
            for i in ids:
                prior = [
                    oldsizes[j] for j in np.unique(oldlab[lab == i]) if j in oldids
                ]
                if sizes[i] < min([f.config.minimum_void_area, *prior]):
                    return False
    return True


def project_organic_3d(env, masks, scores, operation, net_guided, maximum_buildings=2):
    """Prune organic 3D masks for floor usability, connectivity, exterior air, and FAR guidance."""
    f = env.frontier
    mw = min_room_width(env)
    z_occ = np.flatnonzero((f.occupied | masks[1]).any((1, 2)))
    zs = min(env.nz, int(z_occ[-1]) + 2) if len(z_occ) else min(env.nz, 1)
    occ_s = f.occupied[:zs]
    r = masks[0, :zs].copy()
    a = masks[1, :zs].copy()
    s0 = scores[0, :zs]
    s1 = scores[1, :zs]
    skin_s = f.remove_skin[:zs]
    f_ext_s = f.exterior_air[:zs]
    rem_s = f.eligible(-1)[:zs]
    g_s = growth_mask(env)[:zs]
    clean_old = clean_floor_numpy(occ_s, mw) if mw > 1 else None
    low, high = env.acceptable_counts
    for _ in range(6):
        if r.any():
            if (r & ~skin_s).any():
                r &= exterior_air(occ_s & ~r)
            r = f._patch_filter(r, int(r.sum())) if r.any() else r
            if mw > 1 and r.any():
                for _ in range(3):
                    after_r = occ_s & ~r
                    slim_r = (after_r & ~clean_floor_numpy(after_r, mw)) & clean_old
                    if not slim_r.any():
                        break
                    zone = ndi.binary_dilation(
                        slim_r, structure=np.ones((1, 3, 3), bool)
                    )
                    if not (r & zone).any():
                        break
                    r &= ~zone
                r = f._patch_filter(r, int(r.sum())) if r.any() else r
        if a.any():
            a = f._patch_filter(a, int(a.sum())) if a.any() else a
            if mw > 1 and a.any():
                cand = (occ_s & ~r) | a
                cleaned = a & clean_floor_numpy(cand, mw)
                if not np.array_equal(cleaned, a):
                    a = (
                        f._patch_filter(cleaned, int(cleaned.sum()))
                        if cleaned.any()
                        else cleaned
                    )
        if operation == "remove" and not r.any():
            return None
        if operation == "grow" and not a.any():
            return None
        if operation == "exchange" and (not r.any() or not a.any()):
            return None
        changed = (occ_s & ~r) | a
        labels, n = ndi.label(changed, STRUCTURE)
        grounded = np.unique(labels[0])
        grounded = grounded[grounded != 0]
        if n == 0:
            return None
        if len(grounded) == 0 and r[0].any():
            r[0] = False
            continue
        if len(grounded) != n or n > maximum_buildings:
            sizes = np.bincount(labels.ravel())
            keep_ids = (
                grounded[np.argsort(-sizes[grounded])[:maximum_buildings]]
                if len(grounded)
                else np.empty(0, int)
            )
            keep_mask = np.isin(labels, keep_ids)
            bad_solid = changed & ~keep_mask
            if bad_solid.any():
                bad_nbr = ndi.binary_dilation(bad_solid, structure=STRUCTURE)
                repaired = False
                if r.any() and (r & bad_nbr).any():
                    bridge = (
                        r
                        & ndi.binary_dilation(keep_mask, structure=STRUCTURE)
                        & bad_nbr
                        if keep_mask.any()
                        else np.zeros_like(r)
                    )
                    if not bridge.any() and keep_mask.any():
                        bridge = (
                            r
                            & ndi.binary_dilation(
                                keep_mask, structure=STRUCTURE, iterations=2
                            )
                            & bad_nbr
                        )
                    cut_fix = bridge if bridge.any() else (r & bad_nbr)
                    r &= ~cut_fix
                    repaired = True
                if a.any() and (a & bad_solid).any():
                    a &= ~bad_solid
                    repaired = True
                if repaired:
                    continue
        air_after = exterior_air(changed)
        sealed = f_ext_s & ~a & ~air_after
        if sealed.any():
            bad_air = ndi.binary_dilation(sealed, structure=STRUCTURE)
            if (a & bad_air).any():
                a &= ~bad_air
                continue
        if net_guided:
            after = env.count - int(r.sum()) + int(a.sum())
            if env.count > high:
                req_lo, req_hi = low, env.count - 1
            elif env.count < low:
                req_lo, req_hi = env.count + 1, high
            else:
                req_lo, req_hi = low, high
            if after < req_lo:
                need_cut = int(r.sum()) - (req_lo - after)
                if need_cut <= 0:
                    return None
                ids = np.flatnonzero(r)
                keep = ids[np.argsort(-s0.ravel()[ids], kind="stable")[:need_cut]]
                r[:] = False
                r.ravel()[keep] = True
                continue
            if after > req_hi:
                need_add = int(a.sum()) - (after - req_hi)
                if need_add <= 0:
                    return None
                ids = np.flatnonzero(a)
                keep = ids[np.argsort(-s1.ravel()[ids], kind="stable")[:need_add]]
                a[:] = False
                a.ravel()[keep] = True
                continue
        if not _remaining_organic_valid(
            env, f, r, a, changed, air_after, mw, clean_old, rem_s, g_s
        ):
            return None
        if zs == env.nz:
            return Exchange(r.copy(), a.copy(), f.version)
        r_full = np.zeros(env.shape, bool)
        r_full[:zs] = r
        a_full = np.zeros(env.shape, bool)
        a_full[:zs] = a
        return Exchange(r_full, a_full, f.version)
    r_full = np.zeros(env.shape, bool)
    r_full[:zs] = r
    a_full = np.zeros(env.shape, bool)
    a_full[:zs] = a
    move = Exchange(r_full, a_full, f.version)
    return move if valid_move(env, move, maximum_buildings=maximum_buildings) else None


def project(env, masks, scores, operation, repair_remove=False, net_guided=False):
    """Only prune neural selections. Exchanges balance; pure moves change FAR."""
    r, a = masks.copy()
    f = env.frontier
    if getattr(env, "floor_copy_growth", False):
        return project_floor_copy(env, masks, scores, operation, net_guided)
    if getattr(env, "organic_3d_growth", False):
        return project_organic_3d(env, masks, scores, operation, net_guided)
    if repair_remove and (
        operation == "remove" or net_guided and operation == "exchange"
    ):
        # Reuse the proven selection-pruning path; permit two retained grounded
        # masses rather than inheriting the original one-component assumption.
        previous = f.components
        try:
            f.components = max(2, previous)
            action = f.project_mask(r, scores[0], -1, int(r.sum()))
        finally:
            f.components = previous
        if action is None:
            return None
        r = action.mask.copy()
        if operation == "remove":
            move = Exchange(r, a, f.version)
            return move if valid_move(env, move) else None
    for _ in range(4):
        r &= exterior_air(f.occupied & ~r)
        r = f._patch_filter(r, int(r.sum())) if r.any() else r
        a = f._patch_filter(a, int(a.sum())) if a.any() else a
        if operation == "exchange":
            nr, na = int(r.sum()), int(a.sum())
            delta = env.count - env.goal_count
            if net_guided and delta > 0:
                na = min(na, nr)
                nr = min(nr, na + delta)
            elif net_guided and delta < 0:
                nr = min(nr, na)
                na = min(na, nr - delta)
            else:
                nr = na = min(nr, na)
            k = min(nr, na)
            if not k:
                return None
            for mask, confidence, limit in ((r, scores[0], nr), (a, scores[1], na)):
                if mask.sum() > limit:
                    ids = np.flatnonzero(mask)
                    keep = ids[
                        np.argsort(-confidence.ravel()[ids], kind="stable")[:limit]
                    ]
                    mask[:] = False
                    mask.ravel()[keep] = True
        if operation == "remove" and not r.any() or operation == "grow" and not a.any():
            return None
        move = Exchange(r.copy(), a.copy(), f.version)
        if valid_move(env, move):
            return move
    return None


def commit(env, move, audit=False, validated=False):
    if not validated and not valid_move(env, move):
        raise ValueError("Invalid/stale unified edit")
    f = env.frontier
    if getattr(env, "floor_copy_growth", False):
        labels = getattr(env, "copy_labels", np.zeros(env.shape, dtype=np.int64)).copy()
        next_id = int(labels.max()) + 1
        for z in range(1, env.nz):
            plate = move.add[z]
            if not plate.any():
                continue
            fresh = plate & (labels[z - 1] == 0)
            n = int(fresh.sum())
            labels[z - 1][fresh] = np.arange(next_id, next_id + n)
            next_id += n
            labels[z][plate] = labels[z - 1][plate]
        labels[move.remove] = 0
        occ_after = (f.occupied & ~move.remove) | move.add
        has_above = np.zeros(env.shape, bool)
        has_below = np.zeros(env.shape, bool)
        has_above[:-1] = (labels[:-1] > 0) & (labels[:-1] == labels[1:]) & occ_after[1:]
        has_below[1:] = (labels[1:] > 0) & (labels[1:] == labels[:-1]) & occ_after[:-1]
        labels[~(has_above | has_below)] = 0
        env.copy_labels = labels
    f.occupied[move.remove] = False
    f.occupied[move.add] = True
    f.version += 1
    f.refresh()
    env.occupied.masked_fill_(torch.as_tensor(move.remove, device=env.device), False)
    env.occupied.masked_fill_(torch.as_tensor(move.add, device=env.device), True)
    env.count += int(move.add.sum()) - int(move.remove.sum())
    env.edits += 1
    env.last_changed_voxels = int(move.add.sum() + move.remove.sum())
    env.voxels_edited += env.last_changed_voxels
    env.last_operation = (
        "exchange"
        if move.remove.any() and move.add.any()
        else "remove"
        if move.remove.any()
        else "grow"
    )
    env.termination = None  # FAR is acceptance, not an automatic episode boundary.
    env.update_depth()
    if audit:
        env.evaluate()


class UnifiedNetwork(Network):
    def __init__(self):
        super().__init__(10)
        self.goal_encoder = nn.Linear(5, 24)
        self.critic = nn.Sequential(nn.Linear(53, 32), nn.SiLU(), nn.Linear(32, 1))
        self.operation_head = nn.Linear(53, 4)
        self.patch_log_std = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.operation_head.weight)
        nn.init.zeros_(self.operation_head.bias)
        # Learnable initial tendency toward FAR, never an availability/phase
        # rule. Exchange and upward growth remain possible above target.
        with torch.no_grad():
            self.operation_head.weight[0, 48:50] = torch.tensor([8.0, -8.0])
            self.operation_head.weight[1, 48:50] = torch.tensor([-8.0, 8.0])

    def forward(self, observation, goal):
        encoded = self.encoder(observation)
        floor = encoded.mean((-1, -2))
        context = torch.cat((floor.mean(0), floor.amax(0), goal))
        fields = self.mask_head(
            encoded + self.goal_encoder(goal)[None, :, None, None]
        ).permute(1, 0, 2, 3)
        return fields, self.critic(context).squeeze(), self.operation_head(context)


class Organic3DNetwork(nn.Module):
    """Fast no-permute 2D+1D volumetric actor-critic with tiled vertical sliding window."""

    def __init__(self, input_channels=12):
        super().__init__()
        self.input_channels = int(input_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 24, 3, stride=2, padding=1), nn.SiLU()
        )
        self.down = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU())
        self.ctx_xy1 = nn.Conv2d(32, 32, 3, padding=1)
        self.ctx_z1 = nn.Conv2d(32, 96, 1, bias=False)
        self.ctx_z1_bias = nn.Parameter(torch.zeros(32))
        self.ctx_xy2 = nn.Conv2d(32, 32, 3, padding=1)
        self.ctx_z2 = nn.Conv2d(32, 96, 1, bias=False)
        self.ctx_z2_bias = nn.Parameter(torch.zeros(32))
        self.goal_encoder = nn.Linear(5, 32)
        self.local_skip = nn.Conv2d(input_channels, 16, 1)
        self.head = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1), nn.SiLU(), nn.Conv2d(24, 2, 1)
        )
        self.critic = nn.Sequential(nn.Linear(69, 32), nn.SiLU(), nn.Linear(32, 1))
        self.operation_head = nn.Linear(69, 4)
        self.patch_log_std = nn.Parameter(torch.full((2,), -0.4))
        z_bound = 1.0 / math.sqrt(32 * 3)
        for conv, bias in (
            (self.ctx_z1, self.ctx_z1_bias),
            (self.ctx_z2, self.ctx_z2_bias),
        ):
            nn.init.uniform_(conv.weight, -z_bound, z_bound)
            nn.init.uniform_(bias, -z_bound, z_bound)
        nn.init.zeros_(self.goal_encoder.weight)
        nn.init.zeros_(self.goal_encoder.bias)
        nn.init.normal_(self.head[-1].weight, std=0.05)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.operation_head.weight)
        nn.init.zeros_(self.operation_head.bias)
        with torch.no_grad():
            self.operation_head.weight[0, 64:66] = torch.tensor([3.0, -3.0])
            self.operation_head.weight[1, 64:66] = torch.tensor([-3.0, 3.0])
            self.operation_head.bias[2] = 0.6

    @property
    def actor(self):
        return self.head

    @staticmethod
    def _zmix(v, conv, bias, B, Z):
        vc, vp, vn = conv(v).view(B, Z, 96, v.shape[-2], v.shape[-1]).chunk(3, dim=2)
        u = vc + bias.view(1, 1, 32, 1, 1)
        if Z > 1:
            z0 = torch.zeros_like(vp[:, :1])
            u = (
                u
                + torch.cat((z0, vp[:, :-1]), dim=1)
                + torch.cat((vn[:, 1:], z0), dim=1)
            )
        return u.reshape(B * Z, 32, v.shape[-2], v.shape[-1])

    def _encode_window(self, obs_w, goal_b):
        B, Zw, C, H, W = obs_w.shape
        flat = obs_w.reshape(B * Zw, C, H, W)
        x = self.down(self.stem(flat))
        v = F.silu(self.ctx_xy1(x))
        v = F.silu(v + self._zmix(v, self.ctx_z1, self.ctx_z1_bias, B, Zw))
        v = F.silu(self.ctx_xy2(v))
        v = F.silu(v + self._zmix(v, self.ctx_z2, self.ctx_z2_bias, B, Zw))
        return v.view(B, Zw, 32, v.shape[-2], v.shape[-1])

    def _head_slice(self, v_keep, obs_keep, goal_exp):
        B, Zk, C, H, W = obs_keep.shape
        flat_k = obs_keep.reshape(B * Zk, C, H, W)
        vk = v_keep.reshape(B * Zk, 32, v_keep.shape[-2], v_keep.shape[-1])
        ky, kx = min(4, H), min(4, W)
        oc = F.avg_pool2d(
            flat_k,
            kernel_size=(ky, kx),
            stride=4,
            ceil_mode=True,
            count_include_pad=False,
        )
        if oc.shape[-2:] != vk.shape[-2:]:
            oc = F.interpolate(
                oc, size=vk.shape[-2:], mode="bilinear", align_corners=False
            )
        loc = F.silu(self.local_skip(oc))
        gk = goal_exp.expand(-1, Zk, -1, -1, -1).reshape(B * Zk, 32, 1, 1)
        cond = torch.cat((vk + gk, loc), dim=1)
        return (
            self.head(cond)
            .view(B, Zk, 2, vk.shape[-2], vk.shape[-1])
            .permute(0, 2, 1, 3, 4)
        )

    def forward_coarse_batch(self, obs_b, goal_b, stride=(2, 4, 4), window_floors=16):
        B, Z, C, H, W = obs_b.shape
        win = max(6, int(window_floors))
        halo = 2
        goal_exp = self.goal_encoder(goal_b)[:, None, :, None, None]
        if Z <= win:
            v_bz = self._encode_window(obs_b, goal_b)
            fl = v_bz.mean((-1, -2))
            context = torch.cat((fl.mean(1), fl.amax(1), goal_b), dim=-1)
            coarse_z = self._head_slice(v_bz, obs_b, goal_exp)
        else:
            cz_parts = []
            floor_sum = None
            floor_max = None
            k0 = 0
            while k0 < Z:
                w_start = max(0, k0 - halo)
                w_end = min(Z, w_start + win)
                k1 = Z if w_end == Z else w_end - halo
                keep_s = k0 - w_start
                keep_e = k1 - w_start
                v_win = self._encode_window(obs_b[:, w_start:w_end], goal_b)
                v_keep = v_win[:, keep_s:keep_e]
                fl = v_keep.mean((-1, -2))
                f_sum = fl.sum(1)
                f_max = fl.amax(1)
                floor_sum = f_sum if floor_sum is None else (floor_sum + f_sum)
                floor_max = (
                    f_max if floor_max is None else torch.maximum(floor_max, f_max)
                )
                cz_parts.append(self._head_slice(v_keep, obs_b[:, k0:k1], goal_exp))
                k0 = k1
            coarse_z = torch.cat(cz_parts, dim=2)
            context = torch.cat((floor_sum / Z, floor_max, goal_b), dim=-1)
        target = (
            max(1, math.ceil(Z / stride[0])),
            max(1, math.ceil(H / stride[1])),
            max(1, math.ceil(W / stride[2])),
        )
        if tuple(stride) == (2, 4, 4) and coarse_z.shape[-2:] == target[1:]:
            kz = min(2, coarse_z.shape[2])
            coarse = F.avg_pool3d(
                coarse_z,
                (kz, 1, 1),
                stride=(2, 1, 1),
                ceil_mode=True,
                count_include_pad=False,
            )
        else:
            coarse = F.interpolate(
                coarse_z.reshape(B * 2, 1, Z, coarse_z.shape[-2], coarse_z.shape[-1]),
                size=target,
                mode="trilinear",
                align_corners=False,
            ).reshape(B, 2, *target)
        coarse = 3.0 * torch.tanh(coarse / 3.0)
        return coarse, self.critic(context).squeeze(-1), self.operation_head(context)

    def forward_coarse(self, observation, goal, stride=(2, 4, 4), window_floors=16):
        coarse, value, op_logits = self.forward_coarse_batch(
            observation[None], goal[None], stride=stride, window_floors=window_floors
        )
        return coarse[0], value[0], op_logits[0]

    def forward(self, observation, goal):
        coarse, value, op_logits = self.forward_coarse(observation, goal)
        fields = F.interpolate(
            coarse[:, None],
            size=(observation.shape[0], observation.shape[-2], observation.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )[:, 0]
        return fields, value, op_logits


def _ensure_external_potential(e):
    if not hasattr(e, "external_potential"):
        e.precompute_external()
        normals = e.suns.new_tensor(
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        )
        w = (e.suns @ normals.T).clamp_min(0) * e.suns.new_tensor(
            (3.0, 3.0, 3.0, 3.0, 1.0, 1.0)
        )[None]
        e.external_potential = (
            e.static_receivers.values.reshape(len(e.suns), 6, *e.shape).float()
            * w[:, :, None, None, None]
        ).sum((0, 1)) / w.sum().clamp_min(1)


def _organic_goal_features(e):
    limit = getattr(e, "unified_limit", 1)
    spent = getattr(e, "unified_actions", e.edits)
    g_scale = max(10.0, 3.0 * float(e.target_far))
    return [
        min(2.0, float(e.far) / g_scale),
        min(2.0, float(e.target_far) / g_scale),
        min(2.0, float(e.far_error) / g_scale),
        max(0, limit - spent) / max(1, limit),
        spent / max(1, limit),
    ]


def _pooled_edit_eligibility(w_rem, w_add, consumed, kernel, stride):
    eligible = torch.stack(((w_rem > 0).float(), (w_add > 0).float()))[consumed]
    return F.max_pool3d(eligible[:, None], kernel, stride=stride, ceil_mode=True)[:, 0]


def compute_spatial_desirability(
    e, w_rem, w_add, ln6_f, consumed, kernel, stride, z_slice=None
):
    """Orientation-neutral room-scale voxel desirability on the coarse latent grid for training credit assignment."""
    with torch.no_grad():
        occ = e.occupied[:z_slice] if z_slice is not None else e.occupied
        ep = (
            e.external_potential[:z_slice]
            if z_slice is not None
            else e.external_potential
        )
        fl = e.facade_light[:z_slice] if z_slice is not None else e.facade_light
        usable = (e.usable[:z_slice] if z_slice is not None else e.usable).float()
        depth_bad = (
            e.depth_badness[:z_slice] if z_slice is not None else e.depth_badness
        )
        light = e.light[:z_slice] if z_slice is not None else e.light
        core_m = None
        daylight_cells = occ
        if getattr(e, "cores_enabled", False) and hasattr(e, "core_mask"):
            core_mask = (
                e.core_mask[:z_slice] if z_slice is not None else e.core_mask
            ) & occ
            core_m = core_mask.float()
            daylight_cells = occ & ~core_mask
            fl = fl.masked_fill(core_mask, 0.0)
        daylight_weight = daylight_cells.float()
        occ_den = F.avg_pool2d(daylight_weight[:, None], 7, stride=1, padding=3)[
            :, 0
        ].clamp_min(1e-4)
        ep_room = (
            F.avg_pool2d((ep * daylight_weight)[:, None], 7, stride=1, padding=3)[:, 0]
            / occ_den
        )
        for _ in range(2):
            fl = torch.where(
                fl > 0, fl, F.max_pool2d(fl[:, None], 3, stride=1, padding=1)[:, 0]
            )
            if core_m is not None:
                fl = fl.masked_fill(core_mask, 0.0)
        fl_room = (
            F.avg_pool2d((fl * daylight_weight)[:, None], 7, stride=1, padding=3)[:, 0]
            / occ_den
        )
        usable_good = usable * (1.0 - ln6_f)
        if core_m is not None:
            core_d = (
                e.core_distance[:z_slice] if z_slice is not None else e.core_distance
            )
            core_bad = (
                e.core_dist_badness[:z_slice]
                if z_slice is not None
                else e.core_dist_badness
            )
            over_core = (core_d > float(getattr(e, "max_core_dist", 12))).float()
            usable_good = usable_good * (1.0 - core_m) * (1.0 - over_core)
            grounded = (occ.to(torch.int32).cumprod(dim=0) > 0).float()
            bay_solid = F.avg_pool2d(grounded[:, None], 5, stride=1, padding=2)[:, 0]
            q_occ = (
                0.65 * usable_good
                + 1.10 * core_m
                + (1.0 - core_m) * (1.30 * (0.5 * ep + 0.5 * ep_room) + 0.55 * fl_room)
                - 0.30 * depth_bad
                - 0.80 * ln6_f
                - 0.75 * over_core
                - 1.50 * core_bad
                - 0.95 * (1.0 - grounded)
                + 0.85 * bay_solid * core_bad
            )
            core_above = torch.zeros_like(core_m)
            if core_m.shape[0] > 1:
                core_above[1:] = core_m[:-1]
            grounded_below = torch.ones_like(grounded)
            if grounded.shape[0] > 1:
                grounded_below[1:] = grounded[:-1]
            q_add = (
                (1.0 - core_above) * (0.75 * ep + 0.25 * light)
                + 0.45 * core_above
                + 0.40 * grounded_below
            )
        else:
            q_occ = (
                0.55 * usable_good
                + 1.40 * (0.5 * ep + 0.5 * ep_room)
                + 0.55 * fl_room
                - 0.30 * depth_bad
                - 0.80 * ln6_f
            )
            q_add = 0.75 * ep + 0.25 * light
        des_list = []
        for ch in consumed:
            w_m = (w_rem > 0).float() if ch == 0 else (w_add > 0).float()
            n_e = w_m.sum().clamp_min(1.0)
            q = q_occ if ch == 0 else q_add
            mu = (q * w_m).sum() / n_e
            sd = (((q - mu).square() * w_m).sum() / n_e).sqrt().clamp_min(1e-4)
            d_full = ((mu - q) / sd if ch == 0 else (q - mu) / sd) * w_m
            num_c = F.avg_pool3d(
                d_full[None, None],
                kernel,
                stride=stride,
                ceil_mode=True,
                count_include_pad=False,
            )[0, 0]
            den_c = F.avg_pool3d(
                w_m[None, None],
                kernel,
                stride=stride,
                ceil_mode=True,
                count_include_pad=False,
            )[0, 0]
            des_c = torch.where(
                den_c > 1e-6, num_c / den_c.clamp_min(1e-6), torch.zeros_like(num_c)
            )
            des_list.append(des_c)
        return torch.stack(des_list, dim=0)


def _learned_quota(score, weights, max_count, min_step, floor_frac=0.35):
    """Cap thresholded positive-logit voxel selection within [min_count, max_count]."""
    min_count = min(max_count, max(min_step, round(floor_frac * max_count)))
    pos = int(((score > 0.5 * weights) & (weights > 0)).sum().item())
    return min(max_count, max(min_count, pos))


class UnifiedLearner(Learner):
    def __init__(
        self,
        device="cpu",
        seed=123,
        config=None,
        amount_feedback=False,
        repair_remove=True,
        net_guided=False,
        floor_copy=False,
        organic_3d=False,
        spatial_aux_weight=1.2,
        decoder_retry_schedule="fixed",
    ):
        if floor_copy and organic_3d:
            raise ValueError("Cannot enable both floor_copy and organic_3d")
        if decoder_retry_schedule not in ("fixed", "adaptive", "learned_cap"):
            raise ValueError("Invalid decoder retry schedule")
        super().__init__(device, seed)
        self.config = config or PatchConfig(mode="combined", allocation="feedback")
        self.amount_feedback = bool(amount_feedback)
        self.repair_remove = bool(repair_remove)
        self.net_guided = bool(net_guided)
        self.floor_copy = bool(floor_copy)
        self.organic_3d = bool(organic_3d)
        self.spatial_aux_weight = float(spatial_aux_weight)
        self.decoder_retry_schedule = str(decoder_retry_schedule)
        self.evaluation_temperature = 2.5
        self.evaluation_noise_scale = 0.08
        if not math.isfinite(self.spatial_aux_weight) or self.spatial_aux_weight < 0:
            raise ValueError("Invalid auxiliary weight")
        if self.config.mode not in ("combined", "correlated"):
            raise ValueError("Unified policy requires correlated masks")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = (
                Organic3DNetwork(14) if self.organic_3d else UnifiedNetwork()
            ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1.2e-3 if self.organic_3d else 1e-3,
            weight_decay=1e-4 if self.organic_3d else 0.0,
        )

    def observation(self, e, z_slice=None):
        # Cache immutable external exposure once. Keep direct sun and diffuse
        # facade access separate; both edit eligibility fields are explicit.
        _ensure_external_potential(e)
        remove = e.frontier.weights_tensor(-1, self.device)
        grow = torch.as_tensor(
            copy_eligibility(e) if self.floor_copy else growth_mask(e),
            device=self.device,
            dtype=torch.float32,
        )
        ep = e.external_potential
        fl = e.facade_light
        if z_slice is not None:
            remove = remove[:z_slice]
            grow = grow[:z_slice]
            ep = ep[:z_slice]
            fl = fl[:z_slice]
        channels = [
            e.observation(include_depth=True, z_slice=z_slice),
            remove[:, None],
            grow[:, None],
            ep[:, None],
            fl[:, None],
        ]
        if self.organic_3d:
            occ_sub = e.occupied[:z_slice] if z_slice is not None else e.occupied
            usb_sub = e.usable[:z_slice] if z_slice is not None else e.usable
            if e.plot_area >= 400:
                c_key = (e.frontier.version, z_slice)
                cache = getattr(e, "_ln6_cache", None)
                if cache is not None and cache[0] == c_key:
                    ln6_bool = cache[1]
                else:
                    allowed_ln6 = _allowed_long_narrow(e, 6, 2)
                    aln_sub = (
                        allowed_ln6[:z_slice] if z_slice is not None else allowed_ln6
                    )
                    ln6_bool = long_narrow_mask(occ_sub, 6, 2) & ~aln_sub
                    e._ln6_cache = (c_key, ln6_bool)
            else:
                ln6_bool = torch.zeros_like(occ_sub, dtype=torch.bool)
            if getattr(e, "cores_enabled", False) and hasattr(e, "core_mask"):
                cm_sub = (
                    e.core_mask[:z_slice] if z_slice is not None else e.core_mask
                ) & occ_sub
                cb_sub = torch.where(
                    occ_sub,
                    e.core_dist_badness[:z_slice]
                    if z_slice is not None
                    else e.core_dist_badness,
                    0.0,
                )
                core_ch = (cm_sub.float()[:, None], cb_sub[:, None])
            else:
                z_zero = torch.zeros_like(occ_sub, dtype=torch.float32)[:, None]
                core_ch = (z_zero, z_zero)
            channels.extend(
                (ln6_bool.float()[:, None], usb_sub.float()[:, None], *core_ch)
            )
        return torch.cat(channels, 1)

    def _organic_goal(self, e):
        return torch.tensor(
            _organic_goal_features(e), device=self.device, dtype=torch.float32
        )

    def forward(self, e):
        if self.organic_3d:
            return self.model(self.observation(e), self._organic_goal(e))
        limit = getattr(e, "unified_limit", 1)
        spent = getattr(e, "unified_actions", e.edits)
        goal = torch.tensor(
            [
                *e.goal_features,
                max(0, limit - spent) / max(1, limit),
                spent / max(1, limit),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        return self.model(self.observation(e), goal)

    def fields(self, e):
        fields, value, _ = self.forward(e)
        return fields, value

    def legal_operations(self, e, net_guided=None):
        guided = self.net_guided if net_guided is None else net_guided
        r = e.frontier.eligible(-1).any()
        a = (copy_eligibility(e) if self.floor_copy else growth_mask(e)).any()
        remove = r
        grow = a
        if guided:
            low, high = e.acceptable_counts
            remove = r and e.count > low
            grow = a and e.count < high
        if self.floor_copy and guided:

            def _min_plate(p):
                if e.plot_area < 400:
                    return int(p.sum())
                pz = int(np.flatnonzero(p.any((1, 2)))[0])
                return min(int(p.sum()), min_plate_area(e, pz))

            grow = any(
                e.count + _min_plate(plate) <= e.acceptable_counts[1]
                for plate in copy_plates(e)
            )
        allow_exchange = bool(r and a)
        if (
            self.organic_3d
            and self.decoder_retry_schedule == "learned_cap"
            and (remove or grow)
            and (
                abs(float(e.far) - float(e.target_far)) > 0.75
                or (
                    not e.far_satisfied
                    and (
                        getattr(e, "unified_limit", 8)
                        - getattr(e, "unified_actions", e.edits)
                    )
                    <= 4
                )
            )
        ):
            allow_exchange = False
        return torch.tensor(
            [remove, grow, allow_exchange, e.far_satisfied],
            device=self.device,
            dtype=torch.bool,
        )

    def choose(self, e, training=True, force_operation=None, retries=8):
        e.floor_copy_growth = self.floor_copy
        e.organic_3d_growth = self.organic_3d
        if self.organic_3d:
            return self._choose_organic_3d(
                e, training=training, force_operation=force_operation, retries=retries
            )
        start = time.perf_counter()
        self.last_attempt_recorded = False
        self.last_operation = None
        self.last_rejection = None
        self.last_generated_count = 0
        with torch.set_grad_enabled(training):
            fields, value, logits = self.forward(e)
            guided = self.net_guided and force_operation is None
            legal = self.legal_operations(e, net_guided=guided)
            if training and e.edits == 0 and bool(legal[:3].any()):
                legal = legal.clone()
                legal[3] = False
            if not bool(legal.any()):
                self.last_rejection = "no_admissible_edits"
                return None
            distribution = Categorical(logits=logits.masked_fill(~legal, float("-inf")))
            if force_operation is None:
                op = distribution.sample()
            else:
                if training:
                    raise ValueError(
                        "Forced operations are geometry probes, not on-policy training"
                    )
                op = torch.tensor(OPERATIONS.index(force_operation), device=self.device)
                if not bool(legal[op]):
                    raise ValueError("Requested operation is unavailable")
            operation = OPERATIONS[int(op.cpu())]
            self.last_operation = operation
            logp = distribution.log_prob(op)
            entropy = distribution.entropy()
            move = None
            logs = []
            if operation != "finish":
                weights = torch.stack(
                    (
                        e.frontier.weights_tensor(-1, self.device),
                        torch.as_tensor(
                            copy_eligibility(e) if self.floor_copy else growth_mask(e),
                            device=self.device,
                            dtype=torch.float32,
                        ),
                    )
                )
                kernel = tuple(min(n, s) for n, s in zip(e.shape, self.config.stride))
                mean = F.avg_pool3d(
                    fields[:, None],
                    kernel,
                    stride=self.config.stride,
                    ceil_mode=True,
                    count_include_pad=False,
                )[:, 0]
                consumed = (
                    [0]
                    if operation == "remove"
                    else [1]
                    if operation == "grow"
                    else [0, 1]
                )
                dist = Normal(
                    mean[consumed],
                    self.model.patch_log_std[consumed]
                    .clamp(-2, 1)
                    .exp()[:, None, None, None],
                    validate_args=False,
                )
                remaining = abs(e.count - e.goal_count)
                left = max(
                    1,
                    getattr(e, "unified_limit", 8)
                    - getattr(e, "unified_actions", e.edits)
                    - 4,
                )
                toward = (operation == "remove" and e.count > e.goal_count) or (
                    operation == "grow" and e.count < e.goal_count
                )
                amount = (
                    max(4, math.ceil(remaining / left / 0.65))
                    if toward
                    else max(4, round(e.target_count * 0.025))
                )
                if toward and self.amount_feedback:
                    productive = max(
                        1,
                        e.workload.limit
                        - e.workload.margin
                        - getattr(e, "unified_actions", e.edits),
                    )
                    amount = max(
                        4, math.ceil(remaining / productive / e.workload.retention)
                    )
                if toward:
                    cap_band = (
                        (e.count - e.acceptable_counts[0])
                        if (
                            self.floor_copy
                            and guided
                            and operation == "remove"
                            and e.count > e.goal_count
                        )
                        else remaining
                    )
                    amount = min(amount, cap_band)
                if guided and e.far_satisfied:
                    low, high = e.acceptable_counts
                    if operation == "remove":
                        amount = min(amount, e.count - low)
                    elif operation == "grow":
                        amount = min(amount, high - e.count)
                net_amount = 0
                if guided and operation == "exchange" and remaining:
                    productive = max(
                        1,
                        e.workload.limit
                        - e.workload.margin
                        - getattr(e, "unified_actions", e.edits),
                    )
                    net_amount = min(
                        remaining,
                        math.ceil(remaining / productive / e.workload.retention),
                    )
                if self.config.maximum_edit_voxels:
                    amount = min(amount, self.config.maximum_edit_voxels)
                plates = (
                    copy_plates(e) if self.floor_copy and operation != "remove" else []
                )
                if (
                    self.floor_copy
                    and e.plot_area >= 400
                    and not hasattr(e, "external_potential")
                ):
                    e.precompute_external()
                mw = min_room_width(e) if self.floor_copy else 1
                clean_occ = clean_floor_tensor(e.occupied, mw) if mw > 1 else None
                skin = (
                    torch.as_tensor(e.frontier.remove_skin, device=self.device)
                    if (self.floor_copy and e.plot_area >= 400)
                    else None
                )
                for attempt in range(retries):
                    latent = dist.sample()
                    decoded = F.interpolate(
                        latent[:, None],
                        size=e.shape,
                        mode="trilinear",
                        align_corners=False,
                    )[:, 0]
                    if self.config.mode == "combined":
                        decoded = gaussian_field(decoded, self.config.sigma)
                    scores = torch.zeros_like(fields)
                    masks = torch.zeros(
                        fields.shape, device=self.device, dtype=torch.bool
                    )
                    count = max(
                        1,
                        round(
                            amount
                            * (1.0, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0625)[
                                min(attempt, 7)
                            ]
                        ),
                    )
                    # Decode growth as a coherent floor plate or tower sub-plate.
                    plate = None
                    if plates:
                        add_field = decoded[consumed.index(1)].sigmoid()
                        if e.plot_area >= 400:
                            ext_pos = 0.5 * (
                                e.external_potential + e.facade_light
                            ).clamp(0, 1)
                            depth_col = e.depth_badness.mean(0, keepdim=True)
                            add_field = (
                                0.50 * add_field
                                + 0.30 * e.external_potential
                                + 0.20 * e.facade_light
                                - 0.25 * depth_col
                                + 0.05 * ext_pos
                            ).clamp_min(0)
                        plate_scores = torch.stack(
                            [
                                add_field[torch.as_tensor(p, device=self.device)].mean()
                                for p in plates
                            ]
                        )
                        plate = plates[int(plate_scores.argmax().cpu())]
                        if e.plot_area >= 400:
                            pz = int(np.flatnonzero(plate.any((1, 2)))[0])
                            desired_tower = max(
                                min_plate_area(e, pz), int(round(0.30 * e.plot_area))
                            )
                            if guided and operation == "grow":
                                desired_tower = min(
                                    desired_tower,
                                    max(
                                        min_plate_area(e, pz),
                                        e.acceptable_counts[1] - e.count,
                                    ),
                                )
                            if int(plate.sum()) > desired_tower:
                                candidate = extract_tower_plate(
                                    e, plate, add_field, desired_area=desired_tower
                                )
                                if candidate.any():
                                    plate = candidate
                        masks[1] = torch.as_tensor(plate, device=self.device)
                        scores[1] = add_field * weights[1]
                    for index, channel in enumerate(consumed):
                        if plate is not None and channel == 1:
                            continue
                        field = decoded[index]
                        if channel == 0 and self.config.vertical_interior:
                            interior = torch.as_tensor(
                                e.frontier.eligible(-1) & ~e.frontier.facade_band,
                                device=self.device,
                            )
                            field = torch.where(
                                interior,
                                field.mean(0, keepdim=True).expand_as(field),
                                field,
                            )
                        base_score = field.sigmoid()
                        if self.floor_copy and channel == 0 and e.plot_area >= 400:
                            light_combo = 0.5 * (
                                e.external_potential + e.facade_light
                            ).clamp(0, 1)
                            surf_light = light_combo * skin.float()
                            for _ in range(2):
                                prop = F.max_pool2d(
                                    surf_light[:, None], 3, stride=1, padding=1
                                )[:, 0]
                                surf_light = torch.where(
                                    surf_light > 0, surf_light, prop
                                )
                            light_combo = torch.where(
                                skin,
                                light_combo,
                                torch.maximum(light_combo, surf_light),
                            )
                            base_score = (
                                0.55 * base_score
                                + 0.25 * e.depth_badness
                                + 0.20 * (1.0 - light_combo)
                            )
                        scores[channel] = base_score * weights[channel]
                        available = int((weights[channel] > 0).sum().cpu())
                        if self.floor_copy and channel == 0:
                            # Rank cuts with upper-tail column averages so dark/deep shafts
                            # carve full-height while upper floors can still step back.
                            occupied = e.occupied
                            group_ids = torch.as_tensor(
                                getattr(
                                    e, "copy_labels", np.zeros(e.shape, dtype=np.int64)
                                ),
                                device=self.device,
                            ).flatten()
                            if bool((group_ids > 0).any()):
                                # Native tensor reduction gives linked cells the
                                # same neural score before selecting a mask.
                                total = torch.zeros(
                                    int(group_ids.max().cpu()) + 1, device=self.device
                                )
                                count_group = torch.zeros_like(total)
                                total.scatter_add_(
                                    0, group_ids, scores[channel].flatten()
                                )
                                count_group.scatter_add_(
                                    0,
                                    group_ids,
                                    torch.ones_like(scores[channel].flatten()),
                                )
                                shared = (total / count_group.clamp_min(1))[
                                    group_ids
                                ].reshape(e.shape)
                                scores[channel] = torch.where(
                                    group_ids.reshape(e.shape) > 0,
                                    shared,
                                    scores[channel],
                                )
                            occ_f = occupied.float()
                            col_mean = (scores[channel] * occ_f).sum(
                                0, keepdim=True
                            ) / occ_f.sum(0, keepdim=True).clamp_min(1.0)
                            col_score = torch.where(
                                occupied,
                                0.85 * col_mean + 0.15 * scores[channel],
                                torch.zeros_like(field),
                            )
                            col_score = torch.where(
                                occupied & (weights[channel] <= 0),
                                torch.zeros_like(field),
                                col_score,
                            )
                            if mw > 1 and skin is not None:
                                s_occ = torch.where(
                                    occupied, col_score, torch.zeros_like(col_score)
                                )
                                bound = torch.where(
                                    skin, s_occ, torch.zeros_like(s_occ)
                                )
                                near = skin
                                for _ in range(2):
                                    nbr_b = F.max_pool2d(
                                        bound[:, None], 3, stride=1, padding=1
                                    )[:, 0]
                                    nbr_n = (
                                        F.max_pool2d(
                                            near.float()[:, None],
                                            3,
                                            stride=1,
                                            padding=1,
                                        )[:, 0]
                                        > 0.5
                                    )
                                    bound = torch.where(near, bound, nbr_b * 0.999)
                                    near = near | nbr_n
                                inner = (weights[channel] > 0) & ~skin & near
                                col_score = torch.where(
                                    inner, torch.minimum(col_score, bound), col_score
                                )
                            col_score = torch.where(
                                occupied, col_score, torch.ones_like(field)
                            )
                            ordered = torch.cummin(
                                col_score.flip(0), dim=0
                            ).values.flip(0)
                            height = (
                                torch.arange(e.nz, device=self.device)[:, None, None]
                                * 1e-6
                            )
                            scores[channel] = torch.where(
                                weights[channel] > 0, ordered + height, 0
                            )
                        extra = (
                            net_amount
                            if channel == (0 if e.count > e.goal_count else 1)
                            else 0
                        )
                        n = min(
                            count
                            + round(
                                extra
                                * (1.0, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0625)[
                                    min(attempt, 7)
                                ]
                            ),
                            available,
                        )
                        if plate is not None and channel == 0:
                            # Remove enough area to offset the copied plate and
                            # move toward FAR. Retry scaling changes net work,
                            # never fragments the addition into random roof posts.
                            delta = e.count - e.goal_count if self.net_guided else 0
                            net = round(
                                net_amount
                                * (1.0, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0625)[
                                    min(attempt, 7)
                                ]
                            )
                            plate_need = (
                                math.ceil(
                                    int(plate.sum())
                                    / max(0.6, min(1.0, e.workload.retention))
                                )
                                if (self.amount_feedback and delta > 0)
                                else int(plate.sum())
                            )
                            desired = plate_need + (
                                net
                                if delta > 0
                                else -min(net, int(plate.sum()) - 1)
                                if delta < 0
                                else 0
                            )
                            n = min(available, max(1, desired))
                        if self.config.maximum_edit_voxels:
                            n = min(
                                n,
                                max(
                                    0,
                                    self.config.maximum_edit_voxels - int(plate.sum()),
                                )
                                if plate is not None
                                else self.config.maximum_edit_voxels,
                            )
                        ids = (
                            scores[channel]
                            .masked_fill(weights[channel] <= 0, -1)
                            .flatten()
                            .topk(n)
                            .indices
                        )
                        masks[channel].flatten()[ids] = True
                        if mw > 1 and channel == 0 and n > 0:
                            if guided and (e.far_satisfied or e.count > e.goal_count):
                                max_n = min(
                                    available,
                                    max(
                                        n,
                                        (e.count - e.acceptable_counts[0])
                                        + (
                                            int(plate.sum()) if plate is not None else 0
                                        ),
                                    ),
                                )
                            elif e.count > e.goal_count:
                                max_n = min(
                                    available,
                                    max(
                                        n,
                                        remaining
                                        + (
                                            int(plate.sum()) if plate is not None else 0
                                        ),
                                    ),
                                )
                            else:
                                max_n = min(
                                    available,
                                    max(
                                        n, int(plate.sum()) if plate is not None else n
                                    ),
                                )
                            if self.config.maximum_edit_voxels:
                                max_n = min(
                                    max_n,
                                    max(
                                        0,
                                        self.config.maximum_edit_voxels
                                        - int(plate.sum()),
                                    )
                                    if plate is not None
                                    else self.config.maximum_edit_voxels,
                                )
                            s_adj = scores[channel].clone()
                            for _ in range(2):
                                after_t = e.occupied & ~masks[channel]
                                slim_t = (
                                    after_t & ~clean_floor_tensor(after_t, mw)
                                ) & clean_occ
                                if not bool(slim_t.any()):
                                    break
                                add_cand = (
                                    (
                                        torch.cummax(
                                            (slim_t & (weights[channel] > 0)).int(),
                                            dim=0,
                                        ).values
                                        > 0
                                    )
                                    & e.occupied
                                    & (weights[channel] > 0)
                                )
                                if bool((group_ids > 0).any()):
                                    g_hit = torch.zeros(
                                        int(group_ids.max().cpu()) + 1,
                                        device=self.device,
                                    )
                                    g_hit.scatter_add_(
                                        0, group_ids, add_cand.flatten().float()
                                    )
                                    g_hit[0] = 0
                                    add_cand = (
                                        (
                                            add_cand
                                            | (g_hit[group_ids].reshape(e.shape) > 0)
                                        )
                                        & e.occupied
                                        & (weights[channel] > 0)
                                    )
                                new_bits = add_cand & ~masks[channel]
                                if not bool(new_bits.any()):
                                    break
                                n_eff = min(
                                    max_n,
                                    int(masks[channel].sum().cpu())
                                    + int(new_bits.sum().cpu()),
                                )
                                s_adj = torch.where(
                                    add_cand | masks[channel], s_adj.amax() + 1.0, s_adj
                                )
                                s_adj = torch.where(
                                    e.occupied & (weights[channel] > 0),
                                    s_adj,
                                    torch.zeros_like(s_adj),
                                )
                                s_adj = torch.cummin(
                                    torch.where(
                                        e.occupied, s_adj, torch.ones_like(s_adj)
                                    ).flip(0),
                                    dim=0,
                                ).values.flip(0)
                                s_adj = torch.where(
                                    weights[channel] > 0, s_adj + height, 0
                                )
                                ids = (
                                    s_adj.masked_fill(weights[channel] <= 0, -1)
                                    .flatten()
                                    .topk(n_eff)
                                    .indices
                                )
                                masks[channel].zero_()
                                masks[channel].flatten()[ids] = True
                            scores[channel] = s_adj
                    if training:
                        logs.append(
                            dist.log_prob(latent).sum() / math.sqrt(latent.numel())
                        )
                    self.last_generated_count = int(masks.sum().cpu())
                    move = project(
                        e,
                        masks.cpu().numpy(),
                        scores.detach().cpu().numpy(),
                        operation,
                        self.repair_remove,
                        guided,
                    )
                    if move is not None:
                        break
                if training:
                    logp = logp + torch.stack(logs).sum()
                    entropy = entropy + dist.entropy().mean()
                if move is None:
                    self.last_rejection = "mask_generation_failed"
            if training:
                self.tape.append([logp, value, entropy, None])
                self.last_attempt_recorded = True
        self.generation_seconds = time.perf_counter() - start
        return move

    def _choose_organic_3d(self, e, training=True, force_operation=None, retries=8):
        start = time.perf_counter()
        self.last_attempt_recorded = False
        self.last_operation = None
        self.last_rejection = None
        self.last_generated_count = 0
        with torch.no_grad():
            occ_z = e.occupied.any(dim=(1, 2)).nonzero(as_tuple=False)
            z_top = int(occ_z[-1].item()) + 1 if len(occ_z) else 0
            z_act = min(e.nz, max(min(e.nz, 10), ((z_top + 4) // 4) * 4))
            obs = self.observation(e, z_slice=z_act)
            goal = self._organic_goal(e)
            mean_all, value, logits = self.model.forward_coarse(
                obs, goal, stride=self.config.stride
            )
            guided = self.net_guided and force_operation is None
            legal = self.legal_operations(e, net_guided=guided)
            if training and e.edits == 0 and bool(legal[:3].any()):
                legal = legal.clone()
                legal[3] = False
            if not bool(legal.any()):
                self.last_rejection = "no_admissible_edits"
                return None
            eff_logits = logits if training else logits * self.evaluation_temperature
            distribution = Categorical(
                logits=eff_logits.masked_fill(~legal, float("-inf"))
            )
            if force_operation is None:
                op = distribution.sample()
            else:
                if training:
                    raise ValueError(
                        "Forced operations are geometry probes, not on-policy training"
                    )
                op = torch.tensor(OPERATIONS.index(force_operation), device=self.device)
                if not bool(legal[op]):
                    raise ValueError("Requested operation is unavailable")
            operation = OPERATIONS[int(op.cpu())]
            self.last_operation = operation
            move = None
            latent = None
            consumed = None
            elig_c = None
            local_adv = None
            latent_draws = []
            if operation != "finish":
                w_rem = e.frontier.weights_tensor(-1, self.device)[:z_act]
                add_np = growth_mask(e)[:z_act]
                w_add = torch.as_tensor(add_np, device=self.device, dtype=torch.float32)
                avail_rem = int(e.frontier.eligible(-1)[:z_act].sum())
                avail_add = int(add_np.sum())
                occ_s = e.occupied[:z_act]
                kernel = tuple(
                    min(n, s) for n, s in zip(occ_s.shape, self.config.stride)
                )
                consumed = (
                    [0]
                    if operation == "remove"
                    else [1]
                    if operation == "grow"
                    else [0, 1]
                )
                mean = mean_all[consumed]
                std = (
                    self.model.patch_log_std[consumed]
                    .clamp(-1.4, -0.3)
                    .exp()[:, None, None, None]
                )
                dist = Normal(mean, std, validate_args=False)
                low, high = e.acceptable_counts
                remaining = abs(e.count - e.goal_count)
                if self.amount_feedback:
                    productive = max(
                        1,
                        e.workload.limit
                        - e.workload.margin
                        - getattr(e, "unified_actions", e.edits),
                    )
                    ret = max(0.55, min(1.0, e.workload.retention))
                    step_net = (
                        max(4, math.ceil(remaining / productive / ret))
                        if remaining
                        else 0
                    )
                else:
                    left = max(
                        1,
                        getattr(e, "unified_limit", 8)
                        - getattr(e, "unified_actions", e.edits)
                        - 4,
                    )
                    step_net = (
                        max(4, math.ceil(remaining / left / 0.65)) if remaining else 0
                    )
                reloc = max(
                    16,
                    min(
                        int(round(e.plot_area * 0.35)),
                        max(16, round(e.target_count * 0.08)),
                    ),
                )
                base_rem = 0
                base_add = 0
                if operation == "remove":
                    if e.count > e.goal_count:
                        cap_band = (e.count - low) if guided else remaining
                        base_rem = min(cap_band, step_net)
                    else:
                        base_rem = max(4, round(e.target_count * 0.04))
                        if guided and e.far_satisfied:
                            base_rem = min(base_rem, e.count - low)
                    base_rem = min(avail_rem, max(0, base_rem))
                elif operation == "grow":
                    if e.count < e.goal_count:
                        cap_band = (high - e.count) if guided else remaining
                        base_add = min(cap_band, step_net)
                    else:
                        base_add = max(4, round(e.target_count * 0.05))
                        if guided and e.far_satisfied:
                            base_add = min(base_add, high - e.count)
                    base_add = min(avail_add, max(0, base_add))
                else:
                    if guided and e.count > e.goal_count:
                        base_add = min(avail_add, reloc)
                        base_rem = min(
                            avail_rem, base_add + min(e.count - low, max(4, step_net))
                        )
                    elif guided and e.count < e.goal_count:
                        base_rem = min(avail_rem, reloc)
                        base_add = min(
                            avail_add, base_rem + min(high - e.count, max(4, step_net))
                        )
                    else:
                        both = min(reloc, avail_rem, avail_add)
                        base_rem = both
                        base_add = both
                if self.config.maximum_edit_voxels:
                    if operation == "exchange":
                        tot = base_rem + base_add
                        if tot > self.config.maximum_edit_voxels and tot > 0:
                            scale_cap = self.config.maximum_edit_voxels / tot
                            base_rem = max(1, int(base_rem * scale_cap))
                            base_add = max(
                                1, self.config.maximum_edit_voxels - base_rem
                            )
                    elif operation == "remove":
                        base_rem = min(base_rem, self.config.maximum_edit_voxels)
                    else:
                        base_add = min(base_add, self.config.maximum_edit_voxels)
                mw = min_room_width(e)
                clean_occ = clean_floor_tensor(occ_s, mw) if mw > 1 else None
                skin = (
                    torch.as_tensor(e.frontier.remove_skin[:z_act], device=self.device)
                    if mw > 1
                    else None
                )
                allowed_ln6 = (
                    _allowed_long_narrow(e, 6, 2)[:z_act]
                    if e.plot_area >= 400
                    else None
                )
                ln6_before = None
                if e.plot_area >= 400 and e.ny >= 6 and e.nx >= 6:
                    cache = getattr(e, "_ln6_cache", None)
                    if cache is not None and cache[0] == (e.frontier.version, z_act):
                        ln6_before = cache[1]
                    else:
                        ln6_before = long_narrow_mask(occ_s, 6, 2) & ~allowed_ln6
                ln6_f = (
                    ln6_before.float()
                    if ln6_before is not None
                    else torch.zeros_like(e.facade_light[:z_act])
                )
                has_core_guard = bool(
                    getattr(e, "cores_enabled", False) and getattr(e, "cores", ())
                )
                core_m_sub = (
                    e.core_mask[:z_act]
                    if (has_core_guard and hasattr(e, "core_mask"))
                    else None
                )
                active_cores = (
                    [
                        c
                        for c in e.cores
                        if int(c.floors) > 0
                        and e.target_count > c.size_y * c.size_x * int(c.floors) * 1.5
                    ]
                    if has_core_guard
                    else ()
                )
                learned_cap = self.decoder_retry_schedule == "learned_cap"
                scales = (
                    (1.0, 0.6, 0.36, 0.216, 0.1296, 0.07776, 0.046656, 0.027994)
                    if self.decoder_retry_schedule in ("adaptive", "learned_cap")
                    else (1.0, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0625)
                )
                rem_steps = getattr(e, "unified_limit", 8) - getattr(
                    e, "unified_actions", e.edits
                )
                quota_floor = (
                    1.0
                    if (not e.far_satisfied and rem_steps <= 4)
                    else (
                        0.35 if (e.far_satisfied or operation == "exchange") else 0.65
                    )
                )
                for attempt in range(retries):
                    latent = (
                        dist.sample()
                        if (training or attempt > 0)
                        else (
                            mean
                            + self.evaluation_noise_scale * std * torch.randn_like(mean)
                        )
                    )
                    if training:
                        latent_draws.append(latent.detach())
                    decoded = F.interpolate(
                        latent[:, None],
                        size=occ_s.shape,
                        mode="trilinear",
                        align_corners=False,
                    )[:, 0]
                    if self.config.mode == "combined":
                        decoded = gaussian_field(decoded, self.config.sigma)
                    sc = scales[min(attempt, 7)]
                    scores = torch.zeros(
                        (2, *occ_s.shape), device=self.device, dtype=torch.float32
                    )
                    masks = torch.zeros(
                        (2, *occ_s.shape), device=self.device, dtype=torch.bool
                    )
                    if 0 in consumed and avail_rem > 0 and base_rem > 0:
                        max_r = min(avail_rem, max(1, round(base_rem * sc)))
                        s0 = decoded[consumed.index(0)].sigmoid() * w_rem
                        if core_m_sub is not None:
                            s0 = torch.where(core_m_sub, s0 * 0.30, s0)
                        if mw > 1 and skin is not None:
                            bound = torch.where(skin, s0, torch.zeros_like(s0))
                            near = skin
                            for _ in range(max(1, self.config.facade_layers)):
                                nbr_b = F.max_pool2d(
                                    bound[:, None], 3, stride=1, padding=1
                                )[:, 0]
                                nbr_n = (
                                    F.max_pool2d(
                                        near.float()[:, None], 3, stride=1, padding=1
                                    )[:, 0]
                                    > 0.5
                                )
                                bound = torch.where(near, bound, nbr_b * 0.999)
                                near = near | nbr_n
                            inner = (w_rem > 0) & ~skin & near
                            s0 = torch.where(inner, torch.minimum(s0, bound), s0)
                        n_r = (
                            _learned_quota(s0, w_rem, max_r, 4, floor_frac=quota_floor)
                            if learned_cap
                            else max_r
                        )
                        scores[0] = s0
                        r_t = torch.zeros_like(occ_s)
                        ids = s0.masked_fill(w_rem <= 0, -1).flatten().topk(n_r).indices
                        r_t.flatten()[ids] = True
                        if mw > 1:
                            if ln6_before is not None and attempt < 3:
                                after_t = occ_s & ~r_t
                                new_ln6 = (
                                    long_narrow_mask(after_t, 6, 2)
                                    & ~ln6_before
                                    & ~allowed_ln6
                                )
                                zone6 = (
                                    F.max_pool2d(
                                        new_ln6.float()[:, None], 3, stride=1, padding=1
                                    )[:, 0]
                                    > 0.5
                                )
                                r_t = r_t & ~zone6
                            after_t = occ_s & ~r_t
                            slim_t = (
                                after_t & ~clean_floor_tensor(after_t, mw)
                            ) & clean_occ
                            r_t = r_t | (slim_t & (w_rem > 0))
                            for _ in range(2):
                                after_t = occ_s & ~r_t
                                slim_t = (
                                    after_t & ~clean_floor_tensor(after_t, mw)
                                ) & clean_occ
                                zone = (
                                    F.max_pool2d(
                                        slim_t.float()[:, None], 3, stride=1, padding=1
                                    )[:, 0]
                                    > 0.5
                                )
                                r_t = r_t & ~zone
                        if active_cores and bool(r_t.any()):
                            hits = []
                            best_keep = None
                            best_keep_sc = float("inf")
                            for kc in active_cores:
                                kcz = min(z_act, int(kc.floors))
                                sl = r_t[
                                    :kcz,
                                    kc.y0 : kc.y0 + kc.size_y,
                                    kc.x0 : kc.x0 + kc.size_x,
                                ]
                                hit = bool(sl.any())
                                hits.append(hit)
                                if hit:
                                    sc_val = float(
                                        s0[
                                            :kcz,
                                            kc.y0 : kc.y0 + kc.size_y,
                                            kc.x0 : kc.x0 + kc.size_x,
                                        ].mean()
                                    )
                                    if sc_val < best_keep_sc:
                                        best_keep_sc = sc_val
                                        best_keep = (kc, kcz)
                            if all(hits) and best_keep is not None:
                                kc, kcz = best_keep
                                r_t[
                                    :kcz,
                                    kc.y0 : kc.y0 + kc.size_y,
                                    kc.x0 : kc.x0 + kc.size_x,
                                ] = False
                        masks[0] = r_t
                    if 1 in consumed and avail_add > 0 and base_add > 0:
                        max_a = min(avail_add, max(1, round(base_add * sc)))
                        s1 = decoded[consumed.index(1)].sigmoid() * w_add
                        n_a = (
                            _learned_quota(s1, w_add, max_a, 4, floor_frac=quota_floor)
                            if learned_cap
                            else max_a
                        )
                        scores[1] = s1
                        a_t = torch.zeros_like(occ_s)
                        ids = s1.masked_fill(w_add <= 0, -1).flatten().topk(n_a).indices
                        a_t.flatten()[ids] = True
                        if mw > 1:
                            base_occ = occ_s & ~masks[0]
                            a_t = a_t & clean_floor_tensor(base_occ | a_t, mw)
                            if ln6_before is not None:
                                cand = base_occ | a_t
                                has_core6 = (
                                    F.avg_pool2d(cand.float()[:, None], 6, stride=1)
                                    > 1.0 - 1e-4
                                ).amax((-1, -2, -3))
                                new_ln6 = (
                                    long_narrow_mask(cand, 6, 2)
                                    & ~long_narrow_mask(base_occ, 6, 2)
                                    & ~allowed_ln6
                                    & has_core6[:, None, None]
                                )
                                a_t = a_t & ~new_ln6
                                a_t = a_t & clean_floor_tensor(base_occ | a_t, mw)
                        masks[1] = a_t
                    if z_act == e.nz:
                        masks_np = masks.cpu().numpy()
                        scores_np = scores.cpu().numpy()
                    else:
                        masks_np = np.zeros((2, *e.shape), dtype=bool)
                        scores_np = np.zeros((2, *e.shape), dtype=np.float32)
                        masks_np[:, :z_act] = masks.cpu().numpy()
                        scores_np[:, :z_act] = scores.cpu().numpy()
                    self.last_generated_count = int(masks_np.sum())
                    move = project_organic_3d(e, masks_np, scores_np, operation, guided)
                    if move is not None:
                        if training:
                            elig_c = _pooled_edit_eligibility(
                                w_rem, w_add, consumed, kernel, self.config.stride
                            )
                            des_c = compute_spatial_desirability(
                                e,
                                w_rem,
                                w_add,
                                ln6_f,
                                consumed,
                                kernel,
                                self.config.stride,
                                z_slice=z_act,
                            )
                            la_raw = (
                                (latent.sigmoid() - mean.detach().sigmoid())
                                * des_c
                                * elig_c
                            )
                            local_adv = torch.zeros_like(la_raw)
                            for c in range(la_raw.shape[0]):
                                w_c = elig_c[c]
                                n_c = w_c.sum().clamp_min(1.0)
                                sd_c = (
                                    (la_raw[c].square().sum() / n_c)
                                    .sqrt()
                                    .clamp_min(1e-4)
                                )
                                local_adv[c] = la_raw[c] / sd_c
                        break
                if move is None:
                    self.last_rejection = "mask_generation_failed"
            if training:
                self.tape.append(
                    dict(
                        obs=obs.detach(),
                        goal=goal.detach(),
                        op=op.detach(),
                        legal=legal.detach(),
                        value=value.detach(),
                        latent=latent.detach() if move is not None else None,
                        latents=latent_draws,
                        drawn_channels=consumed,
                        consumed=consumed if move is not None else None,
                        elig_c=elig_c.detach() if elig_c is not None else None,
                        local_adv=local_adv.detach() if local_adv is not None else None,
                        reward=None,
                        done=False,
                    )
                )
                self.last_attempt_recorded = True
        self.generation_seconds = time.perf_counter() - start
        return move

    def record(self, reward):
        if not self.organic_3d:
            return super().record(reward)
        if self.tape:
            self.tape[-1]["reward"] = torch.as_tensor(
                reward, dtype=torch.float32, device=self.device
            ).detach()

    def mark_done(self):
        if not self.tape:
            return
        last = self.tape[-1]
        if isinstance(last, dict):
            last["done"] = True
        elif len(last) >= 5:
            last[4] = True

    def add_reward(self, reward):
        if not self.tape:
            return
        r = torch.as_tensor(reward, dtype=torch.float32, device=self.device).detach()
        last = self.tape[-1]
        if isinstance(last, dict):
            last["reward"] = r if last["reward"] is None else (last["reward"] + r)
        else:
            if last[3] is None:
                last[3] = r
            else:
                last[3] = last[3] + r

    def finish(self, bootstrap=0.0):
        if not self.organic_3d:
            return super().finish(bootstrap)
        if not self.tape:
            return dict(
                updates=self.updates,
                directMaskUpdates=self.direct_updates,
                importedUpdates=self.imported_updates,
                loss=0.0,
                gradientNorm=0.0,
            )
        adv, returns = generalized_advantages(self.tape, bootstrap, self.device)

        groups = {}
        for t, entry in enumerate(self.tape):
            groups.setdefault(tuple(entry["obs"].shape), []).append(t)

        values_out = [None] * len(self.tape)
        ent_terms = []
        actor_terms = []
        stds = self.model.patch_log_std.clamp(-1.4, -0.3).exp()
        for _z, idxs in groups.items():
            obs_b = torch.stack([self.tape[t]["obs"] for t in idxs], dim=0)
            goal_b = torch.stack([self.tape[t]["goal"] for t in idxs], dim=0)
            mean_b, val_b, logits_b = self.model.forward_coarse_batch(
                obs_b, goal_b, stride=self.config.stride
            )
            for k, t in enumerate(idxs):
                entry = self.tape[t]
                values_out[t] = val_b[k]
                dist_op = Categorical(
                    logits=logits_b[k].masked_fill(~entry["legal"], float("-inf"))
                )
                logp = dist_op.log_prob(entry["op"])
                ent_terms.append(dist_op.entropy())
                a_ep = adv[t].detach()
                term = -(logp * a_ep)
                # The augmented action is the operation plus ALL Gaussian
                # draws consumed by decoding, including rejected retries. Score
                # the full latent grid: smoothing can carry information across
                # eligibility boundaries, so masking log probabilities is invalid.
                draws = entry.get("latents", []) or (
                    [entry["latent"]] if entry["latent"] is not None else []
                )
                if draws:
                    consumed = entry.get("drawn_channels") or entry["consumed"]
                    mean = mean_b[k, consumed]
                    std = stds[consumed][:, None, None, None]
                    dist_s = Normal(mean, std, validate_args=False)
                    path_logp = torch.stack(
                        [dist_s.log_prob(draw).sum() for draw in draws]
                    ).sum()
                    term = term - path_logp * a_ep
                    if entry["latent"] is not None and self.spatial_aux_weight:
                        elig_c = entry["elig_c"]
                        ecnt = elig_c.sum().clamp_min(1.0)
                        sp_lp = dist_s.log_prob(entry["latent"]) * elig_c
                        # This is an explicit physics-shaped auxiliary objective,
                        # not an unbiased control variate for episode return.
                        term = term - self.spatial_aux_weight * (
                            (sp_lp * entry["local_adv"]).sum() / ecnt.sqrt()
                        )
                actor_terms.append(term)
        values_t = torch.stack(values_out)
        actor = torch.stack(actor_terms).mean()
        critic = F.smooth_l1_loss(values_t, returns)
        ent_coef = 0.02 if self.decoder_retry_schedule == "learned_cap" else 0.005
        loss = actor + 0.5 * critic - ent_coef * torch.stack(ent_terms).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), 2.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.updates += 1
        self.direct_updates += 1
        result = dict(
            updates=self.updates,
            directMaskUpdates=self.direct_updates,
            importedUpdates=self.imported_updates,
            loss=float(loss.detach().cpu()),
            actorLoss=float(actor.detach().cpu()),
            gradientNorm=float(norm.detach().cpu()),
        )
        self.clear()
        return result

    def state(self):
        data = dict(
            format="voxelform.unified.v1",
            config=asdict(self.config),
            amountFeedback=self.amount_feedback,
            repairRemove=self.repair_remove,
            netGuided=self.net_guided,
            floorCopy=self.floor_copy,
            organic3d=self.organic_3d,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            updates=self.updates,
            directMaskUpdates=self.direct_updates,
            importedUpdates=0,
        )
        if self.organic_3d:
            data["organic3dChannels"] = 14
            data["organic3dArch"] = "coarse_2d1d_v4"
            data["samplingSemantics"] = "gaussian-retry-path.v1"
            data["spatialAuxWeight"] = self.spatial_aux_weight
            data["evaluationTemperature"] = self.evaluation_temperature
            data["evaluationNoiseScale"] = self.evaluation_noise_scale
            data["decoderRetrySchedule"] = self.decoder_retry_schedule
        return data

    def load(self, state):
        if state.get("format") != "voxelform.unified.v1":
            raise ValueError("Unified action checkpoint required")
        is_o3d = bool(state.get("organic3d", False))
        if is_o3d and (
            state.get("organic3dArch") != "coarse_2d1d_v4"
            or int(state.get("organic3dChannels", 0)) != 14
        ):
            raise ValueError("Incompatible organic 3D checkpoint architecture")
        other = UnifiedLearner(
            str(self.device),
            config=PatchConfig(**state["config"]),
            amount_feedback=state.get("amountFeedback", False),
            repair_remove=state.get("repairRemove", False),
            net_guided=state.get("netGuided", False),
            floor_copy=state.get("floorCopy", False),
            organic_3d=is_o3d,
            spatial_aux_weight=state.get("spatialAuxWeight", 1.2),
            decoder_retry_schedule=state.get("decoderRetrySchedule", "fixed"),
        )
        other.model.load_state_dict(state["model"])
        other.optimizer.load_state_dict(state["optimizer"])
        other.updates = int(state["updates"])
        other.direct_updates = int(state["directMaskUpdates"])
        other.evaluation_temperature = float(state.get("evaluationTemperature", 2.5))
        other.evaluation_noise_scale = float(state.get("evaluationNoiseScale", 0.08))
        self.__dict__.update(other.__dict__)
