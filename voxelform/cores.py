from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .core_plans import CORE_LAYOUT_VERSION, plan_library

import numpy as np
import torch
import torch.nn.functional as F

UNREACHABLE_DIST: float = 999.0
SQRT2: float = math.sqrt(2.0)
DEFAULT_MIN_CORE_SPACING: int = 10
DEFAULT_MAX_CORE_DIST: int = 12


@dataclass(frozen=True)
class CoreTemplate:
    tier: str
    label: str
    min_height: int
    max_height: int
    width: int
    length: int
    program: str
    zones: tuple[dict, ...] = ()
    attachment_sides: tuple[str, ...] = ()
    metric_plan: dict | None = None

    @property
    def area(self) -> int:
        return self.width * self.length

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "label": self.label,
            "minHeight": self.min_height,
            "maxHeight": self.max_height,
            "width": self.width,
            "length": self.length,
            "area": self.area,
            "program": self.program,
            "zones": [dict(z) for z in self.zones],
            "attachmentSides": list(self.attachment_sides),
            "layoutVersion": CORE_LAYOUT_VERSION,
            "plan": self.metric_plan,
        }


# One-metre reservations are rounded OUT from the metric component plans.
# The height schedule is a configurable research prior, never a code assertion.
CORE_TEMPLATES: tuple[CoreTemplate, ...] = tuple(
    CoreTemplate(tier=p['tier'],label=f"{p['minHeight']}-{p['maxHeight']}F ({p['width']}x{p['length']} m grid · {p['stairCount']} stair + {p['liftCount']} lift)",
        min_height=p['minHeight'],max_height=p['maxHeight'],width=p['width'],length=p['length'],
        program=f"{p['stairCount']} dog-leg stair(s), {p['liftCount']} lift(s); {p['metricWidth']:.2f}x{p['metricLength']:.2f} m metric enclosure",
        zones=tuple(p['zones']),attachment_sides=('S',),metric_plan=p)
    for p in plan_library()
)

@dataclass(frozen=True)
class PlacedCore:
    id: int
    tier: str
    label: str
    y0: int
    x0: int
    size_y: int
    size_x: int
    floors: int
    footprint_area: int
    volume: int
    center_y: float
    center_x: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier,
            "label": self.label,
            "y0": self.y0,
            "x0": self.x0,
            "sizeY": self.size_y,
            "sizeX": self.size_x,
            "floors": self.floors,
            "footprintArea": self.footprint_area,
            "volume": self.volume,
            "centerY": round(self.center_y, 2),
            "centerX": round(self.center_x, 2),
        }


@dataclass
class CoreResult:
    cores: tuple[PlacedCore, ...]
    core_mask: torch.Tensor
    core_distance: torch.Tensor
    core_dist_badness: torch.Tensor
    metrics: dict[str, float | int]


def template_for_height(floors: int) -> CoreTemplate | None:
    h = int(floors)
    if h <= 0:
        return None
    for tmpl in CORE_TEMPLATES:
        if tmpl.min_height <= h <= tmpl.max_height:
            return tmpl
    return None  # Do not silently extrapolate a lift schedule beyond its range.


def core_templates_summary() -> list[dict]:
    return [tmpl.to_dict() for tmpl in CORE_TEMPLATES]


def core_edge_gap(a: PlacedCore, b: PlacedCore) -> float:
    """Clear Euclidean edge-to-edge gap in voxels between two core footprints."""
    dy = max(0, b.y0 - (a.y0 + a.size_y), a.y0 - (b.y0 + b.size_y))
    dx = max(0, b.x0 - (a.x0 + a.size_x), a.x0 - (b.x0 + b.size_x))
    return math.hypot(dx, dy)


def compute_core_distance(
    occupied: torch.Tensor,
    core_mask: torch.Tensor,
    *,
    max_iters: int | None = None,
    return_device: torch.device | str | None = None,
) -> torch.Tensor:
    """Shortest per-floor 8-neighbor geodesic distance through occupied voxels.

    Supports both ``core_mask`` of shape ``(Z, Y, X)`` and batched candidate
    masks of shape ``(B, Z, Y, X)``. Deduplicates unique 2D ``(occupied, core)``
    floor slices and relaxes the graph on CPU before returning to ``return_device``.
    Default convergence is exact for this 8-neighbour graph (no corner cutting).
    An explicit max_iters bounds screening work and may leave distant cells unresolved.
    """
    out_device = occupied.device if return_device is None else torch.device(return_device)
    occ_cpu = occupied.detach().to("cpu", torch.bool)
    batched = core_mask.dim() == 4
    core_cpu = core_mask.detach().to("cpu", torch.bool)
    if not batched:
        core_cpu = core_cpu.unsqueeze(0)
    B = core_cpu.shape[0]
    Z, Y, X = occ_cpu.shape

    out = torch.zeros((B, Z, Y, X), dtype=torch.float32)
    active_z = torch.nonzero(occ_cpu.any(dim=(1, 2)), as_tuple=False)
    if active_z.numel() == 0:
        res = out if batched else out.squeeze(0)
        return res.to(out_device)

    z_top = int(active_z[-1].item()) + 1
    occ_sub = occ_cpu[:z_top]
    core_sub = core_cpu[:, :z_top] & occ_sub.unsqueeze(0)

    if not bool(core_sub.any().item()):
        out[:, :z_top] = torch.where(occ_sub.unsqueeze(0), UNREACHABLE_DIST, 0.0)
        res = out if batched else out.squeeze(0)
        return res.to(out_device)

    N = B * z_top
    occ_flat = occ_sub.unsqueeze(0).expand(B, -1, -1, -1).reshape(N, Y, X)
    core_flat = core_sub.reshape(N, Y, X)
    sub_out = torch.where(occ_flat, UNREACHABLE_DIST, 0.0)

    active_idx = torch.nonzero(core_flat.any(dim=(1, 2)) & occ_flat.any(dim=(1, 2)), as_tuple=False).flatten()
    if active_idx.numel() == 0:
        out[:, :z_top] = sub_out.view(B, z_top, Y, X)
        res = out if batched else out.squeeze(0)
        return res.to(out_device)

    occ_a = occ_flat[active_idx].numpy()
    core_a = core_flat[active_idx].numpy()
    if len(active_idx) > 1:
        packed = np.packbits(
            np.concatenate([occ_a.reshape(len(active_idx), -1), core_a.reshape(len(active_idx), -1)], axis=1),
            axis=1,
        )
        _, uniq_first, inv = np.unique(packed, axis=0, return_index=True, return_inverse=True)
        occ_u = torch.from_numpy(occ_a[uniq_first])
        core_u = torch.from_numpy(core_a[uniq_first])
    else:
        inv = None
        occ_u = torch.from_numpy(occ_a)
        core_u = torch.from_numpy(core_a)

    dist = torch.where(core_u, 0.0, UNREACHABLE_DIST)
    if max_iters is None:
        max_iters = Y * X  # A simple occupied path can exceed the bounding-box diameter.

    # Diagonal moves cannot squeeze through two blocked orthogonal neighbours.
    padded_occ=F.pad(occ_u,(1,1,1,1),value=False)
    on=padded_occ[:, :-2, 1:-1];os=padded_occ[:, 2:, 1:-1]
    ow=padded_occ[:, 1:-1, :-2];oe=padded_occ[:, 1:-1, 2:]
    diag_legal=(on&ow,on&oe,os&ow,os&oe)
    check_every = 6
    inf_val = UNREACHABLE_DIST
    for step in range(max_iters):
        p = F.pad(dist, (1, 1, 1, 1), value=inf_val)
        n = p[:, :-2, 1:-1]
        s = p[:, 2:, 1:-1]
        w = p[:, 1:-1, :-2]
        e = p[:, 1:-1, 2:]
        ortho = torch.minimum(torch.minimum(n, s), torch.minimum(w, e)) + 1.0

        nw = p[:, :-2, :-2]
        ne = p[:, :-2, 2:]
        sw = p[:, 2:, :-2]
        se = p[:, 2:, 2:]
        diags=[torch.where(ok,value,inf_val) for ok,value in zip(diag_legal,(nw,ne,sw,se))]
        diag = torch.minimum(torch.minimum(diags[0], diags[1]), torch.minimum(diags[2], diags[3])) + SQRT2

        cand = torch.minimum(dist, torch.minimum(ortho, diag))
        nxt = torch.where(occ_u, cand, inf_val)
        nxt = torch.where(core_u, 0.0, nxt)

        if (step + 1) % check_every == 0 and torch.equal(nxt, dist):
            dist = nxt
            break
        dist = nxt

    dist_u = torch.where(occ_u, dist, 0.0)
    if inv is not None:
        sub_out[active_idx] = dist_u[torch.from_numpy(inv)]
    else:
        sub_out[active_idx] = dist_u
    out[:, :z_top] = sub_out.view(B, z_top, Y, X)
    res = out if batched else out.squeeze(0)
    return res.to(out_device)


def core_distance_badness(
    core_distance: torch.Tensor,
    occupied: torch.Tensor,
    max_core_dist: float,
    core_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth [0, 1] penalty map for occupied voxels exceeding ``max_core_dist``."""
    occ = occupied.bool()
    limit = max(1.0, float(max_core_dist))
    scale = max(4.0, 0.5 * limit)
    excess = (core_distance - limit).clamp(min=0.0)
    over = core_distance > limit
    bad = torch.where(over, 0.15 + 0.85 * (excess / scale).clamp(0.0, 1.0), 0.0)
    bad = torch.where(core_distance >= (UNREACHABLE_DIST - 1.0), 1.0, bad)
    bad = torch.where(occ, bad, 0.0)
    if core_mask is not None:
        bad = torch.where(core_mask.bool(), 0.0, bad)
    return bad


def summarize_core_metrics(
    occupied: torch.Tensor,
    core_mask: torch.Tensor,
    core_distance: torch.Tensor,
    core_badness: torch.Tensor,
    cores: Sequence[PlacedCore],
    max_core_dist: float,
) -> dict[str, float | int]:
    occ = occupied.bool()
    occ_count = int(occ.sum().item())
    core_b = core_mask.bool() & occ
    core_voxels = int(core_b.sum().item())

    if occ_count == 0:
        return {
            "coreCount": len(cores),
            "coreVoxels": 0,
            "coreFootprintArea": sum(c.footprint_area for c in cores),
            "coreRatio": 0.0,
            "coreDistanceMean": 0.0,
            "coreDistanceMax": 0.0,
            "coreUncoveredFraction": 0.0,
            "coreUnreachableFraction": 0.0,
            "coreDistBadness": 0.0,
            "coreMinSpacingActual": 0.0,
        }

    limit = float(max_core_dist)
    reachable_occ = occ & (core_distance < (UNREACHABLE_DIST - 1.0))
    noncore_reachable = reachable_occ & ~core_b

    dist_mean = float(core_distance[noncore_reachable].mean().item()) if bool(noncore_reachable.any().item()) else 0.0
    dist_max = float(core_distance[reachable_occ].max().item()) if bool(reachable_occ.any().item()) else 0.0
    uncovered = occ & (core_distance > limit)
    unreachable = occ & (core_distance >= (UNREACHABLE_DIST - 1.0))

    min_gap = 0.0
    if len(cores) >= 2:
        gaps = [
            core_edge_gap(cores[i], cores[j])
            for i in range(len(cores))
            for j in range(i + 1, len(cores))
        ]
        min_gap = min(gaps)

    return {
        "coreCount": len(cores),
        "coreVoxels": core_voxels,
        "coreFootprintArea": sum(c.footprint_area for c in cores),
        "coreRatio": core_voxels / occ_count,
        "coreDistanceMean": dist_mean,
        "coreDistanceMax": dist_max,
        "coreUncoveredFraction": float(uncovered.sum().item()) / occ_count,
        "coreUnreachableFraction": float(unreachable.sum().item()) / occ_count,
        "coreDistBadness": float(core_badness[occ].mean().item()),
        "coreMinSpacingActual": min_gap,
    }


def _integral_2d(arr: np.ndarray) -> np.ndarray:
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)
    return ii


def _box_sum_centered(ii: np.ndarray, cy: np.ndarray, cx: np.ndarray, radius: int) -> np.ndarray:
    H = ii.shape[0] - 1
    W = ii.shape[1] - 1
    y0 = np.clip(cy - radius, 0, H)
    y1 = np.clip(cy + radius + 1, 0, H)
    x0 = np.clip(cx - radius, 0, W)
    x1 = np.clip(cx + radius + 1, 0, W)
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def _pick_spaced_indices(scores: np.ndarray, cy: np.ndarray, cx: np.ndarray, k: int, min_sep: float = 3.0) -> list[int]:
    order = np.argsort(-scores)
    chosen: list[int] = []
    for idx in order:
        i = int(idx)
        if scores[i] <= -1e15:
            break
        ok = True
        for prev in chosen:
            if math.hypot(float(cy[i] - cy[prev]), float(cx[i] - cx[prev])) < min_sep:
                ok = False
                break
        if ok:
            chosen.append(i)
            if len(chosen) >= k:
                break
    if not chosen and len(order) > 0 and scores[int(order[0])] > -1e15:
        chosen.append(int(order[0]))
    return chosen


def solve_cores(
    occupied: torch.Tensor,
    *,
    min_core_spacing: int = DEFAULT_MIN_CORE_SPACING,
    max_core_dist: int = DEFAULT_MAX_CORE_DIST,
    max_cores: int = 12,
) -> CoreResult:
    """Greedily place height-graded rectangular vertical cores and measure floor distance."""
    device = occupied.device
    occ_cpu = occupied.detach().to("cpu", torch.bool)
    Z, Y, X = occ_cpu.shape
    occ_count = int(occ_cpu.sum().item())

    if occ_count == 0:
        empty_b = torch.zeros((Z, Y, X), dtype=torch.bool, device=device)
        empty_f = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
        metrics = summarize_core_metrics(occ_cpu, empty_b.cpu(), empty_f.cpu(), empty_f.cpu(), (), max_core_dist)
        return CoreResult(cores=(), core_mask=empty_b, core_distance=empty_f, core_dist_badness=empty_f, metrics=metrics)

    # Continuous vertical column height anchored at ground z=0
    occ_np = occ_cpu.numpy()
    grounded_h = occ_np.cumprod(axis=0).sum(axis=0, dtype=np.int32)
    max_grounded = int(grounded_h.max())

    if max_grounded <= 0:
        core_mask_c = torch.zeros_like(occ_cpu, dtype=torch.bool)
        dist_c = compute_core_distance(occ_cpu, core_mask_c, return_device="cpu")
        bad_c = core_distance_badness(dist_c, occ_cpu, max_core_dist, core_mask_c)
        metrics = summarize_core_metrics(occ_cpu, core_mask_c, dist_c, bad_c, (), max_core_dist)
        return CoreResult(
            cores=(),
            core_mask=core_mask_c.to(device),
            core_distance=dist_c.to(device),
            core_dist_badness=bad_c.to(device),
            metrics=metrics,
        )

    occ_base_ii = _integral_2d((grounded_h > 0).astype(np.float64))

    # Enumerate every valid anchored footprint candidate across all tiers & orientations
    cand_tmpl: list[CoreTemplate] = []
    cand_y0: list[np.ndarray] = []
    cand_x0: list[np.ndarray] = []
    cand_sy: list[np.ndarray] = []
    cand_sx: list[np.ndarray] = []
    cand_h: list[np.ndarray] = []
    cand_attach: list[np.ndarray] = []
    cand_flush: list[np.ndarray] = []

    seen_dims: set[tuple[str, int, int]] = set()
    for tmpl in CORE_TEMPLATES:
        if tmpl.min_height > max_grounded:
            continue
        for rh, rw in ((tmpl.width, tmpl.length), (tmpl.length, tmpl.width)):
            key = (tmpl.tier, rh, rw)
            if key in seen_dims or rh > Y or rw > X:
                continue
            seen_dims.add(key)

            win = np.lib.stride_tricks.sliding_window_view(grounded_h, (rh, rw))
            foot_min = win.min(axis=(-2, -1))
            foot_max = win.max(axis=(-2, -1))
            over_cnt = (win > foot_min[..., None, None]).sum(axis=(-2, -1))
            max_over_cells = (rh * rw) // 3

            tier_match = (foot_min >= tmpl.min_height) & (
                (foot_min <= tmpl.max_height)
            )
            step_ok = (over_cnt == 0) | ((over_cnt <= max_over_cells) & ((foot_max - foot_min) <= 2))

            vy_len, vx_len = foot_min.shape
            yy, xx = np.meshgrid(np.arange(vy_len), np.arange(vx_len), indexing="ij")
            hy0 = np.clip(yy - 1, 0, Y)
            hy1 = np.clip(yy + rh + 1, 0, Y)
            hx0 = np.clip(xx - 1, 0, X)
            hx1 = np.clip(xx + rw + 1, 0, X)
            halo_occ = (
                occ_base_ii[hy1, hx1]
                - occ_base_ii[hy0, hx1]
                - occ_base_ii[hy1, hx0]
                + occ_base_ii[hy0, hx0]
            )
            attach = halo_occ - float(rh * rw)
            valid = tier_match & step_ok & (attach >= 1.0)

            ys, xs = np.nonzero(valid)
            if len(ys) == 0:
                continue

            cand_tmpl.extend([tmpl] * len(ys))
            cand_y0.append(ys.astype(np.int32))
            cand_x0.append(xs.astype(np.int32))
            cand_sy.append(np.full(len(ys), rh, dtype=np.int32))
            cand_sx.append(np.full(len(ys), rw, dtype=np.int32))
            cand_h.append(foot_min[ys, xs].astype(np.int32))
            cand_attach.append(attach[ys, xs].astype(np.float64))
            cand_flush.append((over_cnt[ys, xs] == 0).astype(np.float64))

    if not cand_tmpl:
        core_mask_c = torch.zeros_like(occ_cpu, dtype=torch.bool)
        dist_c = compute_core_distance(occ_cpu, core_mask_c, return_device="cpu")
        bad_c = core_distance_badness(dist_c, occ_cpu, max_core_dist, core_mask_c)
        metrics = summarize_core_metrics(occ_cpu, core_mask_c, dist_c, bad_c, (), max_core_dist)
        return CoreResult(
            cores=(),
            core_mask=core_mask_c.to(device),
            core_distance=dist_c.to(device),
            core_dist_badness=bad_c.to(device),
            metrics=metrics,
        )

    y0_arr = np.concatenate(cand_y0)
    x0_arr = np.concatenate(cand_x0)
    sy_arr = np.concatenate(cand_sy)
    sx_arr = np.concatenate(cand_sx)
    h_arr = np.concatenate(cand_h)
    att_arr = np.concatenate(cand_attach)
    flush_arr = np.concatenate(cand_flush)
    cy_arr = np.round(y0_arr + 0.5 * (sy_arr - 1)).astype(np.int32)
    cx_arr = np.round(x0_arr + 0.5 * (sx_arr - 1)).astype(np.int32)

    radius = max(1, int(max_core_dist))
    placed: list[PlacedCore] = []
    cur_core_mask = torch.zeros_like(occ_cpu, dtype=torch.bool)
    cur_dist = torch.where(occ_cpu, UNREACHABLE_DIST, 0.0)
    cur_uncovered_cnt = occ_count

    spacing_req = float(max(0, int(min_core_spacing)))
    spacing_sq = spacing_req * spacing_req - 1e-9
    floor_weights = (1.0 + 1.5 * np.arange(Z, dtype=np.float64) / max(1, Z - 1))[:, None, None]
    screen_iters = int(max_core_dist) + 3
    eligible = np.ones(len(cand_tmpl), dtype=bool)

    for _ in range(max_cores):
        need = (occ_cpu & (cur_dist > float(max_core_dist))).numpy()
        if not bool(need.any()):
            break

        if placed and spacing_req > 0.0:
            prev = placed[-1]
            dy = np.maximum(0, np.maximum(prev.y0 - (y0_arr + sy_arr), y0_arr - (prev.y0 + prev.size_y)))
            dx = np.maximum(0, np.maximum(prev.x0 - (x0_arr + sx_arr), x0_arr - (prev.x0 + prev.size_x)))
            eligible &= (dx * dx + dy * dy) >= spacing_sq
        if not bool(eligible.any()):
            break

        need_cols = need.sum(axis=0).astype(np.float64)
        high_need_cols = (need * floor_weights).sum(axis=0)
        unw_score = _box_sum_centered(_integral_2d(need_cols), cy_arr, cx_arr, radius)
        hi_score = _box_sum_centered(_integral_2d(high_need_cols), cy_arr, cx_arr, radius)

        base_bonus = 2.0 * flush_arr + 0.25 * np.minimum(att_arr, 12.0) + 0.05 * h_arr
        s_unw = np.where(eligible, unw_score + base_bonus, -1e18)
        s_hi = np.where(eligible, hi_score + 8.0 * h_arr + base_bonus, -1e18)

        shortlist_idx: list[int] = []
        seen_idx: set[int] = set()
        for pool_scores in (s_unw, s_hi):
            for idx in _pick_spaced_indices(pool_scores, cy_arr, cx_arr, k=2, min_sep=4.0):
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    shortlist_idx.append(idx)
                    if len(shortlist_idx) >= 3:
                        break
            if len(shortlist_idx) >= 3:
                break

        if not shortlist_idx:
            break

        # Evaluate candidate additions incrementally for <= max_core_dist coverage
        B = len(shortlist_idx)
        cand_only_masks = torch.zeros((B, Z, Y, X), dtype=torch.bool)
        for bi, ci in enumerate(shortlist_idx):
            yy0 = int(y0_arr[ci])
            xx0 = int(x0_arr[ci])
            ssy = int(sy_arr[ci])
            ssx = int(sx_arr[ci])
            ch = int(h_arr[ci])
            cand_only_masks[bi, :ch, yy0 : yy0 + ssy, xx0 : xx0 + ssx] = True

        new_dists = compute_core_distance(occ_cpu, cand_only_masks, max_iters=screen_iters, return_device="cpu")
        cand_dists = torch.minimum(cur_dist.unsqueeze(0), new_dists)
        cand_full_masks = cur_core_mask.unsqueeze(0) | cand_only_masks
        new_uncov_b = (occ_cpu.unsqueeze(0) & (cand_dists > float(max_core_dist))).sum(dim=(1, 2, 3))

        best_key: tuple[int, float, float, float, float, int] | None = None
        best_bi = -1
        best_ci = -1

        for bi, ci in enumerate(shortlist_idx):
            uncov = int(new_uncov_b[bi].item())
            gain = cur_uncovered_cnt - uncov
            ch = int(h_arr[ci])
            area = int(sy_arr[ci] * sx_arr[ci])
            vol = area * ch

            if placed and gain < max(area, vol // 3):
                continue

            dist_bi = cand_dists[bi]
            mask_bi = cand_full_masks[bi]
            bad_bi = core_distance_badness(dist_bi, occ_cpu, max_core_dist, mask_bi)
            bad_sum = float(bad_bi[occ_cpu].sum().item())
            reach_noncore = occ_cpu & ~mask_bi & (dist_bi < (UNREACHABLE_DIST - 1.0))
            mean_d = float(dist_bi[reach_noncore].mean().item()) if bool(reach_noncore.any().item()) else 0.0

            key = (
                uncov,
                round(bad_sum, 4),
                round(mean_d, 4),
                -float(flush_arr[ci]),
                -float(att_arr[ci]),
                ci,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_bi = bi
                best_ci = ci

        if best_bi < 0:
            break

        ci = best_ci
        tmpl = cand_tmpl[ci]
        yy0 = int(y0_arr[ci])
        xx0 = int(x0_arr[ci])
        ssy = int(sy_arr[ci])
        ssx = int(sx_arr[ci])
        ch = int(h_arr[ci])

        placed.append(
            PlacedCore(
                id=len(placed) + 1,
                tier=tmpl.tier,
                label=tmpl.label,
                y0=yy0,
                x0=xx0,
                size_y=ssy,
                size_x=ssx,
                floors=ch,
                footprint_area=ssy * ssx,
                volume=ssy * ssx * ch,
                center_y=yy0 + 0.5 * (ssy - 1),
                center_x=xx0 + 0.5 * (ssx - 1),
            )
        )
        cur_core_mask = cand_full_masks[best_bi]
        cur_dist = cand_dists[best_bi]
        cur_uncovered_cnt = int(new_uncov_b[best_bi].item())

    # Final converged grid-graph distance pass after greedy placement finishes
    if placed:
        cur_dist = compute_core_distance(occ_cpu, cur_core_mask, return_device="cpu")
    cur_bad = core_distance_badness(cur_dist, occ_cpu, max_core_dist, cur_core_mask)
    metrics = summarize_core_metrics(occ_cpu, cur_core_mask, cur_dist, cur_bad, placed, max_core_dist)
    return CoreResult(
        cores=tuple(placed),
        core_mask=cur_core_mask.to(device),
        core_distance=cur_dist.to(device),
        core_dist_badness=cur_bad.to(device),
        metrics=metrics,
    )
