"""Additive N-stage coarse-to-fine policy (4x4x1 / 2x2x1 -> 1x1x3 m fine) with 14-ch compact cores."""

from dataclasses import asdict
import math
import time
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from .actions import STRUCTURE, exterior_air
from .cores import DEFAULT_MAX_CORE_DIST, DEFAULT_MIN_CORE_SPACING, solve_cores
from .fields import depth_badness, diffuse_access, facade_sky_access, usable_depth_proxy
from .patch_policy import PatchConfig, PatchFacades, gaussian_field
from .redistribution import Exchange
from .decoder_diagnostics import DecoderDiagnostics
from .objectives import coarse_potential, fine_potential
from .rollout import finish_episode_rollout, generalized_advantages
from .shadows import sync
from .unified import (
    OPERATIONS,
    _ensure_external_potential,
    _organic_goal_features,
    _pooled_edit_eligibility,
    Organic3DNetwork,
    clean_floor_numpy,
    clean_floor_tensor,
    commit as commit_fine,
    compute_spatial_desirability,
    growth_mask,
    long_narrow_mask,
    min_room_width,
)
from .workload import WorkloadBudget


def tier_cascade_blocks(env, schedule="tier_adaptive"):
    """Select progressive coarse-to-fine block schedule by parcel area tier."""
    area = float(getattr(env, "plot_area", 0.0))
    if schedule == "tier_meso2":
        return () if area < 2500.0 else ((1, 2, 2),)
    if schedule == "tier_macro4":
        if area < 2500.0:
            return ()
        if area < 4100.0:
            return ((1, 4, 4), (1, 2, 2))
        return ((1, 8, 8), (1, 4, 4), (1, 2, 2))
    if schedule == "tier_adaptive_xl8":
        if area < 2500.0:
            return ()
        if area < 3800.0:
            return ((1, 2, 2),)
        if area < 4100.0:
            return ((1, 4, 4), (1, 2, 2))
        return ((1, 8, 8), (1, 4, 4), (1, 2, 2))
    if area < 2500.0:
        return ()
    if area < 3800.0:
        return ((1, 2, 2),)
    return ((1, 4, 4), (1, 2, 2))


def _heuristic_op_index(
    env_or_cenv, is_coarse, legal, low_f, high_f, remove, grow, can_finish
):
    cur_vol = env_or_cenv.count_fine if is_coarse else env_or_cenv.count
    mid_vol = 0.5 * (low_f + high_f)
    if can_finish and env_or_cenv.edits >= 1:
        return 3
    if remove and (not grow or cur_vol > mid_vol):
        return 0
    if grow and (not remove or cur_vol < mid_vol):
        return 1
    if bool(legal[2]):
        return 2
    return int(legal.nonzero(as_tuple=False)[0].item())


def _learned_quota(score, weights, max_count, min_step, floor_frac=0.35):
    """Cap thresholded positive-logit voxel selection within [min_count, max_count]."""
    min_count = min(max_count, max(min_step, round(floor_frac * max_count)))
    pos = int(((score > 0.5 * weights) & (weights > 0)).sum().item())
    return min(max_count, max(min_count, pos))


def _pool_boundary_2d(arr, kz, kx_y):
    """Pool a 2D boundary slice over kz x kx_y tiles cleanly via edge replication."""
    h, w = arr.shape[-2], arr.shape[-1]
    ph = (kz - h % kz) % kz
    pw = (kx_y - w % kx_y) % kx_y
    x = arr[None, None].float()
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return F.avg_pool2d(x, (kz, kx_y), stride=(kz, kx_y))[0, 0]


class CoarseEnv:
    """Lightweight coarse-grid wrapper over a fine `Environment` for macro/meso stages."""

    def __init__(self, fine_env, block=(2, 2, 2), handoff_tol=None):
        bz, by, bx = (int(v) for v in block)
        if by != bx or by not in (2, 4, 8) or bz not in (1, 2, 3):
            raise ValueError(
                "CoarseEnv supports block=(1|2|3, 2|4|8, 2|4|8) with equal XY"
            )
        self.env = fine_env
        self.block = (bz, by, bx)
        self.bz = bz
        self.bxy = by
        self.device = fine_env.device
        default_tol = (
            (0.45 if bz == 1 else 0.60)
            if by == 2
            else ((0.90 if bz == 1 else 1.15) if by == 4 else 1.50)
        )
        self.handoff_tol = float(default_tol if handoff_tol is None else handoff_tol)

        Z, Y, X = fine_env.shape
        Zc, Yc, Xc = math.ceil(Z / bz), math.ceil(Y / by), math.ceil(X / by)
        self.shape = (Zc, Yc, Xc)
        self.nz, self.ny, self.nx = self.shape
        self.plot_area = float(fine_env.plot_area)
        self.target_far = float(fine_env.target_far)
        self.target_count_fine = int(fine_env.target_count)
        self.far_tolerance = float(fine_env.far_tolerance)

        # Pad fine grids cleanly to exact multiples of (bz, bxy, bxy)
        pz = Zc * bz - Z
        py = Yc * by - Y
        px = Xc * by - X
        pad3 = (0, px, 0, py, 0, pz)
        self.pad3 = pad3
        self.cores_enabled = bool(getattr(fine_env, "cores_enabled", False))
        self.min_core_spacing = int(
            getattr(fine_env, "min_core_spacing", DEFAULT_MIN_CORE_SPACING)
        )
        self.max_core_dist = int(
            getattr(fine_env, "max_core_dist", DEFAULT_MAX_CORE_DIST)
        )
        self.core_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        self.core_dist_badness = torch.zeros(
            self.shape, dtype=torch.float32, device=self.device
        )
        self.core_score = 0.0

        if not hasattr(fine_env, "external_potential"):
            fine_env.precompute_external()
            normals = fine_env.suns.new_tensor(
                ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            )
            w = (fine_env.suns @ normals.T).clamp_min(0) * fine_env.suns.new_tensor(
                (3.0, 3.0, 3.0, 3.0, 1.0, 1.0)
            )[None]
            fine_env.external_potential = (
                fine_env.static_receivers.values.reshape(
                    len(fine_env.suns), 6, *fine_env.shape
                ).float()
                * w[:, :, None, None, None]
            ).sum((0, 1)) / w.sum().clamp_min(1)

        al_p = F.pad(fine_env.allowed.float(), pad3, value=0.0)
        ep_p = F.pad(fine_env.external_potential, pad3, value=0.0)
        ns_p = F.pad(fine_env.noise, pad3, value=0.0)
        al_blk = al_p.view(Zc, bz, Yc, by, Xc, by).sum(dim=(1, 3, 5))
        self.cell_vol = al_blk.to(torch.int32)
        self.cell_vol_np = self.cell_vol.detach().cpu().numpy()
        self.allowed = self.cell_vol > 0
        self.allowed_np = self.allowed.detach().cpu().numpy()
        self.capacity_coarse = int(self.allowed.sum().detach().cpu())
        self.capacity = int(fine_env.capacity)

        den = al_blk.clamp_min(1.0)
        self.external_potential = (ep_p * al_p).view(Zc, bz, Yc, by, Xc, by).sum(
            dim=(1, 3, 5)
        ) / den
        self.noise = (ns_p * al_p).view(Zc, bz, Yc, by, Xc, by).sum(dim=(1, 3, 5)) / den

        zc_m = (
            (torch.arange(Zc, device=self.device, dtype=torch.float32) + 0.5) * 3.0 * bz
        )
        self.rel_z = (
            torch.tanh(zc_m / 36.0)[:, None, None].expand(self.shape).contiguous()
        )

        # Pool external sky openness onto coarse boundaries (ground face 0 stays 0)
        sb = fine_env.sky_boundary
        sky_c = torch.zeros(
            (Zc + 2, Yc + 2, Xc + 2), device=self.device, dtype=torch.float32
        )
        sky_c[-1, 1:-1, 1:-1] = _pool_boundary_2d(sb[-1, 1:-1, 1:-1], by, by)
        sky_c[1:-1, 0, 1:-1] = _pool_boundary_2d(sb[1:-1, 0, 1:-1], bz, by)
        sky_c[1:-1, -1, 1:-1] = _pool_boundary_2d(sb[1:-1, -1, 1:-1], bz, by)
        sky_c[1:-1, 1:-1, 0] = _pool_boundary_2d(sb[1:-1, 1:-1, 0], bz, by)
        sky_c[1:-1, 1:-1, -1] = _pool_boundary_2d(sb[1:-1, 1:-1, -1], bz, by)
        self.sky_boundary = sky_c

        self.config = PatchConfig(
            mode="combined",
            allocation="feedback",
            eligibility="depth",
            facade_layers=2 if by == 2 else 1,
            minimum_patch=2 if by == 2 else 1,
            maximum_patches=4,
            maximum_voids=1,
            minimum_void_area=2 if by == 2 else 1,
            stride=(max(1, 2 // bz), max(1, 4 // by), max(1, 4 // by)),
            sigma=(max(1.0, 2.0 / bz), max(0.5, 2.0 / by), max(0.5, 2.0 / by)),
            batched=True,
            vertical_interior=False,
            budget_reserve_fraction=0.2,
        )
        self.avg_vol = float(self.capacity) / max(1, self.capacity_coarse)
        self.target_coarse = max(
            1, int(round(self.target_count_fine / max(1e-6, self.avg_vol)))
        )
        self._init_cache = None

    @property
    def far(self):
        return self.count_fine / self.plot_area

    @property
    def far_error(self):
        return abs(self.count_fine - self.goal_count_fine) / self.plot_area

    @property
    def acceptable_fine_counts(self):
        low = max(
            1, math.floor((self.target_far - self.handoff_tol) * self.plot_area + 0.5)
        )
        high = min(
            self.capacity,
            math.floor((self.target_far + self.handoff_tol) * self.plot_area + 0.5),
        )
        return (low, min(self.capacity, max(low, high)))

    @property
    def acceptable_counts(self):
        # Workload/choose helper operates cleanly in coarse cell counts; exact fine
        # volume bounds live in acceptable_fine_counts.
        low = max(
            1,
            int(
                math.floor(
                    (self.target_far - self.handoff_tol)
                    * self.plot_area
                    / max(1e-6, self.avg_vol)
                )
            ),
        )
        high = min(
            self.capacity_coarse,
            int(
                math.ceil(
                    (self.target_far + self.handoff_tol)
                    * self.plot_area
                    / max(1e-6, self.avg_vol)
                )
            ),
        )
        return (low, max(low, high))

    @property
    def goal_count(self):
        low, high = self.acceptable_counts
        return min(high, max(low, self.count))

    @property
    def goal_count_fine(self):
        low, high = self.acceptable_fine_counts
        return min(high, max(low, self.count_fine))

    @property
    def near_far(self):
        low, high = self.acceptable_fine_counts
        return low <= self.count_fine <= high

    @property
    def far_satisfied(self):
        return self.near_far

    @property
    def terminal(self):
        return self.termination is not None

    def _update_depth(self):
        dt = getattr(self.frontier, "depth_tensor", None)
        if dt is not None:
            self.depth = dt * float(self.bxy)
        else:
            self.depth = torch.as_tensor(
                self.frontier.depth_numpy * float(self.bxy), device=self.device
            )
        self.depth_badness = depth_badness(self.depth)

    def _update_cores(self):
        if not self.cores_enabled:
            self.core_mask = torch.zeros(
                self.shape, dtype=torch.bool, device=self.device
            )
            self.core_distance = torch.zeros(
                self.shape, dtype=torch.float32, device=self.device
            )
            self.core_dist_badness = torch.zeros(
                self.shape, dtype=torch.float32, device=self.device
            )
            self.core_score = 0.0
            return
        Z, Y, X = self.env.shape
        Zc, Yc, Xc = self.shape
        occ_fine = (
            self.occupied.repeat_interleave(self.bz, dim=0)
            .repeat_interleave(self.bxy, dim=1)
            .repeat_interleave(self.bxy, dim=2)[:Z, :Y, :X]
            & self.env.allowed
        )
        res = solve_cores(
            occ_fine,
            min_core_spacing=self.min_core_spacing,
            max_core_dist=self.max_core_dist,
        )
        cm_f = F.pad((res.core_mask & occ_fine).float(), self.pad3, value=0.0)
        cd_f = F.pad(
            torch.where(occ_fine, res.core_distance, 0.0), self.pad3, value=0.0
        )
        cb_f = F.pad(
            torch.where(occ_fine, res.core_dist_badness, 0.0), self.pad3, value=0.0
        )
        occ_f = F.pad(occ_fine.float(), self.pad3, value=0.0)
        den = (
            occ_f.view(Zc, self.bz, Yc, self.bxy, Xc, self.bxy)
            .sum(dim=(1, 3, 5))
            .clamp_min(1.0)
        )
        cm_p = (
            cm_f.view(Zc, self.bz, Yc, self.bxy, Xc, self.bxy).sum(dim=(1, 3, 5)) / den
        )
        cd_p = (
            cd_f.view(Zc, self.bz, Yc, self.bxy, Xc, self.bxy).sum(dim=(1, 3, 5)) / den
        )
        cb_p = (
            cb_f.view(Zc, self.bz, Yc, self.bxy, Xc, self.bxy).sum(dim=(1, 3, 5)) / den
        )
        self.core_mask = (cm_p > 0.25) & self.occupied
        self.core_distance = torch.where(self.occupied, cd_p, 0.0)
        self.core_dist_badness = torch.where(self.occupied, cb_p, 0.0)
        m = res.metrics
        self.core_score = max(
            0.0,
            (1.0 - float(m["coreUncoveredFraction"]))
            * (1.0 - 0.5 * float(m["coreDistBadness"])),
        )

    def evaluate_fast(self):
        occ_z = self.occupied.any(dim=(1, 2)).nonzero(as_tuple=False)
        z_top = int(occ_z[-1].item()) + 1 if len(occ_z) else 1
        z_hi = min(self.nz, z_top + 2)
        sp = (float(self.bxy), float(self.bxy), 3.0 * self.bz)
        init_f = (
            self.light
            if (
                getattr(self, "light", None) is not None
                and self.light.shape == self.shape
            )
            else None
        )
        if z_hi < self.nz:
            bnd = self.sky_boundary[: z_hi + 2].clone()
            bnd[-1, 1:-1, 1:-1] = (
                init_f[z_hi]
                if init_f is not None
                else self.sky_boundary[-1, 1:-1, 1:-1]
            )
            sub = diffuse_access(
                self.occupied[:z_hi],
                bnd,
                iterations=14,
                spacing=sp,
                init=None if init_f is None else init_f[:z_hi],
            )
            if init_f is not None:
                lnew = init_f.clone()
                lnew[:z_hi] = sub
                self.light = lnew
            else:
                self.light = torch.zeros(
                    self.shape, device=self.device, dtype=torch.float32
                )
                self.light[:z_hi] = sub
                self.light[z_hi:] = self.sky_boundary[1 + z_hi : -1, 1:-1, 1:-1]
        else:
            self.light = diffuse_access(
                self.occupied, self.sky_boundary, iterations=14, spacing=sp, init=init_f
            )
        lateral = torch.as_tensor(
            self.frontier.remove_faces[[1, 0, 3, 2]], device=self.device
        )
        _, self.facade_light = facade_sky_access(
            self.occupied, self.light, self.sky_boundary, lateral
        )
        self.usable = usable_depth_proxy(
            self.occupied,
            self.light,
            depth=max(2, 8 // self.bxy),
            boundary=self.sky_boundary,
        )
        self._update_cores()

    def restore_occupancy(self, occ_arr):
        occ = (
            torch.as_tensor(occ_arr, dtype=torch.bool, device=self.device).reshape(
                self.shape
            )
            & self.allowed
        )
        self.occupied = occ
        self.count = int(occ.sum().detach().cpu())
        self.count_fine = int((occ * self.cell_vol).sum().detach().cpu())
        self.start_count = self.count
        self.start_count_fine = self.count_fine
        self.edits = 0
        self.unified_actions = 0
        self.termination = None
        self.organic_3d_growth = True
        self.light = None
        self.frontier = PatchFacades(
            occ.detach().cpu().numpy(),
            self.allowed_np,
            self.env.seed,
            config=self.config,
            depth_device=self.device,
        )
        self._update_depth()
        self.evaluate_fast()
        self.workload = WorkloadBudget.create(self, self.config)
        self.unified_limit = max(6, self.workload.limit + 4)
        self._ln_cache = None

    def reset(self, initial_storeys):
        init_c = min(self.nz, max(1, int(round(initial_storeys / self.bz))))
        if self._init_cache is not None and self._init_cache[0] == init_c:
            c = self._init_cache[1]
            self.occupied = c["occupied"].clone()
            self.count = c["count"]
            self.count_fine = c["count_fine"]
            self.start_count = c["start_count"]
            self.start_count_fine = c["start_count_fine"]
            self.edits = 0
            self.unified_actions = 0
            self.termination = None
            f = self.frontier
            fc = c["frontier"]
            f.occupied = fc["occupied"].copy()
            f.exterior_air = fc["exterior_air"].copy()
            f.remove_faces = fc["remove_faces"].copy()
            f.add_faces = fc["add_faces"].copy()
            f.remove_skin = fc["remove_skin"].copy()
            f.add_skin = fc["add_skin"].copy()
            f.components = fc["components"]
            f.count = fc["count"]
            f.depth_tensor = fc["depth_tensor"].clone()
            f.old_voids = (
                [(lab.copy(), list(ids), sz.copy()) for lab, ids, sz in fc["old_voids"]]
                if fc["old_voids"] is not None
                else None
            )
            f.facade_band = fc["facade_band"].copy()
            f._remove_eligible = fc["_remove_eligible"].copy()
            f._prev_occ = f.occupied.copy()
            f._prev_air = f.exterior_air.copy()
            f._weight_tensors = {}
            f.version += 1
            self.depth = c["depth"].clone()
            self.depth_badness = c["depth_badness"].clone()
            self.light = c["light"].clone()
            self.facade_light = c["facade_light"].clone()
            self.usable = c["usable"].clone()
            self.core_mask = c["core_mask"].clone()
            self.core_distance = c["core_distance"].clone()
            self.core_dist_badness = c["core_dist_badness"].clone()
            self.core_score = float(c["core_score"])
            self.workload = WorkloadBudget.create(self, self.config)
            self.unified_limit = max(6, self.workload.limit + 4)
            self._ln_cache = None
            self.organic_3d_growth = True
            return

        occ = self.allowed.clone()
        occ[init_c:] = False
        self.occupied = occ
        self.count = int(occ.sum().detach().cpu())
        self.count_fine = int((occ * self.cell_vol).sum().detach().cpu())
        self.start_count = self.count
        self.start_count_fine = self.count_fine
        self.edits = 0
        self.unified_actions = 0
        self.termination = None
        self.organic_3d_growth = True
        self.light = None
        self.frontier = PatchFacades(
            occ.detach().cpu().numpy(),
            self.allowed_np,
            self.env.seed,
            config=self.config,
            depth_device=self.device,
        )
        self._update_depth()
        self.evaluate_fast()
        self.workload = WorkloadBudget.create(self, self.config)
        self.unified_limit = max(6, self.workload.limit + 4)
        self._ln_cache = None
        f = self.frontier
        self._init_cache = (
            init_c,
            dict(
                occupied=occ.clone(),
                count=self.count,
                count_fine=self.count_fine,
                start_count=self.start_count,
                start_count_fine=self.start_count_fine,
                depth=self.depth.clone(),
                depth_badness=self.depth_badness.clone(),
                light=self.light.clone(),
                facade_light=self.facade_light.clone(),
                usable=self.usable.clone(),
                core_mask=self.core_mask.clone(),
                core_distance=self.core_distance.clone(),
                core_dist_badness=self.core_dist_badness.clone(),
                core_score=self.core_score,
                frontier=dict(
                    occupied=f.occupied.copy(),
                    exterior_air=f.exterior_air.copy(),
                    remove_faces=f.remove_faces.copy(),
                    add_faces=f.add_faces.copy(),
                    remove_skin=f.remove_skin.copy(),
                    add_skin=f.add_skin.copy(),
                    components=int(f.components),
                    count=int(f.count),
                    depth_tensor=f.depth_tensor.clone(),
                    old_voids=[
                        (lab.copy(), list(ids), sz.copy())
                        for lab, ids, sz in f.old_voids
                    ]
                    if f.old_voids is not None
                    else None,
                    facade_band=f.facade_band.copy(),
                    _remove_eligible=f._remove_eligible.copy(),
                ),
            ),
        )

    def apply_move(self, move):
        f = self.frontier
        f.occupied[move.remove] = False
        f.occupied[move.add] = True
        f.version += 1
        f.refresh()
        rm_t = torch.as_tensor(move.remove, device=self.device)
        ad_t = torch.as_tensor(move.add, device=self.device)
        self.occupied.masked_fill_(rm_t, False)
        self.occupied.masked_fill_(ad_t, True)
        self.count = int(f.occupied.sum())
        self.count_fine = int((self.occupied * self.cell_vol).sum().detach().cpu())
        self.edits += 1
        self._update_depth()
        self.evaluate_fast()

    def observation(self, z_slice=None, include_cores=False):
        occ = self.occupied[:z_slice] if z_slice is not None else self.occupied
        remove = self.frontier.weights_tensor(-1, self.device)
        grow = torch.as_tensor(
            growth_mask(self), device=self.device, dtype=torch.float32
        )
        ep = self.external_potential
        fl = self.facade_light
        light = self.light
        noise = self.noise
        allowed = self.allowed.float()
        rel_z = self.rel_z
        depth_b = self.depth_badness
        usable = self.usable
        if z_slice is not None:
            remove = remove[:z_slice]
            grow = grow[:z_slice]
            ep = ep[:z_slice]
            fl = fl[:z_slice]
            light = light[:z_slice]
            noise = noise[:z_slice]
            allowed = allowed[:z_slice]
            rel_z = rel_z[:z_slice]
            depth_b = depth_b[:z_slice]
            usable = usable[:z_slice]

        if self.bxy == 2 and self.plot_area >= 400 and self.ny >= 3 and self.nx >= 3:
            c_key = (self.frontier.version, z_slice)
            cache = self._ln_cache
            if cache is not None and cache[0] == c_key:
                ln_bool = cache[1]
            else:
                if not hasattr(self, "_allowed_ln3"):
                    self._allowed_ln3 = long_narrow_mask(self.allowed, width=3, stub=1)
                aln = (
                    self._allowed_ln3[:z_slice]
                    if z_slice is not None
                    else self._allowed_ln3
                )
                ln_bool = long_narrow_mask(occ, width=3, stub=1) & ~aln
                self._ln_cache = (c_key, ln_bool)
        else:
            ln_bool = torch.zeros_like(occ, dtype=torch.bool)

        chans = [
            occ.float(),
            light,
            noise,
            allowed,
            rel_z,
            depth_b,
            remove,
            grow,
            ep,
            fl,
            ln_bool.float(),
            usable.float(),
        ]
        if include_cores:
            if self.cores_enabled:
                cm = (
                    self.core_mask[:z_slice] if z_slice is not None else self.core_mask
                ) & occ
                cb = torch.where(
                    occ,
                    self.core_dist_badness[:z_slice]
                    if z_slice is not None
                    else self.core_dist_badness,
                    0.0,
                )
                chans.extend((cm.float(), cb))
            else:
                zeros = torch.zeros_like(light)
                chans.extend((zeros, zeros))
        return torch.stack(chans, dim=1)

    def upsample_to_coarse(self, target_cenv):
        rz = max(1, self.bz // target_cenv.bz)
        rxy = max(1, self.bxy // target_cenv.bxy)
        Zt, Yt, Xt = target_cenv.shape
        exp = (
            self.occupied.repeat_interleave(rz, dim=0)
            .repeat_interleave(rxy, dim=1)
            .repeat_interleave(rxy, dim=2)
        )
        occ_t = (exp[:Zt, :Yt, :Xt] & target_cenv.allowed).detach().cpu().numpy()
        labels, n = ndi.label(occ_t, STRUCTURE)
        grounded = np.unique(labels[0])
        grounded = grounded[grounded != 0]
        if n == 0 or len(grounded) == 0:
            occ_t = target_cenv.allowed_np.copy()
            occ_t[min(Zt, max(1, int(round(self.target_far / target_cenv.bz)))) :] = (
                False
            )
        elif len(grounded) != n or n > 2:
            sizes = np.bincount(labels.ravel())
            keep_ids = grounded[np.argsort(-sizes[grounded])[:2]]
            occ_t = occ_t & np.isin(labels, keep_ids)
        target_cenv.restore_occupancy(occ_t)

    def upsample_to_fine(self):
        Z, Y, X = self.env.shape
        exp = (
            self.occupied.repeat_interleave(self.bz, dim=0)
            .repeat_interleave(self.bxy, dim=1)
            .repeat_interleave(self.bxy, dim=2)
        )
        occ_fine = exp[:Z, :Y, :X] & self.env.allowed
        occ_np = occ_fine.detach().cpu().numpy()
        mw = min_room_width(self.env)
        if mw > 1:
            al_np = self.env.frontier.allowed
            clean_al = clean_floor_numpy(al_np, mw)
            occ_np = occ_np & (clean_floor_numpy(occ_np, mw) | ~clean_al)
        labels, n = ndi.label(occ_np, STRUCTURE)
        grounded = np.unique(labels[0])
        grounded = grounded[grounded != 0]
        if n == 0 or len(grounded) == 0:
            fallback = self.env.allowed.detach().cpu().numpy().copy()
            fallback[min(Z, max(1, int(round(self.target_far)))) :] = False
            occ_np = fallback
        elif len(grounded) != n or n > 2:
            sizes = np.bincount(labels.ravel())
            keep_ids = grounded[np.argsort(-sizes[grounded])[:2]]
            occ_np = occ_np & np.isin(labels, keep_ids)
        self.env.restore_occupancy(occ_np)
        self.env.copy_labels = np.zeros(self.env.shape, dtype=np.int64)
        self.env._ln6_cache = None
        self.env.organic_3d_growth = True
        self.env.floor_copy_growth = False
        # Compact Stage 2 fine refinement budget
        self.env.unified_limit = max(6, self.env.workload.limit + 4)
        self.env.unified_actions = 0
        self.env.termination = None


def _project_coarse_valid(cenv, f, r, a, changed, air_after, mw, clean_old, rem_s, g_s):
    zs = changed.shape[0]
    if not (r.any() or a.any()) or (r & a).any():
        return False
    if (r & ~rem_s).any() or (a & ~g_s).any():
        return False
    for mask in (r, a):
        if not mask.any():
            continue
        labels, n = ndi.label(mask, np.ones((3, 3, 3), bool))
        if f.config.maximum_patches and n > f.config.maximum_patches:
            return False
        minimum = (
            1
            if int(mask.sum())
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


def project_coarse_3d(
    cenv, masks, scores, operation, net_guided=True, maximum_buildings=2
):
    """Project coarse stage masks with cell_vol-exact FAR progress & stem preservation."""
    f = cenv.frontier
    mw = (2 if cenv.bxy == 2 else 1) if cenv.plot_area >= 400 else 1
    z_occ = np.flatnonzero((f.occupied | masks[1]).any((1, 2)))
    zs = min(cenv.nz, int(z_occ[-1]) + 2) if len(z_occ) else min(cenv.nz, 1)
    occ_s = f.occupied[:zs]
    vol_s = cenv.cell_vol_np[:zs]
    r = masks[0, :zs].copy()
    a = masks[1, :zs].copy()
    s0 = scores[0, :zs]
    s1 = scores[1, :zs]
    skin_s = f.remove_skin[:zs]
    f_ext_s = f.exterior_air[:zs]
    rem_s = f.eligible(-1)[:zs]
    g_s = growth_mask(cenv)[:zs]
    clean_old = clean_floor_numpy(occ_s, mw) if mw > 1 else None
    low_f, high_f = cenv.acceptable_fine_counts
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
                    r &= ~(bridge if bridge.any() else (r & bad_nbr))
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
            rem_f = int(vol_s[r].sum()) if r.any() else 0
            add_f = int(vol_s[a].sum()) if a.any() else 0
            after_f = cenv.count_fine - rem_f + add_f
            if cenv.count_fine > high_f:
                req_lo, req_hi = low_f, cenv.count_fine - 1
            elif cenv.count_fine < low_f:
                req_lo, req_hi = cenv.count_fine + 1, high_f
            else:
                req_lo, req_hi = low_f, high_f
            if after_f < req_lo:
                need_drop = req_lo - after_f
                ids = np.flatnonzero(r)
                if not len(ids):
                    return None
                order = ids[np.argsort(s0.ravel()[ids], kind="stable")]
                vs = vol_s.ravel()[order]
                csum = np.cumsum(vs)
                stop = int(np.searchsorted(csum, need_drop, side="left")) + 1
                if stop >= len(order):
                    return None
                r.ravel()[order[:stop]] = False
                continue
            if after_f > req_hi:
                need_drop = after_f - req_hi
                ids = np.flatnonzero(a)
                if not len(ids):
                    return None
                order = ids[np.argsort(s1.ravel()[ids], kind="stable")]
                vs = vol_s.ravel()[order]
                csum = np.cumsum(vs)
                stop = int(np.searchsorted(csum, need_drop, side="left")) + 1
                if stop >= len(order):
                    return None
                a.ravel()[order[:stop]] = False
                continue
        if not _project_coarse_valid(
            cenv, f, r, a, changed, air_after, mw, clean_old, rem_s, g_s
        ):
            return None
        if zs == cenv.nz:
            return Exchange(r.copy(), a.copy(), f.version)
        r_full = np.zeros(cenv.shape, bool)
        a_full = np.zeros(cenv.shape, bool)
        r_full[:zs] = r
        a_full[:zs] = a
        return Exchange(r_full, a_full, f.version)
    return None


class CoarseToFineNetwork(Organic3DNetwork):
    """Shared 2D+1D trunk with macro (4 m), coarse (2 m), and fine (1 m) stems."""

    def __init__(self, input_channels=12):
        super().__init__(input_channels)
        self.macro_stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1), nn.SiLU()
        )
        self.coarse_stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1), nn.SiLU()
        )
        self.goal_encoder = nn.Linear(6, 32)
        self.critic = nn.Sequential(nn.Linear(70, 32), nn.SiLU(), nn.Linear(32, 1))
        self.operation_head = nn.Linear(70, 4)
        nn.init.zeros_(self.goal_encoder.weight)
        nn.init.zeros_(self.goal_encoder.bias)
        nn.init.zeros_(self.operation_head.weight)
        nn.init.zeros_(self.operation_head.bias)
        with torch.no_grad():
            self.operation_head.weight[0, 64:66] = torch.tensor([3.0, -3.0])
            self.operation_head.weight[1, 64:66] = torch.tensor([-3.0, 3.0])
            self.operation_head.bias[2] = 0.6

    def _encode_stage_window(self, obs_w, scale):
        B, Zw, C, H, W = obs_w.shape
        flat = obs_w.reshape(B * Zw, C, H, W)
        if scale == 8:
            x = self.macro_stem(F.interpolate(flat, scale_factor=2, mode="nearest"))
        elif scale == 4:
            x = self.macro_stem(flat)
        elif scale == 2:
            x = self.coarse_stem(flat)
        else:
            x = self.down(self.stem(flat))
        v = F.silu(self.ctx_xy1(x))
        v = F.silu(v + self._zmix(v, self.ctx_z1, self.ctx_z1_bias, B, Zw))
        v = F.silu(self.ctx_xy2(v))
        v = F.silu(v + self._zmix(v, self.ctx_z2, self.ctx_z2_bias, B, Zw))
        return v.view(B, Zw, 32, v.shape[-2], v.shape[-1])

    def _head_stage_slice(self, v_keep, obs_keep, goal_exp, scale):
        B, Zk, C, H, W = obs_keep.shape
        flat_k = obs_keep.reshape(B * Zk, C, H, W)
        vk = v_keep.reshape(B * Zk, 32, v_keep.shape[-2], v_keep.shape[-1])
        if scale >= 4:
            oc = flat_k
        else:
            s = 2 if scale == 2 else 4
            ky, kx = min(s, H), min(s, W)
            oc = F.avg_pool2d(
                flat_k,
                kernel_size=(ky, kx),
                stride=s,
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

    def forward_stage_batch(
        self,
        obs_b,
        goal_b,
        is_coarse=False,
        stride=(2, 4, 4),
        window_floors=16,
        scale=None,
    ):
        st_scale = (2 if is_coarse else 1) if scale is None else int(scale)
        B, Z, C, H, W = obs_b.shape
        win = max(6, int(window_floors))
        halo = 2
        goal_exp = self.goal_encoder(goal_b)[:, None, :, None, None]
        if Z <= win:
            v_bz = self._encode_stage_window(obs_b, st_scale)
            fl = v_bz.mean((-1, -2))
            context = torch.cat((fl.mean(1), fl.amax(1), goal_b), dim=-1)
            coarse_z = self._head_stage_slice(v_bz, obs_b, goal_exp, st_scale)
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
                v_win = self._encode_stage_window(obs_b[:, w_start:w_end], st_scale)
                v_keep = v_win[:, keep_s:keep_e]
                fl = v_keep.mean((-1, -2))
                f_sum = fl.sum(1)
                f_max = fl.amax(1)
                floor_sum = f_sum if floor_sum is None else (floor_sum + f_sum)
                floor_max = (
                    f_max if floor_max is None else torch.maximum(floor_max, f_max)
                )
                cz_parts.append(
                    self._head_stage_slice(v_keep, obs_b[:, k0:k1], goal_exp, st_scale)
                )
                k0 = k1
            coarse_z = torch.cat(cz_parts, dim=2)
            context = torch.cat((floor_sum / Z, floor_max, goal_b), dim=-1)

        target = (
            max(1, math.ceil(Z / stride[0])),
            max(1, math.ceil(H / stride[1])),
            max(1, math.ceil(W / stride[2])),
        )
        if stride[0] == 1 and coarse_z.shape[2:] == target:
            coarse = coarse_z
        elif stride[0] == 2 and coarse_z.shape[-2:] == target[1:]:
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


class CoarseToFineLearner:
    """Multi-stage actor-critic training arbitrary coarse stages + fine refinement in unified GAE."""

    def __init__(
        self,
        device="cpu",
        seed=123,
        block=(2, 2, 2),
        fine_config=None,
        input_channels=12,
        spatial_aux_weight=1.2,
        evaluation_temperature=2.5,
        evaluation_noise_scale=0.08,
        lr=1.2e-3,
        heuristic_only=False,
        training_objective="actor_critic",
        decoder_retry_schedule="legacy",
    ):
        self.device = torch.device(device)
        self.seed = int(seed)
        self.block = tuple(int(v) for v in block)
        self.bz = self.block[0]
        self.input_channels = int(input_channels)
        self.spatial_aux_weight = float(spatial_aux_weight)
        self.evaluation_temperature = float(evaluation_temperature)
        self.evaluation_noise_scale = float(evaluation_noise_scale)
        self.lr = float(lr)
        self.heuristic_only = bool(heuristic_only)
        self.training_objective = training_objective
        self.decoder_retry_schedule = decoder_retry_schedule
        self.decoder_diagnostics: DecoderDiagnostics | None = None
        if not math.isfinite(self.spatial_aux_weight) or self.spatial_aux_weight < 0.0:
            raise ValueError("Invalid auxiliary weight")
        self.fine_config = fine_config or PatchConfig(
            mode="combined",
            allocation="feedback",
            eligibility="depth",
            facade_layers=2,
            minimum_patch=4,
            maximum_patches=4,
            maximum_voids=1,
            minimum_void_area=4,
            batched=True,
            vertical_interior=False,
            budget_reserve_fraction=0.2,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = CoarseToFineNetwork(self.input_channels).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self.tape = []
        self.updates = 0
        self.direct_updates = 0
        self.last_operation = None
        self.last_rejection = None
        self.last_attempt_recorded = False
        self.last_generated_count = 0
        self.teachers: dict[str, CoarseToFineNetwork] = {}

    def set_teachers(self, teachers: dict[str, CoarseToFineNetwork]):
        """Register frozen tier-expert teacher networks for on-policy distillation."""
        self.teachers = {}
        for key, net in teachers.items():
            net = net.to(self.device)
            net.eval()
            net.requires_grad_(False)
            self.teachers[str(key)] = net

    def _select_teacher(self, env_or_cenv) -> CoarseToFineNetwork | None:
        if not self.teachers:
            return None
        area = float(getattr(env_or_cenv, "plot_area", 0.0))
        if area < 2500.0:
            order = ("XS_M", "ALL")
        elif area < 3800.0:
            order = ("L", "L_XL", "ALL")
        else:
            order = ("XL", "L_XL", "ALL")
        for key in order:
            if key in self.teachers:
                return self.teachers[key]
        return None

    def clear(self):
        self.tape = []

    def _goal_vec(self, env_or_cenv, scale):
        stage_feat = 1.0 if scale >= 4 else (0.5 if scale == 2 else 0.0)
        return torch.tensor(
            [*_organic_goal_features(env_or_cenv), stage_feat],
            device=self.device,
            dtype=torch.float32,
        )

    def _fine_obs(self, e, z_slice=None):
        e.organic_3d_growth = True
        _ensure_external_potential(e)
        remove = e.frontier.weights_tensor(-1, self.device)
        grow = torch.as_tensor(growth_mask(e), device=self.device, dtype=torch.float32)
        ep = e.external_potential
        fl = e.facade_light
        occ_sub = e.occupied
        usb_sub = e.usable
        if z_slice is not None:
            remove = remove[:z_slice]
            grow = grow[:z_slice]
            ep = ep[:z_slice]
            fl = fl[:z_slice]
            occ_sub = occ_sub[:z_slice]
            usb_sub = usb_sub[:z_slice]
        if e.plot_area >= 400 and e.ny >= 6 and e.nx >= 6:
            c_key = (e.frontier.version, z_slice)
            cache = getattr(e, "_ln6_cache", None)
            if cache is not None and cache[0] == c_key:
                ln6 = cache[1]
            else:
                if not hasattr(e, "_allowed_ln6_t"):
                    e._allowed_ln6_t = long_narrow_mask(e.allowed, 6, 2)
                aln = (
                    e._allowed_ln6_t[:z_slice]
                    if z_slice is not None
                    else e._allowed_ln6_t
                )
                ln6 = long_narrow_mask(occ_sub, 6, 2) & ~aln
                e._ln6_cache = (c_key, ln6)
        else:
            ln6 = torch.zeros_like(occ_sub, dtype=torch.bool)
        channels = [
            e.observation(include_depth=True, z_slice=z_slice),
            remove[:, None],
            grow[:, None],
            ep[:, None],
            fl[:, None],
            ln6.float()[:, None],
            usb_sub.float()[:, None],
        ]
        if self.input_channels == 14:
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
                channels.extend((cm_sub.float()[:, None], cb_sub[:, None]))
            else:
                z_zero = torch.zeros_like(occ_sub, dtype=torch.float32)[:, None]
                channels.extend((z_zero, z_zero))
        return torch.cat(channels, 1)

    def choose_stage(self, env_or_cenv, is_coarse, training=True, retries=8):
        self.last_operation = None
        self.last_rejection = None
        self.last_attempt_recorded = False
        self.last_generated_count = 0
        imitation = training and self.training_objective == "imitation"
        on_policy_distill = training and self.training_objective == "on_policy_distill"
        greedy_rollout = self.heuristic_only or imitation
        learned_cap = self.decoder_retry_schedule == "learned_cap"
        cfg = env_or_cenv.config if is_coarse else self.fine_config
        stride = cfg.stride
        scale = int(env_or_cenv.bxy) if is_coarse else 1
        bz = int(env_or_cenv.bz) if is_coarse else 1
        mw = (
            ((2 if scale == 2 else 1) if env_or_cenv.plot_area >= 400 else 1)
            if is_coarse
            else min_room_width(env_or_cenv)
        )
        ln_w, ln_stub = (3, 1) if scale == 2 else ((1, 0) if scale >= 4 else (6, 2))
        z_floor_min = (6 if bz == 1 else 4) if is_coarse else 10

        with torch.no_grad():
            occ_z = env_or_cenv.occupied.any(dim=(1, 2)).nonzero(as_tuple=False)
            z_top = int(occ_z[-1].item()) + 1 if len(occ_z) else 0
            z_step = 4 if (not is_coarse or bz == 1) else 2
            z_act = min(
                env_or_cenv.nz,
                max(
                    min(env_or_cenv.nz, z_floor_min),
                    ((z_top + z_step) // z_step) * z_step,
                ),
            )
            obs = (
                env_or_cenv.observation(
                    z_slice=z_act, include_cores=(self.input_channels == 14)
                )
                if is_coarse
                else self._fine_obs(env_or_cenv, z_slice=z_act)
            )
            goal = self._goal_vec(env_or_cenv, scale)
            if greedy_rollout:
                mean_all = None
                value = torch.zeros((), device=self.device, dtype=torch.float32)
                logits = torch.zeros(4, device=self.device, dtype=torch.float32)
            else:
                mean_all, value, logits = self.model.forward_stage_batch(
                    obs[None],
                    goal[None],
                    is_coarse=is_coarse,
                    stride=stride,
                    scale=scale,
                )
                mean_all, value, logits = mean_all[0], value[0], logits[0]

            r_avail = bool(env_or_cenv.frontier.eligible(-1).any())
            a_avail = bool(growth_mask(env_or_cenv).any())
            if is_coarse:
                low_f, high_f = env_or_cenv.acceptable_fine_counts
                remove = r_avail and env_or_cenv.count_fine > low_f
                grow = a_avail and env_or_cenv.count_fine < high_f
                can_finish = bool(
                    env_or_cenv.near_far
                    or (
                        learned_cap
                        and not greedy_rollout
                        and env_or_cenv.edits >= 2
                        and abs(env_or_cenv.far - env_or_cenv.target_far) <= 0.70
                    )
                )
            else:
                low_f, high_f = env_or_cenv.acceptable_counts
                remove = r_avail and env_or_cenv.count > low_f
                grow = a_avail and env_or_cenv.count < high_f
                can_finish = bool(env_or_cenv.far_satisfied)

            allow_exchange = r_avail and a_avail
            if (
                not is_coarse
                and learned_cap
                and not env_or_cenv.far_satisfied
                and (env_or_cenv.unified_limit - env_or_cenv.unified_actions) <= 4
                and (remove or grow)
            ):
                allow_exchange = False
            legal = torch.tensor(
                [remove, grow, allow_exchange, can_finish],
                device=self.device,
                dtype=torch.bool,
            )
            if training and env_or_cenv.edits == 0 and bool(legal[:3].any()):
                legal = legal.clone()
                legal[3] = False
            if not bool(legal.any()):
                self.last_rejection = "no_admissible_edits"
                return None

            heuristic_op = None
            if training or greedy_rollout:
                h_op_idx = _heuristic_op_index(
                    env_or_cenv,
                    is_coarse,
                    legal,
                    low_f,
                    high_f,
                    remove,
                    grow,
                    bool(env_or_cenv.near_far) if is_coarse else can_finish,
                )
                if training:
                    heuristic_op = torch.tensor(
                        h_op_idx, device=self.device, dtype=torch.long
                    )

            if greedy_rollout:
                op = torch.tensor(h_op_idx, device=self.device, dtype=torch.long)
            else:
                eff_logits = (
                    logits if training else (logits * self.evaluation_temperature)
                )
                dist_op = Categorical(
                    logits=eff_logits.masked_fill(~legal, float("-inf"))
                )
                op = dist_op.sample()
            operation = OPERATIONS[int(op.cpu())]
            self.last_operation = operation

            move = None
            latent = None
            latent_draws = []
            consumed = None
            elig_c = None
            local_adv = None
            teacher_mean = None
            teacher_logits = None
            teacher_net = None
            t_mean_all = None
            if on_policy_distill:
                teacher_net = self._select_teacher(env_or_cenv)
                if teacher_net is not None:
                    t_mean_all, _, t_logits = teacher_net.forward_stage_batch(
                        obs[None],
                        goal[None],
                        is_coarse=is_coarse,
                        stride=stride,
                        scale=scale,
                    )
                    teacher_logits = t_logits[0].detach()
            if operation != "finish":
                w_rem = env_or_cenv.frontier.weights_tensor(-1, self.device)[:z_act]
                add_np = growth_mask(env_or_cenv)[:z_act]
                w_add = torch.as_tensor(add_np, device=self.device, dtype=torch.float32)
                avail_rem = int(env_or_cenv.frontier.eligible(-1)[:z_act].sum())
                avail_add = int(add_np.sum())
                occ_s = env_or_cenv.occupied[:z_act]
                kernel = tuple(min(n, s) for n, s in zip(occ_s.shape, stride))
                consumed = (
                    [0]
                    if operation == "remove"
                    else [1]
                    if operation == "grow"
                    else [0, 1]
                )
                std = (
                    self.model.patch_log_std[consumed]
                    .clamp(-1.4, -0.3)
                    .exp()[:, None, None, None]
                )
                if not greedy_rollout:
                    mean = mean_all[consumed]
                    dist = Normal(mean, std, validate_args=False)

                if is_coarse:
                    avg_v = max(1e-6, env_or_cenv.avg_vol)
                    c_low = max(1, int(round(low_f / avg_v)))
                    c_high = min(
                        env_or_cenv.capacity_coarse, int(round(high_f / avg_v))
                    )
                    c_goal = min(c_high, max(c_low, env_or_cenv.count))
                    remaining = max(
                        0, abs(env_or_cenv.count_fine - env_or_cenv.goal_count_fine)
                    )
                    rem_c = max(0, int(round(remaining / avg_v)))
                    productive = max(
                        1,
                        env_or_cenv.workload.limit
                        - env_or_cenv.workload.margin
                        - env_or_cenv.unified_actions,
                    )
                    ret = max(0.55, min(1.0, env_or_cenv.workload.retention))
                    min_step = 1 if scale == 4 else 2
                    step_net = (
                        max(min_step, math.ceil(rem_c / productive / ret))
                        if rem_c
                        else 0
                    )
                    reloc = max(
                        min_step,
                        min(
                            int(
                                round(
                                    (env_or_cenv.plot_area / float(scale * scale))
                                    * 0.35
                                )
                            ),
                            max(min_step, round(env_or_cenv.target_coarse * 0.10)),
                        ),
                    )
                else:
                    c_low, c_high = low_f, high_f
                    c_goal = env_or_cenv.goal_count
                    remaining = abs(env_or_cenv.count - c_goal)
                    rem_c = remaining
                    productive = max(
                        1,
                        env_or_cenv.workload.limit
                        - env_or_cenv.workload.margin
                        - env_or_cenv.unified_actions,
                    )
                    ret = max(0.55, min(1.0, env_or_cenv.workload.retention))
                    step_net = (
                        max(4, math.ceil(rem_c / productive / ret)) if rem_c else 0
                    )
                    reloc = max(
                        16,
                        min(
                            int(round(env_or_cenv.plot_area * 0.35)),
                            max(16, round(env_or_cenv.target_count * 0.08)),
                        ),
                    )
                    min_step = 4

                base_rem = 0
                base_add = 0
                cur_c = env_or_cenv.count
                if operation == "remove":
                    if cur_c > c_goal:
                        base_rem = min(max(1, cur_c - c_low), step_net)
                    else:
                        base_rem = min(
                            max(1, cur_c - c_low), max(min_step, round(c_goal * 0.04))
                        )
                    base_rem = min(avail_rem, max(0, base_rem))
                elif operation == "grow":
                    if cur_c < c_goal:
                        base_add = min(max(1, c_high - cur_c), step_net)
                    else:
                        base_add = min(
                            max(1, c_high - cur_c), max(min_step, round(c_goal * 0.05))
                        )
                    base_add = min(avail_add, max(0, base_add))
                else:
                    if cur_c > c_goal:
                        base_add = min(avail_add, reloc)
                        base_rem = min(
                            avail_rem,
                            base_add
                            + min(max(1, cur_c - c_low), max(min_step, step_net)),
                        )
                    elif cur_c < c_goal:
                        base_rem = min(avail_rem, reloc)
                        base_add = min(
                            avail_add,
                            base_rem
                            + min(max(1, c_high - cur_c), max(min_step, step_net)),
                        )
                    else:
                        both = min(reloc, avail_rem, avail_add)
                        base_rem = both
                        base_add = both

                clean_occ = clean_floor_tensor(occ_s, mw) if mw > 1 else None
                skin = (
                    torch.as_tensor(
                        env_or_cenv.frontier.remove_skin[:z_act], device=self.device
                    )
                    if mw > 1
                    else None
                )
                ln_before = None
                allowed_ln = None
                if (
                    env_or_cenv.plot_area >= 400
                    and ln_w > 1
                    and env_or_cenv.ny >= ln_w
                    and env_or_cenv.nx >= ln_w
                ):
                    allowed_ln = long_narrow_mask(env_or_cenv.allowed, ln_w, ln_stub)[
                        :z_act
                    ]
                    ln_before = long_narrow_mask(occ_s, ln_w, ln_stub) & ~allowed_ln
                ln_f = (
                    ln_before.float()
                    if ln_before is not None
                    else torch.zeros_like(env_or_cenv.facade_light[:z_act])
                )
                if greedy_rollout:
                    des_c = compute_spatial_desirability(
                        env_or_cenv,
                        w_rem,
                        w_add,
                        ln_f,
                        consumed,
                        kernel,
                        stride,
                        z_slice=z_act,
                    )
                    mean = (2.0 * des_c).clamp(-2.8, 2.8)
                    dist = Normal(mean, std, validate_args=False)
                    if imitation:
                        teacher_mean = mean
                        elig_c = _pooled_edit_eligibility(
                            w_rem, w_add, consumed, kernel, stride
                        )
                has_core_guard = (not is_coarse) and bool(
                    getattr(env_or_cenv, "cores_enabled", False)
                    and getattr(env_or_cenv, "cores", ())
                )
                core_m_sub = (
                    env_or_cenv.core_mask[:z_act]
                    if (has_core_guard and hasattr(env_or_cenv, "core_mask"))
                    else None
                )
                active_cores = (
                    [
                        c
                        for c in env_or_cenv.cores
                        if int(c.floors) > 0
                        and env_or_cenv.target_count
                        > c.size_y * c.size_x * int(c.floors) * 1.5
                    ]
                    if has_core_guard
                    else ()
                )

                scales = (
                    (1.0, 0.6, 0.36, 0.216, 0.1296, 0.07776, 0.046656, 0.027994)
                    if self.decoder_retry_schedule in ("adaptive", "learned_cap")
                    else (1.0, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0625)
                )
                if not is_coarse and not env_or_cenv.far_satisfied and productive <= 4:
                    quota_floor = 1.0
                elif is_coarse or env_or_cenv.far_satisfied or operation == "exchange":
                    quota_floor = 0.35
                else:
                    quota_floor = 0.65
                for attempt in range(retries):
                    diagnostics = self.decoder_diagnostics
                    if diagnostics is not None:
                        proposal_start = time.perf_counter()
                    n_r = n_a = 0
                    latent = (
                        dist.sample()
                        if ((training and not imitation) or attempt > 0)
                        else (
                            mean
                            + (0.0 if imitation else self.evaluation_noise_scale)
                            * std
                            * torch.randn_like(mean)
                        )
                    )
                    if training and not imitation:
                        latent_draws.append(latent.detach())
                    decoded = F.interpolate(
                        latent[:, None],
                        size=occ_s.shape,
                        mode="trilinear",
                        align_corners=False,
                    )[:, 0]
                    if cfg.mode == "combined":
                        decoded = gaussian_field(decoded, cfg.sigma)
                    sc = scales[min(attempt, len(scales) - 1)]
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
                            for _ in range(max(1, cfg.facade_layers)):
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
                            _learned_quota(
                                s0, w_rem, max_r, min_step, floor_frac=quota_floor
                            )
                            if (learned_cap and not greedy_rollout)
                            else max_r
                        )
                        scores[0] = s0
                        r_t = torch.zeros_like(occ_s)
                        ids = s0.masked_fill(w_rem <= 0, -1).flatten().topk(n_r).indices
                        r_t.flatten()[ids] = True
                        if mw > 1:
                            if ln_before is not None and attempt < 3:
                                after_t = occ_s & ~r_t
                                new_ln = (
                                    long_narrow_mask(after_t, ln_w, ln_stub)
                                    & ~ln_before
                                    & ~allowed_ln
                                )
                                zone_ln = (
                                    F.max_pool2d(
                                        new_ln.float()[:, None], 3, stride=1, padding=1
                                    )[:, 0]
                                    > 0.5
                                )
                                r_t = r_t & ~zone_ln
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
                            _learned_quota(
                                s1, w_add, max_a, min_step, floor_frac=quota_floor
                            )
                            if (learned_cap and not greedy_rollout)
                            else max_a
                        )
                        scores[1] = s1
                        a_t = torch.zeros_like(occ_s)
                        ids = s1.masked_fill(w_add <= 0, -1).flatten().topk(n_a).indices
                        a_t.flatten()[ids] = True
                        if mw > 1:
                            base_occ = occ_s & ~masks[0]
                            a_t = a_t & clean_floor_tensor(base_occ | a_t, mw)
                            if ln_before is not None:
                                cand = base_occ | a_t
                                has_core = (
                                    F.avg_pool2d(cand.float()[:, None], ln_w, stride=1)
                                    > 1.0 - 1e-4
                                ).amax((-1, -2, -3))
                                new_ln = (
                                    long_narrow_mask(cand, ln_w, ln_stub)
                                    & ~long_narrow_mask(base_occ, ln_w, ln_stub)
                                    & ~allowed_ln
                                    & has_core[:, None, None]
                                )
                                a_t = a_t & ~new_ln
                                a_t = a_t & clean_floor_tensor(base_occ | a_t, mw)
                        masks[1] = a_t

                    if z_act == env_or_cenv.nz:
                        masks_np = masks.cpu().numpy()
                        scores_np = scores.cpu().numpy()
                    else:
                        masks_np = np.zeros((2, *env_or_cenv.shape), dtype=bool)
                        scores_np = np.zeros((2, *env_or_cenv.shape), dtype=np.float32)
                        masks_np[:, :z_act] = masks.cpu().numpy()
                        scores_np[:, :z_act] = scores.cpu().numpy()
                    self.last_generated_count = int(masks_np.sum())
                    if diagnostics is not None:
                        projection_start = time.perf_counter()
                    if is_coarse:
                        move = project_coarse_3d(
                            env_or_cenv, masks_np, scores_np, operation, True
                        )
                    else:
                        from .unified import project_organic_3d

                        move = project_organic_3d(
                            env_or_cenv, masks_np, scores_np, operation, True
                        )
                    if diagnostics is not None:
                        diagnostics.record(
                            scale=scale,
                            operation=operation,
                            retry=attempt,
                            requested=n_r + n_a,
                            masks=masks_np,
                            proposed=self.last_generated_count,
                            move=move,
                            proposal_seconds=projection_start - proposal_start,
                            projection_seconds=time.perf_counter() - projection_start,
                        )
                    if move is not None:
                        if training and on_policy_distill:
                            elig_c = _pooled_edit_eligibility(
                                w_rem, w_add, consumed, kernel, stride
                            )
                            if t_mean_all is not None:
                                teacher_mean = t_mean_all[0, consumed].detach()
                            else:
                                des_c = compute_spatial_desirability(
                                    env_or_cenv,
                                    w_rem,
                                    w_add,
                                    ln_f,
                                    consumed,
                                    kernel,
                                    stride,
                                    z_slice=z_act,
                                )
                                teacher_mean = (2.0 * des_c).clamp(-2.8, 2.8).detach()
                        elif training and self.spatial_aux_weight > 0.0:
                            elig_c = _pooled_edit_eligibility(
                                w_rem, w_add, consumed, kernel, stride
                            )
                            des_c = compute_spatial_desirability(
                                env_or_cenv,
                                w_rem,
                                w_add,
                                ln_f,
                                consumed,
                                kernel,
                                stride,
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
                        is_coarse=bool(is_coarse),
                        scale=int(scale),
                        stride=tuple(stride),
                        op=op.detach(),
                        legal=legal.detach(),
                        value=value.detach(),
                        latent=latent.detach() if move is not None else None,
                        latents=latent_draws,
                        drawn_channels=consumed,
                        consumed=consumed if move is not None else None,
                        elig_c=elig_c.detach() if elig_c is not None else None,
                        local_adv=local_adv.detach() if local_adv is not None else None,
                        teacher_mean=teacher_mean,
                        teacher_logits=teacher_logits,
                        heuristic_op=heuristic_op,
                        aux_weight=self.spatial_aux_weight,
                        reward=None,
                        done=False,
                    )
                )
                self.last_attempt_recorded = True
        return move

    def record(self, reward):
        if not self.tape:
            return
        self.tape[-1]["reward"] = torch.as_tensor(
            reward, dtype=torch.float32, device=self.device
        ).detach()

    def add_reward(self, reward):
        if not self.tape:
            return
        r = torch.as_tensor(reward, dtype=torch.float32, device=self.device).detach()
        last = self.tape[-1]
        last["reward"] = r if last["reward"] is None else (last["reward"] + r)

    def mark_done(self):
        if self.tape:
            self.tape[-1]["done"] = True

    @staticmethod
    def _masked_spatial_loss(pred, target, eligible):
        error = F.smooth_l1_loss(pred, target, reduction="none")
        return (error * eligible).sum() / eligible.sum().clamp_min(1.0)

    def finish(self, bootstrap=0.0):
        if not self.tape:
            return dict(updates=self.updates, loss=0.0, gradientNorm=0.0)
        imitation = self.training_objective == "imitation"
        on_policy_distill = self.training_objective == "on_policy_distill"
        if not imitation:
            adv, returns = generalized_advantages(self.tape, bootstrap, self.device)

        groups = {}
        for t, entry in enumerate(self.tape):
            key = (
                entry["is_coarse"],
                entry.get("scale", 2 if entry["is_coarse"] else 1),
                entry["stride"],
                tuple(entry["obs"].shape),
            )
            groups.setdefault(key, []).append(t)

        values_out = [None] * len(self.tape)
        ent_terms = []
        actor_terms = []
        stds = self.model.patch_log_std.clamp(-1.4, -0.3).exp()
        for (is_c, st_scale, stride, _shape), idxs in groups.items():
            obs_b = torch.stack([self.tape[t]["obs"] for t in idxs], dim=0)
            goal_b = torch.stack([self.tape[t]["goal"] for t in idxs], dim=0)
            mean_b, val_b, logits_b = self.model.forward_stage_batch(
                obs_b, goal_b, is_coarse=is_c, stride=stride, scale=st_scale
            )
            for k, t in enumerate(idxs):
                entry = self.tape[t]
                values_out[t] = val_b[k]
                dist_op = Categorical(
                    logits=logits_b[k].masked_fill(~entry["legal"], float("-inf"))
                )
                logp = dist_op.log_prob(entry["op"])
                if imitation:
                    term = -logp
                    if entry["teacher_mean"] is not None:
                        term = term + self._masked_spatial_loss(
                            mean_b[k, entry["drawn_channels"]],
                            entry["teacher_mean"],
                            entry["elig_c"],
                        )
                    actor_terms.append(term)
                    continue
                ent_terms.append(dist_op.entropy())
                a_ep = adv[t].detach()
                aux_weight = entry.get("aux_weight", self.spatial_aux_weight)
                rl_scale = 0.25 if (on_policy_distill and aux_weight > 0.0) else 1.0
                term = -(logp * a_ep) * rl_scale
                draws = entry.get("latents") or (
                    [entry["latent"]] if entry["latent"] is not None else []
                )
                if draws:
                    consumed = entry.get("drawn_channels") or entry["consumed"]
                    mean = mean_b[k, consumed]
                    std = stds[consumed][:, None, None, None]
                    dist_s = Normal(mean, std, validate_args=False)
                    # Scale-normalize path log-prob so XS/S/M/L/XL keep balanced gradients.
                    scale_norm = math.sqrt(max(1, draws[0].numel()))
                    path_logp = torch.stack(
                        [dist_s.log_prob(draw).sum() / scale_norm for draw in draws]
                    ).sum()
                    term = term - path_logp * a_ep * rl_scale
                    if aux_weight > 0.0 and entry["elig_c"] is not None:
                        if on_policy_distill and entry.get("teacher_mean") is not None:
                            term = term + (
                                4.0 * aux_weight
                            ) * self._masked_spatial_loss(
                                mean, entry["teacher_mean"], entry["elig_c"]
                            )
                        elif (
                            not on_policy_distill
                            and entry["latent"] is not None
                            and entry.get("local_adv") is not None
                        ):
                            elig_c = entry["elig_c"]
                            ecnt = elig_c.sum().clamp_min(1.0)
                            sp_lp = dist_s.log_prob(entry["latent"]) * elig_c
                            term = term - aux_weight * (
                                (sp_lp * entry["local_adv"]).sum() / ecnt.sqrt()
                            )
                if on_policy_distill and aux_weight > 0.0:
                    if entry.get("teacher_logits") is not None:
                        valid_mask = entry["legal"]
                        p_t = F.softmax(
                            entry["teacher_logits"].masked_fill(
                                ~valid_mask, float("-inf")
                            ),
                            dim=-1,
                        )
                        log_p_s = F.log_softmax(
                            logits_b[k].masked_fill(~valid_mask, float("-inf")),
                            dim=-1,
                        )
                        op_kl = (
                            p_t[valid_mask]
                            * (
                                p_t[valid_mask].clamp_min(1e-8).log()
                                - log_p_s[valid_mask]
                            )
                        ).sum()
                        term = term + (0.5 * aux_weight) * op_kl
                    elif entry.get("heuristic_op") is not None:
                        term = term - 0.15 * aux_weight * dist_op.log_prob(
                            entry["heuristic_op"]
                        )
                actor_terms.append(term)
        actor = torch.stack(actor_terms).mean()
        loss = actor
        if not imitation:
            critic = F.smooth_l1_loss(torch.stack(values_out), returns)
            loss = loss + 0.5 * critic - 0.005 * torch.stack(ent_terms).mean()
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
            loss=float(loss.detach().cpu()),
            actorLoss=float(actor.detach().cpu()),
            gradientNorm=float(norm.detach().cpu()),
        )
        self.clear()
        return result

    def state(self):
        return dict(
            format="voxelform.multiscale.v1",
            samplingSemantics="gaussian-retry-path-normalized.v1",
            fineConfig=asdict(self.fine_config),
            block=list(self.block),
            inputChannels=self.input_channels,
            spatialAuxWeight=self.spatial_aux_weight,
            trainingObjective=self.training_objective,
            decoderRetrySchedule=self.decoder_retry_schedule,
            evaluationTemperature=self.evaluation_temperature,
            evaluationNoiseScale=self.evaluation_noise_scale,
            lr=self.lr,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            updates=self.updates,
            directMaskUpdates=self.direct_updates,
        )

    def load(self, state):
        if state.get("format") != "voxelform.multiscale.v1":
            raise ValueError("Multiscale checkpoint required")
        ch = int(state.get("inputChannels", 12))
        if ch != self.input_channels:
            self.input_channels = ch
            self.model = CoarseToFineNetwork(ch).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(state.get("lr", self.lr)),
                weight_decay=1e-4,
            )
        self.spatial_aux_weight = float(
            state.get("spatialAuxWeight", self.spatial_aux_weight)
        )
        self.evaluation_temperature = float(
            state.get("evaluationTemperature", self.evaluation_temperature)
        )
        self.evaluation_noise_scale = float(
            state.get("evaluationNoiseScale", self.evaluation_noise_scale)
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))
        self.direct_updates = int(state.get("directMaskUpdates", 0))
        self.clear()


def run_cascade_episode(
    env,
    coarse_envs,
    learner,
    initial_storeys,
    objective_fn,
    training=True,
    rollout_steps=16,
    audit_steps=5,
    step_reward_fn=None,
    terminal_reward_fn=None,
    reward_profile="default",
):
    """Run an arbitrary N-stage coarse -> ... -> fine episode and return timing + exact fine metrics."""
    t0 = time.perf_counter()
    tape_start = len(learner.tape)
    coarse_envs = list(coarse_envs)
    dense_potential = reward_profile == "dense_potential"
    stage_steps = []
    trace = []
    attempt_idx = 0
    budget_total = 0
    reward_sum = 0.0

    if coarse_envs:
        coarse_envs[0].reset(initial_storeys)
        start_fine = coarse_envs[0].start_count_fine
        for idx, ce in enumerate(coarse_envs):
            budget_total += ce.unified_limit
            c_steps = 0
            prev_core = float(getattr(ce, "core_score", 0.0))
            prev_pot = coarse_potential(ce) if (training and dense_potential) else 0.0
            while not ce.near_far and ce.unified_actions < ce.unified_limit:
                move = learner.choose_stage(ce, is_coarse=True, training=training)
                ce.unified_actions += 1
                attempt_idx += 1
                op = learner.last_operation or "none"
                if op == "finish":
                    rew = 0.0
                    if training and learner.last_attempt_recorded:
                        learner.record(rew)
                    reward_sum += rew
                    trace.append(
                        dict(
                            attempt=attempt_idx,
                            stage=f"coarse_{ce.bxy}x{ce.bz}",
                            operation=op,
                            accepted=True,
                            removed=0,
                            added=0,
                            far=ce.far,
                        )
                    )
                    break
                if move is None:
                    rew = -0.02
                    if training and learner.last_attempt_recorded:
                        learner.record(rew)
                    reward_sum += rew
                    trace.append(
                        dict(
                            attempt=attempt_idx,
                            stage=f"coarse_{ce.bxy}x{ce.bz}",
                            operation=op,
                            accepted=False,
                            removed=0,
                            added=0,
                            far=ce.far,
                        )
                    )
                    continue
                ce.apply_move(move)
                c_steps += 1
                if learner.fine_config.allocation == "feedback":
                    ce.workload.observe(
                        int(move.remove.sum() + move.add.sum()),
                        int(move.remove.sum() + move.add.sum()),
                    )
                if dense_potential:
                    if training:
                        cur_pot = coarse_potential(ce)
                        rew = 1.5 * (cur_pot - prev_pot) - 0.002
                        prev_pot = cur_pot
                    else:
                        rew = -0.002
                else:
                    delta = 0.0
                    if training and getattr(ce, "cores_enabled", False):
                        cur_core = float(getattr(ce, "core_score", 0.0))
                        delta = cur_core - prev_core
                        prev_core = cur_core
                    rew = 0.35 * delta - 0.002 if step_reward_fn is not None else -0.002
                if training and learner.last_attempt_recorded:
                    learner.record(rew)
                reward_sum += rew
                trace.append(
                    dict(
                        attempt=attempt_idx,
                        stage=f"coarse_{ce.bxy}x{ce.bz}",
                        operation=op,
                        accepted=True,
                        removed=int(move.remove.sum()),
                        added=int(move.add.sum()),
                        far=ce.far,
                    )
                )
            stage_steps.append(c_steps)
            if idx + 1 < len(coarse_envs):
                ce.upsample_to_coarse(coarse_envs[idx + 1])
            else:
                ce.upsample_to_fine()
        env.start_count = start_fine
    else:
        solid = env.allowed.clone()
        solid[initial_storeys:] = False
        env.restore_occupancy(solid)
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        env._ln6_cache = None
        env.unified_limit = max(8, env.workload.limit + 8)
        env.unified_actions = 0
        env.termination = None
        start_fine = env.start_count

    budget_total += env.unified_limit
    fine_steps = 0
    obj0, d0 = objective_fn(env)
    prev_core_fine = float(d0.get("coreScore") or 0.0)
    prev_fine_pot = (
        fine_potential(env, obj0, start_fine) if (training and dense_potential) else 0.0
    )
    while not env.terminal:
        move = learner.choose_stage(env, is_coarse=False, training=training)
        env.unified_actions += 1
        attempt_idx += 1
        op = learner.last_operation or "none"
        if op == "finish":
            env.termination = "far_reached"
            rew = 0.0
            if training and learner.last_attempt_recorded:
                learner.record(rew)
            reward_sum += rew
            trace.append(
                dict(
                    attempt=attempt_idx,
                    stage="fine",
                    operation=op,
                    accepted=True,
                    removed=0,
                    added=0,
                    far=env.far,
                )
            )
            break
        if move is None:
            rew = -0.01 if step_reward_fn is not None else -0.02
            if training and learner.last_attempt_recorded:
                learner.record(rew)
            reward_sum += rew
            trace.append(
                dict(
                    attempt=attempt_idx,
                    stage="fine",
                    operation=op,
                    accepted=False,
                    removed=0,
                    added=0,
                    far=env.far,
                )
            )
            if env.unified_actions >= env.unified_limit:
                env.termination = (
                    "far_reached" if env.far_satisfied else "step_budget_exhausted"
                )
                break
            continue
        commit_fine(env, move, validated=True)
        fine_steps += 1
        if learner.fine_config.allocation == "feedback":
            env.workload.observe(
                int(move.remove.sum() + move.add.sum()),
                int(move.remove.sum() + move.add.sum()),
            )
        if dense_potential:
            if env.edits % audit_steps == 0:
                env.evaluate()
            elif training and getattr(env, "cores_enabled", False):
                env.update_cores()
            if training:
                cur_obj, _ = objective_fn(env)
                cur_fine_pot = fine_potential(env, cur_obj, start_fine)
                rew = 2.0 * (cur_fine_pot - prev_fine_pot) + (
                    -0.005 if env.far_satisfied else 0.0
                )
                prev_fine_pot = cur_fine_pot
            else:
                rew = -0.005 if env.far_satisfied else 0.0
        else:
            delta = 0.0
            if (
                training
                and getattr(env, "cores_enabled", False)
                and step_reward_fn is not None
            ):
                env.update_cores()
                cm = env.core_metrics
                c_sc = max(
                    0.0,
                    (1.0 - float(cm["coreUncoveredFraction"]))
                    * (1.0 - 0.5 * float(cm["coreDistBadness"])),
                )
                delta = c_sc - prev_core_fine
                prev_core_fine = c_sc
            rew = (
                step_reward_fn(env, delta)
                if step_reward_fn is not None
                else (-0.005 if env.far_satisfied else 0.0)
            )
        if training and learner.last_attempt_recorded:
            learner.record(rew)
        reward_sum += rew
        trace.append(
            dict(
                attempt=attempt_idx,
                stage="fine",
                operation=op,
                accepted=True,
                removed=int(move.remove.sum()),
                added=int(move.add.sum()),
                far=env.far,
            )
        )
        if env.unified_actions >= env.unified_limit:
            env.termination = (
                "far_reached" if env.far_satisfied else "step_budget_exhausted"
            )
            break
        if not dense_potential and env.edits % audit_steps == 0:
            env.evaluate()

    if not env.terminal:
        env.termination = (
            "far_reached" if env.far_satisfied else "step_budget_exhausted"
        )
    if env.last_audit_edit != env.edits:
        env.evaluate()
    obj, diag = objective_fn(env)
    learning = None
    if terminal_reward_fn is not None:
        term_rew = terminal_reward_fn(env, obj)
    else:
        far_pen = (
            0.0
            if env.far_satisfied
            else -(
                0.35
                + 0.85
                * abs(env.count - env.goal_count)
                / max(1, abs(start_fine - env.goal_count))
            )
        )
        term_rew = 5.0 * (obj + far_pen)
    reward_sum += term_rew
    if training:
        learning = finish_episode_rollout(learner, tape_start, term_rew, rollout_steps)
    sync(env.device)
    dt = time.perf_counter() - t0
    m = env.metrics()
    fa = [int(x) for x in env.occupied.sum((1, 2)).detach().cpu().tolist()]
    return dict(
        seconds=round(dt, 4),
        stageSteps=stage_steps + [fine_steps],
        coarseSteps=sum(stage_steps),
        fineSteps=fine_steps,
        totalSteps=sum(stage_steps) + fine_steps,
        steps=fine_steps + sum(stage_steps),
        attempts=attempt_idx,
        budget=budget_total,
        reward=reward_sum,
        success=bool(env.termination == "far_reached"),
        termination=env.termination,
        far=round(float(m["far"]), 4),
        farError=float(env.far_error),
        organicObjective=round(float(obj), 4),
        rawObjective=float(obj),
        usableGoodArea=round(float(diag["usableGoodArea"]), 1),
        directSunProxy=round(float(m["directSunProxy"]), 4),
        diffuseFacadeProxy=round(float(m["diffuseFacadeProxy"]), 4),
        longNarrowFraction=round(float(diag["longNarrowFraction"]), 4),
        slimVoxels=int(diag["slimVoxels"]),
        cantileverVoxels=int(diag["cantileverVoxels"]),
        topFloor=max((i + 1 for i, v in enumerate(fa) if v > 0), default=0),
        floorAreas=fa,
        diagnostics=diag,
        learning=learning,
        trace=trace,
    )


def run_episode(
    env,
    cenv,
    learner,
    initial_storeys,
    objective_fn,
    training=True,
    rollout_steps=16,
    audit_steps=5,
):
    """Run one two-stage coarse-to-fine episode."""
    return run_cascade_episode(
        env,
        [cenv],
        learner,
        initial_storeys,
        objective_fn,
        training=training,
        rollout_steps=rollout_steps,
        audit_steps=audit_steps,
    )
