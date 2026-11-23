"""Export a trained VoxelForm-RL raw showcase massing, vertical cores, and urban context for Blender."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
import torch

from voxelform.coarse_to_fine import (
    CoarseEnv,
    CoarseToFineLearner,
    run_cascade_episode,
    tier_cascade_blocks,
)
from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.experiment import (
    CORE_LAYOUT_VERSION,
    PROTOCOL,
    ExperimentConfig,
    seed_everything,
)
from voxelform.objectives import organic_objective
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.shadows import device_name
from voxelform.unified import UnifiedLearner, commit

ROOT = Path(__file__).resolve().parent.parent
STOREY_HEIGHT = 3.0

SHOWCASE_PRESETS: dict[str, int] = {
    "m_prs_0315": 9003,
    "m_hk_0199": 9005,
    "m_nyc_0003": 9002,
    "xs_tokyo_0133": 9022,
    "s_sgp_0345": 9002,
    "m_tokyo_0135": 9007,
    "l_bcn_0074": 9008,
    "xl_tokyo_0185": 9009,
}


class _IndexedMeshBuilder:
    """Accumulate disjoint quads or extruded prisms into a compact indexed mesh."""

    def __init__(self):
        self.verts: list[list[float]] = []
        self.faces: list[list[int]] = []
        self._vmap: dict[tuple[float, float, float], int] = {}

    def _get_vert(self, x: float, y: float, z: float) -> int:
        key = (round(float(x), 4), round(float(y), 4), round(float(z), 4))
        idx = self._vmap.get(key)
        if idx is None:
            idx = len(self.verts)
            self._vmap[key] = idx
            self.verts.append([key[0], key[1], key[2]])
        return idx

    def add_quad(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        p3: tuple[float, float, float],
    ):
        self.faces.append(
            [
                self._get_vert(*p0),
                self._get_vert(*p1),
                self._get_vert(*p2),
                self._get_vert(*p3),
            ]
        )

    def add_prism(
        self,
        geom,
        z0: float,
        z1: float,
        cx: float = 0.0,
        cy: float = 0.0,
        *,
        z_split: float | None = None,
        cap_bottom: bool = True,
    ):
        if z1 <= z0 + 1e-5 or geom is None or geom.is_empty:
            return
        for poly in _iter_polygons(geom):
            _extrude_polygon_into(
                self, poly, z0, z1, cx, cy, z_split=z_split, cap_bottom=cap_bottom
            )

    def to_dict(self) -> dict:
        return {"vertices": self.verts, "faces": self.faces}


def _iter_polygons(geom):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        if geom.area >= 0.02:
            yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for sub in geom.geoms:
            yield from _iter_polygons(sub)


def _iter_lines(geom, min_len: float = 3.0):
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        if geom.length >= min_len:
            yield geom
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for sub in geom.geoms:
            yield from _iter_lines(sub, min_len=min_len)


def _extrude_polygon_into(
    builder: _IndexedMeshBuilder,
    poly: Polygon,
    z0: float,
    z1: float,
    cx: float,
    cy: float,
    *,
    z_split: float | None = None,
    cap_bottom: bool = True,
):
    z_levels = (
        [z0, z_split, z1]
        if (z_split is not None and z0 + 0.05 < z_split < z1 - 0.05)
        else [z0, z1]
    )
    rings = [poly.exterior] + list(poly.interiors)
    ring_maps = []
    for ring_idx, ring in enumerate(rings):
        pts = list(ring.coords)[:-1]
        if len(pts) < 3:
            continue
        # Normalize exterior CCW and holes CW before emitting outward walls
        is_ccw = shapely.LinearRing(pts).is_ccw
        if (ring_idx == 0 and not is_ccw) or (ring_idx > 0 and is_ccw):
            pts = list(reversed(pts))
        bases = []
        for zl in z_levels:
            base = len(builder.verts)
            for px, py in pts:
                builder.verts.append(
                    [round(float(px) - cx, 4), round(float(py) - cy, 4), round(zl, 4)]
                )
            bases.append(base)
        n = len(pts)
        for lvl in range(len(z_levels) - 1):
            b0, b1 = bases[lvl], bases[lvl + 1]
            for i in range(n):
                j = (i + 1) % n
                builder.faces.append([b0 + i, b0 + j, b1 + j, b1 + i])
        ring_maps.append((pts, bases[0], bases[-1]))

    if not ring_maps:
        return
    # Exact vertex map from snapped coords to cap vertex indices so shell stays watertight
    coord_to_idx: dict[tuple[float, float], tuple[int, int]] = {}
    for pts, b_bot, b_top in ring_maps:
        for idx, (px, py) in enumerate(pts):
            coord_to_idx[(round(float(px), 4), round(float(py), 4))] = (
                b_bot + idx,
                b_top + idx,
            )

    for tri in shapely.constrained_delaunay_triangles(poly).geoms:
        if tri.area <= 1e-4 or not poly.covers(tri.representative_point()):
            continue
        tpts = list(tri.exterior.coords)[:-1]
        if len(tpts) != 3:
            continue
        # Ensure CCW order for +Z top cap, CW for -Z bottom cap
        if not shapely.LinearRing(tpts).is_ccw:
            tpts = list(reversed(tpts))
        bot_ids, top_ids = [], []
        for px, py in tpts:
            key = (round(float(px), 4), round(float(py), 4))
            pair = coord_to_idx.get(key)
            if pair is None:
                ib = len(builder.verts)
                builder.verts.append(
                    [round(float(px) - cx, 4), round(float(py) - cy, 4), round(z0, 4)]
                )
                it = len(builder.verts)
                builder.verts.append(
                    [round(float(px) - cx, 4), round(float(py) - cy, 4), round(z1, 4)]
                )
                pair = (ib, it)
                coord_to_idx[key] = pair
            bot_ids.append(pair[0])
            top_ids.append(pair[1])
        builder.faces.append(top_ids)
        if cap_bottom:
            builder.faces.append([bot_ids[0], bot_ids[2], bot_ids[1]])


def _greedy_merge_2d(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Decompose a 2D boolean mask into disjoint maximal axis-aligned (r0, c0, r1, c1) rectangles."""
    if not grid.any():
        return []
    rem = grid.copy()
    nr, nc = rem.shape
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(nr):
        c0 = 0
        while c0 < nc:
            if not rem[r0, c0]:
                c0 += 1
                continue
            c1 = c0 + 1
            while c1 < nc and rem[r0, c1]:
                c1 += 1
            r1 = r0 + 1
            while r1 < nr and rem[r1, c0:c1].all():
                r1 += 1
            rem[r0:r1, c0:c1] = False
            rects.append((r0, c0, r1, c1))
            c0 = c1
    return rects


def _hero_grid_module(hero_style: str) -> int:
    return {"curtain_frame": 2, "shadow_box": 3, "ribbon_louver": 3, "raw": 0}.get(
        hero_style, 2
    )


def _iter_wall_rects(
    slice_2d: np.ndarray,
    merge_vertical: bool,
    grid_module: int = 0,
) -> list[tuple[int, int, int, int]]:
    if not slice_2d.any():
        return []
    if merge_vertical:
        return _greedy_merge_2d(slice_2d)
    rects: list[tuple[int, int, int, int]] = []
    nz, nu = slice_2d.shape
    for z in range(nz):
        row = slice_2d[z]
        if not row.any():
            continue
        u = 0
        while u < nu:
            if not row[u]:
                u += 1
                continue
            u1 = u + 1
            while u1 < nu and row[u1]:
                u1 += 1
            if grid_module > 1:
                cur = u
                while cur < u1:
                    nxt = min(u1, ((cur // grid_module) + 1) * grid_module)
                    rects.append((z, cur, z + 1, nxt))
                    cur = nxt
            else:
                rects.append((z, u, z + 1, u1))
            u = u1
    return rects


def _floor_z(z: int, gf_height: float) -> float:
    return 0.0 if z == 0 else float(gf_height) + (z - 1) * STOREY_HEIGHT


def _mask_to_geom(mask_2d: np.ndarray):
    ny, nx = mask_2d.shape
    boxes = []
    for y in range(ny):
        row = mask_2d[y]
        if not row.any():
            continue
        xs = np.flatnonzero(row)
        start = prev = int(xs[0])
        for x in xs[1:]:
            xi = int(x)
            if xi != prev + 1:
                boxes.append(box(start, y, prev + 1, y + 1))
                start = xi
            prev = xi
        boxes.append(box(start, y, prev + 1, y + 1))
    if not boxes:
        return Polygon()
    g = unary_union(boxes)
    return g if g.is_valid else g.buffer(0)


def _context_gf_height(height: float) -> float:
    if height >= 28.0:
        return 5.2
    if height >= 18.0:
        return 4.8
    if height >= 10.0:
        return 4.4
    return min(float(height), 4.0)


def _build_exterior_shell(
    mask: np.ndarray,
    solid: np.ndarray,
    cx: float,
    cy: float,
    *,
    gf_height: float = STOREY_HEIGHT,
    include_ground: bool,
    merge_vertical_walls: bool,
    grid_module: int = 0,
    include_roof: bool = True,
) -> _IndexedMeshBuilder:
    """Emit outward CCW quads for faces of `mask` whose neighbor in `solid` is empty."""
    builder = _IndexedMeshBuilder()
    if not mask.any():
        return builder

    nz, ny, nx = mask.shape
    pad = np.pad(solid, 1, constant_values=False)

    exp_px = mask & ~pad[1:-1, 1:-1, 2:]
    exp_mx = mask & ~pad[1:-1, 1:-1, :-2]
    exp_py = mask & ~pad[1:-1, 2:, 1:-1]
    exp_my = mask & ~pad[1:-1, :-2, 1:-1]
    exp_pz = mask & ~pad[2:, 1:-1, 1:-1]
    exp_mz = mask & ~pad[:-2, 1:-1, 1:-1]
    if not include_ground:
        exp_mz[0] = False

    for x in range(nx):
        if exp_px[:, :, x].any():
            xp = float(x + 1) - cx
            for z0, y0, z1, y1 in _iter_wall_rects(
                exp_px[:, :, x], merge_vertical_walls, grid_module
            ):
                za, zb = _floor_z(z0, gf_height), _floor_z(z1, gf_height)
                builder.add_quad(
                    (xp, float(y0) - cy, za),
                    (xp, float(y1) - cy, za),
                    (xp, float(y1) - cy, zb),
                    (xp, float(y0) - cy, zb),
                )
        if exp_mx[:, :, x].any():
            xm = float(x) - cx
            for z0, y0, z1, y1 in _iter_wall_rects(
                exp_mx[:, :, x], merge_vertical_walls, grid_module
            ):
                za, zb = _floor_z(z0, gf_height), _floor_z(z1, gf_height)
                builder.add_quad(
                    (xm, float(y0) - cy, za),
                    (xm, float(y0) - cy, zb),
                    (xm, float(y1) - cy, zb),
                    (xm, float(y1) - cy, za),
                )

    for y in range(ny):
        if exp_py[:, y, :].any():
            yp = float(y + 1) - cy
            for z0, x0, z1, x1 in _iter_wall_rects(
                exp_py[:, y, :], merge_vertical_walls, grid_module
            ):
                za, zb = _floor_z(z0, gf_height), _floor_z(z1, gf_height)
                builder.add_quad(
                    (float(x0) - cx, yp, za),
                    (float(x0) - cx, yp, zb),
                    (float(x1) - cx, yp, zb),
                    (float(x1) - cx, yp, za),
                )
        if exp_my[:, y, :].any():
            ym = float(y) - cy
            for z0, x0, z1, x1 in _iter_wall_rects(
                exp_my[:, y, :], merge_vertical_walls, grid_module
            ):
                za, zb = _floor_z(z0, gf_height), _floor_z(z1, gf_height)
                builder.add_quad(
                    (float(x0) - cx, ym, za),
                    (float(x1) - cx, ym, za),
                    (float(x1) - cx, ym, zb),
                    (float(x0) - cx, ym, zb),
                )

    for z in range(nz):
        if include_roof and exp_pz[z].any():
            zp = _floor_z(z + 1, gf_height)
            for y0, x0, y1, x1 in _greedy_merge_2d(exp_pz[z]):
                builder.add_quad(
                    (float(x0) - cx, float(y0) - cy, zp),
                    (float(x1) - cx, float(y0) - cy, zp),
                    (float(x1) - cx, float(y1) - cy, zp),
                    (float(x0) - cx, float(y1) - cy, zp),
                )
        if exp_mz[z].any():
            zm = _floor_z(z, gf_height)
            for y0, x0, y1, x1 in _greedy_merge_2d(exp_mz[z]):
                builder.add_quad(
                    (float(x0) - cx, float(y0) - cy, zm),
                    (float(x0) - cx, float(y1) - cy, zm),
                    (float(x1) - cx, float(y1) - cy, zm),
                    (float(x1) - cx, float(y0) - cy, zm),
                )

    return builder


def _extract_context_building_geom(b: dict):
    verts_raw = b.get("vertices") or []
    faces_raw = b.get("faces") or []
    if not verts_raw or not faces_raw:
        return None, 0.0, 0.0
    verts = np.asarray(verts_raw, dtype=np.float32)
    z_min = float(verts[:, 2].min())
    z_max = float(b.get("height") or verts[:, 2].max())
    if z_max <= z_min + 0.25:
        return None, z_min, z_max
    faces = np.asarray(faces_raw, dtype=np.int32)
    polys = []
    for f in faces:
        if len(f) < 3:
            continue
        zf = verts[f, 2]
        if np.all(zf <= z_min + 0.1) or np.all(zf >= z_max - 0.1):
            p = Polygon(verts[f, :2])
            if not p.is_valid:
                p = p.buffer(0)
            if p.area > 0.05:
                polys.append(p)
    if not polys:
        return None, z_min, z_max
    g = unary_union(polys)
    return (g if g.is_valid else g.buffer(0)), z_min, z_max


def _patch_cfg() -> PatchConfig:
    return PatchConfig(
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


def _load_policy(
    run_dir: Path, temp: float, noise: float, checkpoint: Path | None = None
):
    cfg = ExperimentConfig.read(run_dir / "config.json")
    dev = device_name(cfg.device)
    ckpt_path = checkpoint or (run_dir / f"seed-{cfg.seeds[0]}" / "latest.pt")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if (
        saved.get("format") != PROTOCOL
        or saved.get("coreLayoutVersion") != CORE_LAYOUT_VERSION
    ):
        raise ValueError("Incompatible checkpoint protocol or core layout version")
    seed = saved["seed"]
    if seed not in cfg.seeds:
        raise ValueError("Checkpoint seed is outside the configured run")
    manifest_path = run_dir / "manifest.json"
    manifest_hash = (
        json.loads(manifest_path.read_text()).get("configHash")
        if manifest_path.is_file()
        else None
    )
    if saved.get("configHash") not in (cfg.fingerprint, manifest_hash):
        raise ValueError("Checkpoint configHash does not match run config")

    if cfg.multiscale_schedule != "none":
        policy = CoarseToFineLearner(
            dev,
            seed=seed,
            block=(1, 2, 2),
            fine_config=_patch_cfg(),
            input_channels=14 if cfg.cores_enabled else 12,
            spatial_aux_weight=cfg.spatial_aux_weight,
            evaluation_temperature=temp,
            evaluation_noise_scale=noise,
            lr=1.2e-3,
            decoder_retry_schedule=cfg.decoder_retry_schedule,
            spatial_aux_mode=cfg.spatial_aux_mode,
        )
    else:
        policy = UnifiedLearner(
            dev,
            seed=seed,
            config=_patch_cfg(),
            amount_feedback=True,
            net_guided=True,
            organic_3d=True,
            spatial_aux_weight=cfg.spatial_aux_weight,
        )
    policy.load(saved["policy"])
    policy.evaluation_temperature = float(temp)
    policy.evaluation_noise_scale = float(noise)
    return cfg, dev, policy


def _run_trained_episode(
    cfg: ExperimentConfig, dev: str, policy, site_id: str, eval_seed: int
):
    context = ContextLibrary().load(
        site_id, floors=cfg.ceiling_storeys, grid_alignment="auto_fit"
    )
    env = Environment(
        (cfg.ceiling_storeys, cfg.synthetic_width, cfg.synthetic_length),
        seed=9001,
        context=context,
        device=dev,
        sun_count=8,
        target_far=cfg.target_far,
        far_tolerance=cfg.far_tolerance,
        frontier_factory=functools.partial(
            PatchFacades, config=_patch_cfg(), depth_device=dev
        ),
        fast_self_shadows=True,
        fast_diffuse=True,
        cores_enabled=cfg.cores_enabled,
        min_core_spacing=cfg.min_core_spacing,
        max_core_dist=cfg.max_core_dist,
    )
    env.precompute_external()
    seed_everything(eval_seed, cfg.strict_determinism)

    if isinstance(policy, CoarseToFineLearner):
        coarse_envs = (
            tuple(
                CoarseEnv(env, blk)
                for blk in tier_cascade_blocks(env, schedule=cfg.multiscale_schedule)
            )
            if cfg.multiscale_schedule != "none"
            else ()
        )
        res = run_cascade_episode(
            env,
            coarse_envs,
            policy,
            cfg.initial_storeys,
            organic_objective,
            training=False,
            rollout_steps=cfg.rollout_steps,
            audit_steps=cfg.audit_steps,
            reward_profile=cfg.reward_profile,
        )
        obj = float(res["rawObjective"])
        diag = res["diagnostics"]
    else:
        solid = env.allowed.clone()
        solid[cfg.initial_storeys :] = False
        env.restore_occupancy(solid)
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        env._ln6_cache = None
        env.unified_limit = max(8, env.workload.limit + 8)
        env.unified_actions = 0
        env.termination = None
        while not env.terminal:
            move = policy.choose(env, training=False)
            env.unified_actions += 1
            if policy.last_operation == "finish":
                env.termination = "far_reached"
                break
            if move is not None:
                commit(env, move, validated=True)
                if env.edits % cfg.audit_steps == 0:
                    env.evaluate()
        if not env.terminal:
            env.termination = (
                "far_reached" if env.far_satisfied else "step_budget_exhausted"
            )
        env.evaluate()
        obj, diag = organic_objective(env)

    occ = env.occupied.detach().cpu().numpy().astype(bool)
    core_mask = (env.core_mask.detach().cpu().numpy().astype(bool)) & occ
    return env, occ, core_mask, float(obj), diag


def _probe_road_center_offset(
    mx: float,
    my: float,
    nx: float,
    ny: float,
    road_union,
) -> tuple[float, float] | None:
    """Measure local perpendicular street width (signed_center_offset, width) at (mx, my)."""
    for sign in (1.0, -1.0):
        sx, sy = sign * nx, sign * ny
        start = Point(mx + 0.1 * sx, my + 0.1 * sy)
        hit = LineString(
            [(start.x, start.y), (mx + 14.0 * sx, my + 14.0 * sy)]
        ).intersection(road_union)
        if hit.is_empty:
            continue
        parts = (
            [hit] if hit.geom_type == "LineString" else list(getattr(hit, "geoms", ()))
        )
        for part in parts:
            if (
                part.geom_type == "LineString"
                and start.distance(Point(part.coords[0])) <= 0.35
            ):
                w = float(part.length) + 0.1
                if 3.2 <= w <= 13.5:
                    return sign * (0.5 * w), w
    return None


def _far_enough(
    x: float, y: float, existing: list[tuple[float, float]], min_dist: float
) -> bool:
    min_d2 = min_dist * min_dist
    return all((x - px) ** 2 + (y - py) ** 2 >= min_d2 for px, py in existing)


def _oriented_stripe_poly(
    cx: float, cy: float, angle: float, length: float, width: float
) -> Polygon:
    hl, hw = 0.5 * length, 0.5 * width
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for lx, ly in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)):
        pts.append((cx + ca * lx - sa * ly, cy + sa * lx + ca * ly))
    return Polygon(pts)


def _build_road_markings(road_union, board, building_blockers) -> _IndexedMeshBuilder:
    """Place dashes on a fixed phase along each straight road direction."""
    markings = _IndexedMeshBuilder()
    road_center = road_union.buffer(-2.1)
    if road_center.is_empty:
        return markings

    pitch = 5.6
    allowed = road_union.intersection(board.buffer(-1.0)).difference(building_blockers)
    centers: list[tuple[float, float]] = []
    for poly in _iter_polygons(road_union.simplify(0.08)):
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                length = math.hypot(x1 - x0, y1 - y0)
                if length < 3.0:
                    continue
                ux, uy = (x1 - x0) / length, (y1 - y0) / length
                nx, ny = -uy, ux
                heading = math.atan2(uy, ux)
                start = x0 * ux + y0 * uy
                # Adjacent or opposite curbs on a straight street share the
                # same stations, regardless of where the polygon is split.
                for station in range(
                    math.ceil(start / pitch), math.floor((start + length) / pitch) + 1
                ):
                    distance = station * pitch - start
                    bx, by = x0 + distance * ux, y0 + distance * uy
                    probed = _probe_road_center_offset(bx, by, nx, ny, road_union)
                    if probed is None or probed[1] < 4.4:
                        continue
                    offset, _ = probed
                    px, py = bx + offset * nx, by + offset * ny
                    if not road_center.covers(Point(px, py)) or not _far_enough(
                        px, py, centers, 4.2
                    ):
                        continue
                    stripe = _oriented_stripe_poly(px, py, heading, 2.4, 0.16)
                    if allowed.covers(stripe):
                        markings.add_prism(stripe, 0.022, 0.032)
                        centers.append((px, py))
    return markings


def _extract_street_assets(
    hero_2d,
    ctx_bld_2d: list,
    road_2d: list,
    green_2d: list,
    parcel_2d,
    *,
    board_half: float = 160.0,
    seed: int = 9003,
) -> tuple[_IndexedMeshBuilder, _IndexedMeshBuilder, dict]:
    """Generate full-board non-road sidewalks, full road markings, two-way vehicles, props, and paths."""
    rng = random.Random(seed)
    all_bld_exact = unary_union(
        ([hero_2d] if not hero_2d.is_empty else []) + ctx_bld_2d
    ).buffer(0)
    blockers = []
    if not hero_2d.is_empty:
        blockers.append(hero_2d.buffer(0.85))
    for g in ctx_bld_2d:
        blockers.append(g.buffer(0.65))
    all_bld_union = unary_union(blockers).buffer(0) if blockers else Polygon()

    road_union = unary_union(road_2d).buffer(0) if road_2d else Polygon()
    green_union = unary_union(green_2d).buffer(0) if green_2d else Polygon()
    board_box = box(-board_half, -board_half, board_half, board_half)

    sidewalk_walkable = (
        board_box.difference(road_union) if not road_union.is_empty else board_box
    )
    if not green_union.is_empty and not sidewalk_walkable.is_empty:
        sidewalk_walkable = sidewalk_walkable.difference(green_union)
    sidewalk_walkable = (
        sidewalk_walkable.buffer(0) if not sidewalk_walkable.is_empty else Polygon()
    )

    curb_geom = (
        sidewalk_walkable.difference(parcel_2d).buffer(0)
        if (not parcel_2d.is_empty and not sidewalk_walkable.is_empty)
        else sidewalk_walkable
    )

    sidewalk_curbs = _IndexedMeshBuilder()
    if not curb_geom.is_empty:
        sidewalk_curbs.add_prism(curb_geom, 0.0, 0.18)

    road_markings = _build_road_markings(road_union, board_box, all_bld_union)

    # Extract true road curb edges across the entire road network inside board bounds
    bs_cands: list[tuple[float, float, float, float, float, float, Polygon]] = []
    curb_edges: list[
        tuple[float, float, float, float, float, float, float, float, float]
    ] = []
    edge_roi = board_box.buffer(-2.0)
    sidewalk_inner = (
        sidewalk_walkable.buffer(-0.25) if not sidewalk_walkable.is_empty else Polygon()
    )
    if not road_union.is_empty and not sidewalk_walkable.is_empty:
        for poly in _iter_polygons(road_union):
            for ring in [poly.exterior, *poly.interiors]:
                coords = list(ring.coords)
                for i in range(len(coords) - 1):
                    x0, y0 = float(coords[i][0]), float(coords[i][1])
                    x1, y1 = float(coords[i + 1][0]), float(coords[i + 1][1])
                    seg_len = math.hypot(x1 - x0, y1 - y0)
                    if seg_len < 5.0:
                        continue
                    ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
                    lx_out, ly_out = uy, -ux
                    mid_x, mid_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                    if not edge_roi.contains(Point(mid_x, mid_y)):
                        continue
                    nx_road = ny_road = None
                    for sign in (1.0, -1.0):
                        sw_chk = Point(
                            mid_x + sign * 0.9 * lx_out, mid_y + sign * 0.9 * ly_out
                        )
                        rd_chk = Point(
                            mid_x - sign * 0.9 * lx_out, mid_y - sign * 0.9 * ly_out
                        )
                        if not road_union.contains(sw_chk) and road_union.contains(
                            rd_chk
                        ):
                            nx_road, ny_road = -sign * lx_out, -sign * ly_out
                            break
                    if nx_road is None:
                        continue
                    curb_edges.append(
                        (x0, y0, x1, y1, ux, uy, seg_len, nx_road, ny_road)
                    )
                    road_yaw = math.atan2(ny_road, nx_road) - 0.5 * math.pi

                    if seg_len >= 12.0:
                        bx_m = mid_x - 1.35 * nx_road
                        by_m = mid_y - 1.35 * ny_road
                        if bx_m * bx_m + by_m * by_m <= 45.0**2:
                            bs_box = _oriented_stripe_poly(
                                bx_m, by_m, road_yaw, 4.2, 2.0
                            )
                            if sidewalk_walkable.contains(bs_box) and (
                                all_bld_union.is_empty or all_bld_union.disjoint(bs_box)
                            ):
                                bs_cands.append(
                                    (
                                        bx_m * bx_m + by_m * by_m,
                                        bx_m,
                                        by_m,
                                        road_yaw,
                                        nx_road,
                                        ny_road,
                                        bs_box,
                                    )
                                )

    # Optional space-checked bus stop near hero (max 1)
    bus_stops: list[dict] = []
    bs_info: tuple[float, float, float, float, float, Polygon] | None = None
    bs_cands.sort(key=lambda c: c[0])
    if bs_cands:
        _, bx_m, by_m, ryaw, nx_road, ny_road, bs_box = bs_cands[0]
        z_bs = (
            0.19
            if (not parcel_2d.is_empty and parcel_2d.contains(Point(bx_m, by_m)))
            else 0.18
        )
        bus_stops.append(
            {"pos": [round(bx_m, 3), round(by_m, 3), z_bs], "yaw": round(ryaw, 4)}
        )
        bs_info = (bx_m, by_m, ryaw, nx_road, ny_road, bs_box)

    # Sparse two-way vehicles distributed across every road on the full board
    bus_candidates: list[tuple[float, float, float, Polygon]] = []
    car_candidates: list[tuple[float, float, float, Polygon]] = []
    if not road_union.is_empty:
        veh_roi = board_box.buffer(-3.0)
        for poly in _iter_polygons(road_union):
            for ring in [poly.exterior, *poly.interiors]:
                coords = list(ring.coords)
                for i in range(len(coords) - 1):
                    x0, y0 = float(coords[i][0]), float(coords[i][1])
                    x1, y1 = float(coords[i + 1][0]), float(coords[i + 1][1])
                    seg_len = math.hypot(x1 - x0, y1 - y0)
                    if seg_len < 7.5:
                        continue
                    ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
                    ang = math.atan2(uy, ux)
                    nx_l, ny_l = -uy, ux
                    probed = _probe_road_center_offset(
                        0.5 * (x0 + x1), 0.5 * (y0 + y1), nx_l, ny_l, road_union
                    )
                    if probed is None:
                        continue
                    center_off, road_w = probed
                    c_sign = 1.0 if center_off >= 0.0 else -1.0

                    if seg_len >= 20.0 and road_w >= 5.6:
                        bus_mag = min(1.50, max(1.05, 0.19 * road_w))
                        for base_ang, dir_sign in ((ang, -1.0), (ang + math.pi, 1.0)):
                            b_off = center_off + dir_sign * c_sign * bus_mag
                            dist = rng.uniform(7.5, 11.5)
                            while dist <= seg_len - 7.5:
                                bx_b = x0 + ux * dist + b_off * nx_l
                                by_b = y0 + uy * dist + b_off * ny_l
                                byaw = base_ang + rng.uniform(-0.02, 0.02)
                                bus_box = _oriented_stripe_poly(
                                    bx_b, by_b, byaw, 13.8, 3.3
                                )
                                if (
                                    bx_b * bx_b + by_b * by_b <= 60.0**2
                                    and veh_roi.contains(bus_box)
                                    and road_union.contains(bus_box)
                                    and (
                                        all_bld_union.is_empty
                                        or all_bld_union.disjoint(bus_box)
                                    )
                                ):
                                    bus_candidates.append((bx_b, by_b, byaw, bus_box))
                                dist += rng.uniform(8.0, 12.0)

                    lane_pairs = (
                        ((ang, -1.0), (ang + math.pi, 1.0))
                        if road_w >= 5.5
                        else ((ang, -1.0),)
                    )
                    for base_ang, dir_sign in lane_pairs:
                        dist = rng.uniform(3.2, 6.5)
                        while dist <= seg_len - 3.2:
                            lane_mag = (
                                min(1.85, max(1.45, 0.25 * road_w))
                                + rng.uniform(-0.14, 0.14)
                                if road_w >= 5.5
                                else min(1.30, 0.20 * road_w) + rng.uniform(-0.10, 0.10)
                            )
                            c_off = center_off + dir_sign * c_sign * lane_mag
                            cx_car = x0 + ux * dist + c_off * nx_l
                            cy_car = y0 + uy * dist + c_off * ny_l
                            cyaw = base_ang + rng.uniform(-0.025, 0.025)
                            car_box = _oriented_stripe_poly(
                                cx_car, cy_car, cyaw, 6.0, 2.5
                            )
                            if (
                                veh_roi.contains(car_box)
                                and road_union.contains(car_box)
                                and (
                                    all_bld_union.is_empty
                                    or all_bld_union.disjoint(car_box)
                                )
                            ):
                                car_candidates.append((cx_car, cy_car, cyaw, car_box))
                            dist += rng.uniform(3.5, 6.8)

    chosen_cars: list[dict] = []
    placed_boxes: list[Polygon] = []
    placed_dirs: list[tuple[float, float, float, float, float]] = []
    bus_anchor: tuple[float, float] | None = None
    if bus_candidates:
        if bs_info is not None:
            bus_candidates.sort(
                key=lambda c: math.hypot(c[0] - bs_info[0], c[1] - bs_info[1])
            )
        else:
            bus_candidates.sort(key=lambda c: c[0] * c[0] + c[1] * c[1])
        bx_b, by_b, bang, bbox = rng.choice(
            bus_candidates[: min(4, len(bus_candidates))]
        )
        chosen_cars.append(
            {
                "pos": [round(bx_b, 3), round(by_b, 3), 0.02],
                "yaw": round(bang, 4),
                "variant": "bus",
            }
        )
        placed_boxes.append(bbox.buffer(1.4))
        placed_dirs.append((bx_b, by_b, bang, math.cos(bang), math.sin(bang)))
        bus_anchor = (bx_b, by_b)

    # Place 1 showcase yellow taxi near hero (<= 65 m), well clear of the bus
    hero_taxi_cands = [
        c for c in car_candidates if c[0] * c[0] + c[1] * c[1] <= 65.0**2
    ]
    hero_taxi_cands.sort(key=lambda c: math.hypot(c[0], c[1]) * rng.uniform(0.6, 1.4))
    for cx_car, cy_car, cyaw, car_box in hero_taxi_cands:
        if (
            bus_anchor is not None
            and math.hypot(cx_car - bus_anchor[0], cy_car - bus_anchor[1]) < 26.0
        ):
            continue
        padded_box = car_box.buffer(1.2)
        if any(not padded_box.disjoint(pb) for pb in placed_boxes):
            continue
        chosen_cars.append(
            {
                "pos": [round(cx_car, 3), round(cy_car, 3), 0.02],
                "yaw": round(cyaw, 4),
                "variant": "taxi",
            }
        )
        placed_boxes.append(padded_box)
        placed_dirs.append((cx_car, cy_car, cyaw, math.cos(cyaw), math.sin(cyaw)))
        break

    # Shuffle remaining candidates uniformly across the full board, up to 10 sparse cars
    rng.shuffle(car_candidates)
    pool_variants = ["tesla", "fiat", "savana", "tesla", "taxi"]
    extra_idx = 0
    for cx_car, cy_car, cyaw, car_box in car_candidates:
        if (
            bus_anchor is not None
            and math.hypot(cx_car - bus_anchor[0], cy_car - bus_anchor[1]) < 26.0
        ):
            continue
        padded_box = car_box.buffer(1.2)
        if any(not padded_box.disjoint(pb) for pb in placed_boxes):
            continue
        ux_c, uy_c = math.cos(cyaw), math.sin(cyaw)
        ok_spacing = True
        for px, py, _pyaw, ux_p, uy_p in placed_dirs:
            same_dir = (ux_c * ux_p + uy_c * uy_p) > 0.5
            req_d = rng.uniform(34.0, 60.0) if same_dir else rng.uniform(22.0, 38.0)
            if math.hypot(cx_car - px, cy_car - py) < req_d:
                ok_spacing = False
                break
        if not ok_spacing:
            continue
        variant = pool_variants[extra_idx % len(pool_variants)]
        chosen_cars.append(
            {
                "pos": [round(cx_car, 3), round(cy_car, 3), 0.02],
                "yaw": round(cyaw, 4),
                "variant": variant,
            }
        )
        placed_boxes.append(padded_box)
        placed_dirs.append((cx_car, cy_car, cyaw, ux_c, uy_c))
        extra_idx += 1
        if extra_idx >= 10:
            break

    # Street lamps along curb edges across the entire board (max 45)
    lamps: list[dict] = []
    lamp_pts: list[tuple[float, float]] = []
    for x0, y0, _x1, _y1, ux, uy, seg_len, nx_road, ny_road in curb_edges:
        road_yaw = math.atan2(ny_road, nx_road) - 0.5 * math.pi
        step = 6.0
        while step <= seg_len - 4.0 and len(lamps) < 45:
            px = x0 + ux * step - 0.50 * nx_road
            py = y0 + uy * step - 0.50 * ny_road
            pt = Point(px, py)
            if (
                edge_roi.contains(pt)
                and (sidewalk_inner.is_empty or sidewalk_inner.contains(pt))
                and not all_bld_union.contains(pt)
                and _far_enough(px, py, lamp_pts, 18.0)
                and (bs_info is None or not bs_info[5].contains(pt))
            ):
                z_l = (
                    0.19
                    if (not parcel_2d.is_empty and parcel_2d.contains(pt))
                    else 0.18
                )
                lamps.append(
                    {
                        "pos": [round(px, 3), round(py, 3), z_l],
                        "yaw": round(road_yaw, 4),
                    }
                )
                lamp_pts.append((px, py))
            step += 20.0

    # Collision-clipped pedestrian splines along sidewalks, bus stop apron, and open courtyard
    blocker_parts = [road_union.buffer(0.65)] if not road_union.is_empty else []
    if not hero_2d.is_empty:
        blocker_parts.append(hero_2d.buffer(1.0))
    blocker_parts.extend(g.buffer(0.85) for g in ctx_bld_2d)
    blocker_parts.extend(Point(x, y).buffer(0.85) for x, y in lamp_pts)
    if bs_info is not None:
        blocker_parts.append(bs_info[5].buffer(0.35))
    blocker_parts.extend(g.buffer(0.4) for g in green_2d)
    path_blockers = unary_union(blocker_parts).buffer(0) if blocker_parts else Polygon()

    def _path_z(px: float, py: float) -> float:
        return (
            0.19
            if (not parcel_2d.is_empty and parcel_2d.contains(Point(px, py)))
            else 0.18
        )

    path_roi = edge_roi
    pedestrian_paths: list[dict] = []
    used_path_zones: list[Polygon] = []

    def _add_clipped_line(line: LineString, min_len: float = 3.0) -> None:
        clipped = line.intersection(path_roi)
        if not path_blockers.is_empty and not clipped.is_empty:
            clipped = clipped.difference(path_blockers)
        if used_path_zones and not clipped.is_empty:
            clipped = clipped.difference(unary_union(used_path_zones))
        for seg in _iter_lines(clipped, min_len=min_len):
            pedestrian_paths.append(
                {
                    "points": [
                        [
                            round(float(px), 3),
                            round(float(py), 3),
                            _path_z(float(px), float(py)),
                        ]
                        for px, py in seg.coords
                    ],
                }
            )
            used_path_zones.append(seg.buffer(0.90))

    for x0, y0, x1, y1, _ux, _uy, _seg_len, nx_road, ny_road in curb_edges:
        sw_line = LineString(
            [
                (x0 - 1.35 * nx_road, y0 - 1.35 * ny_road),
                (x1 - 1.35 * nx_road, y1 - 1.35 * ny_road),
            ]
        )
        _add_clipped_line(sw_line, min_len=3.0)

    if bs_info is not None:
        bx_m, by_m, _ryaw, nx_road, ny_road, _ = bs_info
        wx_dir, wy_dir = -ny_road, nx_road
        wait_line = LineString(
            [
                (
                    bx_m + 1.05 * nx_road - 0.9 * wx_dir,
                    by_m + 1.05 * ny_road - 0.9 * wy_dir,
                ),
                (
                    bx_m + 1.05 * nx_road + 0.9 * wx_dir,
                    by_m + 1.05 * ny_road + 0.9 * wy_dir,
                ),
            ]
        )
        _add_clipped_line(wait_line, min_len=1.5)

    if not parcel_2d.is_empty:
        open_plaza = (
            parcel_2d.difference(all_bld_exact).difference(road_union).buffer(-1.0)
        )
        plaza_polys = sorted(
            (p for p in _iter_polygons(open_plaza) if p.area >= 12.0),
            key=lambda p: p.area,
            reverse=True,
        )
        for p_idx, poly in enumerate(plaza_polys[:2]):
            minx, miny, maxx, maxy = poly.bounds
            if (maxx - minx) >= (maxy - miny):
                cy_p = 0.5 * (miny + maxy)
                axis_line = LineString([(minx, cy_p), (maxx, cy_p)])
            else:
                cx_p = 0.5 * (minx + maxx)
                axis_line = LineString([(cx_p, miny), (cx_p, maxy)])
            _add_clipped_line(
                axis_line.intersection(poly), min_len=3.5 if p_idx == 0 else 4.0
            )

    # Place maquette trees inside green spaces (raised Z=0.20 atop green planters)
    trees: list[dict] = []
    tree_pts: list[tuple[float, float]] = []
    for gp in green_2d:
        for poly in _iter_polygons(
            gp.intersection(Point(0.0, 0.0).buffer(68.0))
            .difference(all_bld_union)
            .buffer(-1.2)
        ):
            minx, miny, maxx, maxy = poly.bounds
            tx = minx + 2.0
            while tx < maxx - 1.5 and len(trees) < 6:
                ty = miny + 2.0
                while ty < maxy - 1.5 and len(trees) < 6:
                    if poly.contains(Point(tx, ty)) and _far_enough(
                        tx, ty, tree_pts, 7.0
                    ):
                        trees.append(
                            {
                                "pos": [round(tx, 3), round(ty, 3), 0.20],
                                "scale": round(0.9 + 0.22 * (len(trees) % 3), 2),
                            }
                        )
                        tree_pts.append((tx, ty))
                    ty += 6.5
                tx += 6.5

    return (
        sidewalk_curbs,
        road_markings,
        {
            "lamps": lamps,
            "bus_stops": bus_stops,
            "pedestrian_paths": pedestrian_paths,
            "cars": chosen_cars,
            "trees": trees,
        },
    )


def export_site_payload(
    cfg: ExperimentConfig,
    dev: str,
    policy,
    site_id: str,
    eval_seed: int,
    temp: float,
    noise: float,
    *,
    gf_height: float = 4.2,
    grid_module: int = 2,
    voxel_output: Path | None = None,
) -> dict:
    env, occ, core_mask, obj, diag = _run_trained_episode(
        cfg, dev, policy, site_id, eval_seed
    )
    if voxel_output is not None:
        voxel_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(voxel_output, occupied=occ, cores=core_mask)
    yy, xx = np.nonzero(occ.any(axis=0))
    cx = 0.5 * (float(xx.min()) + float(xx.max() + 1))
    cy = 0.5 * (float(yy.min()) + float(yy.max() + 1))

    # Extract strictly exterior-facing usable mass faces and proud terracotta cores
    mass = _build_exterior_shell(
        occ & ~core_mask,
        occ,
        cx,
        cy,
        gf_height=gf_height,
        include_ground=False,
        merge_vertical_walls=False,
        grid_module=grid_module,
    )
    cores = _build_exterior_shell(
        core_mask,
        core_mask,
        cx,
        cy,
        gf_height=gf_height,
        include_ground=True,
        merge_vertical_walls=True,
        include_roof=False,
    )

    nz = occ.shape[0]
    occ_2d = [_mask_to_geom(occ[z]) for z in range(nz)]
    core_2d = [_mask_to_geom(core_mask[z]) for z in range(nz)]

    # Elevate core crowns +1.05 m proud above their top roof terrace
    for z in range(nz):
        top_c = core_mask[z] & ~(
            core_mask[z + 1] if z + 1 < nz else np.zeros_like(core_mask[z])
        )
        if top_c.any():
            zt = _floor_z(z + 1, gf_height)
            cores.add_prism(
                _mask_to_geom(top_c), zt, zt + 1.05, cx, cy, cap_bottom=False
            )

    hero_parapets = _IndexedMeshBuilder()
    for z in range(nz):
        roof_mask = (
            occ[z]
            & ~core_mask[z]
            & ~(occ[z + 1] if z + 1 < nz else np.zeros_like(occ[z]))
        )
        if not roof_mask.any():
            continue
        roof_2d = _mask_to_geom(roof_mask)
        inner = roof_2d.buffer(-0.24, join_style="mitre")
        parapet_ring = roof_2d.difference(inner) if not inner.is_empty else roof_2d
        if z + 1 < nz and not occ_2d[z + 1].is_empty:
            parapet_ring = parapet_ring.difference(
                occ_2d[z + 1].buffer(0.02, join_style="mitre")
            )
        if not core_2d[z].is_empty:
            parapet_ring = parapet_ring.difference(
                core_2d[z].buffer(0.02, join_style="mitre")
            )
        z_roof = _floor_z(z + 1, gf_height)
        hero_parapets.add_prism(
            parapet_ring, z_roof, z_roof + 0.90, cx, cy, cap_bottom=False
        )

    # Interior 0.30 m floor slabs, recessed upper partitions, and grounded blush GF plinth
    hero_slabs = _IndexedMeshBuilder()
    hero_gf_plinth = _IndexedMeshBuilder()
    for z in range(nz):
        floor_mask = occ[z] & ~core_mask[z]
        if not floor_mask.any():
            continue
        floor_2d = _mask_to_geom(floor_mask)
        if floor_2d.is_empty:
            continue
        z_lo = _floor_z(z, gf_height)
        z_next = _floor_z(z + 1, gf_height)
        slab_2d = floor_2d.buffer(-0.10, join_style="mitre")
        if not slab_2d.is_empty and not core_2d[z].is_empty:
            slab_2d = slab_2d.difference(core_2d[z].buffer(0.02, join_style="mitre"))
        if not slab_2d.is_empty:
            hero_slabs.add_prism(slab_2d, z_lo, min(z_lo + 0.30, z_next - 0.05), cx, cy)

        if z == 0:
            plinth_2d = floor_2d.buffer(-0.55, join_style="mitre")
            if not plinth_2d.is_empty and not core_2d[0].is_empty:
                plinth_2d = plinth_2d.difference(
                    core_2d[0].buffer(0.02, join_style="mitre")
                )
            if not plinth_2d.is_empty and z_next - 0.02 > 0.30:
                hero_gf_plinth.add_prism(plinth_2d, 0.0, z_next - 0.02, cx, cy)
            continue

        part_2d = floor_2d.buffer(-1.6, join_style="mitre")
        if part_2d.is_empty:
            part_2d = floor_2d.buffer(-0.9, join_style="mitre")
        if not part_2d.is_empty and not core_2d[z].is_empty:
            part_2d = part_2d.difference(core_2d[z].buffer(0.02, join_style="mitre"))
        if not part_2d.is_empty and z_next - 0.02 > z_lo + 0.32:
            hero_slabs.add_prism(part_2d, z_lo + 0.30, z_next - 0.02, cx, cy)

    # Extract shifted urban context with GF split + roof parapets
    ctx = env.context or {}
    context_buildings = []
    ctx_bld_2d = []
    ctx_parapets = _IndexedMeshBuilder()
    for idx, b in enumerate(ctx.get("buildings", [])):
        g, z_lo, z_hi = _extract_context_building_geom(b)
        if g is None or g.is_empty:
            continue
        g_loc = translate(g, xoff=-cx, yoff=-cy)
        h_bld = float(z_hi - z_lo)
        ctx_gf_h = _context_gf_height(h_bld)
        z_split = (z_lo + ctx_gf_h) if h_bld > ctx_gf_h + 1.5 else None
        bld = _IndexedMeshBuilder()
        bld.add_prism(g_loc, z_lo, z_hi, 0.0, 0.0, z_split=z_split)
        if bld.verts:
            context_buildings.append(
                dict(
                    name=f"Context_Bld_{idx:02d}",
                    height=round(h_bld, 2),
                    gf_height=round(ctx_gf_h, 3),
                    **bld.to_dict(),
                )
            )
            ctx_bld_2d.append(g_loc)
            inner_p = g_loc.buffer(-0.35, join_style="mitre")
            p_ring = g_loc.difference(inner_p) if not inner_p.is_empty else g_loc
            ctx_parapets.add_prism(
                p_ring, z_hi, z_hi + 0.75, 0.0, 0.0, cap_bottom=False
            )

    road_2d = []
    roads = _IndexedMeshBuilder()
    for r in ctx.get("roads", []):
        vertices = np.asarray(r.get("vertices", []), dtype=np.float64)
        faces = np.asarray(r.get("faces", []), dtype=np.int64)
        if not len(vertices) or not len(faces):
            continue
        triangles = shapely.polygons(vertices[faces, :2] - (cx, cy))
        # A road mesh's vertex buffer can contain caps, sidewalls, and holes.
        # Only its projected faces describe the footprint.
        footprint = shapely.union_all(
            triangles[shapely.area(triangles) > 1e-8], grid_size=1e-4
        )
        if footprint.area > 1.0:
            road_2d.append(footprint)
    roads.add_prism(unary_union(road_2d), 0.0, 0.02)

    green_2d = []
    greens = _IndexedMeshBuilder()
    for gr in ctx.get("greenSpaces", []):
        gv = np.asarray(gr.get("vertices", []), dtype=np.float32)
        if len(gv) < 3:
            continue
        gp = Polygon(gv[:, :2]).buffer(0)
        if not gp.is_empty and gp.area > 1.0:
            gp_loc = translate(gp, xoff=-cx, yoff=-cy)
            green_2d.append(gp_loc)
            greens.add_prism(gp_loc, 0.0, 0.20, 0.0, 0.0)

    parcel_2d = Polygon()
    site_pad = _IndexedMeshBuilder()
    perim = ctx.get("perimeter") or []
    if len(perim) >= 3:
        sp = Polygon(perim).buffer(0)
        if not sp.is_empty:
            parcel_2d = translate(sp, xoff=-cx, yoff=-cy)
            site_pad.add_prism(parcel_2d, 0.0, 0.19, 0.0, 0.0)

    hero_2d = (
        translate(_mask_to_geom(occ.any(axis=0)), xoff=-cx, yoff=-cy)
        if occ.any()
        else Polygon()
    )
    max_ctx_xy = 0.0
    for g in (
        *ctx_bld_2d,
        *road_2d,
        *green_2d,
        *([parcel_2d] if not parcel_2d.is_empty else []),
        *([hero_2d] if not hero_2d.is_empty else []),
    ):
        if not g.is_empty:
            b = g.bounds
            max_ctx_xy = max(max_ctx_xy, abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3]))
    board_half = round(max(160.0, max_ctx_xy + 30.0), 1)

    sidewalk_curbs, road_markings, street_props = _extract_street_assets(
        hero_2d,
        ctx_bld_2d,
        road_2d,
        green_2d,
        parcel_2d,
        board_half=board_half,
        seed=eval_seed,
    )

    m = env.metrics()
    return {
        "metadata": {
            "siteId": site_id,
            "city": ctx.get("city", ""),
            "evalSeed": int(eval_seed),
            "temp": float(temp),
            "noise": float(noise),
            "gfHeight": round(float(gf_height), 3),
            "gridModule": int(grid_module),
            "boardHalf": board_half,
            "centerXY": [round(cx, 4), round(cy, 4)],
            "far": round(float(m["far"]), 4),
            "objective": round(float(obj), 4),
            "usableGoodArea": round(float(diag["usableGoodArea"]), 1),
            "directSunProxy": round(float(m["directSunProxy"]), 4),
            "diffuseFacadeProxy": round(float(m["diffuseFacadeProxy"]), 4),
            "coreCoverage": round(float(diag.get("coreCoverage") or 0.0), 4),
            "coreCount": int(diag.get("coreCount") or 0),
        },
        "hero": {
            "Building_Mass": mass.to_dict(),
            "Hero_Parapets": hero_parapets.to_dict(),
            "Hero_Slabs": hero_slabs.to_dict(),
            "Hero_GF_Plinth": hero_gf_plinth.to_dict(),
            "Vertical_Cores": cores.to_dict(),
        },
        "context": {
            "Site_Parcel": site_pad.to_dict(),
            "Sidewalk_Curb": sidewalk_curbs.to_dict(),
            "Context_Roads": roads.to_dict(),
            "Road_Markings": road_markings.to_dict(),
            "Context_Green": greens.to_dict(),
            "Context_Parapets": ctx_parapets.to_dict(),
            "Context_Buildings": context_buildings,
        },
        "street_props": street_props,
    }


def _find_blender(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if env_bin := os.environ.get("BLENDER_BIN"):
        candidates.append(Path(env_bin))
    if which_bin := shutil.which("blender"):
        candidates.append(Path(which_bin))
    candidates.append(Path("/Applications/Blender.app/Contents/MacOS/Blender"))
    candidates.extend(
        sorted(Path("/Applications").glob("Blender*.app/Contents/MacOS/Blender"))
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Blender executable not found. Pass --blender-bin, set $BLENDER_BIN, "
        "or install Blender under /Applications/Blender.app.",
    )


def _build_blend_file(
    blender_bin: Path,
    template_path: Path,
    json_path: Path,
    blend_out: Path,
    *,
    hero_style: str,
    no_context_facades: bool,
    no_props: bool,
    no_road_markings: bool,
):
    importer = ROOT / "tools" / "blender_import_site.py"
    cmd = [
        str(blender_bin),
        "--background",
        str(template_path),
        "--python-exit-code",
        "1",
        "--python",
        str(importer),
        "--",
        "--input",
        str(json_path),
        "--output",
        str(blend_out),
        "--hero-style",
        hero_style,
    ]
    if no_context_facades:
        cmd.append("--no-context-facades")
    if no_props:
        cmd.append("--no-props")
    if no_road_markings:
        cmd.append("--no-road-markings")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs" / "real-aux-tier-600ep"
    )
    parser.add_argument("--site", nargs="+", default=["m_prs_0315"])
    parser.add_argument(
        "--checkpoint", type=Path, help="Use a checkpoint from --run-dir"
    )
    parser.add_argument(
        "--save-voxels",
        action="store_true",
        help="Save the exact masks used for the Blender export",
    )
    parser.add_argument("--all-showcase", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--gf-height", type=float, default=4.2)
    parser.add_argument(
        "--hero-style",
        choices=("curtain_frame", "shadow_box", "ribbon_louver", "raw"),
        default="curtain_frame",
    )
    parser.add_argument("--no-context-facades", action="store_true")
    parser.add_argument("--no-props", action="store_true")
    parser.add_argument("--no-road-markings", action="store_true")
    parser.add_argument(
        "--template", type=Path, default=ROOT / "blender" / "Template.blend"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "blender")
    parser.add_argument("--out-blend", type=Path, default=None)
    parser.add_argument("--blender-bin", type=Path, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    sites = list(SHOWCASE_PRESETS) if args.all_showcase else list(args.site)
    if args.out_blend is not None and len(sites) > 1:
        raise ValueError("--out-blend can only be used when exporting a single site")
    if not args.json_only and not args.template.is_file():
        parser.error(
            f"Blender template is missing: {args.template}. "
            "Set --template to a local copy or use --json-only."
        )

    blender_bin = None if args.json_only else _find_blender(args.blender_bin)
    cfg, dev, policy = _load_policy(
        args.run_dir, args.temp, args.noise, args.checkpoint
    )

    grid_module = _hero_grid_module(args.hero_style)
    for site_id in sites:
        eval_seed = (
            int(args.seed)
            if args.seed is not None
            else SHOWCASE_PRESETS.get(site_id, 9001)
        )
        json_path = args.out_dir / "data" / f"{site_id}_seed{eval_seed}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = export_site_payload(
            cfg,
            dev,
            policy,
            site_id,
            eval_seed,
            args.temp,
            args.noise,
            gf_height=args.gf_height,
            grid_module=grid_module,
            voxel_output=json_path.with_suffix(".npz") if args.save_voxels else None,
        )
        json_path.write_text(json.dumps(payload))
        print(f"Exported {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
        print("Metadata:", json.dumps(payload["metadata"]))
        for k, v in payload["hero"].items():
            print(f"  hero/{k}: {len(v['vertices'])} verts, {len(v['faces'])} faces")
        sp = payload["street_props"]
        print(
            f"  context: {len(payload['context']['Context_Buildings'])} buildings, "
            f"roads={len(payload['context']['Context_Roads']['faces'])} faces, "
            f"props=(lamps={len(sp['lamps'])}, bs={len(sp.get('bus_stops', ()))}, "
            f"paths={len(sp.get('pedestrian_paths', ()))}, "
            f"cars={len(sp['cars'])}, trees={len(sp['trees'])})",
        )

        if not args.json_only:
            blend_out = args.out_blend or (args.out_dir / f"{site_id}.blend")
            if blend_out.resolve() == args.template.resolve():
                raise ValueError(f"Refusing to overwrite template file: {blend_out}")
            _build_blend_file(
                blender_bin,
                args.template,
                json_path,
                blend_out,
                hero_style=args.hero_style,
                no_context_facades=args.no_context_facades,
                no_props=args.no_props,
                no_road_markings=args.no_road_markings,
            )


if __name__ == "__main__":
    main()
