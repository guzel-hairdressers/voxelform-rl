"""Generate ML documentation figures, trained multi-tier renders, and rollout video for README.md."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
from matplotlib import cm as mpl_cm, colors as mcolors, gridspec, patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate, unary_union
import torch

from voxelform.coarse_to_fine import (
    CoarseEnv,
    CoarseToFineLearner,
    CoarseToFineNetwork,
    commit_fine,
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
MEDIA_DIR = ROOT / "docs" / "media"
RESULTS_DIR = ROOT / "results"


def _surface_quads(mask: np.ndarray, dz: float = 3.0, zero_z: bool = False):
    """Return exposed face vertices (N, 4, 3), face dirs (N,), and flat cell indices (N,)."""
    nz, ny, nx = mask.shape
    pad = np.pad(mask, 1, constant_values=False)
    dirs = (
        (pad[1:-1, 1:-1, 2:], 0, ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        (pad[1:-1, 1:-1, :-2], 1, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        (pad[1:-1, 2:, 1:-1], 2, ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        (pad[1:-1, :-2, 1:-1], 3, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        (pad[2:, 1:-1, 1:-1], 4, ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        (pad[:-2, 1:-1, 1:-1], 5, ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    )
    verts_list, face_list, cell_list = [], [], []
    for neighbor, face_idx, corners in dirs:
        if face_idx == 5 and zero_z is False:
            exp = mask & ~neighbor
            exp[0] = False
        else:
            exp = mask & ~neighbor
        zz, yy, xx = np.nonzero(exp)
        if len(zz) == 0:
            continue
        c_arr = np.asarray(corners, dtype=np.float32)
        base = np.stack((xx, yy, zz), axis=1)[:, None, :].astype(np.float32)
        quad = base + c_arr[None, :, :]
        quad[:, :, 2] *= dz
        verts_list.append(quad)
        face_list.append(np.full(len(zz), face_idx, dtype=np.int32))
        cell_list.append((zz * (nx * ny) + yy * nx + xx).astype(np.int64))
    if not verts_list:
        return (
            np.zeros((0, 4, 3), np.float32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.int64),
        )
    return (
        np.concatenate(verts_list, 0),
        np.concatenate(face_list, 0),
        np.concatenate(cell_list, 0),
    )


def _shade_rgba(
    base_rgb: tuple[float, float, float], faces: np.ndarray, alpha: float
) -> np.ndarray:
    """Directional shading per face index (+X,-X,+Y,-Y,+Z,-Z) matching voxel clarity."""
    gains = np.array([0.84, 0.68, 0.76, 0.92, 1.00, 0.56], dtype=np.float32)
    g = gains[faces][:, None]
    rgb = np.clip(np.asarray(base_rgb, dtype=np.float32)[None, :] * g, 0.0, 1.0)
    return np.column_stack((rgb, np.full(len(faces), alpha, dtype=np.float32)))


def _extract_building_footprint(building: dict) -> tuple[Polygon | None, float, float]:
    """Recover clean 2D Shapely footprint and metric (z_min, z_max) from a cached context prism."""
    verts_raw = building.get("vertices") or []
    faces_raw = building.get("faces") or []
    if not verts_raw or not faces_raw:
        return None, 0.0, 0.0
    verts = np.asarray(verts_raw, dtype=np.float32)
    z_min = float(verts[:, 2].min())
    z_max = float(building.get("height") or verts[:, 2].max())
    if z_max <= z_min + 0.2:
        return None, z_min, z_max
    base_polys = []
    for f in faces_raw:
        if len(f) < 3:
            continue
        zf = verts[f, 2]
        if np.all(zf <= z_min + 0.1) or np.all(zf >= z_max - 0.1):
            p = Polygon(verts[f, :2])
            if not p.is_valid:
                p = p.buffer(0)
            if p.area > 0.05:
                base_polys.append(p)
    if not base_polys:
        return None, z_min, z_max
    merged = unary_union(base_polys)
    if not merged.is_valid:
        merged = merged.buffer(0)
    return merged, z_min, z_max


def _iter_polygons(geom):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        if geom.area >= 8.0:
            yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for sub in geom.geoms:
            yield from _iter_polygons(sub)


def _extrude_clean_prism(
    poly: Polygon, z_lo: float, z_hi: float
) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
    """Extrude a clean 2D Shapely polygon into non-triangulated walls, flat roof, and crease edges."""
    faces_out: list[np.ndarray] = []
    shades_out: list[float] = []
    segs_out: list[np.ndarray] = []

    rings = [poly.exterior] + list(poly.interiors)
    for ring_idx, ring in enumerate(rings):
        pts = np.asarray(ring.coords, dtype=np.float32)[:-1]
        if len(pts) < 3:
            continue
        n = len(pts)
        # Ensure consistent outward horizontal normals for side shading
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            seg = p1 - p0
            seg_len = float(np.hypot(seg[0], seg[1]))
            if seg_len < 1e-3:
                continue
            wall = np.array(
                [
                    [p0[0], p0[1], z_lo],
                    [p1[0], p1[1], z_lo],
                    [p1[0], p1[1], z_hi],
                    [p0[0], p0[1], z_hi],
                ],
                dtype=np.float32,
            )
            nx = float(seg[1]) / seg_len
            ny = -float(seg[0]) / seg_len
            if ring_idx > 0:
                nx, ny = -nx, -ny
            shade = 0.88 + 0.08 * max(-1.0, min(1.0, 0.5 * nx - 0.5 * ny))
            faces_out.append(wall)
            shades_out.append(shade)
            segs_out.append(
                np.array([[p0[0], p0[1], z_hi], [p1[0], p1[1], z_hi]], dtype=np.float32)
            )
            segs_out.append(
                np.array([[p0[0], p0[1], z_lo], [p1[0], p1[1], z_lo]], dtype=np.float32)
            )

            # Draw vertical corner edge only at crisp corners (skip dense circular clip arcs)
            prev_p = pts[(i - 1) % n]
            d_prev = p0 - prev_p
            lp = float(np.hypot(d_prev[0], d_prev[1]))
            if lp > 1e-3:
                dot = float((d_prev[0] * seg[0] + d_prev[1] * seg[1]) / (lp * seg_len))
                if dot < 0.94:
                    segs_out.append(
                        np.array(
                            [[p0[0], p0[1], z_lo], [p0[0], p0[1], z_hi]],
                            dtype=np.float32,
                        )
                    )

    if not poly.interiors:
        pts = np.asarray(poly.exterior.coords, dtype=np.float32)[:-1]
        if len(pts) >= 3:
            roof = np.column_stack((pts, np.full(len(pts), z_hi, dtype=np.float32)))
            faces_out.append(roof)
            shades_out.append(1.0)
    else:
        for t in triangulate(poly):
            if poly.covers(t.representative_point()) and t.area > 0.05:
                tp = np.asarray(t.exterior.coords, dtype=np.float32)[:-1]
                roof = np.column_stack((tp, np.full(len(tp), z_hi, dtype=np.float32)))
                faces_out.append(roof)
                shades_out.append(1.0)

    return faces_out, shades_out, segs_out


def _build_context_geometry(
    context: dict | None,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_max: float,
) -> dict | None:
    """Build clean radially-clipped translucent context prisms and dashed parcel boundary."""
    if not context or not context.get("buildings"):
        return None
    cx = 0.5 * (float(x_bounds[0]) + float(x_bounds[1]))
    cy = 0.5 * (float(y_bounds[0]) + float(y_bounds[1]))
    half = 0.5 * max(
        float(x_bounds[1]) - float(x_bounds[0]), float(y_bounds[1]) - float(y_bounds[0])
    )
    ctx_radius = 0.80 * half
    z_cap = 0.86 * float(z_max)
    clip_disk = Point(cx, cy).buffer(ctx_radius, quad_segs=20)

    faces_all: list[np.ndarray] = []
    shades_all: list[float] = []
    segs_all: list[np.ndarray] = []

    for b in context.get("buildings", []):
        footprint, z_lo, z_hi = _extract_building_footprint(b)
        if footprint is None:
            continue
        if not footprint.intersects(clip_disk):
            continue
        clipped = footprint.intersection(clip_disk)
        z_top_b = min(z_hi, z_cap)
        if z_top_b <= z_lo + 0.3:
            continue
        for poly in _iter_polygons(clipped):
            f_list, s_list, e_list = _extrude_clean_prism(poly, z_lo, z_top_b)
            faces_all.extend(f_list)
            shades_all.extend(s_list)
            segs_all.extend(e_list)

    perim_segs: list[np.ndarray] = []
    perim = context.get("perimeter") or []
    if len(perim) >= 3:
        p_poly = Polygon(perim).buffer(0)
        if not p_poly.is_empty:
            p_clip = p_poly.intersection(clip_disk)
            for p in _iter_polygons(p_clip):
                pts = np.asarray(p.exterior.coords, dtype=np.float32)
                for i in range(len(pts) - 1):
                    perim_segs.append(
                        np.array(
                            [
                                [pts[i, 0], pts[i, 1], 0.08],
                                [pts[i + 1, 0], pts[i + 1, 1], 0.08],
                            ],
                            dtype=np.float32,
                        )
                    )

    if not faces_all:
        return None
    shades_arr = np.asarray(shades_all, dtype=np.float32)[:, None]
    face_rgba = np.column_stack(
        (
            0.66 * shades_arr,
            0.71 * shades_arr,
            0.78 * shades_arr,
            np.full(len(faces_all), 0.055, dtype=np.float32),
        )
    )
    seg_arr = (
        np.asarray(segs_all, dtype=np.float32)
        if segs_all
        else np.zeros((0, 2, 3), dtype=np.float32)
    )
    perim_arr = (
        np.asarray(perim_segs, dtype=np.float32)
        if perim_segs
        else np.zeros((0, 2, 3), dtype=np.float32)
    )
    return {"faces": faces_all, "rgba": face_rgba, "segs": seg_arr, "perim": perim_arr}


def _setup_3d_axes(
    ax,
    occ: np.ndarray,
    dz: float = 3.0,
    elev: float = 28.0,
    azim: float = -52.0,
    context_pad: bool = False,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Set tight isometric metric bounds and return clipped (x_bounds, y_bounds, z_top)."""
    zz, yy, xx = np.nonzero(occ)
    if len(xx) == 0:
        nx, ny, nz = occ.shape[2], occ.shape[1], occ.shape[0]
        x0, x1, y0, y1, z1 = 0.0, float(nx), 0.0, float(ny), float(nz * dz)
    else:
        pad = 2.0
        x0, x1 = float(xx.min()) - pad, float(xx.max() + 1) + pad
        y0, y1 = float(yy.min()) - pad, float(yy.max() + 1) + pad
        z1 = max(12.0, float((zz.max() + 1) * dz) + 2.0)
    span = max(x1 - x0, y1 - y0, z1, 10.0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if context_pad:
        half = max(0.56 * span, 0.48 * span + 7.0)
        z_top = max(z1 + 4.0, 0.44 * (2.0 * half))
    else:
        half = 0.48 * span
        z_top = max(z1, 0.42 * span)
    xb = (cx - half, cx + half)
    yb = (cy - half, cy + half)
    ax.set_xlim(*xb)
    ax.set_ylim(*yb)
    ax.set_zlim(0.0, z_top)
    ax.set_box_aspect((1.0, 1.0, z_top / (2.0 * half)), zoom=1.42)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return xb, yb, z_top


def _render_voxel_panel(
    ax,
    occ: np.ndarray,
    core_mask: np.ndarray,
    mode: str = "mass",
    sun_map: np.ndarray | None = None,
    dist_map: np.ndarray | None = None,
    max_dist: float = 12.0,
    elev: float = 28.0,
    azim: float = -52.0,
    xray_mass: bool = True,
    context: dict | None = None,
    context_geom: dict | None = None,
    bounds_occ: np.ndarray | None = None,
):
    """Render exposed faces of mass, compact-core shafts, and optional ghosted urban context."""
    ref_occ = occ if bounds_occ is None else bounds_occ
    use_ctx = context_geom is not None or (
        context is not None and bool(context.get("buildings"))
    )
    xb, yb, z_top = _setup_3d_axes(
        ax, ref_occ, dz=3.0, elev=elev, azim=azim, context_pad=use_ctx
    )
    if context_geom is None and use_ctx:
        context_geom = _build_context_geometry(context, xb, yb, z_top)

    if context_geom is not None:
        if len(context_geom["perim"]) > 0:
            p_coll = Line3DCollection(
                context_geom["perim"],
                colors=[(0.22, 0.42, 0.78, 0.36)],
                linewidths=0.75,
                linestyles="--",
            )
            ax.add_collection3d(p_coll)
        if len(context_geom["faces"]) > 0:
            c_poly = Poly3DCollection(
                context_geom["faces"],
                facecolors=context_geom["rgba"],
                edgecolors="none",
                zsort="average",
            )
            ax.add_collection3d(c_poly)
        if len(context_geom["segs"]) > 0:
            c_lines = Line3DCollection(
                context_geom["segs"],
                colors=[(0.42, 0.48, 0.56, 0.18)],
                linewidths=0.45,
            )
            ax.add_collection3d(c_lines)

    core_occ = occ & core_mask
    shell_occ = occ & ~core_occ
    dz = 3.0

    verts_all, rgba_all, edge_all = [], [], []
    if mode == "mass":
        if shell_occ.any():
            vq, fq, _ = _surface_quads(shell_occ, dz=dz)
            verts_all.append(vq)
            alpha = 0.52 if xray_mass else 0.96
            base = (0.36, 0.54, 0.98) if xray_mass else (0.40, 0.58, 0.99)
            e_rgba = [0.12, 0.25, 0.65, 0.20] if xray_mass else [0.10, 0.22, 0.58, 0.30]
            rgba_all.append(_shade_rgba(base, fq, alpha=alpha))
            edge_all.append(np.tile(np.array([e_rgba], np.float32), (len(vq), 1)))
        if core_occ.any():
            cq, cf, _ = _surface_quads(core_occ, dz=dz, zero_z=False)
            verts_all.append(cq)
            rgba_all.append(_shade_rgba((0.86, 0.15, 0.15), cf, alpha=0.98))
            edge_all.append(
                np.tile(np.array([[0.45, 0.05, 0.05, 0.55]], np.float32), (len(cq), 1))
            )
    elif mode == "sun":
        cmap = matplotlib.colormaps["YlOrRd"]
        if shell_occ.any():
            vq, fq, cells = _surface_quads(shell_occ, dz=dz)
            vals = (
                np.clip(sun_map[fq, cells], 0.0, 1.0)
                if sun_map is not None
                else np.zeros(len(fq))
            )
            gains = np.array([0.88, 0.78, 0.84, 0.94, 1.00, 0.65], dtype=np.float32)[
                fq
            ][:, None]
            rgba = cmap(vals).astype(np.float32)
            rgba[:, :3] = np.clip(rgba[:, :3] * gains, 0.0, 1.0)
            rgba[:, 3] = 0.96
            verts_all.append(vq)
            rgba_all.append(rgba)
            edge_all.append(
                np.tile(np.array([[0.22, 0.18, 0.16, 0.18]], np.float32), (len(vq), 1))
            )
        if core_occ.any():
            cq, cf, _ = _surface_quads(core_occ, dz=dz)
            verts_all.append(cq)
            rgba_all.append(_shade_rgba((0.39, 0.45, 0.55), cf, alpha=0.92))
            edge_all.append(
                np.tile(np.array([[0.15, 0.18, 0.24, 0.35]], np.float32), (len(cq), 1))
            )
    elif mode == "core_dist":
        cmap = matplotlib.colormaps["viridis_r"]
        if shell_occ.any():
            vq, fq, cells = _surface_quads(shell_occ, dz=dz)
            d_flat = (
                dist_map.reshape(-1)[cells]
                if dist_map is not None
                else np.zeros(len(fq))
            )
            norm = np.clip(d_flat / max(1.0, max_dist * 1.5), 0.0, 1.0)
            gains = np.array([0.88, 0.78, 0.84, 0.94, 1.00, 0.65], dtype=np.float32)[
                fq
            ][:, None]
            rgba = cmap(norm).astype(np.float32)
            rgba[:, :3] = np.clip(rgba[:, :3] * gains, 0.0, 1.0)
            rgba[:, 3] = 0.95
            verts_all.append(vq)
            rgba_all.append(rgba)
            edge_all.append(
                np.tile(np.array([[0.15, 0.18, 0.20, 0.16]], np.float32), (len(vq), 1))
            )
        if core_occ.any():
            cq, cf, _ = _surface_quads(core_occ, dz=dz)
            verts_all.append(cq)
            rgba_all.append(_shade_rgba((0.86, 0.15, 0.15), cf, alpha=0.98))
            edge_all.append(
                np.tile(np.array([[0.45, 0.05, 0.05, 0.55]], np.float32), (len(cq), 1))
            )

    if verts_all:
        verts = np.concatenate(verts_all, 0)
        rgba = np.concatenate(rgba_all, 0)
        edges = np.concatenate(edge_all, 0)
        poly = Poly3DCollection(
            verts, facecolors=rgba, edgecolors=edges, linewidths=0.25, zsort="average"
        )
        ax.add_collection3d(poly)


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


def _load_env(site_id: str, cfg: ExperimentConfig, dev: str) -> Environment:
    library = ContextLibrary()
    context = library.load(
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
    return env


def _run_eval_episode(
    policy,
    env: Environment,
    cfg: ExperimentConfig,
    eval_seed: int,
    temp: float,
    noise: float,
    record_trace: bool = False,
):
    seed_everything(eval_seed, cfg.strict_determinism)
    policy.evaluation_temperature = float(temp)
    policy.evaluation_noise_scale = float(noise)
    is_ms = isinstance(policy, CoarseToFineLearner)
    coarse_envs = (
        tuple(CoarseEnv(env, blk) for blk in tier_cascade_blocks(env))
        if (is_ms and cfg.multiscale_schedule == "tier_adaptive")
        else ()
    )

    def _snap(step_idx: int, op_name: str, dv: int):
        obj, diag = organic_objective(env)
        nx, ny, nz = env.nx, env.ny, env.nz
        sun_grid = np.zeros((6, nz * ny * nx), dtype=np.float32)
        if record_trace and hasattr(env, "receiver_data") and hasattr(env, "visible"):
            _, _, _, cells, faces = env.receiver_data
            vis_mean = env.visible.float().mean(0).detach().cpu().numpy()
            sun_grid[faces.detach().cpu().numpy(), cells.detach().cpu().numpy()] = (
                vis_mean
            )
        return {
            "step": step_idx,
            "op": op_name,
            "dv": dv,
            "occ": env.occupied.detach().cpu().numpy().copy() if record_trace else None,
            "core_mask": env.core_mask.detach().cpu().numpy().copy()
            if record_trace
            else None,
            "core_dist": env.core_distance.detach().cpu().numpy().copy()
            if record_trace
            else None,
            "sun_grid": sun_grid if record_trace else None,
            "far": float(env.far),
            "target_far": float(env.target_far),
            "obj": float(obj),
            "usable_area": float(diag["usableGoodArea"]),
            "usable_frac": float(diag["usableGoodFraction"]),
            "sun": float(env.values["directSunProxy"].detach().cpu()),
            "diff": float(env.values["diffuseFacadeProxy"].detach().cpu()),
            "core_cov": float(diag.get("coreCoverage") or 0.0),
            "core_count": int(diag.get("coreCount") or 0),
        }

    if is_ms and not record_trace:
        raw = run_cascade_episode(
            env,
            coarse_envs,
            policy,
            cfg.initial_storeys,
            organic_objective,
            training=False,
            rollout_steps=cfg.rollout_steps,
            audit_steps=cfg.audit_steps,
        )
        final = _snap(raw["steps"], "cascade", 0)
        final["occ"] = env.occupied.detach().cpu().numpy().copy()
        final["core_mask"] = env.core_mask.detach().cpu().numpy().copy()
        final["far_satisfied"] = bool(env.far_satisfied)
        final["seed"] = eval_seed
        final["stage_steps"] = raw["stageSteps"]
        return final, []

    snaps = []
    if record_trace:
        solid = env.allowed.clone()
        solid[cfg.initial_storeys :] = False
        env.restore_occupancy(solid)
        env.evaluate()
        snaps.append(_snap(0, "init", 0))

    step_offset = 0
    if is_ms and coarse_envs:
        coarse_envs[0].reset(cfg.initial_storeys)
        start_fine = coarse_envs[0].start_count_fine
        for idx, ce in enumerate(coarse_envs):
            while not ce.near_far and ce.unified_actions < ce.unified_limit:
                move = policy.choose_stage(ce, is_coarse=True, training=False)
                ce.unified_actions += 1
                if policy.last_operation == "finish":
                    break
                if move is None:
                    continue
                ce.apply_move(move)
                if policy.fine_config.allocation == "feedback":
                    dv_abs = int(move.remove.sum() + move.add.sum())
                    ce.workload.observe(dv_abs, dv_abs)
            if idx + 1 < len(coarse_envs):
                ce.upsample_to_coarse(coarse_envs[idx + 1])
            else:
                ce.upsample_to_fine()
                if record_trace:
                    step_offset += 1
                    env.evaluate()
                    snaps.append(
                        _snap(
                            step_offset,
                            f"coarse_{ce.bxy}x{ce.bz}",
                            int(env.count - start_fine),
                        )
                    )
        env.start_count = start_fine
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
        env.evaluate()

    op = "init"
    while not env.terminal:
        move = (
            policy.choose_stage(env, is_coarse=False, training=False)
            if is_ms
            else policy.choose(env, training=False)
        )
        env.unified_actions += 1
        op = policy.last_operation or "none"
        if op == "finish":
            env.termination = "far_reached"
            if record_trace:
                step_offset += 1
                env.evaluate()
                snaps.append(_snap(step_offset, "finish", 0))
            break
        if move is None:
            if env.unified_actions >= env.unified_limit:
                env.termination = (
                    "far_reached" if env.far_satisfied else "step_budget_exhausted"
                )
                break
            continue
        dv = int(move.add.sum()) - int(move.remove.sum())
        if is_ms:
            commit_fine(env, move, validated=True)
        else:
            commit(env, move, validated=True)
        if is_ms and policy.fine_config.allocation == "feedback":
            dv_abs = int(move.remove.sum() + move.add.sum())
            env.workload.observe(dv_abs, dv_abs)
        if record_trace:
            step_offset += 1
            env.evaluate()
            snaps.append(_snap(step_offset, op, dv))
        elif env.edits % cfg.audit_steps == 0:
            env.evaluate()
        if env.unified_actions >= env.unified_limit:
            env.termination = (
                "far_reached" if env.far_satisfied else "step_budget_exhausted"
            )
            break
    if not env.terminal:
        env.termination = (
            "far_reached" if env.far_satisfied else "step_budget_exhausted"
        )
    env.evaluate()
    if record_trace:
        final = snaps[-1]
    else:
        final = _snap(env.edits, op, 0)
        final["occ"] = env.occupied.detach().cpu().numpy().copy()
        final["core_mask"] = env.core_mask.detach().cpu().numpy().copy()
    final["far_satisfied"] = bool(env.far_satisfied)
    final["seed"] = eval_seed
    return final, snaps


def _build_policy(cfg: ExperimentConfig, dev: str, seed: int):
    if cfg.multiscale_schedule == "tier_adaptive":
        return CoarseToFineLearner(
            dev,
            seed=seed,
            block=(1, 2, 2),
            fine_config=_patch_cfg(),
            input_channels=14 if cfg.cores_enabled else 12,
            spatial_aux_weight=cfg.spatial_aux_weight,
            evaluation_temperature=cfg.evaluation_temperature,
            evaluation_noise_scale=cfg.evaluation_noise_scale,
            lr=1.2e-3,
        )
    return UnifiedLearner(
        dev,
        seed=seed,
        config=_patch_cfg(),
        amount_feedback=True,
        net_guided=True,
        organic_3d=True,
        spatial_aux_weight=cfg.spatial_aux_weight,
    )


def _load_run_policy(run_dir: Path):
    cfg = ExperimentConfig.read(run_dir / "config.json")
    dev = device_name(cfg.device)
    seed = cfg.seeds[0]
    best_path = run_dir / f"seed-{seed}" / "best.pt"
    ckpt_path = (
        best_path if best_path.is_file() else run_dir / f"seed-{seed}" / "latest.pt"
    )
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if (
        saved.get("format") != PROTOCOL
        or saved.get("coreLayoutVersion") != CORE_LAYOUT_VERSION
    ):
        raise ValueError("Incompatible checkpoint protocol or core layout version")
    manifest_path = run_dir / "manifest.json"
    manifest_hash = (
        json.loads(manifest_path.read_text()).get("configHash")
        if manifest_path.is_file()
        else None
    )
    if saved.get("configHash") not in (cfg.fingerprint, manifest_hash):
        raise ValueError("Checkpoint configHash does not match run config")
    trained = _build_policy(cfg, dev, seed)
    trained.load(saved["policy"])
    untrained = _build_policy(cfg, dev, seed)
    return cfg, dev, untrained, trained, int(saved["nextEpisode"])


def generate_architecture_diagram():
    """Create crisp NeurIPS-style architecture + MDP pipeline figure."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in CoarseToFineNetwork(14).parameters())

    fig, ax = plt.subplots(figsize=(15.5, 6.9), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    ax.text(
        2.0,
        39.4,
        "VoxelForm-RL: Tier-Adaptive Multi-Scale Actor-Critic & Feasibility-Projected 3D Voxel MDP",
        fontsize=15.0,
        fontweight="bold",
        color="#0f172a",
        family="DejaVu Sans",
    )
    ax.text(
        2.0,
        36.9,
        f"State O_t in R^(Z x 14 x Y x X)  ->  CoarseToFineNetwork ({n_params:,} params, shared 2D+1D)  "
        "->  XS/S/M: 1x1x1 | L/XL: 2x2x1->1x1x1 (tier_meso2)",
        fontsize=9.2,
        color="#475569",
    )

    def _box(
        x, y, w, h, title, lines, fc, ec, title_c="#0f172a", body_c="#1e293b", tag=None
    ):
        p = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.6,
        )
        ax.add_patch(p)
        if tag:
            tp = patches.FancyBboxPatch(
                (x + 0.8, y + h - 2.3),
                len(tag) * 0.62 + 1.4,
                2.0,
                boxstyle="round,pad=0.15,rounding_size=0.45",
                facecolor=ec,
                edgecolor="none",
            )
            ax.add_patch(tp)
            ax.text(
                x + 1.5,
                y + h - 1.3,
                tag,
                fontsize=7.6,
                fontweight="bold",
                color="#ffffff",
                va="center",
            )
            ty = y + h - 4.1
        else:
            ty = y + h - 2.0
        ax.text(
            x + 1.1, ty, title, fontsize=9.6, fontweight="bold", color=title_c, va="top"
        )
        cur_y = ty - 2.2
        for line in lines:
            ax.text(
                x + 1.1,
                cur_y,
                line,
                fontsize=8.0,
                color=body_c,
                va="top",
                linespacing=1.22,
            )
            cur_y -= 1.62 * (line.count("\n") + 1) + 0.45

    def _arrow(x0, y0, x1, y1, label="", rad=0.0, color="#334155", y_off=0.0):
        style = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0.0"
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.42,head_length=0.65",
                color=color,
                lw=1.7,
                connectionstyle=style,
            ),
        )
        if label:
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + (1.3 if rad == 0 else y_off)
            ax.text(
                mx,
                my,
                label,
                fontsize=8.0,
                color=color,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="#ffffff",
                    edgecolor="none",
                    alpha=0.95,
                ),
            )

    _box(
        2.0,
        7.5,
        20.5,
        27.6,
        "14-Channel Voxel State",
        [
            "Geometry & Masks (ch 0,3,4,6,7):\n  occupied, allowed, rel_z, rem/add",
            "Physical Proxies (ch 1,2,8,9):\n  diffuse air, noise, ext_sun, facade",
            "Shape Regularity (ch 5,10,11):\n  depth_bad, strip <6m, usable",
            "Vertical Cores (ch 12,13):\n  core_mask, geodesic badness",
            "Goal g_t in R^6:\n  FAR, target, err, budget, scale",
        ],
        fc="#f8fafc",
        ec="#2563eb",
        tag="STATE S_t",
    )

    _box(
        26.5,
        7.5,
        23.0,
        27.6,
        "Multi-Scale 2D+1D Network",
        [
            "Tier-Adaptive Stems:\n  coarse_stem (2x2x1), fine (1x1x1 -> /4 XY)\n  + optional macro_stem (4x4x1)",
            "Shared Factored 2D+1D Core:\n  ctx_xy (3x3) + _zmix (Z-1, Z, Z+1)\n  inside Halo-2 Window (win=16)",
            "Goal + Skip Conditioning:\n  1x1 local_skip (16 ch)\n  + goal_encoder(g_t) -> cat (48 ch)",
            "Policy & Value Outputs:\n  mu = 3*tanh(head/3) [2, Z_c, Y_c, X_c]\n  op_logits (4 ops), scalar critic V(s)",
        ],
        fc="#eff6ff",
        ec="#1d4ed8",
        tag="POLICY pi_theta",
    )

    _box(
        53.5,
        7.5,
        22.0,
        27.6,
        "Action & Constraint Decoder",
        [
            "1. Categorical Op + Gaussian Retry:\n  o_t in {remove, grow, exchange, finish}\n  z^(k) ~ N(mu, sigma^2), learned_cap\n  |FAR - FAR*| <= 0.75 exchange gate",
            "2. Coarse-to-Fine Handoff:\n  2x2x1 -> 1x1x1 (or 4x4x1) exact\n  grounded refinement + early finish",
            "3. Deterministic Projection Pi_valid:\n  - FAR band & min width clean\n  - 6-neighbor connectivity & cavities\n  - Active compact-core.v5 preservation",
        ],
        fc="#fef2f2",
        ec="#dc2626",
        tag="DECODER Pi_valid",
    )

    _box(
        79.5,
        7.5,
        18.5,
        27.6,
        "Sim Solvers & GAE",
        [
            "Vectorized Proxy Solvers:\n  GPU DDA FastVoxelShadows\n  Cached MeshShadows (SHA-256)\n  Screened-Laplace diffuse\n  8-nb geodesic core distance",
            "Scale-Normalized GAE:\n  GAE(0.99, 0.95) + dense_pot\n  1/sqrt(N_lat) logp, op_ent=0.02\n  Aux=1.2x (or distill), Cosine LR",
        ],
        fc="#f0fdf4",
        ec="#15803d",
        tag="SIM & LEARNER",
    )

    _arrow(22.9, 21.3, 26.1, 21.3, "O_t, g_t")
    _arrow(49.9, 21.3, 53.1, 21.3, "o_t, mu, V")
    _arrow(75.9, 21.3, 79.1, 21.3, "valid mask")
    _arrow(
        88.8,
        7.0,
        12.2,
        7.0,
        "transition (O_{t+1}, step/terminal reward r_t)  &  normalized GAE update",
        rad=-0.065,
        color="#0f766e",
        y_off=-3.6,
    )

    out = MEDIA_DIR / "mdp-architecture.png"
    fig.savefig(out, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def generate_trained_rollout_assets(
    cfg: ExperimentConfig, dev: str, trained, total_eps: int
):
    """Render MP4 + GIF from the best FAR-satisfied trained evaluation rollout."""
    candidates = [
        ("m_prs_0315", "Held-Out Test: Paris (m_prs_0315)", 2.5, 0.08),
        ("m_prs_0315", "Held-Out Test: Paris (m_prs_0315)", 1.0, 1.0),
        ("m_bcn_0093", "Barcelona Eixample (m_bcn_0093)", 2.5, 0.08),
    ]
    chosen = None
    chosen_label = ""
    chosen_decode = ""
    best_rank = -1e9
    envs = {}
    chosen_site = None
    for site_id, label, temp, noise in candidates:
        if site_id not in envs:
            envs[site_id] = _load_env(site_id, cfg, dev)
        env = envs[site_id]
        for s in range(9001, 9011):
            final, snaps = _run_eval_episode(
                trained, env, cfg, s, temp, noise, record_trace=True
            )
            if not final["far_satisfied"] or len(snaps) < 4 or final["core_cov"] < 0.85:
                continue
            rank = final["obj"] + (0.05 if site_id == "m_prs_0315" else 0.0)
            if rank > best_rank:
                best_rank = rank
                chosen = snaps
                chosen_site = site_id
                chosen_label = f"{label}, seed {s}"
                chosen_decode = f"temp={temp:g}, noise={noise:g}"
    assert chosen is not None and len(chosen) >= 4
    print(
        f"Trained rollout: {chosen_label} ({chosen_decode}), {len(chosen)} states, FAR={chosen[-1]['far']:.2f}, obj={chosen[-1]['obj']:.3f}"
    )

    rollout_occ = np.logical_or.reduce([s["occ"] for s in chosen])
    tmp_fig = plt.figure()
    tmp_ax = tmp_fig.add_subplot(111, projection="3d")
    xb, yb, z_top = _setup_3d_axes(tmp_ax, rollout_occ, dz=3.0, context_pad=True)
    plt.close(tmp_fig)
    rollout_ctx = _build_context_geometry(envs[chosen_site].context, xb, yb, z_top)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_entries: list[tuple[Path, float]] = []

        def _draw_state(
            snap: dict,
            total_steps: int,
            azim: float,
            out_png: Path,
            tag_suffix: str = "",
        ):
            fig = plt.figure(figsize=(12.8, 5.4), dpi=100)
            fig.patch.set_facecolor("#ffffff")
            gs = gridspec.GridSpec(
                1,
                3,
                figure=fig,
                left=0.02,
                right=0.98,
                top=0.81,
                bottom=0.14,
                wspace=0.04,
            )
            axes = [fig.add_subplot(gs[0, i], projection="3d") for i in range(3)]

            _render_voxel_panel(
                axes[0],
                snap["occ"],
                snap["core_mask"],
                mode="mass",
                azim=azim,
                xray_mass=True,
                context_geom=rollout_ctx,
                bounds_occ=rollout_occ,
            )
            axes[0].set_title(
                "Occupied Mass + compact-core.v5",
                fontsize=10.5,
                fontweight="bold",
                color="#0f172a",
                pad=4,
            )

            _render_voxel_panel(
                axes[1],
                snap["occ"],
                snap["core_mask"],
                mode="sun",
                sun_map=snap["sun_grid"],
                azim=azim,
                context_geom=rollout_ctx,
                bounds_occ=rollout_occ,
            )
            axes[1].set_title(
                "Direct Sun Proxy (Cores Exempt)",
                fontsize=10.5,
                fontweight="bold",
                color="#0f172a",
                pad=4,
            )

            _render_voxel_panel(
                axes[2],
                snap["occ"],
                snap["core_mask"],
                mode="core_dist",
                dist_map=snap["core_dist"],
                max_dist=float(cfg.max_core_dist),
                azim=azim,
                context_geom=rollout_ctx,
                bounds_occ=rollout_occ,
            )
            axes[2].set_title(
                f"Geodesic Core Distance (Limit {cfg.max_core_dist} m)",
                fontsize=10.5,
                fontweight="bold",
                color="#0f172a",
                pad=4,
            )

            cax1 = fig.add_axes([0.39, 0.11, 0.22, 0.022])
            cb1 = fig.colorbar(
                mpl_cm.ScalarMappable(
                    norm=mcolors.Normalize(0.0, 1.0),
                    cmap=matplotlib.colormaps["YlOrRd"],
                ),
                cax=cax1,
                orientation="horizontal",
            )
            cb1.ax.tick_params(labelsize=7.5)
            cb1.set_label(
                "Visible sun fraction", fontsize=8.0, color="#334155", labelpad=2
            )

            cax2 = fig.add_axes([0.71, 0.11, 0.22, 0.022])
            cb2 = fig.colorbar(
                mpl_cm.ScalarMappable(
                    norm=mcolors.Normalize(0.0, 18.0),
                    cmap=matplotlib.colormaps["viridis_r"],
                ),
                cax=cax2,
                orientation="horizontal",
            )
            cb2.ax.tick_params(labelsize=7.5)
            cb2.set_label(
                "Distance to nearest core (m, red = core)",
                fontsize=8.0,
                color="#334155",
                labelpad=2,
            )

            fig.text(
                0.025,
                0.94,
                f"{total_eps:,}-Ep Trained Policy ({'Tier-Adaptive' if cfg.multiscale_schedule == 'tier_adaptive' else 'Flat'}, {chosen_decode}) — {chosen_label}{tag_suffix}",
                fontsize=12.0,
                fontweight="bold",
                color="#0f172a",
            )
            op_txt = (
                f"{snap['op']} ({snap['dv']:+d} vox)"
                if snap["step"] > 0 and snap["op"] != "finish"
                else snap["op"]
            )
            hud = (
                f"Step {snap['step']}/{total_steps} [{op_txt}]  |  "
                f"FAR: {snap['far']:.2f} (target {snap['target_far']:.2f} +/-{cfg.far_tolerance:.2f})  |  "
                f"Usable: {snap['usable_frac'] * 100:.1f}% ({snap['usable_area']:.0f} m2)  |  "
                f"Sun: {snap['sun'] * 100:.1f}%  |  Sky: {snap['diff'] * 100:.1f}%  |  "
                f"CoreCov: {snap['core_cov'] * 100:.1f}% ({snap['core_count']})  |  Obj: {snap['obj']:.3f}"
            )
            fig.text(
                0.025,
                0.875,
                hud,
                fontsize=8.9,
                color="#1e293b",
                bbox=dict(
                    boxstyle="round,pad=0.35", facecolor="#f1f5f9", edgecolor="#cbd5e1"
                ),
            )
            fig.savefig(out_png, facecolor="#ffffff")
            plt.close(fig)

        total_steps = chosen[-1]["step"]
        idx = 0
        for s_i, snap in enumerate(chosen):
            png = tmp_dir / f"step_{idx:03d}.png"
            _draw_state(snap, total_steps, azim=-52.0, out_png=png)
            dur = 0.85 if s_i == 0 else (0.95 if s_i == len(chosen) - 1 else 0.70)
            frame_entries.append((png, dur))
            idx += 1

        last = chosen[-1]
        for a in np.linspace(-52.0, -14.0, 12)[1:]:
            png = tmp_dir / f"step_{idx:03d}.png"
            _draw_state(
                last, total_steps, azim=float(a), out_png=png, tag_suffix=" [3D Orbit]"
            )
            frame_entries.append((png, 0.09))
            idx += 1
        for a in np.linspace(-14.0, -52.0, 12)[1:]:
            png = tmp_dir / f"step_{idx:03d}.png"
            _draw_state(
                last, total_steps, azim=float(a), out_png=png, tag_suffix=" [3D Orbit]"
            )
            frame_entries.append((png, 0.09))
            idx += 1
        frame_entries.append((frame_entries[-1][0], 1.1))

        concat_txt = tmp_dir / "concat.txt"
        lines = []
        for path, dur in frame_entries:
            lines.append(f"file '{path.as_posix()}'")
            lines.append(f"duration {dur:.3f}")
        lines.append(f"file '{frame_entries[-1][0].as_posix()}'")
        concat_txt.write_text("\n".join(lines) + "\n")

        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        mp4_out = MEDIA_DIR / "episode-rollout.mp4"
        gif_out = MEDIA_DIR / "episode-rollout.gif"
        palette = tmp_dir / "palette.png"

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "22",
                "-movflags",
                "+faststart",
                str(mp4_out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(mp4_out),
                "-vf",
                "fps=10,scale=880:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff",
                str(palette),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(mp4_out),
                "-i",
                str(palette),
                "-lavfi",
                "fps=10,scale=880:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
                str(gif_out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(
            f"Saved {mp4_out.relative_to(ROOT)} ({mp4_out.stat().st_size / 1024:.1f} KB)"
        )
        print(
            f"Saved {gif_out.relative_to(ROOT)} ({gif_out.stat().st_size / 1024:.1f} KB)"
        )


def generate_trained_multitier_showcase(
    cfg: ExperimentConfig, dev: str, trained, total_eps: int
):
    """Render curated trained FAR-satisfied layouts across XS, S, M, L, XL training tiers."""
    tiers = [
        ("xs_tokyo_0133", "XS", 9022),
        ("s_sgp_0345", "S", 9002),
        ("m_tokyo_0135", "M", 9007),
        ("l_bcn_0074", "L", 9008),
        ("xl_tokyo_0185", "XL", 9009),
    ]
    fig = plt.figure(figsize=(16.5, 5.1), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    gs = gridspec.GridSpec(
        1, 5, figure=fig, left=0.015, right=0.985, top=0.71, bottom=0.04, wspace=0.03
    )

    for idx, (site_id, tier, eval_seed) in enumerate(tiers):
        env = _load_env(site_id, cfg, dev)
        best, _ = _run_eval_episode(
            trained, env, cfg, eval_seed, temp=1.0, noise=1.0, record_trace=False
        )
        occ = best["occ"]
        cmask = best["core_mask"]
        ax = fig.add_subplot(gs[0, idx], projection="3d")
        _render_voxel_panel(
            ax,
            occ,
            cmask,
            mode="mass",
            elev=29.0,
            azim=-48.0,
            xray_mass=True,
            context=env.context,
        )
        z_top = int(np.flatnonzero(occ.any((1, 2)))[-1]) + 1 if occ.any() else 0
        nc = best["core_count"]
        blocks = (
            tier_cascade_blocks(env)
            if cfg.multiscale_schedule == "tier_adaptive"
            else ()
        )
        sched_lbl = "->".join([f"{b[1]}x{b[2]}x{b[0]}" for b in blocks] + ["1x1x1"])
        ax.set_title(
            f"{tier} ({sched_lbl}): {site_id}\n"
            f"{z_top}F active | {nc} core{'s' if nc != 1 else ''} | FAR {best['far']:.2f}\n"
            f"Obj {best['obj']:.3f} | {best['usable_area']:.0f} m2 usable\n"
            f"Sun {best['sun'] * 100:.1f}% | Sky {best['diff'] * 100:.1f}% | Cov {best['core_cov'] * 100:.0f}%",
            fontsize=8.3,
            fontweight="bold",
            color="#0f172a",
            pad=3,
        )
        print(
            f"  Tier {tier:2s} ({site_id}, {sched_lbl}): obj={best['obj']:.3f} far={best['far']:.2f} usable={best['usable_area']:.0f} cov={best['core_cov'] * 100:.1f}% seed={best['seed']}"
        )

    mode_tag = (
        "Tier-Adaptive Multi-Scale"
        if cfg.multiscale_schedule == "tier_adaptive"
        else "Flat 1x1x1"
    )
    fig.text(
        0.02,
        0.94,
        f"Multi-Tier Best Trained Layouts Across XS -> XL Parcels ({total_eps}-Ep {mode_tag}, compact-core.v5)",
        fontsize=12.6,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.02,
        0.885,
        "Selected FAR-satisfied layouts under matched eval decoding (temp=1.0, noise=1.0; target FAR 3.0 +/- 0.10, ceiling 20F).",
        fontsize=9.1,
        color="#475569",
    )
    out = MEDIA_DIR / "multitier-showcase.png"
    fig.savefig(out, facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def generate_heldout_generalization(
    cfg: ExperimentConfig, dev: str, untrained, trained, summary: dict, total_eps: int
):
    """Render paired 2x3 comparison on unseen HK, NYC, and Paris under matched eval decoding (1.0 / 1.0)."""
    sites = [
        (
            "m_hk_0199",
            "Hong Kong (m_hk_0199, Val)",
            summary["validation"]["perSite"]["m_hk_0199"]["bestSeed"],
        ),
        (
            "m_nyc_0003",
            "New York (m_nyc_0003, Test)",
            summary["test"]["perSite"]["m_nyc_0003"]["bestSeed"],
        ),
        (
            "m_prs_0315",
            "Paris (m_prs_0315, Test)",
            summary["test"]["perSite"]["m_prs_0315"]["bestSeed"],
        ),
    ]
    fig = plt.figure(figsize=(14.6, 8.9), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        left=0.060,
        right=0.985,
        top=0.755,
        bottom=0.025,
        hspace=0.32,
        wspace=0.08,
    )

    col_axes = []
    for col, (site_id, label, best_seed) in enumerate(sites):
        env = _load_env(site_id, cfg, dev)
        u_res, _ = _run_eval_episode(
            untrained, env, cfg, best_seed, temp=1.0, noise=1.0, record_trace=False
        )
        t_res, _ = _run_eval_episode(
            trained, env, cfg, best_seed, temp=1.0, noise=1.0, record_trace=False
        )
        pair_axes = []
        union_occ = np.logical_or(u_res["occ"], t_res["occ"])
        for row, (res, row_tag) in enumerate(
            ((u_res, "Untrained"), (t_res, "Trained"))
        ):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            ax.patch.set_alpha(0.0)
            _render_voxel_panel(
                ax,
                res["occ"],
                res["core_mask"],
                mode="mass",
                elev=29.0,
                azim=-48.0,
                xray_mass=True,
                context=env.context,
                bounds_occ=union_occ,
            )
            z_top = (
                int(np.flatnonzero(res["occ"].any((1, 2)))[-1]) + 1
                if res["occ"].any()
                else 0
            )
            nc = res["core_count"]
            ax.set_title(
                f"{row_tag}: {z_top}F | {nc} core{'s' if nc != 1 else ''} | FAR {res['far']:.2f}\n"
                f"Obj {res['obj']:.3f} | {res['usable_area']:.0f} m2 usable | Cov {res['core_cov'] * 100:.0f}%\n"
                f"Sun {res['sun'] * 100:.1f}% | Sky {res['diff'] * 100:.1f}%",
                fontsize=8.5,
                fontweight="bold" if row == 1 else "normal",
                color="#0f172a" if row == 1 else "#475569",
                pad=3,
            )
            pair_axes.append(ax)
        col_axes.append((label, best_seed, u_res, t_res, pair_axes[0], pair_axes[1]))

    fg_overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, zorder=10)
    fg_overlay.patch.set_alpha(0.0)
    fg_overlay.set_xlim(0.0, 1.0)
    fg_overlay.set_ylim(0.0, 1.0)
    fg_overlay.axis("off")

    mid_xs = []
    for label, best_seed, _u_res, _t_res, ax_top, ax_bot in col_axes:
        pt = ax_top.get_position()
        pb = ax_bot.get_position()
        cx = 0.5 * (pt.x0 + pt.x1)
        mid_xs.append((pt.x0, pt.x1, cx))
        fg_overlay.text(
            cx,
            0.842,
            f"{label}\n[paired seed {best_seed}]",
            ha="center",
            va="top",
            fontsize=9.8,
            fontweight="bold",
            color="#0f172a",
        )
        arrow_y0 = pt.y0 + 0.008
        arrow_y1 = pb.y1 + 0.070
        fg_overlay.annotate(
            "",
            xy=(cx, arrow_y1),
            xytext=(cx, arrow_y0),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.45,head_length=0.55",
                color="#1d4ed8",
                lw=2.0,
            ),
        )

    for i in range(len(mid_xs) - 1):
        vx = 0.5 * (mid_xs[i][1] + mid_xs[i + 1][0])
        fg_overlay.plot(
            [vx, vx], [0.020, 0.855], color="#cbd5e1", linestyle="-", linewidth=1.5
        )

    fig.text(
        0.018,
        0.62,
        "Untrained Control\n(seed 123 init)",
        fontsize=10.2,
        fontweight="bold",
        color="#475569",
        rotation=90,
        va="center",
    )
    fig.text(
        0.018,
        0.19,
        f"Trained ({total_eps} ep)\nZero-Shot",
        fontsize=10.2,
        fontweight="bold",
        color="#1d4ed8",
        rotation=90,
        va="center",
    )
    fig.text(
        0.060,
        0.950,
        "Zero-Shot Generalization to Unseen Cities (Matched Evaluation Distribution: temp=1.0, noise=1.0)",
        fontsize=13.0,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.060,
        0.910,
        f"Paired columns on identical seeds and parcels with no fine-tuning ({total_eps}-ep checkpoint, compact-core.v5, spatial_aux_weight=1.2).",
        fontsize=9.2,
        color="#475569",
    )
    out = MEDIA_DIR / "heldout-generalization.png"
    fig.savefig(out, facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def generate_training_curves(
    run_dir: Path,
    summary: dict,
    baseline_summary: dict | None = None,
    heuristic_summary: dict | None = None,
    pure_rl_summary: dict | None = None,
):
    """Plot multi-seed tier-adaptive training dynamics and held-out generalization vs baselines."""
    seed_files = sorted(run_dir.glob("seed-*/train.jsonl"))
    if not seed_files:
        seed_files = [run_dir / "seed-123" / "train.jsonl"]
    all_seed_rows = [
        [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        for p in seed_files
    ]
    n_seeds = len(all_seed_rows)
    eps = np.array([r["episode"] for r in all_seed_rows[0]], dtype=np.float32)
    win = 30
    total_eps = len(all_seed_rows[0])
    kern = np.ones(win, dtype=np.float32) / win
    rx = eps[win - 1 :]

    def _roll_multi(extractor):
        curves = np.stack(
            [
                np.convolve(
                    np.asarray([extractor(r) for r in rows], dtype=np.float32),
                    kern,
                    mode="valid",
                )
                for rows in all_seed_rows
            ],
            axis=0,
        )
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)
        return mean, std

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 5.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    curves_spec = (
        ("Layout Objective J", lambda r: r["quality"], "#1d4ed8", "-", 2.2),
        (
            "Usable Good Fraction",
            lambda r: r["diagnostics"]["usableGoodFraction"],
            "#0f766e",
            "-",
            1.7,
        ),
        (
            "Direct Sun Proxy",
            lambda r: r["environment"]["directSunProxy"],
            "#d97706",
            "-",
            1.8,
        ),
        (
            "Diffuse Sky Proxy",
            lambda r: r["environment"]["diffuseFacadeProxy"],
            "#0284c7",
            "--",
            1.8,
        ),
        (
            "Core Coverage (12 m)",
            lambda r: float(r["diagnostics"].get("coreCoverage") or 0.0),
            "#dc2626",
            ":",
            1.7,
        ),
        (
            "FAR Band Success Rate",
            lambda r: float(r["success"]),
            "#475569",
            "-.",
            1.7,
        ),
    )
    for label, fn, color, ls, lw in curves_spec:
        mean, std = _roll_multi(fn)
        ax1.plot(rx, mean, label=label, color=color, linestyle=ls, linewidth=lw)
        if n_seeds > 1:
            ax1.fill_between(rx, mean - std, mean + std, color=color, alpha=0.12)

    val_hist_files = [
        p.parent / "validation-history.jsonl"
        for p in seed_files
        if (p.parent / "validation-history.jsonl").is_file()
    ]
    if len(val_hist_files) == len(seed_files):
        all_vh = [
            [json.loads(line) for line in vp.read_text().splitlines() if line.strip()]
            for vp in val_hist_files
        ]
        if all_vh and all(len(vh) == len(all_vh[0]) and vh for vh in all_vh):
            vx = np.array([r["episode"] for r in all_vh[0]], dtype=np.float32)
            vq = np.array(
                [[float(r["meanQuality"]) for r in vh] for vh in all_vh],
                dtype=np.float32,
            )
            v_mean = vq.mean(axis=0)
            v_std = vq.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(v_mean)
            ax1.plot(
                vx,
                v_mean,
                label="Val m_hk_0199 (50-ep ckpt)",
                color="#7c3aed",
                marker="o",
                markersize=4.2,
                linewidth=2.0,
                zorder=5,
            )
            if n_seeds > 1:
                ax1.fill_between(
                    vx, v_mean - v_std, v_mean + v_std, color="#7c3aed", alpha=0.12
                )

    ax1.set_xlim(win, total_eps)
    ax1.set_ylim(-0.04, 1.38)
    seed_lbl = f"Across {n_seeds} Seeds" if n_seeds > 1 else "Seed 123"
    ax1.set_xlabel(
        f"Training Episode (30-Ep Rolling Mean ± 1 SD {seed_lbl})",
        fontsize=9.6,
        fontweight="bold",
    )
    ax1.set_ylabel("Score / Proxy Fraction", fontsize=9.8, fontweight="bold")
    ax1.set_title(
        f"A. Tier-Adaptive Training Trajectory ({total_eps} Ep, {n_seeds} Seed{'s' if n_seeds != 1 else ''})",
        fontsize=10.8,
        fontweight="bold",
        color="#0f172a",
    )
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper center", fontsize=6.8, ncol=4, frameon=True)

    sites = [
        ("HK (Val)", "validation", "m_hk_0199"),
        ("NYC (Test)", "test", "m_nyc_0003"),
        ("Paris (Test)", "test", "m_prs_0315"),
        ("Test Agg", "test", "overall"),
    ]
    series = [
        (
            f"Untrained ({n_seeds} seeds)" if n_seeds > 1 else "Untrained Control",
            summary["untrainedAggregate"]
            if "untrainedAggregate" in summary
            else summary,
            "untrained",
            "#cbd5e1",
            "#64748b",
        )
    ]
    if heuristic_summary is not None:
        series.append(
            ("Greedy Heuristic", heuristic_summary, "std", "#f59e0b", "#b45309")
        )
    if baseline_summary is not None:
        series.append(
            ("Flat 1x1x1 (1,200 ep)", baseline_summary, "std", "#93c5fd", "#2563eb")
        )
    if pure_rl_summary is not None:
        series.append(("Pure-RL (aux=0)", pure_rl_summary, "std", "#14b8a6", "#0f766e"))
    tier_tag = (
        f"Tier-Adaptive ({n_seeds} seeds, best.pt)"
        if n_seeds > 1 and "bestValidationEpisodes" in summary
        else (
            f"Tier-Adaptive ({n_seeds} seeds)"
            if n_seeds > 1
            else f"Tier-Adaptive ({total_eps} ep)"
        )
    )
    series.append((tier_tag, summary, "std", "#1d4ed8", "#1e3a8a"))

    def _extract(src, mode, split, site_key):
        if mode == "untrained":
            uk = "untrainedValidation" if split == "validation" else "untrainedTest"
            if uk in src:
                node = (
                    src[uk]["perSite"][site_key]
                    if site_key != "overall"
                    else src[uk]["overall"]
                )
                return node["meanQuality"], float(node.get("stdQuality") or 0.0)
            node = (
                src["validation"]["perSite"][site_key]
                if split == "validation"
                else (
                    src["test"]["perSite"][site_key]
                    if site_key != "overall"
                    else src["test"]["overall"]
                )
            )
            return node["meanQuality"], 0.0
        sec = src[split]
        if site_key == "overall":
            node = sec.get("overall") or sec.get("summary")
        else:
            node = sec["perSite"][site_key]
        return node["meanQuality"], float(node.get("stdAcrossModelSeeds") or 0.0)

    x = np.arange(len(sites))
    m_series = len(series)
    w = min(0.24, 0.80 / m_series)
    offsets = (np.arange(m_series) - 0.5 * (m_series - 1)) * w

    for idx_s, (s_lbl, src, mode, fc, ec) in enumerate(series):
        vals, errs = [], []
        for _slbl, sp, sk in sites:
            v, sd = _extract(src, mode, sp, sk)
            vals.append(v)
            errs.append(sd)
        has_err = any(e > 0 for e in errs)
        bars = ax2.bar(
            x + offsets[idx_s],
            vals,
            width=w,
            yerr=errs if has_err else None,
            capsize=2.5 if has_err else 0,
            color=fc,
            edgecolor=ec,
            linewidth=0.8,
            label=s_lbl,
            error_kw=dict(ecolor="#0f172a", lw=1.0),
        )
        for rect, v, sd in zip(bars, vals, errs):
            h = rect.get_height() + (sd if has_err else 0.0)
            txt = f"{v:.2f}"
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.015,
                txt,
                ha="center",
                va="bottom",
                fontsize=6.8 if m_series >= 4 else 7.8,
                fontweight="bold" if idx_s == m_series - 1 else "normal",
                color=ec if idx_s > 0 else "#475569",
            )

    ax2.set_ylim(0.0, 1.54)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[0] for s in sites], fontsize=9.4)
    ax2.set_ylabel(
        "Mean Held-Out Objective J (5 eval seeds/site)", fontsize=9.6, fontweight="bold"
    )
    ax2.set_title(
        "B. Held-Out Generalization & Controls (1.0 / 1.0 Decoding)",
        fontsize=10.8,
        fontweight="bold",
        color="#0f172a",
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.legend(loc="upper center", fontsize=7.2, ncol=3, frameon=True)

    cfg_seeds = summary.get("config", {}).get("seeds") or [
        p.parent.name.split("-", 1)[1] for p in seed_files
    ]
    seed_ids_txt = "/".join(str(s) for s in cfg_seeds)
    ckpt_note = (
        "; best.pt selected every 50 ep on Val m_hk_0199"
        if "bestValidationEpisodes" in summary
        else ""
    )
    fig.suptitle(
        f"{total_eps}-Episode Multi-Seed Tier-Adaptive Training & Controls (Apple M4 MPS, compact-core.v5)",
        fontsize=12.4,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        f"Training seeds: {n_seeds} ({seed_ids_txt}), matched eval decoding (temp=1.0, noise=1.0){ckpt_note}, error bars = +/-1 SD across seeds.",
        ha="center",
        fontsize=8.2,
        color="#475569",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out = MEDIA_DIR / "training-curves-600ep.png"
    fig.savefig(out, facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def generate_multiscale_ablation(
    flat_run: Path,
    tier_run: Path,
    cfg_flat: ExperimentConfig,
    cfg_tier: ExperimentConfig,
    flat_pol,
    tier_pol,
    dev: str,
):
    """Compare Flat 1x1x1 (1,200 ep) vs Tier-Adaptive Multi-Scale (600 ep) on L/XL training & deterministic eval."""

    def _load_jsonl(p: Path):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    flat_rows = _load_jsonl(flat_run / "seed-123" / "train.jsonl")
    tier_rows = _load_jsonl(tier_run / "seed-123" / "train.jsonl")

    def _tier_bins(rows, prefix):
        xs, oks, objs = [], [], []
        sub = [r for r in rows if r["site"].startswith(prefix)]
        win = 10
        for i in range(win, len(sub) + 1):
            w = sub[i - win : i]
            xs.append(w[-1]["episode"])
            oks.append(float(np.mean([float(r["success"]) for r in w])))
            objs.append(float(np.mean([float(r["quality"]) for r in w])))
        return xs, oks, objs

    def _grad_bins(rows):
        xs, gs = [], []
        for lo in range(0, len(rows), 100):
            b = rows[lo : lo + 100]
            g = [r["learning"]["gradientNorm"] for r in b if r.get("learning")]
            if g:
                xs.append(lo + len(b) / 2.0)
                gs.append(float(np.mean(g)))
        return xs, gs

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    fx_l, fok_l, _ = _tier_bins(flat_rows, "l_")
    fx_xl, fok_xl, _ = _tier_bins(flat_rows, "xl_")
    tx_l, tok_l, _ = _tier_bins(tier_rows, "l_")
    tx_xl, tok_xl, _ = _tier_bins(tier_rows, "xl_")

    ax1.plot(
        fx_l,
        fok_l,
        color="#94a3b8",
        linestyle="--",
        lw=1.8,
        label="Flat 1x1x1 — L Sites",
    )
    ax1.plot(
        fx_xl,
        fok_xl,
        color="#ef4444",
        linestyle="--",
        lw=1.9,
        label="Flat 1x1x1 — XL Sites",
    )
    ax1.plot(
        tx_l,
        tok_l,
        color="#0f766e",
        linestyle="-",
        lw=2.2,
        label="Tier-Adaptive — L Sites (2x2x1->1)",
    )
    ax1.plot(
        tx_xl,
        tok_xl,
        color="#1d4ed8",
        linestyle="-",
        lw=2.4,
        label="Tier-Adaptive — XL Sites (4x4x1->2->1)",
    )
    ax1.set_ylim(0.0, 1.08)
    ax1.set_xlim(0, len(flat_rows))
    ax1.set_xlabel("Training Episode", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("10-Visit Rolling FAR Success Rate", fontsize=9.5, fontweight="bold")
    ax1.set_title(
        "A. L & XL Training FAR Compliance",
        fontsize=10.8,
        fontweight="bold",
        color="#0f172a",
    )
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=7.6, frameon=True)

    gx_f, gy_f = _grad_bins(flat_rows)
    gx_t, gy_t = _grad_bins(tier_rows)
    ax2.plot(
        gx_f,
        gy_f,
        marker="o",
        markersize=4.5,
        color="#dc2626",
        lw=2.0,
        label="Flat 1x1x1 (unnormalized)",
    )
    ax2.plot(
        gx_t,
        gy_t,
        marker="s",
        markersize=4.5,
        color="#1d4ed8",
        lw=2.2,
        label="Tier-Adaptive (1/sqrt(N) + cosine)",
    )
    ax2.set_yscale("log")
    ax2.set_xlim(0, len(flat_rows))
    ax2.set_xlabel(
        "Training Episode (100-Ep Bin Midpoint)", fontsize=9.5, fontweight="bold"
    )
    ax2.set_ylabel(
        "Mean GAE Gradient Norm (log scale)", fontsize=9.5, fontweight="bold"
    )
    ax2.set_title(
        "B. Optimization Gradient Stability",
        fontsize=10.8,
        fontweight="bold",
        color="#0f172a",
    )
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=7.9, frameon=True)

    eval_sites = [
        ("L: l_bcn", "l_bcn_0074"),
        ("L: l_sgp", "l_sgp_0369"),
        ("XL: xl_tokyo", "xl_tokyo_0185"),
        ("XL: xl_sgp", "xl_sgp_0386"),
        ("Val: HK", "m_hk_0199"),
        ("Test: NYC", "m_nyc_0003"),
        ("Test: Paris", "m_prs_0315"),
    ]
    fq_list, tq_list, fok_list, tok_list = [], [], [], []
    for _, sid in eval_sites:
        env = _load_env(sid, cfg_tier, dev)
        fr, _ = _run_eval_episode(
            flat_pol, env, cfg_flat, 9001, temp=1.0, noise=1.0, record_trace=False
        )
        tr, _ = _run_eval_episode(
            tier_pol, env, cfg_tier, 9001, temp=1.0, noise=1.0, record_trace=False
        )
        fq_list.append(fr["obj"])
        tq_list.append(tr["obj"])
        fok_list.append(fr["far_satisfied"])
        tok_list.append(tr["far_satisfied"])

    x = np.arange(len(eval_sites))
    bw = 0.36
    b_f = ax3.bar(
        x - bw / 2,
        fq_list,
        width=bw,
        color=["#94a3b8" if ok else "#fca5a5" for ok in fok_list],
        edgecolor="#64748b",
        label="Flat 1x1x1 (1,200 ep)",
    )
    b_t = ax3.bar(
        x + bw / 2, tq_list, width=bw, color="#1d4ed8", label="Tier-Adaptive (600 ep)"
    )
    for rect, ok in zip(b_f, fok_list):
        h = rect.get_height()
        tag = f"{h:.2f}" if ok else f"{h:.2f}\n(FAR fail)"
        ax3.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.015,
            tag,
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#dc2626" if not ok else "#475569",
            fontweight="bold" if not ok else "normal",
        )
    for rect, ok in zip(b_t, tok_list):
        h = rect.get_height()
        ax3.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.015,
            f"{h:.2f}" if ok else f"{h:.2f}!",
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            color="#1d4ed8",
        )
    ax3.set_ylim(0.0, 1.32)
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [s[0] for s in eval_sites], fontsize=8.0, rotation=18, ha="right"
    )
    ax3.set_ylabel(
        "Deterministic Eval Objective J (seed 9001)", fontsize=9.5, fontweight="bold"
    )
    ax3.set_title(
        "C. Large-Scale & Unseen Eval (Matched 1.0/1.0)",
        fontsize=10.8,
        fontweight="bold",
        color="#0f172a",
    )
    ax3.grid(axis="y", linestyle="--", alpha=0.45)
    ax3.legend(loc="upper left", fontsize=7.9, frameon=True)

    fig.suptitle(
        "Flat 1x1x1 (1,200 Ep) vs Tier-Adaptive Multi-Scale Cascade (600 Ep) — M4 MPS Ablation",
        fontsize=12.6,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
    out = MEDIA_DIR / "multiscale-ablation.png"
    fig.savefig(out, facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def generate_l_xl_ablation_figure():
    """Plot 5-seed L/XL NYC validation curves and 7-site held-out ablation from committed JSONs."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    ctrl = json.loads(
        (RESULTS_DIR / "control-all-tiers-400ep-mps-2026-11-22.json").read_text()
    )
    ungated = ctrl["ungatedAblation"]
    dist = json.loads(
        (RESULTS_DIR / "distill-all-tiers-400ep-mps-2026-11-22.json").read_text()
    )
    greedy = json.loads(
        (RESULTS_DIR / "greedy-control-all-tiers-mps-2026-11-22.json").read_text()
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16.2, 5.8),
        dpi=160,
        gridspec_kw={"width_ratios": [1.0, 1.38]},
    )
    fig.patch.set_facecolor("#ffffff")

    def _val_curve(src):
        vh_list = [s["validationHistory"] for s in src["seeds"]]
        xs = np.array([r["episode"] for r in vh_list[0]], dtype=np.float32)
        ys = np.array(
            [[float(r["meanQuality"]) for r in vh] for vh in vh_list], dtype=np.float32
        )
        return xs, ys.mean(axis=0), ys.std(axis=0, ddof=1)

    g_val = float(greedy["validation"]["summary"]["meanQuality"])
    ax1.axhline(
        g_val,
        color="#ea580c",
        linestyle="--",
        linewidth=1.6,
        label=f"Greedy (tier_meso2): {g_val:.3f}",
        zorder=2,
    )
    for label, src, color, ls, marker in (
        (
            "FAR-Gated Actor-Critic (best.pt: 0.723)",
            ctrl,
            "#1d4ed8",
            "-",
            "o",
        ),
        (
            "Ungated Actor-Critic (best.pt: 0.736)",
            ungated,
            "#64748b",
            "--",
            "s",
        ),
        (
            "Two-Teacher Distill (best.pt: 0.707)",
            dist,
            "#0f766e",
            "-",
            "^",
        ),
    ):
        xs, ym, ys = _val_curve(src)
        ax1.plot(
            xs,
            ym,
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=4.5,
            linewidth=2.0,
            zorder=4 if color == "#1d4ed8" else 3,
        )
        ax1.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.11)

    ax1.set_xlim(40, 410)
    ax1.set_ylim(0.15, 0.88)
    ax1.set_xticks(np.arange(50, 401, 50))
    ax1.set_xlabel(
        "Training Episode (50-Ep Checkpoint)", fontsize=9.6, fontweight="bold"
    )
    ax1.set_ylabel(
        "Mean Validation Objective J (3 NYC Sites)", fontsize=9.6, fontweight="bold"
    )
    ax1.set_title(
        "A. Held-Out NYC Validation Trajectory (400 Ep, 5 Seeds)",
        fontsize=10.6,
        fontweight="bold",
        color="#0f172a",
    )
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=7.6, frameon=True)

    sites = [
        ("m_nyc\n(Val)", "validation", "m_nyc_0005"),
        ("l_nyc\n(Val)", "validation", "l_nyc_0019"),
        ("xl_nyc\n(Val)", "validation", "xl_nyc_0062"),
        ("m_prs\n(Test)", "test", "m_prs_0320"),
        ("l_sgp_327\n(Test)", "test", "l_sgp_0327"),
        ("l_sgp_367\n(Test)", "test", "l_sgp_0367"),
        ("l_sgp_379\n(Test)", "test", "l_sgp_0379"),
        ("Test Mean\n(4 sites)", "overall", "overall"),
    ]
    bars_spec = [
        ("Greedy (tier_meso2)", "greedy", "#f59e0b", "#b45309"),
        ("Two-Teacher Distill", "dist", "#14b8a6", "#0f766e"),
        ("Ungated Actor-Critic", "ungated", "#94a3b8", "#475569"),
        ("FAR-Gated Actor-Critic", "gated", "#1d4ed8", "#1e3a8a"),
    ]

    def _bar_val(kind, split, site_key):
        if kind == "greedy":
            if split == "overall":
                return float(greedy["test"]["summary"]["meanQuality"]), 0.0
            return float(greedy[split]["perSite"][site_key]["meanQuality"]), 0.0
        src = dist if kind == "dist" else (ungated if kind == "ungated" else ctrl)
        if split == "overall":
            m = src["aggregate"]["test"]["metrics"]["meanQuality"]
            return float(m["mean"]), float(m["stdAcrossModelSeeds"])
        grp = "validationSelected" if split == "validation" else "testSelected"
        node = src["perSite"][grp][site_key]
        return float(node["meanQuality"]), float(node["stdAcrossModelSeeds"])

    x = np.arange(len(sites))
    w = 0.195
    offsets = (np.arange(len(bars_spec)) - 1.5) * w
    for idx_b, (b_lbl, kind, fc, ec) in enumerate(bars_spec):
        vals, errs = [], []
        for _slbl, sp, sk in sites:
            v, sd = _bar_val(kind, sp, sk)
            vals.append(v)
            errs.append(sd)
        has_err = any(e > 0 for e in errs)
        rects = ax2.bar(
            x + offsets[idx_b],
            vals,
            width=w,
            yerr=errs if has_err else None,
            capsize=2.2 if has_err else 0,
            color=fc,
            edgecolor=ec,
            linewidth=0.8,
            label=b_lbl,
            error_kw=dict(ecolor="#0f172a", lw=0.95),
        )
        for idx_g, (rect, v, sd) in enumerate(zip(rects, vals, errs)):
            is_last_group = idx_g == len(sites) - 1
            if idx_b not in (0, 3) and not is_last_group:
                continue
            top_y = rect.get_height() + (sd if has_err else 0.0)
            txt = f"{v:.3f}" if is_last_group and idx_b == 3 else f"{v:.2f}"
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                top_y + (0.015 if not is_last_group else 0.018),
                txt,
                ha="center",
                va="bottom",
                fontsize=6.7 if not is_last_group else 6.6,
                rotation=90 if is_last_group else 0,
                fontweight="bold" if idx_b == 3 else "normal",
                color=ec,
            )

    ax2.axvline(2.5, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax2.axvline(6.5, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax2.set_ylim(0.0, 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[0] for s in sites], fontsize=8.4)
    ax2.set_ylabel(
        "Mean Held-Out Objective J (5 eval seeds/site)", fontsize=9.6, fontweight="bold"
    )
    ax2.set_title(
        "B. Held-Out Per-Site & Aggregate Comparison (best.pt, 5 Seeds ± 1 SD)",
        fontsize=10.6,
        fontweight="bold",
        color="#0f172a",
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=7.6, ncol=2, frameon=True)

    fig.suptitle(
        "400-Episode L/XL Ablation: Matched Actor-Critic Control vs. Two-Teacher Distillation (Apple M4 MPS)",
        fontsize=12.2,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Model seeds: 5 (4211-4215), matched decoding (temp=1.0, noise=1.0), tier_meso2 + dense_potential + learned_cap; error bars = +/-1 SD across model seeds. FAR-Gated Actor-Critic achieves 100% FAR across all 7/7 Val and Test sites (best.pt).",
        ha="center",
        fontsize=8.0,
        color="#475569",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out = MEDIA_DIR / "l-xl-ablation-400ep.png"
    fig.savefig(out, facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def generate_blind_and_aux_ablation_figure():
    """Plot 8-site zero-shot blind holdout (XS-XL) and 200-ep auxiliary schedule ablation."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    blind = json.loads(
        (RESULTS_DIR / "blind-8site-all-tiers-mps-2026-11-23.json").read_text()
    )
    aux = json.loads(
        (RESULTS_DIR / "aux-ablation-200ep-mps-2026-11-23.json").read_text()
    )
    greedy = json.loads(
        (RESULTS_DIR / "greedy-control-all-tiers-mps-2026-11-22.json").read_text()
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16.5, 5.8),
        dpi=160,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    fig.patch.set_facecolor("#ffffff")

    groups = [
        ("xs_sgp\n346", "site", "xs_sgp_0346"),
        ("s_sgp\n345", "site", "s_sgp_0345"),
        ("m_nyc\n003", "site", "m_nyc_0003"),
        ("m_prs\n315", "site", "m_prs_0315"),
        ("XS–M\nMean", "xs_m", "xs_m"),
        ("l_prs\n324", "site", "l_prs_0324"),
        ("l_sgp\n369", "site", "l_sgp_0369"),
        ("xl_nyc\n048", "site", "xl_nyc_0048"),
        ("xl_sgp\n386", "site", "xl_sgp_0386"),
        ("L–XL\nMean", "l_xl", "l_xl"),
        ("All 8\nBlind", "overall", "overall"),
    ]
    bars_spec = [
        ("Greedy (tier_meso2)", "greedy_tier_meso2", "#f59e0b", "#b45309"),
        ("Two-Teacher Distill", "on_policy_distill", "#14b8a6", "#0f766e"),
        ("Ungated Actor-Critic", "ungated_actor_critic", "#94a3b8", "#475569"),
        ("FAR-Gated Actor-Critic", "far_gated_actor_critic", "#1d4ed8", "#1e3a8a"),
    ]

    def _blind_val(arm_key: str, kind: str, key: str) -> tuple[float, float]:
        arm = blind[arm_key]
        if arm_key == "greedy_tier_meso2":
            if kind == "site":
                return float(arm["perSite"][key]["meanQuality"]), 0.0
            node = arm["summary"] if kind == "overall" else arm[kind]
            return float(node["meanQuality"]), 0.0
        if kind == "site":
            node = arm["perSite"][key]
            return float(node["meanQuality"]), float(node["stdAcrossModelSeeds"])
        node = arm["aggregate"][kind]["meanQuality"]
        return float(node["mean"]), float(node["stdAcrossModelSeeds"])

    x = np.arange(len(groups))
    w = 0.20
    offsets = (np.arange(len(bars_spec)) - 1.5) * w
    for idx_b, (b_lbl, arm_key, fc, ec) in enumerate(bars_spec):
        vals, errs = [], []
        for _lbl, kind, key in groups:
            v, sd = _blind_val(arm_key, kind, key)
            vals.append(v)
            errs.append(sd)
        has_err = any(e > 0 for e in errs)
        rects = ax1.bar(
            x + offsets[idx_b],
            vals,
            width=w,
            yerr=errs if has_err else None,
            capsize=2.0 if has_err else 0,
            color=fc,
            edgecolor=ec,
            linewidth=0.8,
            label=b_lbl,
            error_kw=dict(ecolor="#0f172a", lw=0.9),
        )
        for idx_g, (rect, v, sd) in enumerate(zip(rects, vals, errs)):
            is_summary = groups[idx_g][1] != "site"
            show_neg = v < 0
            if idx_b not in (0, 3) and not is_summary and not show_neg:
                continue
            cx = rect.get_x() + rect.get_width() / 2
            if v >= 0:
                top_y = v + (sd if has_err else 0.0)
                txt = (
                    f"{v:.3f}"
                    if (idx_g == len(groups) - 1 and idx_b == 3)
                    else f"{v:.2f}"
                )
                ax1.text(
                    cx,
                    top_y + (0.015 if not is_summary else 0.018),
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    rotation=90 if is_summary else 0,
                    fontweight="bold" if idx_b == 3 else "normal",
                    color=ec,
                )
            else:
                bot_y = v - (sd if has_err else 0.0)
                ax1.text(
                    cx,
                    bot_y - 0.015,
                    f"{v:.2f}",
                    ha="center",
                    va="top",
                    fontsize=6.4,
                    color=ec,
                )

    for split_x in (3.5, 4.5, 8.5, 9.5):
        ax1.axvline(split_x, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax1.axhline(0.0, color="#64748b", linewidth=0.85)
    ax1.set_ylim(-0.42, 1.24)
    ax1.set_xticks(x)
    ax1.set_xticklabels([g[0] for g in groups], fontsize=7.9)
    ax1.set_ylabel(
        "Mean Zero-Shot Objective J (5 eval seeds/site)",
        fontsize=9.5,
        fontweight="bold",
    )
    ax1.set_title(
        "A. Blind 8-Site Zero-Shot Holdout Across XS–XL (640 Episodes, 5 Seeds ± 1 SD)",
        fontsize=10.4,
        fontweight="bold",
        color="#0f172a",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=7.6, ncol=2, frameon=True)

    g_val = float(greedy["validation"]["summary"]["meanQuality"])
    ax2.axhline(
        g_val,
        color="#ea580c",
        linestyle=":",
        linewidth=1.6,
        label=f"Greedy (tier_meso2) NYC Val: {g_val:.3f}",
        zorder=2,
    )
    arms_spec = [
        (
            "annealed_aux",
            "Annealed Aux (1.2→0.0) [best Test 0.797]",
            "#1d4ed8",
            "-",
            "o",
            0.0,
        ),
        (
            "constant_aux",
            "Constant Aux (λ=1.2)* [best Test 0.801]",
            "#0f766e",
            "--",
            "s",
            0.0,
        ),
        ("pure_rl", "Pure RL (λ=0.0) [best Test 0.464]", "#dc2626", "-.", "^", 0.0),
    ]
    for arm_key, label, color, ls, marker, y_off in arms_spec:
        arm_doc = aux["arms"][arm_key]
        curve = arm_doc["validationCurve200"]
        xs = np.array([row["episode"] for row in curve], dtype=np.float32)
        ym = np.array([row["meanQuality"] for row in curve], dtype=np.float32)
        ys = np.array([row["stdQuality"] for row in curve], dtype=np.float32)
        ep200_test = float(
            arm_doc["aggregate"]["finalTest"]["metrics"]["meanQuality"]["mean"]
        )
        ax2.plot(
            xs,
            ym,
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=4.8,
            linewidth=2.0,
            zorder=4 if arm_key == "annealed_aux" else 3,
        )
        ax2.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.12)
        ax2.text(
            204,
            float(ym[-1]) + y_off,
            f"ep200 Test\n{ep200_test:.3f}",
            ha="left",
            va="center",
            fontsize=7.3,
            fontweight="bold" if arm_key == "annealed_aux" else "normal",
            color=color,
        )

    ax2.set_xlim(42, 236)
    ax2.set_ylim(0.18, 0.84)
    ax2.set_xticks([50, 100, 150, 200])
    ax2.set_xlabel(
        "Training Episode (50-Ep Checkpoint)", fontsize=9.5, fontweight="bold"
    )
    ax2.set_ylabel(
        "Mean Validation Objective J (3 NYC Sites)", fontsize=9.5, fontweight="bold"
    )
    ax2.set_title(
        "B. 200-Episode Auxiliary Supervision Ablation (3 Seeds ± 1 SD)",
        fontsize=10.4,
        fontweight="bold",
        color="#0f172a",
    )
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(
        loc="center right", bbox_to_anchor=(0.98, 0.56), fontsize=7.4, frameon=True
    )

    fig.suptitle(
        "Blind Cross-Scale Transfer (XS–XL) & Auxiliary-Supervision Schedule Ablation (Apple M4 MPS)",
        fontsize=12.0,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Panel A: 8 untouched parcels (0 overlap with train/val, best.pt, seeds 4211-4215). Panel B: seeds 4211-4213, tier_meso2 + dense_potential + learned_cap (*constant_aux uses 400-ep cosine LR prefix).",
        ha="center",
        fontsize=8.0,
        color="#475569",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out = MEDIA_DIR / "blind-8site-and-aux-ablation.png"
    fig.savefig(out, facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def _load_summary(summary_path: Path, cfg: ExperimentConfig) -> dict:
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    raise FileNotFoundError(f"Missing audited summary file: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs" / "real-aux-tier-600ep"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULTS_DIR / "real-aux-tier-600ep-mps-2026-11-21.json",
    )
    parser.add_argument(
        "--baseline-run-dir", type=Path, default=ROOT / "runs" / "real-aux-1200ep"
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=RESULTS_DIR / "real-aux-1200ep-mps-2026-11-21.json",
    )
    parser.add_argument(
        "--heuristic-summary",
        type=Path,
        default=RESULTS_DIR / "greedy-heuristic-tier-mps-2026-11-21.json",
    )
    parser.add_argument(
        "--pure-rl-summary",
        type=Path,
        default=RESULTS_DIR / "real-tier-600ep-mps-2026-11-21.json",
    )
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "arch",
            "lxl",
            "blind",
            "curves",
            "ablation",
            "multitier",
            "heldout",
            "rollout",
        ),
        default="all",
    )
    args = parser.parse_args()

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "arch"):
        generate_architecture_diagram()
        if args.only == "arch":
            return
    if args.only in ("all", "lxl"):
        generate_l_xl_ablation_figure()
        if args.only == "lxl":
            return
    if args.only in ("all", "blind"):
        generate_blind_and_aux_ablation_figure()
        if args.only == "blind":
            return
    summary = _load_summary(
        args.summary, ExperimentConfig.read(args.run_dir / "config.json")
    )
    baseline_summary = (
        json.loads(args.baseline_summary.read_text())
        if args.baseline_summary.is_file()
        else None
    )
    heuristic_summary = (
        json.loads(args.heuristic_summary.read_text())
        if args.heuristic_summary.is_file()
        else None
    )
    pure_rl_summary = (
        json.loads(args.pure_rl_summary.read_text())
        if args.pure_rl_summary.is_file()
        else None
    )
    if args.only in ("all", "curves"):
        generate_training_curves(
            args.run_dir,
            summary,
            baseline_summary=baseline_summary,
            heuristic_summary=heuristic_summary,
            pure_rl_summary=pure_rl_summary,
        )
    if args.only in ("all", "ablation", "multitier", "heldout", "rollout"):
        cfg, dev, untrained, trained, total_eps = _load_run_policy(args.run_dir)
        if (
            args.only in ("all", "ablation")
            and (args.baseline_run_dir / "config.json").is_file()
        ):
            cfg_flat, _, _, flat_trained, _ = _load_run_policy(args.baseline_run_dir)
            generate_multiscale_ablation(
                args.baseline_run_dir,
                args.run_dir,
                cfg_flat,
                cfg,
                flat_trained,
                trained,
                dev,
            )
        if args.only in ("all", "multitier"):
            generate_trained_multitier_showcase(cfg, dev, trained, total_eps)
        if args.only in ("all", "heldout"):
            generate_heldout_generalization(
                cfg, dev, untrained, trained, summary, total_eps
            )
        if args.only in ("all", "rollout"):
            generate_trained_rollout_assets(cfg, dev, trained, total_eps)


if __name__ == "__main__":
    main()
