"""Render a 4x6 VoxelForm-RL portfolio massing atlas with separate Affinity layers.

Features:
- Clean non-triangulated road geometry (unioned, buffer-healed, building-subtracted,
  clipped to orthographic viewport with zero internal triangle seams).
- Depth-classified white surrounding buildings:
  * Behind / left / right of hero: pure white, 100% opaque.
  * Foreground occluding hero: pure white, semi-transparent.
- Translucent white hero voxel massing with vivid yellow cores (#FADE24).
- Separate pixel-locked layers for Affinity Designer / Publisher:
  * atlas_4x6_renders.png      (shaded fills only)
  * atlas_4x6_outlines.png     (hidden-line-removed transparent RGBA outlines)
  * atlas_4x6_outlines.svg     (editable vector outlines)
  * atlas_4x6_affinity.svg     (layered Affinity-ready SVG combining all layers)
  * atlas_4x6_composite.png    (exact preview composite)
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image
from shapely import constrained_delaunay_triangles
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from tools.export_blender_site import _load_policy, _run_trained_episode
from tools.generate_readme_media import (
    _extract_building_footprint,
    _iter_polygons,
    _surface_quads,
)
from voxelform.context import ContextLibrary

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "control_all_tiers"
CKPT_PATH = RUN_DIR / "seed-4214" / "best.pt"
CACHE_DIR = ROOT / "runs" / "portfolio-stage2"
DRAWINGS_DIR = CACHE_DIR / "drawings"
TILES_DIR = DRAWINGS_DIR / "atlas_tiles_4x6"
CACHE_FILE = CACHE_DIR / "atlas_candidates.npz"
META_CACHE_FILE = CACHE_DIR / "atlas_candidates.json"

TIERS = ("xs", "s", "m", "l", "xl")
AZIM_CANDIDATES = (-46.0, 44.0, 134.0, -136.0)
DEFAULT_ELEV = 31.0

# User-curated balanced shuffled 4x6 selection across 6 rows x 4 cols (from 7x7 picks)
DEFAULT_PICKS_24: tuple[int, ...] = (
    35, 2, 46, 12,    # Row 1: L(BCN), XS(Paris), XL(Tokyo), S(SGP)
    23, 47, 15, 38,   # Row 2: M(HK), XL(NYC), S(BCN), L(Paris)
    9, 34, 40, 28,    # Row 3: XS(HK), L(Tokyo), L(SGP), M(NYC)
    48, 26, 18, 5,    # Row 4: XL(SGP), M(BCN), S(NYC), XS(BCN)
    27, 37, 13, 49,   # Row 5: M(Paris), L(HK), S(Tokyo), XL(NYC)
    39, 7, 29, 25,    # Row 6: L(NYC), XS(BCN), M(SGP), M(Tokyo)
)
TIER_ZOOM: dict[str, float] = {"xl": 1.25, "l": 1.25}

SHOWCASE_PRIORITIES: dict[str, tuple[str, ...]] = {
    "xs": ("xs_tokyo_0133", "xs_hk_0196", "xs_nyc_0001", "xs_prs_0290"),
    "s": ("s_sgp_0345", "s_bcn_0112", "s_hk_0202", "s_prs_0302"),
    "m": ("m_tokyo_0135", "m_prs_0315", "m_nyc_0003", "m_bcn_0093", "m_hk_0199"),
    "l": ("l_sgp_0379", "l_bcn_0074", "l_nyc_0019", "l_hk_0198", "l_tokyo_0191"),
    "xl": ("xl_tokyo_0185", "xl_nyc_0062"),
}


def _camera_basis(elev_deg: float, azim_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (u_right, v_up, view_toward_cam) vectors for mplot3d ortho."""
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    toward_cam = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    toward_cam /= np.linalg.norm(toward_cam)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_axis = np.cross(up_world, toward_cam)
    nrm = np.linalg.norm(u_axis)
    if nrm < 1e-8:
        u_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u_axis /= nrm
    v_axis = np.cross(toward_cam, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return u_axis, v_axis, toward_cam


def _project_pts(
    pts: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, view_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = pts @ u_axis
    v = pts @ v_axis
    d = pts @ view_vec
    return u, v, d


def _clean_road_geometry(
    context: dict,
    cx: float,
    cy: float,
    half: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Union, heal, and clean OSM road geometry without cutting at borders or footprints."""
    roads_raw = context.get("roads") or ()
    if not roads_raw:
        return [], []

    view_pad = max(half * 3.5, 220.0)
    view_box = box(cx - view_pad, cy - view_pad, cx + view_pad, cy + view_pad)

    raw_polys: list[Polygon] = []
    for r in roads_raw:
        rv = np.asarray(r.get("vertices") or (), dtype=np.float64)
        rf = r.get("faces") or ()
        if not len(rv) or not rf:
            continue
        for f in rf:
            if len(f) < 3:
                continue
            p = Polygon(rv[f, :2])
            if not p.is_valid:
                p = p.buffer(0)
            if p.area > 0.05 and p.intersects(view_box):
                raw_polys.append(p)

    if not raw_polys:
        return [], []

    merged = unary_union(raw_polys)
    if not merged.is_valid:
        merged = merged.buffer(0)
    if merged.is_empty:
        return [], []

    # Bridge micro-gaps between road triangles and smooth spikes/whiskers cleanly without cutting
    cleaned = (
        merged.buffer(0.25, join_style="bevel")
        .buffer(-0.25, join_style="bevel")
        .buffer(-0.18, join_style="bevel")
        .buffer(0.18, join_style="bevel")
        .simplify(0.25, preserve_topology=True)
    )
    if not cleaned.is_valid:
        cleaned = cleaned.buffer(0)
    if cleaned.is_empty:
        return [], []

    fill_tris: list[np.ndarray] = []
    curb_segs: list[np.ndarray] = []

    for poly in _iter_polygons(cleaned):
        if poly.area < 4.0:
            continue
        # Drop tiny interior islands/triangles inside road junctions before triangulation & curb tracing
        kept_holes = [ring for ring in poly.interiors if Polygon(ring).area >= 8.0]
        clean_poly = Polygon(poly.exterior, kept_holes)
        if not clean_poly.is_valid:
            clean_poly = clean_poly.buffer(0)
        for sub in _iter_polygons(clean_poly):
            if sub.area < 4.0:
                continue
            # Constrained Delaunay guarantees exact 100% area coverage inside concave/holed roads
            cdt = constrained_delaunay_triangles(sub)
            tris = list(cdt.geoms) if hasattr(cdt, "geoms") else [cdt]
            for t in tris:
                if t.area > 1e-3:
                    tp = np.asarray(t.exterior.coords, dtype=np.float32)[:-1]
                    fill_tris.append(np.column_stack((tp, np.full(len(tp), 0.02, dtype=np.float32))))

            # Collect all exterior/interior curb segments uncut (frame clips cleanly)
            for ring in [sub.exterior] + list(sub.interiors):
                pts = np.asarray(ring.coords, dtype=np.float32)
                for i in range(len(pts) - 1):
                    x0, y0 = float(pts[i, 0]), float(pts[i, 1])
                    x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
                    curb_segs.append(np.array([[x0, y0, 0.04], [x1, y1, 0.04]], dtype=np.float32))

    return fill_tris, curb_segs


def _visible_prism_edges(
    poly: Polygon, z_lo: float, z_top: float, view_ax: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split visible prism edges into (outer_silhouette_segs, inner_crease_segs)."""
    p_ccw = orient(poly, sign=1.0)
    sil_segs: list[np.ndarray] = []
    inner_segs: list[np.ndarray] = []
    for ring in [p_ccw.exterior] + list(p_ccw.interiors):
        coords = np.asarray(ring.coords, dtype=np.float64)
        if len(coords) < 4:
            continue
        pts = coords[:-1]
        n_v = len(pts)
        nxt = np.roll(pts, -1, axis=0)
        dxy = nxt - pts
        lengths = np.hypot(dxy[:, 0], dxy[:, 1])
        valid = lengths > 1e-4
        # Outward normal for CCW exterior & CW hole is (dy, -dx, 0)
        normals = np.zeros((n_v, 3), dtype=np.float64)
        normals[valid, 0] = dxy[valid, 1] / lengths[valid]
        normals[valid, 1] = -dxy[valid, 0] / lengths[valid]
        dots = normals @ view_ax

        for i in range(n_v):
            if not valid[i]:
                continue
            x0, y0 = float(pts[i, 0]), float(pts[i, 1])
            x1, y1 = float(nxt[i, 0]), float(nxt[i, 1])
            # Roof edge: front roof meeting back wall -> outer silhouette; meeting front wall -> inner crease
            roof_seg = np.array([[x0, y0, z_top], [x1, y1, z_top]], dtype=np.float32)
            if dots[i] <= 1e-3:
                sil_segs.append(roof_seg)
            else:
                inner_segs.append(roof_seg)
            # Base edge (front wall meeting ground) -> outer silhouette
            if dots[i] > 1e-3:
                sil_segs.append(np.array([[x0, y0, z_lo], [x1, y1, z_lo]], dtype=np.float32))

            # Vertical corner edge between edge (i - 1) and edge i
            prev_i = (i - 1) % n_v
            d_prev = float(dots[prev_i])
            d_curr = float(dots[i])
            is_front = max(d_prev, d_curr) > 1e-3
            cos_turn = float(normals[prev_i] @ normals[i])
            is_silhouette = (d_prev > 1e-3) != (d_curr > 1e-3)
            if is_front and (cos_turn < 0.94 or is_silhouette):
                v_seg = np.array([[x0, y0, z_lo], [x0, y0, z_top]], dtype=np.float32)
                if is_silhouette:
                    sil_segs.append(v_seg)
                else:
                    inner_segs.append(v_seg)
    return sil_segs, inner_segs


OccluderFace = tuple[Polygon, tuple[float, float, float, float], float, float, float, float]


def _extract_prism_occluders(
    poly: Polygon,
    z_lo: float,
    z_top: float,
    u_ax: np.ndarray,
    v_ax: np.ndarray,
    view_ax: np.ndarray,
) -> list[OccluderFace]:
    """Extract front-facing 2D projected faces + exact depth planes d(u,v)=a*u+b*v+c."""
    p_ccw = orient(poly, sign=1.0)
    occs: list[OccluderFace] = []

    # 1. Top roof face (normal = (0, 0, +1))
    denom_top = float(view_ax[2])
    if denom_top > 1e-3:
        ext_pts = np.asarray(p_ccw.exterior.coords, dtype=np.float64)[:-1]
        if len(ext_pts) >= 3:
            ext_3d = np.column_stack([ext_pts, np.full(len(ext_pts), z_top)])
            ext_uv = np.column_stack([ext_3d @ u_ax, ext_3d @ v_ax])
            holes_uv = []
            for h_ring in p_ccw.interiors:
                hp = np.asarray(h_ring.coords, dtype=np.float64)[:-1]
                if len(hp) >= 3:
                    h_3d = np.column_stack([hp, np.full(len(hp), z_top)])
                    holes_uv.append(np.column_stack([h_3d @ u_ax, h_3d @ v_ax]))
            poly_uv = Polygon(ext_uv, holes_uv)
            if not poly_uv.is_valid:
                poly_uv = poly_uv.buffer(0)
            if not poly_uv.is_empty and poly_uv.area > 1e-4:
                a = -float(u_ax[2]) / denom_top
                b = -float(v_ax[2]) / denom_top
                c = float(z_top) / denom_top
                max_d = float((ext_3d @ view_ax).max())
                occs.append((poly_uv, poly_uv.bounds, a, b, c, max_d))

    # 2. Front-facing vertical walls
    for ring in [p_ccw.exterior] + list(p_ccw.interiors):
        coords = np.asarray(ring.coords, dtype=np.float64)
        if len(coords) < 4:
            continue
        pts = coords[:-1]
        nxt = np.roll(pts, -1, axis=0)
        dxy = nxt - pts
        lengths = np.hypot(dxy[:, 0], dxy[:, 1])
        for i in range(len(pts)):
            if lengths[i] <= 1e-4:
                continue
            nx = float(dxy[i, 1] / lengths[i])
            ny = float(-dxy[i, 0] / lengths[i])
            n_vec = np.array([nx, ny, 0.0], dtype=np.float64)
            denom = float(n_vec @ view_ax)
            if denom <= 1e-3:
                continue
            x0, y0 = float(pts[i, 0]), float(pts[i, 1])
            x1, y1 = float(nxt[i, 0]), float(nxt[i, 1])
            quad_3d = np.array(
                [
                    [x0, y0, z_lo],
                    [x1, y1, z_lo],
                    [x1, y1, z_top],
                    [x0, y0, z_top],
                ],
                dtype=np.float64,
            )
            poly_uv = Polygon(np.column_stack([quad_3d @ u_ax, quad_3d @ v_ax]))
            if not poly_uv.is_valid:
                poly_uv = poly_uv.buffer(0)
            if poly_uv.is_empty or poly_uv.area <= 1e-4:
                continue
            a = -float(n_vec @ u_ax) / denom
            b = -float(n_vec @ v_ax) / denom
            c = float(n_vec @ quad_3d[0]) / denom
            max_d = float((quad_3d @ view_ax).max())
            occs.append((poly_uv, poly_uv.bounds, a, b, c, max_d))
    return occs


def _iter_linestrings(geom):
    if geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, (MultiLineString, GeometryCollection)):
        for g in geom.geoms:
            yield from _iter_linestrings(g)


def _clip_segments_hlr(
    segs_3d: list[np.ndarray],
    occluders: list[OccluderFace],
    u_ax: np.ndarray,
    v_ax: np.ndarray,
    view_ax: np.ndarray,
    *,
    eps: float = 0.08,
) -> list[np.ndarray]:
    """Analytically clip 3D line segments against projected occluder depth planes."""
    if not segs_3d or not occluders:
        return segs_3d

    kept: list[np.ndarray] = []
    for seg in segs_3d:
        p0 = np.asarray(seg[0], dtype=np.float64)
        p1 = np.asarray(seg[1], dtype=np.float64)
        u0, v0, d0 = float(p0 @ u_ax), float(p0 @ v_ax), float(p0 @ view_ax)
        u1, v1, d1 = float(p1 @ u_ax), float(p1 @ v_ax), float(p1 @ view_ax)
        du, dv = u1 - u0, v1 - v0
        len2 = du * du + dv * dv
        if len2 <= 1e-8:
            continue

        smin_u, smax_u = (u0, u1) if u0 <= u1 else (u1, u0)
        smin_v, smax_v = (v0, v1) if v0 <= v1 else (v1, v0)
        smin_d = min(d0, d1)
        line_uv = LineString([(u0, v0), (u1, v1)])
        blocked: list[tuple[float, float]] = []

        for poly_uv, (omin_u, omin_v, omax_u, omax_v), a, b, c, max_d in occluders:
            if smin_d > max_d - eps:
                continue
            if smax_u < omin_u or smin_u > omax_u or smax_v < omin_v or smin_v > omax_v:
                continue
            inter = line_uv.intersection(poly_uv)
            if inter.is_empty:
                continue
            for ls in _iter_linestrings(inter):
                coords = ls.coords
                if len(coords) < 2:
                    continue
                qa = coords[0]
                qb = coords[-1]
                ta = max(0.0, min(1.0, ((qa[0] - u0) * du + (qa[1] - v0) * dv) / len2))
                tb = max(0.0, min(1.0, ((qb[0] - u0) * du + (qb[1] - v0) * dv) / len2))
                if ta > tb:
                    ta, tb = tb, ta
                    qa, qb = qb, qa
                if tb - ta <= 1e-5:
                    continue
                del_a = (a * qa[0] + b * qa[1] + c) - (d0 + ta * (d1 - d0))
                del_b = (a * qb[0] + b * qb[1] + c) - (d0 + tb * (d1 - d0))
                if del_a <= eps and del_b <= eps:
                    continue
                if del_a > eps and del_b > eps:
                    blocked.append((ta, tb))
                else:
                    t_cross = ta + (eps - del_a) * (tb - ta) / (del_b - del_a)
                    t_cross = max(ta, min(tb, t_cross))
                    if del_a > eps:
                        if t_cross - ta > 1e-5:
                            blocked.append((ta, t_cross))
                    else:
                        if tb - t_cross > 1e-5:
                            blocked.append((t_cross, tb))

        if not blocked:
            kept.append(seg)
            continue

        blocked.sort()
        merged_b: list[list[float]] = []
        for t_lo, t_hi in blocked:
            if merged_b and t_lo <= merged_b[-1][1] + 1e-5:
                if t_hi > merged_b[-1][1]:
                    merged_b[-1][1] = t_hi
            else:
                merged_b.append([t_lo, t_hi])

        cur = 0.0
        dp = p1 - p0
        seg_uv_len = math.sqrt(len2)
        for t_lo, t_hi in merged_b:
            if t_lo - cur > 1e-4 and (t_lo - cur) * seg_uv_len > 0.02:
                kept.append(np.array([p0 + cur * dp, p0 + t_lo * dp], dtype=np.float32))
            if t_hi > cur:
                cur = t_hi
        if 1.0 - cur > 1e-4 and (1.0 - cur) * seg_uv_len > 0.02:
            kept.append(np.array([p0 + cur * dp, p1], dtype=np.float32))

    return kept


def _extrude_prism_minimal_shade(
    poly: Polygon, z_lo: float, z_hi: float, u_ax: np.ndarray
) -> tuple[list[np.ndarray], list[float]]:
    """Extrude prism with pure #FFFFFF roofs/left walls and minimal very light grey right face."""
    p_ccw = orient(poly, sign=1.0)
    faces_out: list[np.ndarray] = []
    shades_out: list[float] = []

    for ring in [p_ccw.exterior] + list(p_ccw.interiors):
        pts = np.asarray(ring.coords, dtype=np.float32)[:-1]
        n = len(pts)
        if n < 3:
            continue
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            seg_len = math.hypot(dx, dy)
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
            nx = dy / seg_len
            ny = -dx / seg_len
            side = float(nx * u_ax[0] + ny * u_ax[1])
            shade = 1.0 - 0.042 * max(0.0, side)
            faces_out.append(wall)
            shades_out.append(shade)

    if not p_ccw.interiors:
        pts = np.asarray(p_ccw.exterior.coords, dtype=np.float32)[:-1]
        if len(pts) >= 3:
            faces_out.append(np.column_stack((pts, np.full(len(pts), z_hi, dtype=np.float32))))
            shades_out.append(1.0)
    else:
        cdt = constrained_delaunay_triangles(p_ccw)
        tris = list(cdt.geoms) if hasattr(cdt, "geoms") else [cdt]
        for t in tris:
            if t.area > 1e-3:
                tp = np.asarray(t.exterior.coords, dtype=np.float32)[:-1]
                faces_out.append(np.column_stack((tp, np.full(len(tp), z_hi, dtype=np.float32))))
                shades_out.append(1.0)

    return faces_out, shades_out


def _extract_voxel_crease_segs(
    mask: np.ndarray, dz: float, view_ax: np.ndarray
) -> list[np.ndarray]:
    """Extract merged architectural crease segments, dropping all coplanar voxel-grid edges."""
    if not mask.any():
        return []
    vq, fq, _ = _surface_quads(mask, dz=dz)
    face_normals = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )

    boundary_parts: list[np.ndarray] = []
    for d in range(6):
        if float(face_normals[d] @ view_ax) <= 1e-3:
            continue
        qd = vq[fq == d]
        if not len(qd):
            continue
        edges_d = np.concatenate(
            [qd[:, [0, 1], :], qd[:, [1, 2], :], qd[:, [2, 3], :], qd[:, [3, 0], :]],
            axis=0,
        )
        p0 = np.round(edges_d[:, 0, :], 2)
        p1 = np.round(edges_d[:, 1, :], 2)
        swap = (
            (p0[:, 0] > p1[:, 0])
            | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] > p1[:, 1]))
            | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] == p1[:, 1]) & (p0[:, 2] > p1[:, 2]))
        )
        a = np.where(swap[:, None], p1, p0)
        b = np.where(swap[:, None], p0, p1)
        keys = np.concatenate([a, b], axis=1)
        _uniq, uniq_idx, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
        # Shared edges inside the same face direction (counts == 2) are strictly coplanar -> omit
        boundary_parts.append(edges_d[uniq_idx[counts == 1]])

    if not boundary_parts:
        return []
    all_edges = np.concatenate(boundary_parts, axis=0)
    p0 = np.round(all_edges[:, 0, :], 2)
    p1 = np.round(all_edges[:, 1, :], 2)
    swap = (
        (p0[:, 0] > p1[:, 0])
        | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] > p1[:, 1]))
        | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] == p1[:, 1]) & (p0[:, 2] > p1[:, 2]))
    )
    a = np.where(swap[:, None], p1, p0)
    b = np.where(swap[:, None], p0, p1)
    keys = np.concatenate([a, b], axis=1)
    _uniq, uniq_idx = np.unique(keys, axis=0, return_index=True)
    dedup = all_edges[uniq_idx]

    # Merge collinear contiguous axis-aligned segments so lines draw cleanly without breaks
    groups: dict[tuple[int, float, float], list[tuple[float, float, np.ndarray]]] = {}
    for seg in dedup:
        s0 = seg[0].astype(np.float64)
        s1 = seg[1].astype(np.float64)
        diff = np.abs(s1 - s0)
        axis = int(np.argmax(diff))
        if diff[axis] < 1e-4:
            continue
        other = [idx for idx in range(3) if idx != axis]
        gkey = (axis, round(float(s0[other[0]]), 2), round(float(s0[other[1]]), 2))
        t_lo = min(float(s0[axis]), float(s1[axis]))
        t_hi = max(float(s0[axis]), float(s1[axis]))
        groups.setdefault(gkey, []).append((t_lo, t_hi, s0))

    merged: list[np.ndarray] = []
    for (axis, _o0, _o1), intervals in groups.items():
        intervals.sort(key=lambda x: x[0])
        cur_lo, cur_hi, ref_pt = intervals[0]
        for t_lo, t_hi, _pt in intervals[1:]:
            if t_lo <= cur_hi + 1e-3:
                if t_hi > cur_hi:
                    cur_hi = t_hi
            else:
                q0 = ref_pt.copy()
                q1 = ref_pt.copy()
                q0[axis] = cur_lo
                q1[axis] = cur_hi
                merged.append(np.array([q0, q1], dtype=np.float32))
                cur_lo, cur_hi = t_lo, t_hi
        q0 = ref_pt.copy()
        q1 = ref_pt.copy()
        q0[axis] = cur_lo
        q1[axis] = cur_hi
        merged.append(np.array([q0, q1], dtype=np.float32))
    return merged


def _connected_components_3d(occ: np.ndarray) -> list[np.ndarray]:
    """Split a 3D occupancy mask into 6-connected building volume masks."""
    if not occ.any():
        return []
    nz, ny, nx = occ.shape
    visited = np.zeros_like(occ, dtype=bool)
    zs, ys, xs = np.nonzero(occ)
    comps: list[np.ndarray] = []

    for sz, sy, sx in zip(zs, ys, xs, strict=False):
        iz, iy, ix = int(sz), int(sy), int(sx)
        if visited[iz, iy, ix]:
            continue
        comp = np.zeros_like(occ, dtype=bool)
        q: deque[tuple[int, int, int]] = deque([(iz, iy, ix)])
        visited[iz, iy, ix] = True
        comp[iz, iy, ix] = True
        while q:
            cz, cy, cx = q.popleft()
            for dz_s, dy_s, dx_s in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                qz, qy, qx = cz + dz_s, cy + dy_s, cx + dx_s
                if 0 <= qz < nz and 0 <= qy < ny and 0 <= qx < nx and occ[qz, qy, qx] and not visited[qz, qy, qx]:
                    visited[qz, qy, qx] = True
                    comp[qz, qy, qx] = True
                    q.append((qz, qy, qx))
        comps.append(comp)

    comps.sort(key=lambda c: int(c.sum()), reverse=True)
    return comps


def _extract_hero_silhouette_segs(
    occ: np.ndarray,
    dz: float,
    u_ax: np.ndarray,
    v_ax: np.ndarray,
    view_ax: np.ndarray,
) -> tuple[list[np.ndarray], Polygon, float]:
    """Compute separate non-merged 3D outer silhouettes per connected hero volume."""
    comps = _connected_components_3d(occ)
    if not comps:
        return [], Polygon(), 0.0

    face_normals = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    sil_segs: list[np.ndarray] = []
    comp_sil_geoms: list[Polygon] = []
    max_d_all = -1e9

    for comp in comps:
        vq, fq, _ = _surface_quads(comp, dz=dz)
        keep = ((face_normals @ view_ax) > 1e-3)[fq]
        front_q = vq[keep].astype(np.float64)
        if not len(front_q):
            continue

        max_d_all = max(max_d_all, float((front_q.reshape(-1, 3) @ view_ax).max()))
        u_q = front_q @ u_ax
        v_q = front_q @ v_ax
        quad_polys = []
        for k in range(len(front_q)):
            qp = Polygon(np.column_stack([u_q[k], v_q[k]]))
            if not qp.is_valid:
                qp = qp.buffer(0)
            if not qp.is_empty and qp.area > 1e-5:
                quad_polys.append(qp)
        if not quad_polys:
            continue

        comp_sil = unary_union(quad_polys).buffer(1e-3, join_style="mitre").buffer(-1e-3, join_style="mitre")
        if not comp_sil.is_valid:
            comp_sil = comp_sil.buffer(0)
        comp_sil = comp_sil.simplify(1e-3, preserve_topology=True)
        comp_sil_geoms.append(comp_sil)

        bnd_band = comp_sil.boundary.buffer(5e-3, cap_style="flat", join_style="mitre")
        comp_creases = _extract_voxel_crease_segs(comp, dz, view_ax)
        comp_outer_3d: list[np.ndarray] = []

        for seg in comp_creases:
            p0 = seg[0].astype(np.float64)
            p1 = seg[1].astype(np.float64)
            u0, v0 = float(p0 @ u_ax), float(p0 @ v_ax)
            u1, v1 = float(p1 @ u_ax), float(p1 @ v_ax)
            du, dv = u1 - u0, v1 - v0
            len2 = du * du + dv * dv
            if len2 <= 1e-8:
                continue
            inter = LineString([(u0, v0), (u1, v1)]).intersection(bnd_band)
            if inter.is_empty:
                continue
            dp = p1 - p0
            for ls in _iter_linestrings(inter):
                if ls.length <= 0.05:
                    continue
                qa = ls.coords[0]
                qb = ls.coords[-1]
                ta = max(0.0, min(1.0, ((qa[0] - u0) * du + (qa[1] - v0) * dv) / len2))
                tb = max(0.0, min(1.0, ((qb[0] - u0) * du + (qb[1] - v0) * dv) / len2))
                if ta > tb:
                    ta, tb = tb, ta
                if tb - ta > 1e-4:
                    comp_outer_3d.append(np.array([p0 + ta * dp, p0 + tb * dp], dtype=np.float32))

        comp_outer_3d = _clip_segments_against_voxels(
            comp_outer_3d, comp, dz, u_ax, v_ax, view_ax, eps=0.08
        )
        if len(comps) > 1 and comp_outer_3d:
            other = occ & ~comp
            comp_outer_3d = _clip_segments_against_voxels(
                comp_outer_3d, other, dz, u_ax, v_ax, view_ax, eps=0.08
            )
        sil_segs.extend(comp_outer_3d)

    if not comp_sil_geoms:
        return [], Polygon(), 0.0
    sil_geom_all = unary_union(comp_sil_geoms)
    if not sil_geom_all.is_valid:
        sil_geom_all = sil_geom_all.buffer(0)
    return sil_segs, sil_geom_all, max_d_all + 0.15


def _chain_segments_to_polylines(
    segs_3d: list[np.ndarray], tol: float = 0.03
) -> list[np.ndarray]:
    """Chain contiguous 3D segments into continuous polycurves for seamless corners."""
    if not segs_3d:
        return []

    def _pt_key(pt: np.ndarray) -> tuple[int, int, int]:
        q = np.round(pt / tol).astype(np.int64)
        return (int(q[0]), int(q[1]), int(q[2]))

    adj: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    keys: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for idx, seg in enumerate(segs_3d):
        k0 = _pt_key(seg[0])
        k1 = _pt_key(seg[1])
        keys.append((k0, k1))
        adj.setdefault(k0, []).append((idx, 0))
        adj.setdefault(k1, []).append((idx, 1))

    used = [False] * len(segs_3d)
    polylines: list[np.ndarray] = []

    def _walk(start_seg: int, start_end: int) -> None:
        used[start_seg] = True
        pts = [segs_3d[start_seg][start_end], segs_3d[start_seg][1 - start_end]]
        cur_key = keys[start_seg][1 - start_end]
        start_key = keys[start_seg][start_end]
        while True:
            nxt_edge: tuple[int, int] | None = None
            for cand_seg, cand_end in adj.get(cur_key, ()):
                if not used[cand_seg]:
                    nxt_edge = (cand_seg, cand_end)
                    break
            if nxt_edge is None:
                break
            c_seg, c_end = nxt_edge
            used[c_seg] = True
            nxt_pt = segs_3d[c_seg][1 - c_end]
            nxt_key = keys[c_seg][1 - c_end]
            if nxt_key == start_key:
                pts.append(pts[0])
                break
            pts.append(nxt_pt)
            cur_key = nxt_key
        polylines.append(np.asarray(pts, dtype=np.float32))

    # First trace open chains starting at degree != 2 vertices
    for _k, inc in adj.items():
        if len(inc) != 2:
            for s_idx, end_idx in inc:
                if not used[s_idx]:
                    _walk(s_idx, end_idx)

    # Then trace remaining closed loops
    for s_idx in range(len(segs_3d)):
        if not used[s_idx]:
            _walk(s_idx, 0)

    return polylines


def _clip_segments_against_voxels(
    segs_3d: list[np.ndarray],
    occ: np.ndarray,
    dz: float,
    u_ax: np.ndarray,
    v_ax: np.ndarray,
    view_ax: np.ndarray,
    *,
    eps: float = 0.65,
) -> list[np.ndarray]:
    """Clip 3D segments against exact 3D depth planes of front-facing hero voxel quads."""
    if not segs_3d or not occ.any():
        return segs_3d

    vq, fq, _ = _surface_quads(occ, dz=dz)
    face_normals = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    denoms_all = face_normals @ view_ax
    front_mask = denoms_all > 1e-3
    keep = front_mask[fq]
    front_q = vq[keep].astype(np.float64)
    if not len(front_q):
        return segs_3d

    fq_keep = fq[keep]
    n_keep = face_normals[fq_keep]
    denom_keep = denoms_all[fq_keep]

    u_q = front_q @ u_ax
    v_q = front_q @ v_ax
    d_q = front_q @ view_ax
    q_umin = u_q.min(axis=1)
    q_umax = u_q.max(axis=1)
    q_vmin = v_q.min(axis=1)
    q_vmax = v_q.max(axis=1)
    q_max_d = d_q.max(axis=1)

    a_all = -(n_keep @ u_ax) / denom_keep
    b_all = -(n_keep @ v_ax) / denom_keep
    c_all = (n_keep * front_q[:, 0, :]).sum(axis=1) / denom_keep

    kept: list[np.ndarray] = []
    for seg in segs_3d:
        p0 = np.asarray(seg[0], dtype=np.float64)
        p1 = np.asarray(seg[1], dtype=np.float64)
        u0, v0, d0 = float(p0 @ u_ax), float(p0 @ v_ax), float(p0 @ view_ax)
        u1, v1, d1 = float(p1 @ u_ax), float(p1 @ v_ax), float(p1 @ view_ax)
        smin_u, smax_u = (u0, u1) if u0 <= u1 else (u1, u0)
        smin_v, smax_v = (v0, v1) if v0 <= v1 else (v1, v0)
        smin_d = min(d0, d1)

        mask = (
            (smin_d <= q_max_d - eps)
            & (smax_u >= q_umin)
            & (smin_u <= q_umax)
            & (smax_v >= q_vmin)
            & (smin_v <= q_vmax)
        )
        idxs = np.flatnonzero(mask)
        if not len(idxs):
            kept.append(seg)
            continue

        cand_occs: list[OccluderFace] = []
        for idx in idxs:
            qp = Polygon(np.column_stack([u_q[idx], v_q[idx]]))
            if not qp.is_valid:
                qp = qp.buffer(0)
            if not qp.is_empty and qp.area > 1e-5:
                cand_occs.append(
                    (
                        qp,
                        (float(q_umin[idx]), float(q_vmin[idx]), float(q_umax[idx]), float(q_vmax[idx])),
                        float(a_all[idx]),
                        float(b_all[idx]),
                        float(c_all[idx]),
                        float(q_max_d[idx]),
                    )
                )
        kept.extend(_clip_segments_hlr([seg], cand_occs, u_ax, v_ax, view_ax, eps=eps))
    return kept


def _classify_context_buildings(
    context: dict,
    occ: np.ndarray,
    cx: float,
    cy: float,
    half: float,
    elev_deg: float,
    azim_deg: float,
    *,
    dz: float = 3.0,
) -> tuple[list[dict], list[dict], list[Polygon]]:
    """Split surrounding buildings into (visible_bg_side, occluding_fg, building_footprints)."""
    buildings_raw = context.get("buildings") or ()
    footprints_used: list[Polygon] = []
    if not buildings_raw:
        return [], [], footprints_used

    zz, yy, xx = np.nonzero(occ)
    hero_pts = np.stack(
        [xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5, (zz.astype(np.float64) + 0.5) * dz],
        axis=1,
    )
    hero_top_z = float((zz.max() + 1) * dz)
    u_ax, v_ax, view_ax = _camera_basis(elev_deg, azim_deg)
    hu, hv, hd = _project_pts(hero_pts, u_ax, v_ax, view_ax)
    hero_uv = np.column_stack([hu, hv])
    hero_hull = MultiPoint(hero_uv).convex_hull.buffer(0.8)

    ctx_radius = half * 1.45
    z_cap = max(hero_top_z * 2.15, 34.0)
    clip_disk = Point(cx, cy).buffer(ctx_radius, quad_segs=20)

    visible_side: list[dict] = []
    occluding_fg: list[dict] = []

    for b in buildings_raw:
        footprint, z_lo, z_hi = _extract_building_footprint(b)
        if footprint is None or not footprint.intersects(clip_disk):
            continue
        if not clip_disk.contains(footprint.centroid):
            continue
        z_top_b = min(z_hi, z_cap)
        if z_top_b <= z_lo + 0.3:
            continue

        b_faces: list[np.ndarray] = []
        b_shades: list[float] = []
        b_sil_segs: list[np.ndarray] = []
        b_inner_segs: list[np.ndarray] = []
        b_occluders: list[OccluderFace] = []
        pts_3d: list[np.ndarray] = []

        for poly in _iter_polygons(footprint):
            footprints_used.append(poly)
            fl, sl = _extrude_prism_minimal_shade(poly, z_lo, z_top_b, u_ax)
            if not fl:
                continue
            b_faces.extend(fl)
            b_shades.extend(sl)
            psil, pinner = _visible_prism_edges(poly, z_lo, z_top_b, view_ax)
            b_sil_segs.extend(psil)
            b_inner_segs.extend(pinner)
            b_occluders.extend(_extract_prism_occluders(poly, z_lo, z_top_b, u_ax, v_ax, view_ax))
            ext_pts = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
            if len(ext_pts):
                pts_3d.append(np.column_stack([ext_pts, np.full(len(ext_pts), z_lo)]))
                pts_3d.append(np.column_stack([ext_pts, np.full(len(ext_pts), z_top_b)]))

        if not b_faces or not pts_3d:
            continue
        b_arr = np.concatenate(pts_3d, axis=0)
        bu, bv, bd = _project_pts(b_arr, u_ax, v_ax, view_ax)
        b_hull = MultiPoint(np.column_stack([bu, bv])).convex_hull
        overlap = b_hull.intersection(hero_hull)

        is_occluding = False
        if not overlap.is_empty and overlap.area >= 0.5:
            test_uvs = [np.array([overlap.centroid.x, overlap.centroid.y])]
            for px, py, _pd in zip(bu, bv, bd, strict=False):
                if overlap.buffer(1.5).contains(Point(px, py)):
                    test_uvs.append(np.array([px, py]))
            d_hero_hits = []
            for tuv in test_uvs:
                dists_uv = np.hypot(hero_uv[:, 0] - tuv[0], hero_uv[:, 1] - tuv[1])
                near = dists_uv <= 4.5
                if near.any():
                    d_hero_hits.append(float(hd[near].max()))
            if d_hero_hits:
                d_hero_ref = float(np.percentile(d_hero_hits, 25))
                in_ov_b = [
                    float(pd)
                    for px, py, pd in zip(bu, bv, bd, strict=False)
                    if overlap.buffer(2.5).contains(Point(px, py))
                ]
                b_depth_front = max(in_ov_b) if in_ov_b else float(bd.max())
                if b_depth_front > d_hero_ref - 0.5:
                    is_occluding = True

        item = {
            "faces": b_faces,
            "shades": b_shades,
            "sil_segs": b_sil_segs,
            "inner_segs": b_inner_segs,
            "occluders": b_occluders,
            "overlap_area": float(overlap.area) if not overlap.is_empty else 0.0,
        }
        if is_occluding:
            occluding_fg.append(item)
        else:
            visible_side.append(item)

    return visible_side, occluding_fg, footprints_used


def pick_best_azimuth(
    context: dict, occ: np.ndarray, cx: float, cy: float, half: float, elev_deg: float = DEFAULT_ELEV
) -> float:
    """Pick the isometric azimuth minimizing foreground building occlusion over hero massing."""
    best_azim = AZIM_CANDIDATES[0]
    best_score = 1e18
    for idx, az in enumerate(AZIM_CANDIDATES):
        _vis, occ_fg, _fps = _classify_context_buildings(context, occ, cx, cy, half, elev_deg, az)
        ov_area = sum(item["overlap_area"] for item in occ_fg)
        score = len(occ_fg) * 200.0 + ov_area + idx * 0.5
        if score < best_score:
            best_score = score
            best_azim = az
    return best_azim


def render_voxel_scene_on_ax(
    ax,
    occ: np.ndarray,
    core_mask: np.ndarray,
    context: dict,
    *,
    elev_deg: float = DEFAULT_ELEV,
    azim_deg: float | None = None,
    dz: float = 3.0,
    carved_delta: np.ndarray | None = None,
    hero_base_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hero_alpha: float = 0.96,
    silhouette_lw: float = 1.25,
    inner_lw: float = 0.24,
    half_override: float | None = None,
    center_xy: tuple[float, float] | None = None,
    render_mode: str = "combined",  # "combined" | "fills" | "outlines_knockout" | "outlines_vector"
) -> float:
    """Render voxel massing + yellow cores + depth-classified white surroundings.

    Supports separating pure shaded fills (``fills``) from hidden-line-removed
    outlines (``outlines_knockout``) and editable vector strokes (``outlines_vector``)
    so layers overlay 1:1 in Affinity.
    """
    zs, ys, xs = np.nonzero(occ)
    if center_xy is None:
        cx = 0.5 * (float(xs.min()) + float(xs.max() + 1))
        cy = 0.5 * (float(ys.min()) + float(ys.max() + 1))
    else:
        cx, cy = center_xy

    if half_override is None:
        span_x = float(xs.max() - xs.min() + 1)
        span_y = float(ys.max() - ys.min() + 1)
        span_z = float(zs.max() + 1) * dz
        half = max(span_x, span_y, span_z) * 0.62
    else:
        half = half_override

    if azim_deg is None:
        azim_deg = pick_best_azimuth(context, occ, cx, cy, half, elev_deg)

    u_ax, v_ax, view_ax = _camera_basis(elev_deg, azim_deg)
    vis_bg, occ_fg, _bld_footprints = _classify_context_buildings(
        context, occ, cx, cy, half, elev_deg, azim_deg, dz=dz
    )

    sil_segs, sil_geom, d_sil = _extract_hero_silhouette_segs(occ, dz, u_ax, v_ax, view_ax)
    bg_occluders = [occ_face for item in vis_bg for occ_face in item["occluders"]]
    fg_occluders = [occ_face for item in occ_fg for occ_face in item["occluders"]]
    hero_occ_list: list[OccluderFace] = []
    if not sil_geom.is_empty:
        eroded = sil_geom.buffer(-0.08)
        for p in _iter_polygons(eroded):
            if p.area > 0.1:
                hero_occ_list.append((p, p.bounds, 0.0, 0.0, d_sil, d_sil))

    perim = context.get("perimeter") or []
    perim_raw_segs: list[np.ndarray] = []
    if len(perim) >= 3:
        p_poly = Polygon(perim).buffer(0)
        if not p_poly.is_empty:
            clip_sq = box(cx - half * 2.5, cy - half * 2.5, cx + half * 2.5, cy + half * 2.5)
            for p in _iter_polygons(p_poly.intersection(clip_sq)):
                pts = np.asarray(p.exterior.coords, dtype=np.float32)
                for i in range(len(pts) - 1):
                    perim_raw_segs.append(
                        np.array(
                            [[pts[i, 0], pts[i, 1], 0.10], [pts[i + 1, 0], pts[i + 1, 1], 0.10]],
                            dtype=np.float32,
                        )
                    )

    _road_fills, road_curbs = _clean_road_geometry(context, cx, cy, half)

    is_knockout = render_mode == "outlines_knockout"
    is_vector = render_mode == "outlines_vector"
    draw_fills = render_mode in ("combined", "fills", "outlines_knockout")
    draw_lines = render_mode in ("combined", "outlines_knockout", "outlines_vector")

    if is_vector:
        ax.set_facecolor("none")
    else:
        ax.set_facecolor("#FFFFFF")

    ax.computed_zorder = False
    ax.view_init(elev=elev_deg, azim=azim_deg)
    ax.set_proj_type("ortho")
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(0.0, 2.0 * half * 0.66)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.66))
    except Exception:
        pass
    ax.set_axis_off()

    # 1. Pure white uncut ground roads defined by crisp non-triangulated curbs matching context inner weight
    if draw_lines and road_curbs:
        vis_curbs = _clip_segments_hlr(
            road_curbs, bg_occluders + hero_occ_list, u_ax, v_ax, view_ax
        )
        if vis_curbs:
            ax.add_collection3d(
                Line3DCollection(
                    _chain_segments_to_polylines(vis_curbs),
                    colors="#8C95A2",
                    linewidths=inner_lw,
                    alpha=0.72,
                    capstyle="round",
                    joinstyle="round",
                    zorder=1.2,
                )
            )

    # 1b. Continuous parcel boundary matching surrounding opaque buildings' silhouette weight & color
    if draw_lines and perim_raw_segs:
        vis_perim = _clip_segments_hlr(perim_raw_segs, bg_occluders, u_ax, v_ax, view_ax, eps=0.08)
        vis_perim = _clip_segments_against_voxels(
            vis_perim, occ, dz, u_ax, v_ax, view_ax, eps=0.65
        )
        if vis_perim:
            ax.add_collection3d(
                Line3DCollection(
                    _chain_segments_to_polylines(vis_perim),
                    colors=(0.42, 0.46, 0.52, 0.72),
                    linewidths=0.42,
                    linestyles="-",
                    capstyle="round",
                    joinstyle="round",
                    zorder=2.2,
                )
            )

    # 2. Pure white FULLY OPAQUE buildings behind & left/right of hero (minimal 1-face shading)
    if vis_bg:
        bg_faces = [f for item in vis_bg for f in item["faces"]]
        if draw_fills:
            if is_knockout:
                bg_fc = (1.0, 1.0, 1.0, 1.0)
            else:
                sh = np.asarray([s for item in vis_bg for s in item["shades"]], dtype=np.float32)
                bg_fc = np.column_stack((sh, sh, sh, np.ones(len(bg_faces), dtype=np.float32)))
            ax.add_collection3d(
                Poly3DCollection(bg_faces, facecolors=bg_fc, edgecolors="none", zsort="average", zorder=2.0)
            )
        if draw_lines:
            bg_occ = bg_occluders + hero_occ_list
            bg_inner = _clip_segments_hlr(
                [seg for item in vis_bg for seg in item["inner_segs"]],
                bg_occ,
                u_ax,
                v_ax,
                view_ax,
            )
            if bg_inner:
                ax.add_collection3d(
                    Line3DCollection(
                        np.asarray(bg_inner, dtype=np.float32),
                        colors=(0.42, 0.46, 0.52, 0.65),
                        linewidths=inner_lw,
                        capstyle="round",
                        joinstyle="round",
                        zorder=2.1,
                    )
                )
            bg_sil = _clip_segments_hlr(
                [seg for item in vis_bg for seg in item["sil_segs"]],
                bg_occ,
                u_ax,
                v_ax,
                view_ax,
            )
            if bg_sil:
                ax.add_collection3d(
                    Line3DCollection(
                        _chain_segments_to_polylines(bg_sil),
                        colors=(0.42, 0.46, 0.52, 0.72),
                        linewidths=0.42,
                        capstyle="round",
                        joinstyle="round",
                        zorder=2.15,
                    )
                )

    # 3. Hero voxel massing + vivid yellow cores (keep side shading; z=+1 top faces pure opaque white)
    cm = core_mask & occ
    shell = occ & ~cm

    verts_all, rgba_all = [], []
    if draw_fills:
        if not is_knockout and carved_delta is not None and carved_delta.any():
            vq_d, fq_d, _ = _surface_quads(carved_delta, dz=dz)
            gains_d = np.array([0.92, 0.84, 0.88, 0.95, 1.0, 0.75], dtype=np.float32)[fq_d][:, None]
            rgb_d = np.tile(np.array([[0.97, 0.88, 0.24]], np.float32), (len(vq_d), 1)) * gains_d
            rgba_d = np.concatenate([rgb_d, np.full((len(vq_d), 1), 0.28, np.float32)], axis=1)
            verts_all.append(vq_d)
            rgba_all.append(rgba_d)

        if shell.any():
            vq, fq, _ = _surface_quads(shell, dz=dz)
            if is_knockout:
                rgba = np.ones((len(vq), 4), dtype=np.float32)
            else:
                if hero_base_rgb == (1.0, 1.0, 1.0):
                    # Top roof stays 100% pure white (#FFFFFF); vertical facades: 96% white (left) and 92% white (right)
                    fn = np.array(
                        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                        dtype=np.float64,
                    )
                    gains_6 = np.ones(6, dtype=np.float32)
                    for d_i in range(4):
                        if float(fn[d_i] @ view_ax) > 1e-3:
                            gains_6[d_i] = 0.92 if float(fn[d_i] @ u_ax) > 0.1 else 0.96
                else:
                    gains_6 = np.array([0.94, 0.88, 0.91, 0.97, 1.0, 0.84], dtype=np.float32)
                gains = gains_6[fq][:, None]
                base = np.array([hero_base_rgb], np.float32)
                rgb = np.clip(base * gains, 0, 1)
                alpha_arr = np.full((len(vq), 1), hero_alpha, np.float32)
                if hero_base_rgb == (1.0, 1.0, 1.0):
                    top_mask = fq == 4
                    rgb[top_mask] = 1.0
                    alpha_arr[top_mask] = 1.0
                rgba = np.concatenate([rgb, alpha_arr], axis=1)
            verts_all.append(vq)
            rgba_all.append(rgba)

        if cm.any():
            cq, cf, _ = _surface_quads(cm, dz=dz)
            if is_knockout:
                rgba_c = np.ones((len(cq), 4), dtype=np.float32)
            else:
                gains_c = np.array([0.92, 0.84, 0.88, 0.96, 1.0, 0.72], dtype=np.float32)[cf][:, None]
                # Vivid architectural yellow cores (#FADE24)
                base_c = np.array([[0.98, 0.87, 0.14]], np.float32)
                rgba_c = np.concatenate(
                    [np.clip(base_c * gains_c, 0, 1), np.full((len(cq), 1), 0.98, np.float32)], axis=1
                )
            verts_all.append(cq)
            rgba_all.append(rgba_c)

        if verts_all:
            poly_col = Poly3DCollection(
                np.concatenate(verts_all, 0),
                facecolors=np.concatenate(rgba_all, 0),
                edgecolors="none",
                linewidths=0.0,
                antialiaseds=False,
                zsort="average",
                zorder=3.0,
            )
            ax.add_collection3d(poly_col)

    # 3b. Thicker coplanar-free architectural creases + separate continuous outer silhouettes
    if draw_lines:
        hero_creases = _clip_segments_against_voxels(
            _extract_voxel_crease_segs(occ, dz, view_ax), occ, dz, u_ax, v_ax, view_ax, eps=0.08
        )
        if hero_creases:
            ax.add_collection3d(
                Line3DCollection(
                    _chain_segments_to_polylines(hero_creases),
                    colors=(0.18, 0.21, 0.26, 0.84),
                    linewidths=inner_lw,
                    capstyle="round",
                    joinstyle="round",
                    zorder=3.3,
                )
            )

        core_creases = _clip_segments_against_voxels(
            _extract_voxel_crease_segs(cm, dz, view_ax), occ, dz, u_ax, v_ax, view_ax, eps=0.08
        )
        if core_creases:
            ax.add_collection3d(
                Line3DCollection(
                    _chain_segments_to_polylines(core_creases),
                    colors=(0.45, 0.37, 0.04, 0.75),
                    linewidths=inner_lw * 0.85,
                    capstyle="round",
                    joinstyle="round",
                    zorder=3.4,
                )
            )

        if sil_segs:
            ax.add_collection3d(
                Line3DCollection(
                    _chain_segments_to_polylines(sil_segs),
                    colors="#111827",
                    linewidths=silhouette_lw,
                    alpha=0.98,
                    capstyle="round",
                    joinstyle="round",
                    zorder=3.6,
                )
            )

    # 4. Pure white SEMI-TRANSPARENT foreground buildings (minimal 1-face shading + visible-only edges)
    if occ_fg:
        fg_faces = [f for item in occ_fg for f in item["faces"]]
        if draw_fills and not is_knockout:
            sh_fg = np.asarray([s for item in occ_fg for s in item["shades"]], dtype=np.float32)
            fg_rgba = np.column_stack(
                (sh_fg, sh_fg, sh_fg, np.full(len(fg_faces), 0.36, dtype=np.float32))
            )
            ax.add_collection3d(
                Poly3DCollection(fg_faces, facecolors=fg_rgba, edgecolors="none", zsort="average", zorder=4.0)
            )
        if draw_lines:
            fg_occ = fg_occluders + bg_occluders
            fg_inner = _clip_segments_hlr(
                [seg for item in occ_fg for seg in item["inner_segs"]],
                fg_occ,
                u_ax,
                v_ax,
                view_ax,
            )
            if fg_inner:
                ax.add_collection3d(
                    Line3DCollection(
                        np.asarray(fg_inner, dtype=np.float32),
                        colors=(0.42, 0.47, 0.54, 0.42),
                        linewidths=inner_lw,
                        capstyle="round",
                        joinstyle="round",
                        zorder=4.1,
                    )
                )
            fg_sil = _clip_segments_hlr(
                [seg for item in occ_fg for seg in item["sil_segs"]],
                fg_occ,
                u_ax,
                v_ax,
                view_ax,
            )
            if fg_sil:
                ax.add_collection3d(
                    Line3DCollection(
                        _chain_segments_to_polylines(fg_sil),
                        colors=(0.42, 0.47, 0.54, 0.48),
                        linewidths=0.36,
                        capstyle="round",
                        joinstyle="round",
                        zorder=4.15,
                    )
                )

    return azim_deg


def _knockout_white_to_transparent_rgba(png_path: Path) -> None:
    """Convert a white-ground RGB PNG into crisp transparent-background ink RGBA."""
    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32)
    # Ink darkness measures coverage against pure white [255, 255, 255]
    dark = (255.0 - rgb).max(axis=2)
    alpha = np.clip(dark * (255.0 / 217.0), 0.0, 255.0)
    # Recover pure un-premultiplied stroke color where alpha > 0
    a_frac = np.maximum(alpha / 255.0, 1e-4)[..., None]
    orig_rgb = np.clip(255.0 - (255.0 - rgb) / a_frac, 0.0, 255.0)
    rgba = np.dstack([orig_rgb, alpha]).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(png_path)


def _write_layered_affinity_svg(
    renders_png: Path,
    outlines_png: Path,
    outlines_svg: Path,
    out_affinity_svg: Path,
) -> None:
    """Assemble an Affinity-ready multi-layer SVG (Renders + HLR Outlines + Editable Vector)."""
    w_px, h_px = Image.open(renders_png).size
    b64_renders = base64.b64encode(renders_png.read_bytes()).decode("ascii")
    b64_outlines = base64.b64encode(outlines_png.read_bytes()).decode("ascii")

    vec_tree = ET.parse(outlines_svg)
    vec_root = vec_tree.getroot()
    vb = vec_root.attrib.get("viewBox", f"0 0 {w_px} {h_px}").split()
    vb_w = float(vb[2]) if len(vb) == 4 else float(w_px)
    vb_h = float(vb[3]) if len(vb) == 4 else float(h_px)
    sx = float(w_px) / max(1e-6, vb_w)
    sy = float(h_px) / max(1e-6, vb_h)

    # Extract defs + figure elements from Matplotlib SVG, stripping namespace prefixes
    defs_xml = ""
    vec_body_parts: list[str] = []
    for child in vec_root:
        tag = child.tag.split("}")[-1]
        if tag == "defs":
            defs_xml = ET.tostring(child, encoding="unicode").replace("ns0:", "").replace(":ns0", "")
        elif tag not in ("metadata", "namedview"):
            # Strip Matplotlib's full-canvas figure background rectangle (patch_1) so vector lines stay transparent
            if child.attrib.get("id") == "figure_1":
                for sub in list(child):
                    if sub.attrib.get("id") == "patch_1":
                        child.remove(sub)
            raw = ET.tostring(child, encoding="unicode").replace("ns0:", "").replace(":ns0", "")
            vec_body_parts.append(raw)

    svg_doc = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="{w_px}px" height="{h_px}px" viewBox="0 0 {w_px} {h_px}" version="1.1">
  {defs_xml}
  <g id="Layer_Renders" inkscape:label="Renders (Shaded Fills Only)">
    <image x="0" y="0" width="{w_px}" height="{h_px}" preserveAspectRatio="none"
           xlink:href="data:image/png;base64,{b64_renders}" />
  </g>
  <g id="Layer_Outlines_HLR" inkscape:label="Outlines (Hidden-Line Removed Overlay)">
    <image x="0" y="0" width="{w_px}" height="{h_px}" preserveAspectRatio="none"
           xlink:href="data:image/png;base64,{b64_outlines}" />
  </g>
  <g id="Layer_Outlines_Vector" inkscape:label="Outlines (Editable Vector Paths)"
     visibility="hidden" transform="scale({sx:.6f},{sy:.6f})">
    {"".join(vec_body_parts)}
  </g>
</svg>
"""
    out_affinity_svg.write_text(svg_doc)


def _compose_preview(renders_png: Path, outlines_png: Path, out_composite: Path) -> None:
    base = Image.open(renders_png).convert("RGBA")
    over = Image.open(outlines_png).convert("RGBA")
    Image.alpha_composite(base, over).convert("RGB").save(out_composite)


def _build_candidate_pool() -> list[str]:
    """Assemble ~11-12 diverse multi-city sites per tier XS..L plus all XL sites."""
    lib = ContextLibrary()
    by_tier_city: dict[str, dict[str, list[str]]] = {t: {} for t in TIERS}
    valid_ids: set[str] = set()
    for entry in lib.catalog():
        sid = entry["site_id"]
        valid_ids.add(sid)
        t = sid.split("_")[0]
        c = entry["city_code"]
        if t in by_tier_city:
            by_tier_city[t].setdefault(c, []).append(sid)

    pool: list[str] = []
    for t in TIERS:
        seen_t: list[str] = []
        for fav in SHOWCASE_PRIORITIES.get(t, ()):
            if fav in valid_ids and fav not in seen_t:
                seen_t.append(fav)
        if t == "xl":
            for c_list in by_tier_city[t].values():
                for sid in c_list:
                    if sid not in seen_t:
                        seen_t.append(sid)
        else:
            target_n = 12 if t == "l" else 11
            city_queues = [sorted(v) for _k, v in sorted(by_tier_city[t].items())]
            idx = 0
            while len(seen_t) < target_n and any(idx < len(q) for q in city_queues):
                for q in city_queues:
                    if idx < len(q) and q[idx] not in seen_t:
                        seen_t.append(q[idx])
                        if len(seen_t) >= target_n:
                            break
                idx += 1
        pool.extend(seen_t)
    return pool


def _evaluate_and_cache_candidates(force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    if not force and CACHE_FILE.is_file() and META_CACHE_FILE.is_file():
        meta = json.loads(META_CACHE_FILE.read_text())
        arrays = dict(np.load(CACHE_FILE))
        return meta, arrays

    cfg, dev, policy = _load_policy(RUN_DIR, 0.0, 0.0, CKPT_PATH)
    pool = _build_candidate_pool()
    meta_out: dict[str, dict] = {}
    arrays_out: dict[str, np.ndarray] = {}

    for i, sid in enumerate(pool, start=1):
        tier = sid.split("_")[0]
        city_code = sid.split("_")[1]
        eval_seed = 9201 if sid == "l_sgp_0379" else (9100 + i)
        try:
            env, occ, core_mask, obj, diag = _run_trained_episode(cfg, dev, policy, sid, eval_seed)
        except ValueError as exc:
            print(f"[{i:02d}/{len(pool)}] {sid:16s} SKIPPED ({exc})", flush=True)
            continue
        active_z = np.flatnonzero(occ.any(axis=(1, 2)))
        z_top = int(active_z[-1]) + 1 if len(active_z) else 0
        occ_trim = occ[:z_top]
        core_trim = core_mask[:z_top]
        far_val = float(diag.get("far", float(occ.sum()) / max(1.0, float(env.plot_area))))
        meta_out[sid] = {
            "siteId": sid,
            "tier": tier.upper(),
            "city": env.context.get("city", city_code.upper()),
            "cityCode": city_code,
            "seed": eval_seed,
            "far": round(far_val, 3),
            "farSatisfied": bool(env.far_satisfied),
            "objective": round(float(obj), 4),
            "storeys": z_top,
            "coreCount": int(len(env.cores)),
            "usableGoodArea": round(float(diag.get("usableGoodArea", 0.0)), 1),
            "directSunProxy": round(float(diag.get("directSunProxy", 0.0)), 4),
            "context": {
                "city": env.context.get("city", city_code.upper()),
                "buildings": env.context.get("buildings") or (),
                "roads": env.context.get("roads") or (),
                "perimeter": env.context.get("perimeter") or (),
            },
        }
        arrays_out[f"{sid}__occ"] = occ_trim.astype(np.uint8)
        arrays_out[f"{sid}__core"] = core_trim.astype(np.uint8)
        print(
            f"[{i:02d}/{len(pool)}] {sid:16s} FAR={far_val:.2f} F={z_top} cores={len(env.cores)} obj={obj:.3f}",
            flush=True,
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    META_CACHE_FILE.write_text(
        json.dumps(meta_out, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    )
    np.savez_compressed(CACHE_FILE, **arrays_out)
    return meta_out, arrays_out


def _ordered_candidate_ids(meta: dict[str, dict]) -> list[str]:
    tier_rank = {t: i for i, t in enumerate(TIERS)}
    items = sorted(
        meta.values(),
        key=lambda v: (
            tier_rank.get(v["tier"].lower(), 9),
            -float(v["objective"]),
            v["siteId"],
        ),
    )
    return [v["siteId"] for v in items[:49]]


def render_contact_sheet_7x7(meta: dict[str, dict], arrays: dict[str, np.ndarray]) -> list[str]:
    """Render all 49 evaluated candidates in a numbered 7x7 contact sheet for curation."""
    ordered_ids = _ordered_candidate_ids(meta)
    fig = plt.figure(figsize=(16.5, 16.5), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    manifest: list[dict] = []

    for idx, sid in enumerate(ordered_ids, start=1):
        info = meta[sid]
        occ = arrays[f"{sid}__occ"].astype(bool)
        cm = arrays[f"{sid}__core"].astype(bool)
        ctx = info["context"]

        ax = fig.add_subplot(7, 7, idx, projection="3d")
        chosen_azim = render_voxel_scene_on_ax(ax, occ, cm, ctx, render_mode="combined")
        ax.text2D(
            0.04,
            0.94,
            f"#{idx:02d}",
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color="#111827",
            ha="left",
            va="top",
            zorder=20,
            bbox=dict(facecolor="#FFFFFF", edgecolor="#D1D5DB", linewidth=0.35, alpha=0.94, pad=1.0),
        )
        far_flag = "" if info["farSatisfied"] else "*"
        ax.text2D(
            0.03,
            0.03,
            f"{sid} · FAR {info['far']:.2f}{far_flag} · {info['storeys']}F · obj {info['objective']:.2f}",
            transform=ax.transAxes,
            fontsize=4.9,
            color="#111827",
            ha="left",
            va="bottom",
            zorder=20,
            bbox=dict(facecolor="#FFFFFF", edgecolor="none", alpha=0.92, pad=0.8),
        )
        manifest.append(
            {
                "num": idx,
                "siteId": sid,
                "tier": info["tier"],
                "city": info["city"],
                "far": info["far"],
                "farSatisfied": info["farSatisfied"],
                "storeys": info["storeys"],
                "cores": info["coreCount"],
                "objective": info["objective"],
                "azimDeg": chosen_azim,
            }
        )

    fig.subplots_adjust(left=0.006, right=0.994, top=0.994, bottom=0.006, wspace=0.025, hspace=0.025)
    out_png = DRAWINGS_DIR / "atlas_candidates_7x7.png"
    fig.savefig(out_png, dpi=220, facecolor="#FFFFFF")
    plt.close(fig)
    (DRAWINGS_DIR / "atlas_candidates_7x7.json").write_text(json.dumps(manifest, indent=2))
    print(f"Rendered {len(ordered_ids)}-tile 7x7 contact sheet -> {out_png}", flush=True)
    return ordered_ids


def _resolve_picks(pick_arg: str | None, meta: dict[str, dict], ordered_7x7: list[str]) -> list[str]:
    if not pick_arg:
        return [ordered_7x7[n - 1] for n in DEFAULT_PICKS_24]
    picks: list[str] = []
    for raw in pick_arg.split(","):
        tok = raw.strip().lstrip("#")
        if not tok:
            continue
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= len(ordered_7x7):
                picks.append(ordered_7x7[n - 1])
            else:
                raise ValueError(f"Index #{n} out of range 1..{len(ordered_7x7)}")
        elif tok in meta:
            picks.append(tok)
        else:
            raise ValueError(f"Unknown site ID {tok!r}")
    return picks


def render_atlas(
    force_eval: bool = False,
    pick_arg: str | None = None,
    skip_7x7: bool = False,
    only_7x7: bool = False,
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    meta, arrays = _evaluate_and_cache_candidates(force=force_eval)
    DRAWINGS_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    ordered_7x7 = _ordered_candidate_ids(meta)
    if not skip_7x7:
        ordered_7x7 = render_contact_sheet_7x7(meta, arrays)
    if only_7x7:
        return

    selected_ids = _resolve_picks(pick_arg, meta, ordered_7x7)
    if len(selected_ids) != 24:
        raise ValueError(f"Expected 24 sites for 4x6 atlas, got {len(selected_ids)}")

    # Precompute consistent best azimuth & per-tier zoom per site so fills and outlines align to 0.0 px
    azims: dict[str, float] = {}
    halfs: dict[str, float] = {}
    for sid in selected_ids:
        occ = arrays[f"{sid}__occ"].astype(bool)
        zs, ys, xs = np.nonzero(occ)
        cx = 0.5 * (float(xs.min()) + float(xs.max() + 1))
        cy = 0.5 * (float(ys.min()) + float(ys.max() + 1))
        tier_l = meta[sid]["tier"].lower()
        zoom = TIER_ZOOM.get(tier_l, 1.0)
        base_half = max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1), float(zs.max() + 1) * 3.0) * 0.62
        half_s = base_half / zoom
        halfs[sid] = half_s
        azims[sid] = pick_best_azimuth(meta[sid]["context"], occ, cx, cy, half_s, DEFAULT_ELEV)

    fig_size = (10.6, 15.0)
    dpi = 300
    adjust_kw = dict(left=0.008, right=0.992, top=0.992, bottom=0.008, wspace=0.016, hspace=0.016)

    renders_png = DRAWINGS_DIR / "atlas_4x6_renders.png"
    outlines_png = DRAWINGS_DIR / "atlas_4x6_outlines.png"
    outlines_svg = DRAWINGS_DIR / "atlas_4x6_outlines.svg"
    affinity_svg = DRAWINGS_DIR / "atlas_4x6_affinity.svg"
    composite_png = DRAWINGS_DIR / "atlas_4x6_composite.png"

    # 1. Render Fills Only + Hidden-Line-Removed Outlines + Vector SVG Outlines
    for mode, out_path, fmt in (
        ("fills", renders_png, "png"),
        ("outlines_knockout", outlines_png, "png"),
        ("outlines_vector", outlines_svg, "svg"),
    ):
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        fig.patch.set_facecolor("#FFFFFF" if mode != "outlines_vector" else "none")
        for idx, sid in enumerate(selected_ids, start=1):
            info = meta[sid]
            occ = arrays[f"{sid}__occ"].astype(bool)
            cm = arrays[f"{sid}__core"].astype(bool)
            ax = fig.add_subplot(6, 4, idx, projection="3d")
            render_voxel_scene_on_ax(
                ax,
                occ,
                cm,
                info["context"],
                azim_deg=azims[sid],
                half_override=halfs[sid],
                render_mode=mode,
            )
        fig.subplots_adjust(**adjust_kw)
        if fmt == "svg":
            fig.savefig(out_path, format="svg", transparent=True)
        else:
            fig.savefig(out_path, dpi=dpi, facecolor="#FFFFFF")
        plt.close(fig)

    _knockout_white_to_transparent_rgba(outlines_png)
    _compose_preview(renders_png, outlines_png, composite_png)
    _write_layered_affinity_svg(renders_png, outlines_png, outlines_svg, affinity_svg)

    manifest_rows = [
        {
            "index": idx,
            "row": (idx - 1) // 4 + 1,
            "col": (idx - 1) % 4 + 1,
            "candidateNum": ordered_7x7.index(sid) + 1 if sid in ordered_7x7 else None,
            "siteId": sid,
            "tier": meta[sid]["tier"],
            "city": meta[sid]["city"],
            "zoom": TIER_ZOOM.get(meta[sid]["tier"].lower(), 1.0),
            "seed": meta[sid].get("seed"),
            "far": meta[sid]["far"],
            "storeys": meta[sid]["storeys"],
            "cores": meta[sid]["coreCount"],
            "objective": meta[sid]["objective"],
            "usableGoodArea": meta[sid].get("usableGoodArea"),
            "directSunProxy": meta[sid].get("directSunProxy"),
            "azimDeg": azims[sid],
        }
        for idx, sid in enumerate(selected_ids, start=1)
    ]
    (DRAWINGS_DIR / "atlas_4x6_manifest.json").write_text(json.dumps(manifest_rows, indent=2))
    print(f"Exported Affinity 4x6 atlas layers -> {affinity_svg}", flush=True)
    for r in manifest_rows:
        print(
            f"  {r['index']:02d} R{r['row']}C{r['col']} (#{r['candidateNum']:02d}): "
            f"{r['tier']:2s} {r['siteId']:15s} ({r['city']}) zoom={r['zoom']:.2f}x "
            f"FAR={r['far']:.2f} {r['storeys']}F {r['cores']} cores obj={r['objective']:.3f}"
        )


def export_atlas_pdf(out_pdf: Path) -> Path:
    """Export the clean 4x6 atlas without labels as a print-ready PDF (fills + vector linework)."""
    manifest_path = DRAWINGS_DIR / "atlas_4x6_manifest.json"
    renders_png = DRAWINGS_DIR / "atlas_4x6_renders.png"
    for req in (manifest_path, renders_png, META_CACHE_FILE, CACHE_FILE):
        if not req.is_file():
            raise FileNotFoundError(f"Required atlas input missing: {req}")

    manifest_rows = json.loads(manifest_path.read_text())
    if len(manifest_rows) != 24:
        raise ValueError(f"Expected 24 manifest rows, got {len(manifest_rows)}")

    meta = json.loads(META_CACHE_FILE.read_text())
    arrays = dict(np.load(CACHE_FILE))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(10.6, 15.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    renders_arr = np.asarray(Image.open(renders_png).convert("RGB"))
    ax_bg = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
    ax_bg.imshow(
        renders_arr,
        origin="upper",
        aspect="auto",
        interpolation="none",
        extent=(0.0, 1.0, 0.0, 1.0),
    )
    ax_bg.set_xlim(0.0, 1.0)
    ax_bg.set_ylim(0.0, 1.0)
    ax_bg.set_axis_off()

    for idx, entry in enumerate(manifest_rows, start=1):
        sid = entry["siteId"]
        info = meta[sid]
        occ = arrays[f"{sid}__occ"].astype(bool)
        cm = arrays[f"{sid}__core"].astype(bool)
        zs, ys, xs = np.nonzero(occ)
        zoom = float(entry.get("zoom") or TIER_ZOOM.get(info["tier"].lower(), 1.0))
        base_half = (
            max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1), float(zs.max() + 1) * 3.0)
            * 0.62
        )
        half_s = base_half / zoom
        ax = fig.add_subplot(6, 4, idx, projection="3d")
        render_voxel_scene_on_ax(
            ax,
            occ,
            cm,
            info["context"],
            azim_deg=float(entry["azimDeg"]),
            half_override=half_s,
            render_mode="outlines_vector",
        )

    fig.subplots_adjust(left=0.008, right=0.992, top=0.992, bottom=0.008, wspace=0.016, hspace=0.016)
    out_pdf = out_pdf.expanduser()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=300, facecolor="#FFFFFF")
    plt.close(fig)

    canonical_pdf = DRAWINGS_DIR / "atlas_4x6.pdf"
    if out_pdf.resolve() != canonical_pdf.resolve():
        canonical_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_pdf, canonical_pdf)
    print(f"Exported clean 4x6 atlas PDF -> {out_pdf} (copy at {canonical_pdf})", flush=True)
    return out_pdf


def export_atlas_pdf_layers(out_pdf: Path) -> Path:
    """Export pure shaded fills (page 1) and pure vector linework (page 2) as a 2-page PDF."""
    manifest_path = DRAWINGS_DIR / "atlas_4x6_manifest.json"
    renders_png = DRAWINGS_DIR / "atlas_4x6_renders.png"
    for req in (manifest_path, renders_png, META_CACHE_FILE, CACHE_FILE):
        if not req.is_file():
            raise FileNotFoundError(f"Required atlas input missing: {req}")

    manifest_rows = json.loads(manifest_path.read_text())
    if len(manifest_rows) != 24:
        raise ValueError(f"Expected 24 manifest rows, got {len(manifest_rows)}")

    meta = json.loads(META_CACHE_FILE.read_text())
    arrays = dict(np.load(CACHE_FILE))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    out_pdf = out_pdf.expanduser()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Page 1: Pure shaded render layer
        fig_fills = plt.figure(figsize=(10.6, 15.0), dpi=300)
        fig_fills.patch.set_facecolor("#FFFFFF")
        renders_arr = np.asarray(Image.open(renders_png).convert("RGB"))
        ax_bg = fig_fills.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
        ax_bg.imshow(
            renders_arr,
            origin="upper",
            aspect="auto",
            interpolation="none",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        ax_bg.set_xlim(0.0, 1.0)
        ax_bg.set_ylim(0.0, 1.0)
        ax_bg.set_axis_off()
        pdf.savefig(fig_fills, dpi=300, facecolor="#FFFFFF")
        plt.close(fig_fills)

        # Page 2: Pure vector linework layer
        fig_lines = plt.figure(figsize=(10.6, 15.0), dpi=300)
        fig_lines.patch.set_facecolor("none")
        for idx, entry in enumerate(manifest_rows, start=1):
            sid = entry["siteId"]
            info = meta[sid]
            occ = arrays[f"{sid}__occ"].astype(bool)
            cm = arrays[f"{sid}__core"].astype(bool)
            zs, ys, xs = np.nonzero(occ)
            zoom = float(entry.get("zoom") or TIER_ZOOM.get(info["tier"].lower(), 1.0))
            base_half = (
                max(
                    float(xs.max() - xs.min() + 1),
                    float(ys.max() - ys.min() + 1),
                    float(zs.max() + 1) * 3.0,
                )
                * 0.62
            )
            half_s = base_half / zoom
            ax = fig_lines.add_subplot(6, 4, idx, projection="3d")
            render_voxel_scene_on_ax(
                ax,
                occ,
                cm,
                info["context"],
                azim_deg=float(entry["azimDeg"]),
                half_override=half_s,
                render_mode="outlines_vector",
            )
        fig_lines.subplots_adjust(
            left=0.008, right=0.992, top=0.992, bottom=0.008, wspace=0.016, hspace=0.016
        )
        pdf.savefig(fig_lines, dpi=300, facecolor="none", transparent=True)
        plt.close(fig_lines)

    canonical_pdf = DRAWINGS_DIR / "atlas_4x6_layers.pdf"
    if out_pdf.resolve() != canonical_pdf.resolve():
        canonical_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_pdf, canonical_pdf)
    print(f"Exported 2-page atlas layers PDF -> {out_pdf} (copy at {canonical_pdf})", flush=True)
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-7x7", action="store_true")
    parser.add_argument("--only-7x7", action="store_true")
    parser.add_argument(
        "--export-pdf",
        type=Path,
        default=None,
        help="Export the clean 4x6 atlas (no labels) to this PDF path using existing manifest/renders",
    )
    parser.add_argument(
        "--export-pdf-layers",
        type=Path,
        default=None,
        help="Export pure shaded fills (page 1) and vector linework (page 2) as a 2-page PDF",
    )
    parser.add_argument(
        "--pick",
        type=str,
        default=None,
        help="Comma-separated 24 candidate numbers (#1..#49 from 7x7 sheet) or site IDs",
    )
    args = parser.parse_args()
    if args.export_pdf is not None:
        export_atlas_pdf(args.export_pdf)
        return
    if args.export_pdf_layers is not None:
        export_atlas_pdf_layers(args.export_pdf_layers)
        return
    render_atlas(
        force_eval=args.force_eval,
        pick_arg=args.pick,
        skip_7x7=args.skip_7x7,
        only_7x7=args.only_7x7,
    )


if __name__ == "__main__":
    main()
