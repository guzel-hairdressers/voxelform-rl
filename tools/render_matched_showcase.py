"""Draw a voxel axonometric and pair it with a Blender render from the same camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from tools.generate_readme_media import (
    _extract_building_footprint,
    _iter_polygons,
    _shade_rgba,
    _surface_quads,
)
from tools.render_portfolio_atlas import (
    _chain_segments_to_polylines,
    _clip_segments_against_voxels,
    _clip_segments_hlr,
    _extract_hero_silhouette_segs,
    _extract_prism_occluders,
    _extract_voxel_crease_segs,
    _extrude_prism_minimal_shade,
    _visible_prism_edges,
)


def _camera_basis_from_matrix(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_ax = axes[:, 0].astype(np.float64)
    u_ax /= max(1e-9, float(np.linalg.norm(u_ax)))
    v_ax = axes[:, 1].astype(np.float64)
    v_ax /= max(1e-9, float(np.linalg.norm(v_ax)))
    view_ax = np.cross(u_ax, v_ax)
    view_ax /= max(1e-9, float(np.linalg.norm(view_ax)))
    if float(view_ax @ axes[:, 2]) > 0.0:
        view_ax = -view_ax
    return u_ax, v_ax, view_ax


def _to_shifted_hero_pts(
    pts: np.ndarray, cx: float, cy: float, gf_delta: float
) -> np.ndarray:
    out = np.asarray(pts, dtype=np.float64).copy()
    out[..., 0] -= cx
    out[..., 1] -= cy
    if gf_delta != 0.0:
        out[..., 2] += (out[..., 2] > 0.0) * gf_delta
    return out


def draw_voxels(occupied, cores, payload, camera, size):
    """Project the voxel + context meshes in the atlas architectural graphic style."""
    metadata = payload["metadata"]
    cx, cy = float(metadata["centerXY"][0]), float(metadata["centerXY"][1])
    gf_delta = float(metadata["gfHeight"]) - 3.0
    axes = np.asarray(camera["axes"], dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    u_ax, v_ax, view_ax = _camera_basis_from_matrix(axes)

    occ = np.asarray(occupied, dtype=bool)
    core_mask = np.asarray(cores, dtype=bool) & occ
    shell_mask = occ & ~core_mask

    polygons, depths, colors, edges, edge_lws = [], [], [], [], []

    def add_poly(vertices, rgba, edge=(0.0, 0.0, 0.0, 0.0), lw=0.0):
        screen = np.asarray(vertices, dtype=np.float64) @ axes + origin
        polygons.append(screen[:, :2])
        depths.append(float(screen[:, 2].mean()))
        colors.append(rgba)
        edges.append(edge)
        edge_lws.append(lw)

    # Compute shifted hero hull for foreground/background context classification
    zz, yy, xx = np.nonzero(occ)
    hero_pts_world = _to_shifted_hero_pts(
        np.stack(
            [xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5, (zz.astype(np.float64) + 0.5) * 3.0],
            axis=1,
        ),
        cx,
        cy,
        gf_delta,
    )
    hu = hero_pts_world @ u_ax
    hv = hero_pts_world @ v_ax
    hd = hero_pts_world @ view_ax
    hero_uv = np.column_stack([hu, hv])
    hero_hull = MultiPoint(hero_uv).convex_hull.buffer(0.8)

    sil_segs, _sil_geom, _d_sil = _extract_hero_silhouette_segs(occ, 3.0, u_ax, v_ax, view_ax)
    bg_items, fg_items = [], []
    bg_occluders, fg_occluders = [], []

    for building in payload["context"]["Context_Buildings"]:
        footprint, lower, upper = _extract_building_footprint(building)
        if footprint is None:
            continue
        b_faces, b_shades, b_sil, b_inner, b_occ, pts_3d = [], [], [], [], [], []
        for polygon in _iter_polygons(footprint):
            fl, sl = _extrude_prism_minimal_shade(polygon, lower, upper, u_ax)
            if not fl:
                continue
            b_faces.extend(fl)
            b_shades.extend(sl)
            psil, pinner = _visible_prism_edges(polygon, lower, upper, view_ax)
            b_sil.extend(psil)
            b_inner.extend(pinner)
            b_occ.extend(_extract_prism_occluders(polygon, lower, upper, u_ax, v_ax, view_ax))
            ep = np.asarray(polygon.exterior.coords, dtype=np.float64)[:-1]
            if len(ep):
                pts_3d.append(np.column_stack([ep, np.full(len(ep), lower)]))
                pts_3d.append(np.column_stack([ep, np.full(len(ep), upper)]))
        if not b_faces or not pts_3d:
            continue
        b_arr = np.concatenate(pts_3d, axis=0)
        bu = b_arr @ u_ax
        bv = b_arr @ v_ax
        bd = b_arr @ view_ax
        b_hull = MultiPoint(np.column_stack([bu, bv])).convex_hull
        overlap = b_hull.intersection(hero_hull)
        is_occ = False
        if not overlap.is_empty and overlap.area >= 0.5:
            dists_uv = np.hypot(hero_uv[:, 0] - overlap.centroid.x, hero_uv[:, 1] - overlap.centroid.y)
            near = dists_uv <= 6.0
            if near.any() and float(bd.max()) > float(np.percentile(hd[near], 25)) - 0.5:
                is_occ = True
        item = {"faces": b_faces, "shades": b_shades, "sil": b_sil, "inner": b_inner}
        if is_occ:
            fg_items.append(item)
            fg_occluders.extend(b_occ)
        else:
            bg_items.append(item)
            bg_occluders.extend(b_occ)

    for item in bg_items:
        for verts, shade in zip(item["faces"], item["shades"], strict=True):
            add_poly(verts, (shade, shade, shade, 1.0))
    for item in fg_items:
        for verts, shade in zip(item["faces"], item["shades"], strict=True):
            add_poly(verts, (shade, shade, shade, 0.36))

    # Hero shell: pure #FFFFFF roofs/left walls + minimal #F4F4F4 right wall
    if shell_mask.any():
        quads, fq, _ = _surface_quads(shell_mask, dz=3.0)
        quads = _to_shifted_hero_pts(quads, cx, cy, gf_delta)
        fn = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
        gains_6 = np.ones(6, dtype=np.float32)
        for d_i in range(4):
            if float(fn[d_i] @ view_ax) > 1e-3:
                gains_6[d_i] = 0.92 if float(fn[d_i] @ u_ax) > 0.1 else 0.96
        g = gains_6[fq]
        for verts, f_idx, shade in zip(quads, fq, g, strict=True):
            alpha = 1.0 if int(f_idx) == 4 else 0.96
            add_poly(verts, (float(shade), float(shade), float(shade), alpha))

    # Vivid yellow cores (#FADE24)
    if core_mask.any():
        c_quads, cf, _ = _surface_quads(core_mask, dz=3.0)
        c_quads = _to_shifted_hero_pts(c_quads, cx, cy, gf_delta)
        c_rgba = _shade_rgba((0.98, 0.87, 0.14), cf, 0.98)
        for verts, col in zip(c_quads, c_rgba, strict=True):
            add_poly(verts, col)

    width, height = size
    paper = "#ffffff"
    figure = plt.figure(figsize=(width / 150, height / 150), dpi=150, facecolor=paper)
    ax = figure.add_axes((0, 0, 1, 1), facecolor=paper)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    order = np.argsort(depths)[::-1]
    ax.add_collection(
        PolyCollection(
            [polygons[i] for i in order],
            facecolors=np.asarray(colors)[order],
            edgecolors="none",
            linewidths=0.0,
            zorder=2.0,
        )
    )

    def _add_lines(segs_3d, color, lw, alpha=1.0, zorder=3.0, shift_hero=False, chain=True):
        if not segs_3d:
            return
        lines_2d = []
        pts_list = _chain_segments_to_polylines(segs_3d) if chain else segs_3d
        for poly3d in pts_list:
            arr = np.asarray(poly3d, dtype=np.float64)
            if shift_hero:
                arr = _to_shifted_hero_pts(arr, cx, cy, gf_delta)
            scr = arr @ axes + origin
            lines_2d.append(scr[:, :2])
        ax.add_collection(
            LineCollection(
                lines_2d,
                colors=color,
                linewidths=lw,
                alpha=alpha,
                capstyle="round",
                joinstyle="round",
                zorder=zorder,
            )
        )

    # Clip background context lines & parcel perimeter behind hero mass + occluders
    bg_occ_all = list(bg_occluders)
    if not hero_hull.is_empty:
        eroded = hero_hull.buffer(-0.8)
        for p in _iter_polygons(eroded):
            bg_occ_all.append((p, p.bounds, 0.0, 0.0, float(hd.max()), float(hd.max())))

    # Parcel perimeter if present in Site_Parcel
    parcel = payload["context"].get("Site_Parcel") or {}
    if parcel.get("vertices") and parcel.get("faces"):
        pv = np.asarray(parcel["vertices"], dtype=np.float64)
        p_tris = [Polygon(pv[f, :2]) for f in parcel["faces"] if len(f) >= 3]
        p_merged = unary_union([p.buffer(0) for p in p_tris if not p.is_empty])
        perim_segs = []
        for poly in _iter_polygons(p_merged):
            ep = np.asarray(poly.exterior.coords, dtype=np.float32)
            for i in range(len(ep) - 1):
                perim_segs.append(np.array([[ep[i, 0], ep[i, 1], 0.10], [ep[i + 1, 0], ep[i + 1, 1], 0.10]], dtype=np.float32))
        if perim_segs:
            vis_perim = _clip_segments_hlr(perim_segs, bg_occluders, u_ax, v_ax, view_ax, eps=0.08)
            vis_perim_unshifted = [
                np.array([[s[0, 0] + cx, s[0, 1] + cy, s[0, 2]], [s[1, 0] + cx, s[1, 1] + cy, s[1, 2]]], dtype=np.float32)
                for s in vis_perim
            ]
            vis_perim_unshifted = _clip_segments_against_voxels(
                vis_perim_unshifted, occ, 3.0, u_ax, v_ax, view_ax, eps=0.65
            )
            vis_perim = [
                np.array([[s[0, 0] - cx, s[0, 1] - cy, s[0, 2]], [s[1, 0] - cx, s[1, 1] - cy, s[1, 2]]], dtype=np.float32)
                for s in vis_perim_unshifted
            ]
            _add_lines(vis_perim, (0.42, 0.46, 0.52), 0.72, alpha=0.72, zorder=2.1)

    if bg_items:
        bg_inner = _clip_segments_hlr([s for it in bg_items for s in it["inner"]], bg_occ_all, u_ax, v_ax, view_ax)
        bg_sil = _clip_segments_hlr([s for it in bg_items for s in it["sil"]], bg_occ_all, u_ax, v_ax, view_ax)
        _add_lines(bg_inner, (0.42, 0.46, 0.52), 0.45, alpha=0.62, zorder=2.2, chain=False)
        _add_lines(bg_sil, (0.42, 0.46, 0.52), 0.72, alpha=0.72, zorder=2.25)

    # Hero coplanar-free creases + core creases + bold #111827 outer silhouette
    hero_creases = _clip_segments_against_voxels(
        _extract_voxel_crease_segs(occ, 3.0, view_ax), occ, 3.0, u_ax, v_ax, view_ax, eps=0.08
    )
    core_creases = _clip_segments_against_voxels(
        _extract_voxel_crease_segs(core_mask, 3.0, view_ax), occ, 3.0, u_ax, v_ax, view_ax, eps=0.08
    )
    _add_lines(hero_creases, (0.18, 0.21, 0.26), 0.45, alpha=0.84, zorder=3.3, shift_hero=True)
    _add_lines(core_creases, (0.45, 0.37, 0.04), 0.40, alpha=0.75, zorder=3.4, shift_hero=True)
    _add_lines(sil_segs, "#111827", 1.75, alpha=0.98, zorder=3.6, shift_hero=True)

    if fg_items:
        fg_occ = fg_occluders + bg_occluders
        fg_inner = _clip_segments_hlr([s for it in fg_items for s in it["inner"]], fg_occ, u_ax, v_ax, view_ax)
        fg_sil = _clip_segments_hlr([s for it in fg_items for s in it["sil"]], fg_occ, u_ax, v_ax, view_ax)
        _add_lines(fg_inner, (0.42, 0.47, 0.54), 0.42, alpha=0.42, zorder=4.1, chain=False)
        _add_lines(fg_sil, (0.42, 0.47, 0.54), 0.62, alpha=0.48, zorder=4.15)

    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB")
    plt.close(figure)
    return image


def render_plate(snapshot_path, aerial_path, output_path):
    payload = json.loads(snapshot_path.with_suffix(".json").read_text())
    camera = json.loads(aerial_path.with_name("axo-camera.json").read_text())
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        occupied, cores = snapshot["occupied"], snapshot["cores"]
    with Image.open(aerial_path) as source:
        if list(source.size) != camera["size"]:
            raise ValueError(
                "The aerial render does not match the camera's frame size."
            )
        aerial = source.convert("RGB")

    panel_size = (1240, 698)
    voxel = draw_voxels(occupied, cores, payload, camera, panel_size)
    margin, gutter, top = 114, 172, 84
    width, height = panel_size
    plate = Image.new(
        "RGB", (2 * width + 2 * margin + gutter, top + height + 36), "white"
    )
    canvas = ImageDraw.Draw(plate)
    font = ImageFont.truetype(
        findfont(FontProperties(family="DejaVu Sans", weight="bold")), 26
    )
    info = payload["metadata"]
    lattice_label = (
        f"A. VoxelForm-RL 3D Lattice (FAR {info['far']:.2f} · "
        f"{info['coreCount']} Cores · Sun Proxy {info['directSunProxy']:.0%})"
    )
    aerial_label = "B. Procedural Blender Maquette — Parallel Aerial View (Cycles)"
    for offset, label, panel in (
        (margin, lattice_label, voxel),
        (
            margin + width + gutter,
            aerial_label,
            aerial.resize(panel_size, Image.Resampling.LANCZOS),
        ),
    ):
        plate.paste(panel, (offset, top))
        canvas.rectangle(
            (offset, top, offset + width - 1, top + height - 1),
            outline="#c8d3e1",
            width=3,
        )
        canvas.text(
            (offset + width / 2, top - 26),
            label,
            font=font,
            fill="#172033",
            anchor="mm",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plate.save(output_path, quality=94, subsampling=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="NPZ with its export JSON beside it",
    )
    parser.add_argument(
        "--aerial",
        type=Path,
        required=True,
        help="Blender aerial with axo-camera.json beside it",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_plate(args.snapshot, args.aerial, args.output)


if __name__ == "__main__":
    main()
