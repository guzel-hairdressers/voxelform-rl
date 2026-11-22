"""Draw a voxel axonometric and pair it with a Blender render from the same camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.generate_readme_media import (
    _extract_building_footprint,
    _extrude_clean_prism,
    _iter_polygons,
    _shade_rgba,
    _surface_quads,
)


def draw_voxels(occupied, cores, payload, camera, size):
    """Project the existing voxel and context meshes onto the aerial image plane."""
    metadata = payload["metadata"]
    axes = np.asarray(camera["axes"])
    origin = np.asarray(camera["origin"])
    polygons, depths, colors, edges = [], [], [], []

    def add(vertices, rgba, edge):
        screen = vertices @ axes + origin
        polygons.append(screen[:, :2])
        depths.append(screen[:, 2].mean())
        colors.append(rgba)
        edges.append(edge)

    for building in payload["context"]["Context_Buildings"]:
        footprint, lower, upper = _extract_building_footprint(building)
        if footprint is None:
            continue
        for polygon in _iter_polygons(footprint):
            mesh, shades, _ = _extrude_clean_prism(polygon, lower, upper)
            for vertices, shade in zip(mesh, shades, strict=True):
                color = tuple(channel * shade for channel in (0.7, 0.75, 0.82))
                add(vertices, (*color, 0.09), (0.45, 0.53, 0.63, 0.16))

    for mask, base, opacity, edge in (
        (occupied & ~cores, (0.36, 0.54, 0.98), 0.52, (0.12, 0.25, 0.65, 0.20)),
        (occupied & cores, (0.86, 0.15, 0.15), 0.98, (0.45, 0.05, 0.05, 0.55)),
    ):
        quads, direction, _ = _surface_quads(mask)
        quads[..., :2] -= metadata["centerXY"]
        quads[..., 2] += (quads[..., 2] > 0) * (metadata["gfHeight"] - 3)
        for vertices, color in zip(
            quads, _shade_rgba(base, direction, opacity), strict=True
        ):
            add(vertices, color, edge)

    width, height = size
    paper = "#eff3f8"
    figure = plt.figure(figsize=(width / 150, height / 150), dpi=150, facecolor=paper)
    ax = figure.add_axes((0, 0, 1, 1), facecolor=paper)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    order = np.argsort(depths)[::-1]
    ax.add_collection(
        PolyCollection(
            [polygons[i] for i in order],
            facecolors=np.asarray(colors)[order],
            edgecolors=np.asarray(edges)[order],
            linewidths=0.24,
        )
    )
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert(
        "RGB"
    )
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
