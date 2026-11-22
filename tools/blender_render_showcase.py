"""Render an exported maquette from an axonometric and street-level camera.

Run with Blender's Python:
    blender --background site.blend --python tools/blender_render_showcase.py -- \
        --out-dir renders
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from blender_maquette_library import configure_cycles_light_paths  # noqa: E402


def set_camera(camera, position, target):
    camera.location = position
    camera.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def visible_fraction(scene, hero_objects, origin, probes):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    visible = 0
    for target in probes:
        ray = target - origin
        hit, _, _, _, obj, _ = scene.ray_cast(
            depsgraph, origin, ray.normalized(), distance=ray.length + 0.25
        )
        # A probe on the source mass may fall in an open bay after panelization.
        visible += int(not hit or obj.original in hero_objects)
    return visible / len(probes)


def azimuth_directions(angle, default):
    if angle is not None:
        angles = [math.radians(angle)]
    else:
        angles = [
            default + math.radians(offset)
            for offset in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
        ]
    return [Vector((math.cos(phi), math.sin(phi), 0.0)) for phi in angles]


def set_camera_backdrop(world):
    # Keep the HDRI lighting and reflections, with a solid backdrop in the camera.
    tree = world.node_tree
    output = next(node for node in tree.nodes if node.type == "OUTPUT_WORLD")
    surface = output.inputs["Surface"]
    illumination = surface.links[0].from_socket
    camera_ray = tree.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
    backdrop = tree.nodes.new("ShaderNodeBackground")
    backdrop.inputs["Color"].default_value = (0.065, 0.076, 0.098, 1)
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(camera_ray, mix.inputs[0])
    tree.links.new(illumination, mix.inputs[1])
    tree.links.new(backdrop.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], surface)


def render_views(args):
    scene = bpy.context.scene
    camera = scene.camera
    hero = bpy.data.collections.get("01_Hero_Building")
    mass = bpy.data.objects.get("Building_Mass")
    if not camera or not hero or not mass or not mass.data.vertices:
        raise ValueError("Open a maquette scene produced by tools.export_blender_site.")

    vertices = [mass.matrix_world @ vert.co for vert in mass.data.vertices]
    extent_min = Vector([min(pt[i] for pt in vertices) for i in range(3)])
    extent_max = Vector([max(pt[i] for pt in vertices) for i in range(3)])
    center = (extent_min + extent_max) / 2
    span = (extent_max - extent_min).length
    start = camera.location - center
    default_angle = math.atan2(start.y, start.x)
    probes = vertices[:: max(1, len(vertices) // 55)]
    hero_objects = set(hero.objects)

    scene.render.resolution_x = args.width
    scene.render.resolution_y = round(args.width * 0.61)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    camera.data.dof.use_dof = False
    camera.data.shift_x = 0
    camera.data.shift_y = 0
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
    if scene.render.engine == "CYCLES":
        configure_cycles_light_paths(scene)
        set_camera_backdrop(scene.world)
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        if args.metal:
            devices = bpy.context.preferences.addons["cycles"].preferences
            devices.compute_device_type = "METAL"
            devices.get_devices()
            for device in devices.devices:
                device.use = device.type == "METAL"
            if not any(device.use for device in devices.devices):
                raise RuntimeError("No Metal render device found.")
            scene.cycles.device = "GPU"

    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, angle in (("axo", args.axo_angle), ("street", args.street_angle)):
        is_axo = name == "axo"
        camera.data.type = "ORTHO" if is_axo else "PERSP"
        camera.data.lens = 32
        if is_axo:
            target = center
            candidates = [
                center + (direction + Vector((0, 0, 0.83))) * (span * 2.8)
                for direction in azimuth_directions(angle, default_angle)
            ]
        else:
            target = Vector(
                (center.x, center.y, extent_min.z + 0.4 * (extent_max.z - extent_min.z))
            )
            eye = Vector((center.x, center.y, extent_min.z + 1.7))
            candidates = [
                eye + direction * (span * distance)
                for direction in azimuth_directions(angle, default_angle)
                for distance in (0.85, 1.03, 0.73)
            ]
        camera.data.ortho_scale = span * 1.4
        scored = []
        for position in candidates:
            set_camera(camera, position, target)
            corners = [world_to_camera_view(scene, camera, point) for point in vertices]
            if not is_axo and any(
                pt.z <= 0 or min(pt.x, pt.y) < 0.04 or max(pt.x, pt.y) > 0.96
                for pt in corners
            ):
                continue
            visibility = visible_fraction(scene, hero_objects, position, probes)
            scored.append((visibility, position))
        if not scored:
            raise ValueError(f"No camera position frames the building for {name}.")
        visibility, position = max(scored, key=lambda result: result[0])
        set_camera(camera, position, target)
        if is_axo:
            corners = [world_to_camera_view(scene, camera, pt) for pt in vertices]
            width = max(p.x for p in corners) - min(p.x for p in corners)
            height = max(p.y for p in corners) - min(p.y for p in corners)
            camera.data.ortho_scale *= max(width / 0.79, height / 0.72)
        print(
            f"{name}: camera {tuple(round(c, 2) for c in position)}, visible probes {visibility:.0%}",
            flush=True,
        )
        scene.render.filepath = str(output / f"{name}.jpg")
        bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1760)
    parser.add_argument("--engine", choices=("scene", "cycles"), default="scene")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--metal", action="store_true", help="Use Metal for Cycles rendering"
    )
    parser.add_argument(
        "--axo-angle", type=float, help="Azimuth in degrees from the positive X axis"
    )
    parser.add_argument(
        "--street-angle", type=float, help="Azimuth in degrees from the positive X axis"
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    if args.metal and args.engine != "cycles":
        parser.error("--metal requires --engine cycles")
    render_views(args)


if __name__ == "__main__":
    main()
