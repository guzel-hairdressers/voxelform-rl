"""Build an automated architectural-maquette VoxelForm-RL .blend scene from exported JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Vector

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from blender_maquette_library import (  # noqa: E402
    CONTEXT_TIERS,
    HERO_STYLES,
    configure_studio_world,
    ensure_maquette_library,
    set_mod_input,
)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hero-style", default="curtain_frame")
    parser.add_argument("--no-context-facades", action="store_true")
    parser.add_argument("--no-props", action="store_true")
    parser.add_argument("--no-road-markings", action="store_true")
    return parser.parse_args(argv)


def _ensure_collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        parent.children.link(col)
    elif col.name not in parent.children:
        parent.children.link(col)
    return col


def _create_mesh_object(
    name: str,
    verts: list[list[float]],
    faces: list[list[int]],
    collection: bpy.types.Collection,
    material: bpy.types.Material,
    *,
    recalc_normals: bool = True,
) -> bpy.types.Object | None:
    if not verts or not faces:
        return None
    bm = bmesh.new()
    bm_verts = [bm.verts.new(v) for v in verts]
    bm.verts.ensure_lookup_table()
    for face in faces:
        if len(face) < 3:
            continue
        try:
            bm.faces.new([bm_verts[idx] for idx in face])
        except ValueError:
            pass
    if not bm.faces:
        bm.free()
        return None
    if len(bm.faces) == 1:
        bm.faces.ensure_lookup_table()
        bm.faces[0].normal_update()
        if bm.faces[0].normal.z < 0.0:
            bm.faces[0].normal_flip()
    elif recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = False
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _add_gn_mod(
    obj: bpy.types.Object, name: str, group_name: str
) -> bpy.types.Modifier:
    ng = bpy.data.node_groups.get(group_name)
    if ng is None:
        raise RuntimeError(f"Missing Geometry Node group {group_name!r}")
    mod = obj.modifiers.new(name=name, type="NODES")
    mod.node_group = ng
    return mod


def _apply_hero_facades(
    obj: bpy.types.Object | None, meta: dict, hero_style: str
) -> None:
    if obj is None or hero_style == "raw":
        return
    spec = HERO_STYLES.get(hero_style)
    if spec is None:
        raise ValueError(f"Unknown hero style: {hero_style!r}")

    gf_h = float(meta.get("gfHeight", 4.2))
    eval_seed = int(meta.get("evalSeed", 9003))

    m_pan = _add_gn_mod(obj, "Panelize Building", "Panelize Building")
    set_mod_input(m_pan, "GF Height", gf_h)
    set_mod_input(m_pan, "GF Length", spec["gf_length"])
    set_mod_input(m_pan, "Upper Height", 3.0005)
    set_mod_input(m_pan, "Upper Length", spec["upper_length"])

    if spec["mode"] == "curtain_wall":
        m_cw = _add_gn_mod(obj, "Hero Curtain Wall", "Hero Curtain Wall")
        set_mod_input(m_cw, "GF Height", gf_h)
        set_mod_input(m_cw, "Grid Module", float(spec["grid_module"]))
        set_mod_input(m_cw, "Opaque Every N Bays", int(spec.get("opaque_step", 0)))
        set_mod_input(m_cw, "Delete Glass", False)
        set_mod_input(m_cw, "Use Backing", False)
        return

    m_id = _add_gn_mod(obj, "Assign Panel ID", "Assign Panel ID")
    set_mod_input(m_id, "Ground ID", spec["ground_id"])
    set_mod_input(m_id, "Ground Narrow ID", spec["ground_narrow_id"])
    set_mod_input(m_id, "Narrow ID", spec["narrow_id"])
    set_mod_input(m_id, "Upper Min ID", spec["upper_min"])
    set_mod_input(m_id, "Upper Max ID", spec["upper_max"])
    set_mod_input(m_id, "Opaque ID", int(spec.get("opaque_id", -1)))
    set_mod_input(m_id, "Opaque Step", int(spec.get("opaque_step", 0)))
    set_mod_input(m_id, "Ground Max Z", gf_h + 0.1)
    set_mod_input(m_id, "Min Face Area", 1.2)
    set_mod_input(m_id, "Narrow Max Area", 4.6)
    set_mod_input(m_id, "Grid Module", float(spec["grid_module"]))
    set_mod_input(m_id, "Seed", eval_seed)

    m_swap = _add_gn_mod(obj, "Panel Swap", "Panel Swap")
    set_mod_input(m_swap, "Collection", bpy.data.collections[spec["collection"]])
    set_mod_input(m_swap, "Realize and Merge", False)
    _add_gn_mod(obj, "Facade Glazing", "Facade Glazing")


def _apply_context_facades(
    obj: bpy.types.Object | None,
    bld: dict,
    idx: int,
    site_hash: int,
    mono_mat: bpy.types.Material,
) -> None:
    if obj is None:
        return
    h_bld = float(bld.get("height", 12.0))
    ctx_gf_h = float(bld.get("gf_height", 4.4))
    tier = "low" if h_bld < 14.0 else ("mid" if h_bld < 28.0 else "high")
    variants = CONTEXT_TIERS[tier]
    col_name, gf_len, up_len, target_up_h, upper_max_id = variants[
        (site_hash + idx) % len(variants)
    ]

    rem_h = max(0.0, h_bld - ctx_gf_h)
    if rem_h > 1.5:
        n_up = max(1, round(rem_h / target_up_h))
        up_h = rem_h / n_up + 0.002
        eff_gf_h = ctx_gf_h
    else:
        eff_gf_h = h_bld + 0.05
        up_h = h_bld + 0.05

    m_pan = _add_gn_mod(obj, "Panelize Building", "Panelize Building")
    set_mod_input(m_pan, "GF Height", eff_gf_h)
    set_mod_input(m_pan, "GF Length", gf_len)
    set_mod_input(m_pan, "Upper Height", up_h)
    set_mod_input(m_pan, "Upper Length", up_len)

    if col_name is None:
        m_cw = _add_gn_mod(obj, "Hero Curtain Wall", "Hero Curtain Wall")
        set_mod_input(m_cw, "GF Height", eff_gf_h)
        set_mod_input(m_cw, "GF Mullion Material", mono_mat)
        set_mod_input(m_cw, "Upper Mullion Material", mono_mat)
        set_mod_input(m_cw, "Glass Material", mono_mat)
        set_mod_input(m_cw, "Interior Material", mono_mat)
        set_mod_input(m_cw, "Mullion Width", 0.09)
        set_mod_input(m_cw, "Mullion Depth", 0.24)
        set_mod_input(m_cw, "GF Mullion Width", 0.09)
        set_mod_input(m_cw, "GF Mullion Depth", 0.24)
        set_mod_input(m_cw, "Opaque Every N Bays", 0)
        set_mod_input(m_cw, "Delete Glass", False)
        set_mod_input(m_cw, "Use Backing", False)
        return

    m_id = _add_gn_mod(obj, "Assign Panel ID", "Assign Panel ID")
    set_mod_input(m_id, "Ground ID", 0)
    set_mod_input(m_id, "Ground Narrow ID", -1)
    set_mod_input(m_id, "Narrow ID", -1)
    set_mod_input(m_id, "Upper Min ID", 1)
    set_mod_input(m_id, "Upper Max ID", upper_max_id)
    set_mod_input(m_id, "Opaque ID", -1)
    set_mod_input(m_id, "Opaque Step", 0)
    set_mod_input(m_id, "Ground Max Z", eff_gf_h + 0.1)
    set_mod_input(m_id, "Min Face Area", 1.6)
    set_mod_input(m_id, "Narrow Max Area", 4.2)
    set_mod_input(m_id, "Seed", site_hash + idx)

    m_swap = _add_gn_mod(obj, "Panel Swap", "Panel Swap")
    set_mod_input(m_swap, "Collection", bpy.data.collections[col_name])
    set_mod_input(m_swap, "Realize and Merge", False)


def _create_pedestrian_paths(
    paths: list[dict], props_col: bpy.types.Collection, seed: int
) -> None:
    if not paths:
        return
    curve = bpy.data.curves.new("Pedestrian_Paths", type="CURVE")
    curve.dimensions = "3D"
    for entry in paths:
        pts = entry.get("points") or ()
        if len(pts) < 2:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(len(pts) - 1)
        for idx, (px, py, pz) in enumerate(pts):
            spline.points[idx].co = (float(px), float(py), float(pz), 1.0)
    if not curve.splines:
        bpy.data.curves.remove(curve)
        return
    obj = bpy.data.objects.new("Pedestrian_Paths", curve)
    props_col.objects.link(obj)

    mod = _add_gn_mod(obj, "Scatter Humans On Paths", "Scatter Humans On Paths")
    if (humans_col := bpy.data.collections.get("Maquette_Humans")) is not None:
        set_mod_input(mod, "Humans Collection", humans_col)
    set_mod_input(mod, "Spacing", 4.2)
    set_mod_input(mod, "Keep Chance", 0.30)
    set_mod_input(mod, "Along Jitter", 1.65)
    set_mod_input(mod, "Lateral Jitter", 0.32)
    set_mod_input(mod, "Flip Direction", 0.5)
    set_mod_input(mod, "Seed", seed)


def _place_street_props(
    street_props: dict, parent_col: bpy.types.Collection, *, seed: int = 9003
) -> None:
    if not street_props:
        return
    props_col = _ensure_collection("03_Maquette_Props", parent_col)

    def _spawn(
        candidates: tuple[str, ...],
        inst_name: str,
        loc: list[float],
        yaw: float = 0.0,
        scale: float = 1.0,
    ):
        src = None
        for src_name in candidates:
            cand = bpy.data.objects.get(src_name)
            if cand is not None and cand.data is not None:
                src = cand
                break
        if src is None:
            return
        inst = bpy.data.objects.new(inst_name, src.data)
        inst.location = (float(loc[0]), float(loc[1]), float(loc[2]))
        inst.rotation_euler = (0.0, 0.0, float(yaw))
        inst.scale = (scale, scale, scale)
        props_col.objects.link(inst)

    for idx, lamp in enumerate(street_props.get("lamps", ())):
        _spawn(
            ("Prop_StreetLamp",),
            f"Lamp_{idx:02d}",
            lamp["pos"],
            yaw=lamp.get("yaw", 0.0),
        )
    for idx, bs in enumerate(street_props.get("bus_stops", ())):
        _spawn(
            ("Prop_BusStop",), f"BusStop_{idx:02d}", bs["pos"], yaw=bs.get("yaw", 0.0)
        )
    ped_paths = street_props.get("pedestrian_paths", ())
    if ped_paths:
        _create_pedestrian_paths(ped_paths, props_col, seed)
    else:
        for idx, fig in enumerate(street_props.get("figures", ())):
            v = int(fig.get("variant", 0))
            fallback_fig = "Prop_ScaleFigure_A" if v % 2 == 0 else "Prop_ScaleFigure_B"
            _spawn(
                (f"Prop_Human_{v % 6:02d}", fallback_fig),
                f"Figure_{idx:02d}",
                fig["pos"],
                yaw=fig.get("yaw", 0.0),
            )
    car_map = {
        "bus": ("Prop_Bus_SOR", "Prop_Car_Grey"),
        "taxi": ("Prop_Car_Taxi", "Prop_Car_White"),
        "tesla": ("Prop_Car_Tesla3", "Prop_Car_White"),
        "fiat": ("Prop_Car_Fiat500", "Prop_Car_Grey"),
        "savana": ("Prop_Car_Savana", "Prop_Car_Grey"),
        "white": ("Prop_Car_Tesla3", "Prop_Car_White"),
        "grey": ("Prop_Car_Savana", "Prop_Car_Grey"),
    }
    for idx, car in enumerate(street_props.get("cars", ())):
        cands = car_map.get(
            car.get("variant", "tesla"), ("Prop_Car_Tesla3", "Prop_Car_White")
        )
        _spawn(cands, f"Car_{idx:02d}", car["pos"], yaw=car.get("yaw", 0.0))
    for idx, tree in enumerate(street_props.get("trees", ())):
        _spawn(
            ("Prop_Tree_Maquette",),
            f"Tree_{idx:02d}",
            tree["pos"],
            scale=float(tree.get("scale", 1.0)),
        )


def _max_xy_extent(payload: dict) -> float:
    max_xy = 0.0
    for mesh_dict in payload.get("hero", {}).values():
        for x, y, _ in mesh_dict.get("vertices", ()):
            max_xy = max(max_xy, abs(float(x)), abs(float(y)))
    ctx = payload.get("context", {})
    for key in ("Site_Parcel", "Sidewalk_Curb", "Context_Roads", "Context_Green"):
        for x, y, _ in ctx.get(key, {}).get("vertices", ()):
            max_xy = max(max_xy, abs(float(x)), abs(float(y)))
    for bld in ctx.get("Context_Buildings", ()):
        for x, y, _ in bld.get("vertices", ()):
            max_xy = max(max_xy, abs(float(x)), abs(float(y)))
    return max_xy


def _ensure_studio_lighting(ctx_col: bpy.types.Collection) -> None:
    """Configure the scene world with a packed/local studio HDRI, softbox key, and dark backdrop."""
    configure_studio_world(light_col=ctx_col)


def _auto_frame_camera(hero_col: bpy.types.Collection) -> None:
    bpy.context.view_layer.update()
    pts: list[Vector] = []
    for obj in hero_col.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))
    if not pts:
        return

    min_x = min(p.x for p in pts)
    max_x = max(p.x for p in pts)
    min_y = min(p.y for p in pts)
    max_y = max(p.y for p in pts)
    max_z = max(p.z for p in pts)

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    target = Vector((cx, cy, 0.42 * max_z))
    r = max(0.5 * math.hypot(max_x - min_x, max_y - min_y), max_z, 8.0)
    dist = max(3.25 * r, 34.0) * (60.0 / 45.0)

    hero_objs = set(hero_col.objects)
    depsgraph = bpy.context.view_layer.depsgraph
    scene = bpy.context.scene
    probe_pts = (
        target,
        Vector((cx, cy, 2.2)),
        Vector((cx, cy, 0.80 * max_z)),
    )
    directions = (
        Vector((0.72, -0.82, 0.48)),
        Vector((0.86, -0.40, 0.48)),
        Vector((-0.72, -0.82, 0.48)),
        Vector((0.72, 0.82, 0.48)),
        Vector((-0.72, 0.82, 0.48)),
        Vector((0.0, -1.0, 0.48)),
        Vector((1.0, 0.0, 0.48)),
        Vector((-1.0, 0.0, 0.48)),
    )

    best_score = -1
    chosen_pos: Vector | None = None
    for idx, d in enumerate(directions):
        cand = target + d.normalized() * dist
        # Prefer first clear angle when tied by biasing slightly on priority order
        hits_hero = 0
        blocked_close = False
        for pt in probe_pts:
            ray_vec = pt - cand
            ray_len = ray_vec.length
            hit, loc, _norm, _idx, hit_obj, _mat = scene.ray_cast(
                depsgraph,
                cand,
                ray_vec.normalized(),
                distance=ray_len,
            )
            if hit and hit_obj not in hero_objs:
                if (loc - cand).length < 4.0:
                    blocked_close = True
                    break
            else:
                hits_hero += 1
        if blocked_close:
            continue
        score = hits_hero * 10 - idx
        if score > best_score:
            best_score = score
            chosen_pos = cand

    if chosen_pos is None:
        chosen_pos = target + Vector((0.72, -0.82, 0.82)).normalized() * dist

    cam = scene.camera or bpy.data.objects.get("Camera")
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam

    clip_end = max(1000.0, dist * 10.0)
    cam.location = chosen_pos
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 60.0
    cam.data.clip_end = clip_end
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = (target - cam.location).length
    cam.data.dof.aperture_fstop = 8.0

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.clip_end = clip_end


def build_scene(
    json_path: Path,
    out_path: Path,
    *,
    hero_style: str = "curtain_frame",
    no_context_facades: bool = False,
    no_props: bool = False,
    no_road_markings: bool = False,
) -> None:
    current_blend = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    if current_blend is not None and out_path.resolve() == current_blend:
        raise ValueError(f"Refusing to overwrite template file: {out_path}")

    payload = json.loads(json_path.read_text())
    meta = payload.get("metadata", {})
    lib = ensure_maquette_library()
    mats = lib["materials"]

    scene_col = bpy.context.scene.collection
    hero_col = _ensure_collection("01_Hero_Building", scene_col)
    ctx_col = _ensure_collection("02_Urban_Context", scene_col)
    bld_col = _ensure_collection("Context_Buildings", ctx_col)

    hero = payload.get("hero", {})
    mass = hero.get("Building_Mass", {})
    parapets = hero.get("Hero_Parapets", {})
    slabs = hero.get("Hero_Slabs", {})
    gf_plinth = hero.get("Hero_GF_Plinth", {})
    cores = hero.get("Vertical_Cores", {})

    hero_obj = _create_mesh_object(
        "Building_Mass",
        mass.get("vertices", []),
        mass.get("faces", []),
        hero_col,
        mats["Mat_Hero_White"],
        recalc_normals=False,
    )
    _create_mesh_object(
        "Hero_Parapets",
        parapets.get("vertices", []),
        parapets.get("faces", []),
        hero_col,
        mats["Mat_Hero_White"],
    )
    if slabs.get("faces"):
        _create_mesh_object(
            "Hero_Slabs",
            slabs.get("vertices", []),
            slabs["faces"],
            hero_col,
            mats["Mat_Hero_Slab"],
        )
    if gf_plinth.get("faces"):
        _create_mesh_object(
            "Hero_GF_Plinth",
            gf_plinth.get("vertices", []),
            gf_plinth["faces"],
            hero_col,
            mats["Mat_Hero_Accent"],
        )
    _create_mesh_object(
        "Vertical_Cores",
        cores.get("vertices", []),
        cores.get("faces", []),
        hero_col,
        mats["Mat_Core_Blush"],
        recalc_normals=False,
    )
    _apply_hero_facades(hero_obj, meta, hero_style)

    half_ground = float(
        meta.get("boardHalf") or 0.5 * max(240.0, 2.2 * _max_xy_extent(payload))
    )
    ground_verts = [
        [-half_ground, -half_ground, 0.0],
        [half_ground, -half_ground, 0.0],
        [half_ground, half_ground, 0.0],
        [-half_ground, half_ground, 0.0],
    ]
    _create_mesh_object(
        "Ground_Plane",
        ground_verts,
        [[0, 1, 2, 3]],
        ctx_col,
        mats["Mat_Context_Maquette"],
    )

    ctx = payload.get("context", {})
    for key, mat_name in (
        ("Site_Parcel", "Mat_Sidewalk_Maquette"),
        ("Sidewalk_Curb", "Mat_Sidewalk_Maquette"),
        ("Context_Roads", "Mat_Road_Maquette"),
        ("Road_Markings", "Mat_Prop_White"),
        ("Context_Green", "Mat_Green_Maquette"),
        ("Context_Parapets", "Mat_Context_Maquette"),
    ):
        if no_road_markings and key == "Road_Markings":
            continue
        mesh_dict = ctx.get(key, {})
        if mesh_dict.get("faces"):
            _create_mesh_object(
                key,
                mesh_dict.get("vertices", []),
                mesh_dict["faces"],
                ctx_col,
                mats[mat_name],
            )

    site_hash = sum(ord(ch) for ch in str(meta.get("siteId", "site")))
    for idx, bld in enumerate(ctx.get("Context_Buildings", ())):
        if not bld.get("faces"):
            continue
        c_obj = _create_mesh_object(
            bld["name"],
            bld.get("vertices", []),
            bld["faces"],
            bld_col,
            mats["Mat_Context_Maquette"],
        )
        if not no_context_facades:
            _apply_context_facades(
                c_obj, bld, idx, site_hash, mats["Mat_Context_Maquette"]
            )

    if not no_props:
        _place_street_props(
            payload.get("street_props", {}),
            scene_col,
            seed=int(meta.get("evalSeed", 9003)),
        )

    _ensure_studio_lighting(ctx_col)
    _auto_frame_camera(hero_col)

    for ng in bpy.data.node_groups:
        ng.use_fake_user = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    print(f"Saved Blender scene ({hero_style}) -> {out_path}")


def main() -> None:
    args = _parse_args()
    build_scene(
        args.input,
        args.output,
        hero_style=args.hero_style,
        no_context_facades=args.no_context_facades,
        no_props=args.no_props,
        no_road_markings=args.no_road_markings,
    )


if __name__ == "__main__":
    main()
