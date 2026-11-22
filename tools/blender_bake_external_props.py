"""Bake Traffiq vehicles/street elements and Casual Humans into Template.blend as maquette props."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Euler, Vector

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from blender_maquette_library import _ensure_collection, _ensure_materials  # noqa: E402

TRAFFIQ_MODELS = Path(
    os.environ.get(
        "VOXELFORM_TRAFFIQ_DIR",
        Path.home() / "Documents/3D/Assets/Polygoniq/traffiq_full/blends/models",
    )
)
CASUAL_HUMANS_SUB = Path(
    os.environ.get(
        "VOXELFORM_CROWDS_DIR",
        Path.home()
        / "Documents/3D/Assets/Procedural Crowds/Procedural_Crowds_Assets_v2.0.0/Individuals/Casual Humans/SubCategories",
    )
)

SKIP_NAME_TOKENS = (
    "wgt",
    "sensor",
    "collider",
    "license-plate",
    "license_plate",
    "shadow",
)

# (prop_name, relative_path, collection_name_or_None, category, body_mat_name, rot_z_deg)
TRAFFIQ_SPECS: tuple[tuple[str, str, str | None, str, str, float], ...] = (
    (
        "Prop_Car_Taxi",
        "service-vehicles/tq_Car_Ford_Crown-Victoria-Taxi-1998.blend",
        "tq_Car_Ford_Crown-Victoria-Taxi-1998",
        "vehicle",
        "Mat_Car_Accent",
        90.0,
    ),
    (
        "Prop_Bus_SOR",
        "public-transport/tq_Bus_SOR_NB-18-2008.blend",
        "tq_Bus_SOR_NB-18-2008",
        "vehicle",
        "Mat_Prop_White",
        90.0,
    ),
    (
        "Prop_Car_Savana",
        "utility-vehicles/tq_Car_GMC_Savana-Cargo-2022.blend",
        "tq_Car_GMC_Savana-Cargo-2022",
        "vehicle",
        "Mat_Prop_Grey",
        90.0,
    ),
    (
        "Prop_Car_Fiat500",
        "contemporary-cars/tq_Car_Fiat_500-2008.blend",
        "tq_Car_Fiat_500-2008",
        "vehicle",
        "Mat_Prop_White",
        90.0,
    ),
    (
        "Prop_Car_Tesla3",
        "contemporary-cars/tq_Car_Tesla_Model-3-2022.blend",
        "tq_Car_Tesla_Model-3-2022",
        "vehicle",
        "Mat_Prop_White",
        90.0,
    ),
    (
        "Prop_StreetLamp",
        "street-elements/tq_StreetElement_Lamp_Modern-Circle-860cm.blend",
        "tq_StreetElement_Lamp_Modern-Circle-860cm",
        "street_pole",
        "Mat_Dark",
        180.0,
    ),
    (
        "Prop_TrafficLight",
        "street-elements/tq_StreetElement_Traffic-Light_3+P-440cm.blend",
        "tq_StreetElement_Traffic-Light_3+P-440cm",
        "street_pole",
        "Mat_Dark",
        180.0,
    ),
    (
        "Prop_BusStop",
        "street-elements/tq_StreetElement_Bus-Stop_Iron-Modern.blend",
        "tq_StreetElement_Bus-Stop_Iron-Modern",
        "bus_stop",
        "Mat_Dark",
        0.0,
    ),
)

CASUAL_HUMAN_SPECS: tuple[tuple[str, str, float], ...] = (
    ("Prop_Human_00", "Men/Low/Man 05 Pose/Man_05_Pose.blend", 0.0),
    ("Prop_Human_01", "Men/Low/Man 07 Pose/Man_07_Pose.blend", 0.0),
    ("Prop_Human_02", "Men/Low/Man 09 Pose/Man_09_Pose.blend", 0.0),
    ("Prop_Human_03", "Men/Low/Man 11 Pose/Man_11_Pose.blend", 0.0),
    ("Prop_Human_04", "Women/Low/Woman 06 Pose/Woman_06_Pose.blend", 0.0),
    ("Prop_Human_05", "Women/Low/Woman 07 Pose/Woman_07_Pose.blend", 0.0),
)


def _collect_nested_objects(col: bpy.types.Collection) -> set[bpy.types.Object]:
    seen_cols: set[bpy.types.Collection] = set()
    objs: set[bpy.types.Object] = set()

    def _walk(c: bpy.types.Collection):
        if c in seen_cols:
            return
        seen_cols.add(c)
        for o in c.all_objects:
            objs.add(o)
            if (
                o.type == "EMPTY"
                and o.instance_type == "COLLECTION"
                and o.instance_collection is not None
            ):
                _walk(o.instance_collection)

    _walk(col)
    return objs


def _resolve_maquette_mat(
    orig_mat: bpy.types.Material | None,
    obj_name: str,
    category: str,
    body_mat: bpy.types.Material,
    mats: dict[str, bpy.types.Material],
) -> bpy.types.Material:
    if category == "human":
        return mats["Humans"]

    mname = orig_mat.name.lower() if orig_mat else ""
    oname = obj_name.lower()

    if category == "street_pole":
        if "glass" in mname or "emission" in mname or "light" in mname:
            return mats["Mat_Prop_White"]
        return mats["Mat_Dark"]

    if category == "bus_stop":
        if "glass" in mname:
            return mats["Glass"]
        return mats["Mat_Dark"]

    # Vehicle: keep clean Glass windows, dark charcoal wheels/tires/trim/interior, and cohesive painted body
    if "glass" in mname or "windshield" in mname or "window" in mname:
        return mats["Glass"]
    if (
        "wheel" in oname
        or "brake" in oname
        or any(
            k in mname
            for k in (
                "tire",
                "tyre",
                "rubber",
                "black",
                "dark",
                "interior",
                "seat",
                "grill",
                "undercarriage",
                "chassis",
                "plastic",
            )
        )
    ):
        return mats["Mat_Dark"]
    if (
        any(k in mname for k in ("light", "lamp", "emission", "mirror"))
        and "paint" not in mname
    ):
        return mats["Mat_Prop_White"]
    return body_mat


def _bake_single_asset(
    blend_path: Path,
    col_name: str | None,
    target_name: str,
    target_col: bpy.types.Collection,
    *,
    category: str,
    body_mat: bpy.types.Material,
    mats: dict[str, bpy.types.Material],
    rot_z_deg: float,
    max_polys: int = 12000,
) -> dict:
    if not blend_path.is_file():
        raise FileNotFoundError(f"Asset file not found: {blend_path}")

    with bpy.data.libraries.load(str(blend_path), link=False) as (df, dt):
        if col_name and col_name in df.collections:
            dt.collections = [col_name]
        elif df.collections:
            dt.collections = [df.collections[0]]
        else:
            raise RuntimeError(f"No collections found in {blend_path}")

    src_col = dt.collections[0]
    tmp_col = bpy.data.collections.new(f"_bake_tmp_{target_name}")
    bpy.context.scene.collection.children.link(tmp_col)
    tmp_col.children.link(src_col)
    bpy.context.view_layer.update()

    allowed_objs = _collect_nested_objects(src_col)
    deps = bpy.context.evaluated_depsgraph_get()

    bm = bmesh.new()
    out_mats: list[bpy.types.Material] = []

    def _slot_idx(m: bpy.types.Material) -> int:
        if m not in out_mats:
            out_mats.append(m)
        return out_mats.index(m)

    for inst in deps.object_instances:
        obj = inst.object
        if obj.type != "MESH":
            continue
        if inst.is_instance:
            parent_ok = inst.parent in allowed_objs
            inst_ok = obj.original in allowed_objs
            if not (parent_ok or inst_ok):
                continue
        elif obj.original not in allowed_objs:
            continue

        oname_low = obj.name.lower()
        if any(tok in oname_low for tok in SKIP_NAME_TOKENS):
            continue

        me = obj.to_mesh(preserve_all_data_layers=False, depsgraph=deps)
        if me is None or len(me.polygons) == 0:
            if me is not None:
                obj.to_mesh_clear()
            continue

        xform = inst.matrix_world.copy()
        vert_map: dict[int, bmesh.types.BMVert] = {}
        for vi, mv in enumerate(me.vertices):
            vert_map[vi] = bm.verts.new(xform @ mv.co)

        slot_map: dict[int, int] = {}
        for si in range(max(1, len(me.materials))):
            orig_m = me.materials[si] if si < len(me.materials) else None
            maq_m = _resolve_maquette_mat(orig_m, obj.name, category, body_mat, mats)
            slot_map[si] = _slot_idx(maq_m)

        for poly in me.polygons:
            if len(poly.vertices) < 3:
                continue
            try:
                f = bm.faces.new([vert_map[vi] for vi in poly.vertices])
                f.material_index = slot_map.get(poly.material_index, 0)
                f.smooth = category in ("vehicle", "human")
            except ValueError:
                pass
        obj.to_mesh_clear()

    if len(bm.verts) == 0:
        bm.free()
        raise RuntimeError(f"Empty baked mesh for {target_name} from {blend_path}")

    # Rotate into canonical orientation (+X forward for vehicles, +Y facing front/arm for props/humans)
    if abs(rot_z_deg) > 1e-4:
        bmesh.ops.rotate(
            bm,
            verts=bm.verts,
            cent=(0.0, 0.0, 0.0),
            matrix=Euler((0.0, 0.0, math.radians(rot_z_deg))).to_matrix(),
        )

    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    min_z, max_z = min(zs), max(zs)

    if category == "street_pole":
        z_cut = min_z + 0.15 * max(0.1, max_z - min_z)
        base_verts = [v for v in bm.verts if v.co.z <= z_cut] or bm.verts
        cx = 0.5 * (min(v.co.x for v in base_verts) + max(v.co.x for v in base_verts))
        cy = 0.5 * (min(v.co.y for v in base_verts) + max(v.co.y for v in base_verts))
    elif category == "bus_stop":
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
    else:
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))

    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector((-cx, -cy, -min_z)))

    out_mesh = bpy.data.meshes.new(target_name)
    bm.to_mesh(out_mesh)
    bm.free()
    for m in out_mats:
        out_mesh.materials.append(m)

    existing = bpy.data.objects.get(target_name)
    if existing is not None:
        old_mesh = existing.data if existing.type == "MESH" else None
        bpy.data.objects.remove(existing, do_unlink=True)
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    out_obj = bpy.data.objects.new(target_name, out_mesh)
    out_obj.location = (0.0, 0.0, 0.0)
    target_col.objects.link(out_obj)

    # Light collapse decimation + sharp edge split so vehicles stay smooth & fast
    if category == "vehicle":
        if len(out_mesh.polygons) > max_polys:
            dec = out_obj.modifiers.new("Decimate", "DECIMATE")
            dec.ratio = max(0.10, float(max_polys) / float(len(out_mesh.polygons)))
            dec.use_collapse_triangulate = True
            with bpy.context.temp_override(object=out_obj, active_object=out_obj):
                bpy.ops.object.modifier_apply(modifier=dec.name)
        es = out_obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
        es.split_angle = math.radians(35.0)
        with bpy.context.temp_override(object=out_obj, active_object=out_obj):
            bpy.ops.object.modifier_apply(modifier=es.name)

    # Clean up appended temporary objects/collections cleanly
    objs_to_rm = _collect_nested_objects(src_col)
    cols_to_rm: set[bpy.types.Collection] = {tmp_col, src_col}

    def _gather_cols(c: bpy.types.Collection):
        for child in c.children:
            cols_to_rm.add(child)
            _gather_cols(child)

    _gather_cols(src_col)
    if tmp_col.name in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.unlink(tmp_col)
    for o in objs_to_rm:
        bpy.data.objects.remove(o, do_unlink=True)
    for c in cols_to_rm:
        if c.name in bpy.data.collections:
            bpy.data.collections.remove(c)

    mn = [round(min(v.co[i] for v in out_obj.data.vertices), 2) for i in range(3)]
    mx = [round(max(v.co[i] for v in out_obj.data.vertices), 2) for i in range(3)]
    return {
        "name": target_name,
        "verts": len(out_obj.data.vertices),
        "polys": len(out_obj.data.polygons),
        "min": mn,
        "max": mx,
        "size": [round(mx[i] - mn[i], 2) for i in range(3)],
    }


def bake_external_props_into_template(template_path: Path | None = None) -> list[dict]:
    """Bake curated Traffiq + Casual Humans assets cleanly into the current Blend scene."""
    scene_col = bpy.context.scene.collection
    assets_col = _ensure_collection("00_Maquette_Assets", scene_col)
    assets_col.hide_viewport = False
    assets_col.hide_render = True
    props_col = _ensure_collection("Maquette_Props", assets_col)
    props_col.hide_viewport = False
    props_col.hide_render = True

    mats = _ensure_materials()
    stats: list[dict] = []

    for name, rel_path, col_name, cat, mat_key, rot_z in TRAFFIQ_SPECS:
        info = _bake_single_asset(
            TRAFFIQ_MODELS / rel_path,
            col_name,
            name,
            props_col,
            category=cat,
            body_mat=mats[mat_key],
            mats=mats,
            rot_z_deg=rot_z,
            max_polys=16000,
        )
        stats.append(info)

    for name, rel_path, rot_z in CASUAL_HUMAN_SPECS:
        info = _bake_single_asset(
            CASUAL_HUMANS_SUB / rel_path,
            None,
            name,
            props_col,
            category="human",
            body_mat=mats["Humans"],
            mats=mats,
            rot_z_deg=rot_z,
        )
        stats.append(info)

    props_col.hide_viewport = True
    assets_col.hide_viewport = True
    props_col["baked_external_props"] = True
    for o in list(props_col.objects):
        if o.type == "MESH" and o.data:
            o.data.use_fake_user = True
    for ng in bpy.data.node_groups:
        ng.use_fake_user = True

    bpy.ops.outliner.orphans_purge(
        do_local_ids=True, do_linked_ids=True, do_recursive=True
    )

    save_target = (
        Path(template_path)
        if template_path is not None
        else (Path(bpy.data.filepath) if bpy.data.filepath else None)
    )
    if save_target is not None:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(save_target))
    return stats
