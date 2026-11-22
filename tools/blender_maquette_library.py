"""Premade architectural-maquette materials, Geometry Node groups, facade modules, and props."""

from __future__ import annotations

from pathlib import Path
import re

import bmesh
import bpy

KEEP_NODE_GROUPS: frozenset[str] = frozenset(
    {
        "Panelize Facade",
        "Panelize Building",
        "Assign Panel ID",
        "Panel Swap",
        "Glass Facade",
        "Inset Face",
        "Simple Extrusion",
        "Hero Curtain Wall",
        "Scatter Humans On Paths",
    }
)

# Hero presets selectable from CLI (--hero-style)
HERO_STYLES: dict[str, dict] = {
    "curtain_frame": {
        "mode": "curtain_wall",
        "grid_module": 2,
        "gf_length": 2.05,
        "upper_length": 2.05,
        "opaque_step": 5,
    },
    "shadow_box": {
        "mode": "panel_swap",
        "collection": "HERO_ShadowBox",
        "grid_module": 3,
        "gf_length": 3.05,
        "upper_length": 3.05,
        "ground_id": 0,
        "ground_narrow_id": 1,
        "narrow_id": 2,
        "upper_min": 3,
        "upper_max": 5,
        "opaque_id": 6,
        "opaque_step": 4,
    },
    "ribbon_louver": {
        "mode": "panel_swap",
        "collection": "HERO_RibbonLouver",
        "grid_module": 3,
        "gf_length": 3.05,
        "upper_length": 3.05,
        "ground_id": 0,
        "ground_narrow_id": 1,
        "narrow_id": 2,
        "upper_min": 3,
        "upper_max": 4,
        "opaque_id": 5,
        "opaque_step": 4,
    },
}

# Context presets by building height tier (collection_name or None for Context Curtain Mono)
# Tuples: (collection_name | None, gf_len, upper_len, target_upper_floor_h, upper_max_id)
CONTEXT_TIERS: dict[str, tuple[tuple[str | None, float, float, float, int], ...]] = {
    "low": (
        ("CTX_Low_Storefront", 2.8, 2.8, 3.6, 1),
        ("CTX_Low_Punched", 2.4, 2.4, 3.4, 2),
    ),
    "mid": (
        ("CTX_Mid_PunchedGrid", 2.5, 2.5, 3.6, 2),
        ("CTX_Mid_PairedBays", 2.8, 2.8, 3.6, 2),
        ("CTX_Mid_HorizontalBand", 2.6, 2.6, 3.6, 1),
    ),
    "high": (
        ("CTX_High_FlutedTower", 2.4, 2.4, 3.6, 1),
        ("CTX_High_GridTower", 2.4, 2.4, 3.6, 2),
        (None, 2.4, 2.4, 3.6, 0),
    ),
}


def set_mod_input(mod: bpy.types.Modifier, socket_name: str, value) -> None:
    """Assign a Geometry Nodes modifier input socket across Blender 4.x/5.3."""
    ng = mod.node_group
    if ng is None:
        raise ValueError(f"Modifier {mod.name!r} has no node_group")
    for item in ng.interface.items_tree:
        if (
            item.item_type == "SOCKET"
            and item.in_out == "INPUT"
            and item.name == socket_name
        ):
            ident = item.identifier
            if (
                hasattr(mod, "properties")
                and hasattr(mod.properties, "inputs")
                and ident in mod.properties.inputs
            ):
                mod.properties.inputs[ident]["value"] = value
            else:
                mod[ident] = value
            return
    raise KeyError(f"Input socket {socket_name!r} not found in node group {ng.name!r}")


def _in_sock(node: bpy.types.Node, name: str) -> bpy.types.NodeSocket:
    enabled = [s for s in node.inputs if s.name == name and s.enabled]
    if enabled:
        return enabled[0]
    matches = [s for s in node.inputs if s.name == name]
    if matches:
        return matches[0]
    raise KeyError(f"Input socket {name!r} not found on node {node.name!r}")


def _out_sock(node: bpy.types.Node, name: str) -> bpy.types.NodeSocket:
    enabled = [s for s in node.outputs if s.name == name and s.enabled]
    if enabled:
        return enabled[0]
    matches = [s for s in node.outputs if s.name == name]
    if matches:
        return matches[0]
    raise KeyError(f"Output socket {name!r} not found on node {node.name!r}")


def _ensure_collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        parent.children.link(col)
    elif col.name not in parent.children:
        parent.children.link(col)
    return col


def _clear_collection_objects(col: bpy.types.Collection) -> None:
    for obj in list(col.objects):
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _make_principled_material(
    name: str,
    rgba: tuple[float, float, float, float],
    roughness: float,
    *,
    metallic: float = 0.0,
    specular: float = 0.25,
    coat: float = 0.0,
    coat_roughness: float = 0.08,
    ao_factor: float = 0.0,
    ao_dist: float = 2.0,
    grain_scale: float = 0.0,
    bump_strength: float = 0.0,
    roughness_var: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    if coat > 0.0 and "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = coat_roughness

    if ao_factor > 1e-4:
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = 8
        ao.inputs["Distance"].default_value = ao_dist
        ao.inputs["Color"].default_value = rgba
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.clamp_result = True
        mix.inputs["Factor"].default_value = ao_factor
        rgba_in = [s for s in mix.inputs if s.type == "RGBA"]
        rgba_out = [s for s in mix.outputs if s.type == "RGBA"][0]
        rgba_in[0].default_value = rgba
        nt.links.new(ao.outputs["AO"], rgba_in[1])
        nt.links.new(rgba_out, bsdf.inputs["Base Color"])

    if grain_scale > 0.0 and (bump_strength > 1e-4 or roughness_var > 1e-4):
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = grain_scale
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.65
        nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

        if bump_strength > 1e-4:
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = bump_strength
            bump.inputs["Distance"].default_value = 0.008
            nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        if roughness_var > 1e-4:
            map_range = nt.nodes.new("ShaderNodeMapRange")
            map_range.clamp = True
            map_range.inputs["From Min"].default_value = 0.25
            map_range.inputs["From Max"].default_value = 0.75
            map_range.inputs["To Min"].default_value = max(
                0.05, roughness - roughness_var
            )
            map_range.inputs["To Max"].default_value = min(
                0.98, roughness + roughness_var
            )
            nt.links.new(noise.outputs["Fac"], map_range.inputs["Value"])
            nt.links.new(map_range.outputs["Result"], bsdf.inputs["Roughness"])

    mat.diffuse_color = rgba
    mat.use_fake_user = True
    return mat


def _make_sidewalk_pavement_material(
    name: str = "Mat_Sidewalk_Maquette",
) -> bpy.types.Material:
    """Create the paving-tile sidewalk shader matching the user's studio maquette look."""
    rgba = (0.90, 0.89, 0.865, 1.0)
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.86
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.18
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 8
    ao.inputs["Distance"].default_value = 2.0
    ao.inputs["Color"].default_value = rgba

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mix_ao.clamp_factor = True
    mix_ao.clamp_result = True
    mix_ao.inputs["Factor"].default_value = 0.25
    mix_ao_in = [s for s in mix_ao.inputs if s.type == "RGBA"]
    mix_ao_out = [s for s in mix_ao.outputs if s.type == "RGBA"][0]
    mix_ao_in[0].default_value = rgba
    nt.links.new(ao.outputs["AO"], mix_ao_in[1])

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.5
    brick.offset_frequency = 2
    brick.squash = 1.0
    brick.squash_frequency = 2
    brick.inputs["Color1"].default_value = (0.8, 0.8, 0.8, 1.0)
    brick.inputs["Color2"].default_value = (0.2, 0.2, 0.2, 1.0)
    brick.inputs["Mortar"].default_value = (0.0, 0.0, 0.0, 1.0)
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Mortar Size"].default_value = 0.005
    brick.inputs["Mortar Smooth"].default_value = 0.5
    brick.inputs["Bias"].default_value = 0.0
    brick.inputs["Brick Width"].default_value = 1.0
    brick.inputs["Row Height"].default_value = 2.0
    nt.links.new(tex_coord.outputs["Object"], brick.inputs["Vector"])

    mix_brick = nt.nodes.new("ShaderNodeMix")
    mix_brick.data_type = "RGBA"
    mix_brick.blend_type = "MULTIPLY"
    mix_brick.clamp_factor = True
    mix_brick.clamp_result = False
    mix_brick.inputs["Factor"].default_value = 0.25
    mix_brick_in = [s for s in mix_brick.inputs if s.type == "RGBA"]
    mix_brick_out = [s for s in mix_brick.outputs if s.type == "RGBA"][0]
    nt.links.new(mix_ao_out, mix_brick_in[0])
    nt.links.new(brick.outputs["Color"], mix_brick_in[1])
    nt.links.new(mix_brick_out, bsdf.inputs["Base Color"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.05
    bump.inputs["Filter Width"].default_value = 1.0
    nt.links.new(brick.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    mat.diffuse_color = rgba
    mat.use_fake_user = True
    return mat


def _ensure_materials() -> dict[str, bpy.types.Material]:
    specs: dict[str, tuple[tuple[float, float, float, float], float, dict]] = {
        "Mat_Hero_White": (
            (0.9331, 0.9185, 0.8894, 1.0),
            0.44,
            {
                "specular": 0.42,
                "coat": 0.18,
                "coat_roughness": 0.22,
                "ao_factor": 0.42,
                "ao_dist": 2.4,
                "grain_scale": 38.0,
                "bump_strength": 0.045,
                "roughness_var": 0.08,
            },
        ),
        "Mat_Hero_Slab": (
            (0.9331, 0.8809, 0.8158, 1.0),
            0.90,
            {
                "specular": 0.12,
                "ao_factor": 0.58,
                "ao_dist": 3.4,
                "grain_scale": 55.0,
                "bump_strength": 0.04,
            },
        ),
        "Mat_Hero_Accent": (
            (0.85, 0.57, 0.49, 1.0),
            0.85,
            {
                "specular": 0.18,
                "ao_factor": 0.32,
                "ao_dist": 2.5,
                "grain_scale": 160.0,
                "bump_strength": 0.04,
            },
        ),
        "Mat_Core_Blush": (
            (0.4465, 0.2762, 0.2486, 1.0),
            0.85,
            {
                "specular": 0.18,
                "ao_factor": 0.28,
                "ao_dist": 2.0,
                "grain_scale": 160.0,
                "bump_strength": 0.04,
            },
        ),
        "Mat_Context_Maquette": (
            (0.4967, 0.5064, 0.5225, 1.0),
            0.94,
            {
                "specular": 0.08,
                "ao_factor": 0.42,
                "ao_dist": 2.8,
                "grain_scale": 50.0,
                "bump_strength": 0.04,
            },
        ),
        "Mat_Road_Maquette": (
            (0.085, 0.09, 0.10, 1.0),
            0.34,
            {
                "specular": 0.55,
                "ao_factor": 0.25,
                "grain_scale": 95.0,
                "bump_strength": 0.05,
            },
        ),
        "Mat_Green_Maquette": (
            (0.76, 0.79, 0.74, 1.0),
            0.88,
            {
                "specular": 0.15,
                "ao_factor": 0.25,
                "grain_scale": 150.0,
                "bump_strength": 0.03,
            },
        ),
        "Humans": ((0.6921, 0.6921, 0.6921, 1.0), 0.50, {"specular": 0.50}),
        "Mat_Prop_White": ((0.94, 0.93, 0.91, 1.0), 0.82, {"specular": 0.20}),
        "Mat_Prop_Grey": (
            (0.62, 0.63, 0.65, 1.0),
            0.32,
            {"specular": 0.45, "coat": 0.35},
        ),
        "Mat_Car_Accent": (
            (0.92, 0.67, 0.08, 1.0),
            0.22,
            {"specular": 0.50, "coat": 0.55},
        ),
        "Mat_Dark": (
            (0.14, 0.145, 0.155, 1.0),
            0.45,
            {"metallic": 0.10, "specular": 0.35},
        ),
    }
    mats = {
        k: _make_principled_material(k, rgba, rough, **kw)
        for k, (rgba, rough, kw) in specs.items()
    }
    mats["Mat_Sidewalk_Maquette"] = _make_sidewalk_pavement_material()
    glass = bpy.data.materials.get("Glass") or _make_principled_material(
        "Glass", (0.76, 0.84, 0.88, 1.0), 0.08, specular=0.5
    )
    glass.use_fake_user = True
    mats["Glass"] = glass
    return mats


def _cleanup_duplicates_and_unused_geonodes(
    mats: dict[str, bpy.types.Material],
) -> None:
    """Remap duplicate/retired materials & node groups and prune unnecessary Geometry Node groups."""
    remap_mats: dict[str, bpy.types.Material] = {
        "Black Plastic": mats["Mat_Dark"],
        "Black Plastic.001": mats["Mat_Dark"],
        "Mat_Prop_Lamp": mats["Mat_Dark"],
        "Glass.001": mats["Glass"],
        "Mat_Hero_Interior": mats["Mat_Hero_Slab"],
        "Mat_Base_Board": mats["Mat_Context_Maquette"],
        "Mat_Road_Marking": mats["Mat_Prop_White"],
        "Mat_Humans": mats["Humans"],
    }
    for old_name, target_mat in remap_mats.items():
        old_mat = bpy.data.materials.get(old_name)
        if old_mat is not None and old_mat != target_mat:
            old_mat.user_remap(target_mat)
            bpy.data.materials.remove(old_mat, do_unlink=True)

    suffix_re = re.compile(r"^(.+)\.\d{3}$")
    for mat in list(bpy.data.materials):
        m = suffix_re.match(mat.name)
        if m:
            base = bpy.data.materials.get(m.group(1))
            if base is not None and base != mat:
                mat.user_remap(base)
                bpy.data.materials.remove(mat, do_unlink=True)

    for ng in list(bpy.data.node_groups):
        m = suffix_re.match(ng.name)
        if m:
            base = bpy.data.node_groups.get(m.group(1))
            if base is not None and base != ng and base.type == ng.type:
                ng.user_remap(base)
                bpy.data.node_groups.remove(ng, do_unlink=True)

    for ng in list(bpy.data.node_groups):
        if ng.name in KEEP_NODE_GROUPS:
            ng.use_fake_user = True
            continue
        if ng.type == "GEOMETRY":
            ng.use_fake_user = False
            if (
                ng.name.startswith("Auto Smooth")
                or ng.name in {"Context Curtain Mono", "Set Material"}
                or ng.users == 0
            ):
                bpy.data.node_groups.remove(ng, do_unlink=True)


def _recreate_geo_group(name: str) -> bpy.types.NodeTree:
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old, do_unlink=True)
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.use_fake_user = True
    return ng


def _build_panelize_building_group() -> bpy.types.NodeTree:
    """Wrap Template's `Panelize Facade` with distinct Ground Floor vs Upper Floor heights."""
    pf_tree = bpy.data.node_groups.get("Panelize Facade")
    if pf_tree is None:
        raise RuntimeError(
            'Template.blend is missing required "Panelize Facade" Geometry Node group'
        )

    ng = _recreate_geo_group("Panelize Building")
    ng.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    ng.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )

    s_gfh = ng.interface.new_socket(
        "GF Height", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    s_gfh.default_value = 4.2
    s_gfh.min_value = 0.5

    s_gfl = ng.interface.new_socket(
        "GF Length", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    s_gfl.default_value = 2.0
    s_gfl.min_value = 0.5

    s_uph = ng.interface.new_socket(
        "Upper Height", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    s_uph.default_value = 3.0005
    s_uph.min_value = 0.5

    s_upl = ng.interface.new_socket(
        "Upper Length", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    s_upl.default_value = 3.0
    s_upl.min_value = 0.5

    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")

    pos = ng.nodes.new("GeometryNodeInputPosition")
    sep_pos = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(pos, "Position"), _in_sock(sep_pos, "Vector"))

    cmp_z = ng.nodes.new("FunctionNodeCompare")
    cmp_z.data_type = "FLOAT"
    cmp_z.operation = "LESS_THAN"
    ng.links.new(_out_sock(sep_pos, "Z"), _in_sock(cmp_z, "A"))
    ng.links.new(_out_sock(gin, "GF Height"), _in_sock(cmp_z, "B"))

    norm = ng.nodes.new("GeometryNodeInputNormal")
    sep_n = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(norm, "True Normal"), _in_sock(sep_n, "Vector"))

    abs_nz = ng.nodes.new("ShaderNodeMath")
    abs_nz.operation = "ABSOLUTE"
    ng.links.new(_out_sock(sep_n, "Z"), _in_sock(abs_nz, "Value"))

    cmp_wall = ng.nodes.new("FunctionNodeCompare")
    cmp_wall.data_type = "FLOAT"
    cmp_wall.operation = "LESS_THAN"
    ng.links.new(_out_sock(abs_nz, "Value"), _in_sock(cmp_wall, "A"))
    _in_sock(cmp_wall, "B").default_value = 0.35

    is_gf = ng.nodes.new("FunctionNodeBooleanMath")
    is_gf.operation = "AND"
    ng.links.new(_out_sock(cmp_z, "Result"), is_gf.inputs[0])
    ng.links.new(_out_sock(cmp_wall, "Result"), is_gf.inputs[1])

    sep = ng.nodes.new("GeometryNodeSeparateGeometry")
    sep.domain = "FACE"
    ng.links.new(_out_sock(gin, "Geometry"), _in_sock(sep, "Geometry"))
    ng.links.new(_out_sock(is_gf, "Boolean"), _in_sock(sep, "Selection"))

    # Keep GF taller as one band by passing Floor Height slightly above GF Height
    add_eps = ng.nodes.new("ShaderNodeMath")
    add_eps.operation = "ADD"
    ng.links.new(_out_sock(gin, "GF Height"), add_eps.inputs[0])
    add_eps.inputs[1].default_value = 0.05

    pf_gf = ng.nodes.new("GeometryNodeGroup")
    pf_gf.node_tree = pf_tree
    ng.links.new(_out_sock(sep, "Selection"), _in_sock(pf_gf, "Geometry"))
    ng.links.new(_out_sock(add_eps, "Value"), _in_sock(pf_gf, "Floor Height"))
    ng.links.new(_out_sock(gin, "GF Length"), _in_sock(pf_gf, "Length"))

    # Shift upper geometry by -GF Height so Panelize Facade snaps from Z = GF Height, not Z = 0
    neg_gf = ng.nodes.new("ShaderNodeMath")
    neg_gf.operation = "MULTIPLY"
    ng.links.new(_out_sock(gin, "GF Height"), neg_gf.inputs[0])
    neg_gf.inputs[1].default_value = -1.0

    vec_down = ng.nodes.new("ShaderNodeCombineXYZ")
    ng.links.new(_out_sock(neg_gf, "Value"), _in_sock(vec_down, "Z"))

    shift_down = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(_out_sock(sep, "Inverted"), _in_sock(shift_down, "Geometry"))
    ng.links.new(_out_sock(vec_down, "Vector"), _in_sock(shift_down, "Offset"))

    pf_up = ng.nodes.new("GeometryNodeGroup")
    pf_up.node_tree = pf_tree
    ng.links.new(_out_sock(shift_down, "Geometry"), _in_sock(pf_up, "Geometry"))
    ng.links.new(_out_sock(gin, "Upper Height"), _in_sock(pf_up, "Floor Height"))
    ng.links.new(_out_sock(gin, "Upper Length"), _in_sock(pf_up, "Length"))

    vec_up = ng.nodes.new("ShaderNodeCombineXYZ")
    ng.links.new(_out_sock(gin, "GF Height"), _in_sock(vec_up, "Z"))

    shift_up = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(_out_sock(pf_up, "Geometry"), _in_sock(shift_up, "Geometry"))
    ng.links.new(_out_sock(vec_up, "Vector"), _in_sock(shift_up, "Offset"))

    join = ng.nodes.new("GeometryNodeJoinGeometry")
    ng.links.new(_out_sock(pf_gf, "Geometry"), _in_sock(join, "Geometry"))
    ng.links.new(_out_sock(shift_up, "Geometry"), _in_sock(join, "Geometry"))
    ng.links.new(_out_sock(join, "Geometry"), _in_sock(gout, "Geometry"))
    return ng


def _add_column_opaque_mask(
    ng: bpy.types.NodeTree,
    grid_sock: bpy.types.NodeSocket,
    step_sock: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Boolean face field selecting consistent full-height structural bays every N grid modules."""
    norm = ng.nodes.new("GeometryNodeInputNormal")
    sep_n = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(norm, "True Normal"), _in_sock(sep_n, "Vector"))

    abs_nx = ng.nodes.new("ShaderNodeMath")
    abs_nx.operation = "ABSOLUTE"
    ng.links.new(_out_sock(sep_n, "X"), _in_sock(abs_nx, "Value"))

    is_xwall = ng.nodes.new("FunctionNodeCompare")
    is_xwall.data_type = "FLOAT"
    is_xwall.operation = "GREATER_THAN"
    ng.links.new(_out_sock(abs_nx, "Value"), _in_sock(is_xwall, "A"))
    _in_sock(is_xwall, "B").default_value = 0.5

    pos = ng.nodes.new("GeometryNodeInputPosition")
    sep_p = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(pos, "Position"), _in_sock(sep_p, "Vector"))

    coord = ng.nodes.new("GeometryNodeSwitch")
    coord.input_type = "FLOAT"
    ng.links.new(_out_sock(is_xwall, "Result"), _in_sock(coord, "Switch"))
    ng.links.new(_out_sock(sep_p, "X"), _in_sock(coord, "False"))
    ng.links.new(_out_sock(sep_p, "Y"), _in_sock(coord, "True"))

    shift = ng.nodes.new("ShaderNodeMath")
    shift.operation = "ADD"
    ng.links.new(_out_sock(coord, "Output"), shift.inputs[0])
    shift.inputs[1].default_value = 0.15

    div = ng.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    ng.links.new(_out_sock(shift, "Value"), div.inputs[0])
    ng.links.new(grid_sock, div.inputs[1])

    col_idx = ng.nodes.new("ShaderNodeMath")
    col_idx.operation = "FLOOR"
    ng.links.new(_out_sock(div, "Value"), col_idx.inputs[0])

    mod = ng.nodes.new("ShaderNodeMath")
    mod.operation = "FLOORED_MODULO"
    ng.links.new(_out_sock(col_idx, "Value"), mod.inputs[0])
    ng.links.new(step_sock, mod.inputs[1])

    abs_mod = ng.nodes.new("ShaderNodeMath")
    abs_mod.operation = "ABSOLUTE"
    ng.links.new(_out_sock(mod, "Value"), _in_sock(abs_mod, "Value"))

    is_zero = ng.nodes.new("FunctionNodeCompare")
    is_zero.data_type = "FLOAT"
    is_zero.operation = "LESS_THAN"
    ng.links.new(_out_sock(abs_mod, "Value"), _in_sock(is_zero, "A"))
    _in_sock(is_zero, "B").default_value = 0.1

    is_step_pos = ng.nodes.new("FunctionNodeCompare")
    is_step_pos.data_type = "INT"
    is_step_pos.operation = "GREATER_THAN"
    ng.links.new(step_sock, _in_sock(is_step_pos, "A"))
    _in_sock(is_step_pos, "B").default_value = 0

    is_col = ng.nodes.new("FunctionNodeBooleanMath")
    is_col.operation = "AND"
    ng.links.new(_out_sock(is_zero, "Result"), is_col.inputs[0])
    ng.links.new(_out_sock(is_step_pos, "Result"), is_col.inputs[1])
    return _out_sock(is_col, "Boolean")


def _build_assign_panel_id_group() -> bpy.types.NodeTree:
    """Assign `Panel ID` attribute on faces, handling Ground Floor and structured opaque bays."""
    ng = _recreate_geo_group("Assign Panel ID")
    ng.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    ng.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )

    defaults_int = (
        ("Ground ID", 0),
        ("Ground Narrow ID", 1),
        ("Narrow ID", 2),
        ("Upper Min ID", 3),
        ("Upper Max ID", 5),
        ("Opaque ID", -1),
        ("Opaque Step", 0),
    )
    for name, dv in defaults_int:
        s = ng.interface.new_socket(name, in_out="INPUT", socket_type="NodeSocketInt")
        s.default_value = dv

    defaults_float = (
        ("Ground Max Z", 4.3),
        ("Min Face Area", 1.2),
        ("Narrow Max Area", 4.6),
        ("Grid Module", 3.0),
    )
    for name, dv in defaults_float:
        s = ng.interface.new_socket(name, in_out="INPUT", socket_type="NodeSocketFloat")
        s.default_value = dv

    s_seed = ng.interface.new_socket(
        "Seed", in_out="INPUT", socket_type="NodeSocketInt"
    )
    s_seed.default_value = 0

    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")

    norm = ng.nodes.new("GeometryNodeInputNormal")
    sep_n = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(norm, "True Normal"), _in_sock(sep_n, "Vector"))

    abs_nz = ng.nodes.new("ShaderNodeMath")
    abs_nz.operation = "ABSOLUTE"
    ng.links.new(_out_sock(sep_n, "Z"), _in_sock(abs_nz, "Value"))

    is_wall = ng.nodes.new("FunctionNodeCompare")
    is_wall.data_type = "FLOAT"
    is_wall.operation = "LESS_THAN"
    ng.links.new(_out_sock(abs_nz, "Value"), _in_sock(is_wall, "A"))
    _in_sock(is_wall, "B").default_value = 0.35

    area = ng.nodes.new("GeometryNodeInputMeshFaceArea")
    is_big = ng.nodes.new("FunctionNodeCompare")
    is_big.data_type = "FLOAT"
    is_big.operation = "GREATER_EQUAL"
    ng.links.new(_out_sock(area, "Area"), _in_sock(is_big, "A"))
    ng.links.new(_out_sock(gin, "Min Face Area"), _in_sock(is_big, "B"))

    is_valid = ng.nodes.new("FunctionNodeBooleanMath")
    is_valid.operation = "AND"
    ng.links.new(_out_sock(is_wall, "Result"), is_valid.inputs[0])
    ng.links.new(_out_sock(is_big, "Result"), is_valid.inputs[1])

    is_narrow = ng.nodes.new("FunctionNodeCompare")
    is_narrow.data_type = "FLOAT"
    is_narrow.operation = "LESS_THAN"
    ng.links.new(_out_sock(area, "Area"), _in_sock(is_narrow, "A"))
    ng.links.new(_out_sock(gin, "Narrow Max Area"), _in_sock(is_narrow, "B"))

    pos = ng.nodes.new("GeometryNodeInputPosition")
    sep_p = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(pos, "Position"), _in_sock(sep_p, "Vector"))

    is_ground = ng.nodes.new("FunctionNodeCompare")
    is_ground.data_type = "FLOAT"
    is_ground.operation = "LESS_THAN"
    ng.links.new(_out_sock(sep_p, "Z"), _in_sock(is_ground, "A"))
    ng.links.new(_out_sock(gin, "Ground Max Z"), _in_sock(is_ground, "B"))

    idx = ng.nodes.new("GeometryNodeInputIndex")
    rng = ng.nodes.new("FunctionNodeRandomValue")
    rng.data_type = "INT"
    ng.links.new(_out_sock(gin, "Upper Min ID"), _in_sock(rng, "Min"))
    ng.links.new(_out_sock(gin, "Upper Max ID"), _in_sock(rng, "Max"))
    ng.links.new(_out_sock(idx, "Index"), _in_sock(rng, "ID"))
    ng.links.new(_out_sock(gin, "Seed"), _in_sock(rng, "Seed"))

    has_op_id = ng.nodes.new("FunctionNodeCompare")
    has_op_id.data_type = "INT"
    has_op_id.operation = "GREATER_EQUAL"
    ng.links.new(_out_sock(gin, "Opaque ID"), _in_sock(has_op_id, "A"))
    _in_sock(has_op_id, "B").default_value = 0

    col_op_mask = _add_column_opaque_mask(
        ng, _out_sock(gin, "Grid Module"), _out_sock(gin, "Opaque Step")
    )
    use_op_id = ng.nodes.new("FunctionNodeBooleanMath")
    use_op_id.operation = "AND"
    ng.links.new(col_op_mask, use_op_id.inputs[0])
    ng.links.new(_out_sock(has_op_id, "Result"), use_op_id.inputs[1])

    sw_upper = ng.nodes.new("GeometryNodeSwitch")
    sw_upper.input_type = "INT"
    ng.links.new(_out_sock(use_op_id, "Boolean"), _in_sock(sw_upper, "Switch"))
    ng.links.new(_out_sock(rng, "Value"), _in_sock(sw_upper, "False"))
    ng.links.new(_out_sock(gin, "Opaque ID"), _in_sock(sw_upper, "True"))

    sw_wide = ng.nodes.new("GeometryNodeSwitch")
    sw_wide.input_type = "INT"
    ng.links.new(_out_sock(is_ground, "Result"), _in_sock(sw_wide, "Switch"))
    ng.links.new(_out_sock(sw_upper, "Output"), _in_sock(sw_wide, "False"))
    ng.links.new(_out_sock(gin, "Ground ID"), _in_sock(sw_wide, "True"))

    sw_narrow_id = ng.nodes.new("GeometryNodeSwitch")
    sw_narrow_id.input_type = "INT"
    ng.links.new(_out_sock(is_ground, "Result"), _in_sock(sw_narrow_id, "Switch"))
    ng.links.new(_out_sock(gin, "Narrow ID"), _in_sock(sw_narrow_id, "False"))
    ng.links.new(_out_sock(gin, "Ground Narrow ID"), _in_sock(sw_narrow_id, "True"))

    sw_narrow = ng.nodes.new("GeometryNodeSwitch")
    sw_narrow.input_type = "INT"
    ng.links.new(_out_sock(is_narrow, "Result"), _in_sock(sw_narrow, "Switch"))
    ng.links.new(_out_sock(sw_wide, "Output"), _in_sock(sw_narrow, "False"))
    ng.links.new(_out_sock(sw_narrow_id, "Output"), _in_sock(sw_narrow, "True"))

    sw_final = ng.nodes.new("GeometryNodeSwitch")
    sw_final.input_type = "INT"
    ng.links.new(_out_sock(is_valid, "Boolean"), _in_sock(sw_final, "Switch"))
    _in_sock(sw_final, "False").default_value = -1
    ng.links.new(_out_sock(sw_narrow, "Output"), _in_sock(sw_final, "True"))

    store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store.domain = "FACE"
    store.data_type = "INT"
    _in_sock(store, "Name").default_value = "Panel ID"
    ng.links.new(_out_sock(gin, "Geometry"), _in_sock(store, "Geometry"))
    ng.links.new(_out_sock(sw_final, "Output"), _in_sock(store, "Value"))
    ng.links.new(_out_sock(store, "Geometry"), _in_sock(gout, "Geometry"))
    return ng


def _link_glass_facade_width(
    node: bpy.types.Node, from_sock: bpy.types.NodeSocket
) -> None:
    for sock_name in ("Mullion Width", "Mulltion Width"):
        matches = [s for s in node.inputs if s.name == sock_name]
        if matches:
            node.id_data.links.new(from_sock, matches[0])
            return
    raise KeyError("Glass Facade Mullion Width socket not found")


def _build_hero_curtain_wall_group(
    mats: dict[str, bpy.types.Material],
) -> bpy.types.NodeTree:
    """Cohesive 2 m / 1 m curtain wall using Template's `Glass Facade`, recessed board, and opaque relief."""
    gf_tree = bpy.data.node_groups.get("Glass Facade")
    if gf_tree is None:
        raise RuntimeError(
            'Template.blend is missing required "Glass Facade" Geometry Node group'
        )
    se_tree = bpy.data.node_groups.get("Simple Extrusion")
    if se_tree is None:
        raise RuntimeError(
            'Template.blend is missing required "Simple Extrusion" Geometry Node group'
        )

    ng = _recreate_geo_group("Hero Curtain Wall")
    ng.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    ng.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )

    s_gfh = ng.interface.new_socket(
        "GF Height", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    s_gfh.default_value = 4.2

    for name, mat_key in (
        ("GF Mullion Material", "Mat_Hero_Accent"),
        ("Upper Mullion Material", "Mat_Hero_White"),
        ("Glass Material", "Glass"),
        ("Interior Material", "Mat_Hero_Slab"),
    ):
        s = ng.interface.new_socket(
            name, in_out="INPUT", socket_type="NodeSocketMaterial"
        )
        s.default_value = mats[mat_key]

    for name, dv in (
        ("Mullion Width", 0.085),
        ("Mullion Depth", 0.28),
        ("GF Mullion Width", 0.14),
        ("GF Mullion Depth", 0.34),
        ("Backing Depth", 0.36),
        ("Grid Module", 2.0),
    ):
        s = ng.interface.new_socket(name, in_out="INPUT", socket_type="NodeSocketFloat")
        s.default_value = dv

    s_step = ng.interface.new_socket(
        "Opaque Every N Bays", in_out="INPUT", socket_type="NodeSocketInt"
    )
    s_step.default_value = 5

    s_del_glass = ng.interface.new_socket(
        "Delete Glass", in_out="INPUT", socket_type="NodeSocketBool"
    )
    s_del_glass.default_value = False

    s_use_back = ng.interface.new_socket(
        "Use Backing", in_out="INPUT", socket_type="NodeSocketBool"
    )
    s_use_back.default_value = False

    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")

    # Keep roofs, soffits, and tiny slivers solid white; pass clean walls to Glass Facade
    norm = ng.nodes.new("GeometryNodeInputNormal")
    sep_n = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(norm, "True Normal"), _in_sock(sep_n, "Vector"))

    abs_nz = ng.nodes.new("ShaderNodeMath")
    abs_nz.operation = "ABSOLUTE"
    ng.links.new(_out_sock(sep_n, "Z"), _in_sock(abs_nz, "Value"))

    is_horiz = ng.nodes.new("FunctionNodeCompare")
    is_horiz.data_type = "FLOAT"
    is_horiz.operation = "GREATER_THAN"
    ng.links.new(_out_sock(abs_nz, "Value"), _in_sock(is_horiz, "A"))
    _in_sock(is_horiz, "B").default_value = 0.35

    area = ng.nodes.new("GeometryNodeInputMeshFaceArea")
    is_tiny = ng.nodes.new("FunctionNodeCompare")
    is_tiny.data_type = "FLOAT"
    is_tiny.operation = "LESS_THAN"
    ng.links.new(_out_sock(area, "Area"), _in_sock(is_tiny, "A"))
    _in_sock(is_tiny, "B").default_value = 0.30

    keep_solid = ng.nodes.new("FunctionNodeBooleanMath")
    keep_solid.operation = "OR"
    ng.links.new(_out_sock(is_horiz, "Result"), keep_solid.inputs[0])
    ng.links.new(_out_sock(is_tiny, "Result"), keep_solid.inputs[1])

    sep_top = ng.nodes.new("GeometryNodeSeparateGeometry")
    sep_top.domain = "FACE"
    ng.links.new(_out_sock(gin, "Geometry"), _in_sock(sep_top, "Geometry"))
    ng.links.new(_out_sock(keep_solid, "Boolean"), _in_sock(sep_top, "Selection"))

    set_roof_mat = ng.nodes.new("GeometryNodeSetMaterial")
    ng.links.new(_out_sock(sep_top, "Selection"), _in_sock(set_roof_mat, "Geometry"))
    ng.links.new(
        _out_sock(gin, "Upper Mullion Material"), _in_sock(set_roof_mat, "Material")
    )

    # Recessed interior shadow board behind glazing
    neg_back = ng.nodes.new("ShaderNodeMath")
    neg_back.operation = "MULTIPLY"
    ng.links.new(_out_sock(gin, "Backing Depth"), neg_back.inputs[0])
    neg_back.inputs[1].default_value = -1.0

    norm_b = ng.nodes.new("GeometryNodeInputNormal")
    scale_b = ng.nodes.new("ShaderNodeVectorMath")
    scale_b.operation = "SCALE"
    ng.links.new(_out_sock(norm_b, "True Normal"), _in_sock(scale_b, "Vector"))
    ng.links.new(_out_sock(neg_back, "Value"), _in_sock(scale_b, "Scale"))

    pos_back = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(_out_sock(sep_top, "Inverted"), _in_sock(pos_back, "Geometry"))
    ng.links.new(_out_sock(scale_b, "Vector"), _in_sock(pos_back, "Offset"))

    mat_back = ng.nodes.new("GeometryNodeSetMaterial")
    ng.links.new(_out_sock(pos_back, "Geometry"), _in_sock(mat_back, "Geometry"))
    ng.links.new(_out_sock(gin, "Interior Material"), _in_sock(mat_back, "Material"))

    sw_back = ng.nodes.new("GeometryNodeSwitch")
    sw_back.input_type = "GEOMETRY"
    ng.links.new(_out_sock(gin, "Use Backing"), _in_sock(sw_back, "Switch"))
    ng.links.new(_out_sock(mat_back, "Geometry"), _in_sock(sw_back, "True"))

    # Split wall panels into Ground Floor vs Upper Floors
    pos_z = ng.nodes.new("GeometryNodeInputPosition")
    sep_p = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(_out_sock(pos_z, "Position"), _in_sock(sep_p, "Vector"))

    is_gf = ng.nodes.new("FunctionNodeCompare")
    is_gf.data_type = "FLOAT"
    is_gf.operation = "LESS_THAN"
    ng.links.new(_out_sock(sep_p, "Z"), _in_sock(is_gf, "A"))
    ng.links.new(_out_sock(gin, "GF Height"), _in_sock(is_gf, "B"))

    sep_gf = ng.nodes.new("GeometryNodeSeparateGeometry")
    sep_gf.domain = "FACE"
    ng.links.new(_out_sock(sep_top, "Inverted"), _in_sock(sep_gf, "Geometry"))
    ng.links.new(_out_sock(is_gf, "Result"), _in_sock(sep_gf, "Selection"))

    gf_cw = ng.nodes.new("GeometryNodeGroup")
    gf_cw.node_tree = gf_tree
    ng.links.new(_out_sock(sep_gf, "Selection"), _in_sock(gf_cw, "Geometry"))
    ng.links.new(_out_sock(gin, "Glass Material"), _in_sock(gf_cw, "Glass Material"))
    ng.links.new(
        _out_sock(gin, "GF Mullion Material"), _in_sock(gf_cw, "Mullions Material")
    )
    _link_glass_facade_width(gf_cw, _out_sock(gin, "GF Mullion Width"))
    ng.links.new(_out_sock(gin, "GF Mullion Depth"), _in_sock(gf_cw, "Mullion Depth"))
    ng.links.new(_out_sock(gin, "Delete Glass"), _in_sock(gf_cw, "Delete Glass"))
    _in_sock(gf_cw, "Split Edges").default_value = True

    # Route structured full-height column bays (every N grid modules) to solid opaque relief panels
    is_wide_u = ng.nodes.new("FunctionNodeCompare")
    is_wide_u.data_type = "FLOAT"
    is_wide_u.operation = "GREATER_EQUAL"
    ng.links.new(_out_sock(area, "Area"), _in_sock(is_wide_u, "A"))
    _in_sock(is_wide_u, "B").default_value = 4.5

    col_op_mask = _add_column_opaque_mask(
        ng,
        _out_sock(gin, "Grid Module"),
        _out_sock(gin, "Opaque Every N Bays"),
    )

    is_op = ng.nodes.new("FunctionNodeBooleanMath")
    is_op.operation = "AND"
    ng.links.new(_out_sock(is_wide_u, "Result"), is_op.inputs[0])
    ng.links.new(col_op_mask, is_op.inputs[1])

    sep_op = ng.nodes.new("GeometryNodeSeparateGeometry")
    sep_op.domain = "FACE"
    ng.links.new(_out_sock(sep_gf, "Inverted"), _in_sock(sep_op, "Geometry"))
    ng.links.new(_out_sock(is_op, "Boolean"), _in_sock(sep_op, "Selection"))

    up_cw = ng.nodes.new("GeometryNodeGroup")
    up_cw.node_tree = gf_tree
    ng.links.new(_out_sock(sep_op, "Inverted"), _in_sock(up_cw, "Geometry"))
    ng.links.new(_out_sock(gin, "Glass Material"), _in_sock(up_cw, "Glass Material"))
    ng.links.new(
        _out_sock(gin, "Upper Mullion Material"), _in_sock(up_cw, "Mullions Material")
    )
    _link_glass_facade_width(up_cw, _out_sock(gin, "Mullion Width"))
    ng.links.new(_out_sock(gin, "Mullion Depth"), _in_sock(up_cw, "Mullion Depth"))
    ng.links.new(_out_sock(gin, "Delete Glass"), _in_sock(up_cw, "Delete Glass"))
    _in_sock(up_cw, "Split Edges").default_value = True

    split_op = ng.nodes.new("GeometryNodeSplitEdges")
    ng.links.new(_out_sock(sep_op, "Selection"), _in_sock(split_op, "Mesh"))

    scale_op = ng.nodes.new("GeometryNodeScaleElements")
    scale_op.domain = "FACE"
    ng.links.new(_out_sock(split_op, "Mesh"), _in_sock(scale_op, "Geometry"))
    _in_sock(scale_op, "Scale").default_value = 0.992

    norm_op = ng.nodes.new("GeometryNodeInputNormal")
    in_shift = ng.nodes.new("ShaderNodeVectorMath")
    in_shift.operation = "SCALE"
    ng.links.new(_out_sock(norm_op, "True Normal"), _in_sock(in_shift, "Vector"))
    _in_sock(in_shift, "Scale").default_value = -0.04

    pos_op = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(_out_sock(scale_op, "Geometry"), _in_sock(pos_op, "Geometry"))
    ng.links.new(_out_sock(in_shift, "Vector"), _in_sock(pos_op, "Offset"))

    out_ext = ng.nodes.new("ShaderNodeVectorMath")
    out_ext.operation = "SCALE"
    ng.links.new(_out_sock(norm_op, "True Normal"), _in_sock(out_ext, "Vector"))
    _in_sock(out_ext, "Scale").default_value = 0.20

    ext_op = ng.nodes.new("GeometryNodeGroup")
    ext_op.node_tree = se_tree
    ng.links.new(_out_sock(pos_op, "Geometry"), _in_sock(ext_op, "Geometry"))
    ng.links.new(_out_sock(out_ext, "Vector"), _in_sock(ext_op, "Offset"))

    mat_op = ng.nodes.new("GeometryNodeSetMaterial")
    ng.links.new(_out_sock(ext_op, "Geometry"), _in_sock(mat_op, "Geometry"))
    ng.links.new(_out_sock(gin, "Upper Mullion Material"), _in_sock(mat_op, "Material"))

    join = ng.nodes.new("GeometryNodeJoinGeometry")
    for src, sname in (
        (set_roof_mat, "Geometry"),
        (sw_back, "Output"),
        (gf_cw, "Geometry"),
        (up_cw, "Geometry"),
        (mat_op, "Geometry"),
    ):
        ng.links.new(_out_sock(src, sname), _in_sock(join, "Geometry"))
    ng.links.new(_out_sock(join, "Geometry"), _in_sock(gout, "Geometry"))
    return ng


def _add_bm_box(
    bm: bmesh.types.BMesh,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    mat_idx: int,
) -> None:
    if x1 <= x0 + 1e-5 or y1 <= y0 + 1e-5 or z1 <= z0 + 1e-5:
        return
    v = (
        bm.verts.new((x0, y0, z0)),
        bm.verts.new((x1, y0, z0)),
        bm.verts.new((x1, y1, z0)),
        bm.verts.new((x0, y1, z0)),
        bm.verts.new((x0, y0, z1)),
        bm.verts.new((x1, y0, z1)),
        bm.verts.new((x1, y1, z1)),
        bm.verts.new((x0, y1, z1)),
    )
    for quad in (
        (v[0], v[3], v[2], v[1]),
        (v[4], v[5], v[6], v[7]),
        (v[0], v[1], v[5], v[4]),
        (v[1], v[2], v[6], v[5]),
        (v[2], v[3], v[7], v[6]),
        (v[3], v[0], v[4], v[7]),
    ):
        f = bm.faces.new(quad)
        f.material_index = mat_idx
        f.smooth = False


def _add_bm_cylinder(
    bm: bmesh.types.BMesh,
    cx: float,
    cy: float,
    z0: float,
    z1: float,
    radius: float,
    segments: int,
    mat_idx: int,
) -> None:
    from math import cos, pi, sin

    bot, top = [], []
    for k in range(segments):
        ang = 2.0 * pi * k / segments
        px = cx + radius * cos(ang)
        py = cy + radius * sin(ang)
        bot.append(bm.verts.new((px, py, z0)))
        top.append(bm.verts.new((px, py, z1)))
    for k in range(segments):
        nk = (k + 1) % segments
        f = bm.faces.new((bot[k], bot[nk], top[nk], top[k]))
        f.material_index = mat_idx
        f.smooth = False
    f_bot = bm.faces.new(list(reversed(bot)))
    f_bot.material_index = mat_idx
    f_bot.smooth = False
    f_top = bm.faces.new(top)
    f_top.material_index = mat_idx
    f_top.smooth = False


def _finalize_module_object(
    name: str,
    col: bpy.types.Collection,
    bm: bmesh.types.BMesh,
    materials: list[bpy.types.Material],
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for mat in materials:
        mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (0.0, 0.0, 0.0)
    col.objects.link(obj)
    return obj


def _build_framed_module(
    name: str,
    col: bpy.types.Collection,
    *,
    w: float,
    h: float,
    z_front: float,
    z_back: float,
    frame_mat: bpy.types.Material,
    back_mat: bpy.types.Material | None = None,
    glass_mat: bpy.types.Material | None = None,
    mullion_mat: bpy.types.Material | None = None,
    t_side: float = 0.14,
    t_bot: float = 0.14,
    t_top: float = 0.14,
    z_glass: float | None = None,
    n_panes: int = 1,
    mullion_w: float = 0.055,
    louvers: int = 0,
    louver_z_front: float | None = None,
    split_x: float | None = None,
    center_pier: float = 0.0,
    front_slab_z: float | None = None,
) -> bpy.types.Object:
    """Build a watertight maquette facade module centered at bottom-center (0, 0, 0)."""
    back_mat = back_mat or frame_mat
    mullion_mat = mullion_mat or frame_mat
    mats_list: list[bpy.types.Material] = []

    def _mat_idx(m: bpy.types.Material) -> int:
        if m not in mats_list:
            mats_list.append(m)
        return mats_list.index(m)

    i_frame = _mat_idx(frame_mat)
    i_back = _mat_idx(back_mat)
    i_mull = _mat_idx(mullion_mat)
    i_glass = _mat_idx(glass_mat) if glass_mat is not None else -1

    bm = bmesh.new()
    hw = 0.5 * w
    in_x0 = -hw + t_side
    in_x1 = hw - t_side
    in_y0 = t_bot
    in_y1 = h - t_top

    # Back wall sealing the recess so buildings never show see-through holes
    _add_bm_box(bm, -hw, 0.0, z_back - 0.08, hw, h, z_back, i_back)

    # Outer perimeter box frame (returns from z_back to z_front)
    if t_bot > 1e-4:
        _add_bm_box(bm, -hw, 0.0, z_back, hw, in_y0, z_front, i_frame)
    if t_top > 1e-4:
        _add_bm_box(bm, -hw, in_y1, z_back, hw, h, z_front, i_frame)
    if t_side > 1e-4:
        _add_bm_box(bm, -hw, in_y0, z_back, in_x0, in_y1, z_front, i_frame)
        _add_bm_box(bm, in_x1, in_y0, z_back, hw, in_y1, z_front, i_frame)

    if front_slab_z is not None and front_slab_z > z_front + 1e-4:
        _add_bm_box(bm, -hw, 0.0, z_front, hw, 0.28, front_slab_z, i_frame)

    if center_pier > 1e-4:
        hp = 0.5 * center_pier
        _add_bm_box(bm, -hp, in_y0, z_back, hp, in_y1, z_front, i_frame)

    if split_x is not None:
        sx = -hw + (hw * 2.0) * split_x
        _add_bm_box(
            bm,
            sx - 0.07,
            in_y0,
            z_back,
            sx + 0.07,
            in_y1,
            min(z_front, z_back + 0.48),
            i_frame,
        )

    if louvers > 0:
        zf_l = z_front if louver_z_front is None else louver_z_front
        lw = 0.07 if w <= 1.2 else 0.09
        for k in range(1, louvers + 1):
            lx = in_x0 + k * (in_x1 - in_x0) / (louvers + 1)
            _add_bm_box(
                bm,
                lx - 0.5 * lw,
                in_y0,
                max(z_back, zf_l - 0.34),
                lx + 0.5 * lw,
                in_y1,
                zf_l,
                i_frame,
            )

    if glass_mat is not None and in_x1 > in_x0 + 0.05 and in_y1 > in_y0 + 0.05:
        zg = (z_back + 0.14) if z_glass is None else z_glass
        _add_bm_box(bm, in_x0, in_y0, zg - 0.02, in_x1, in_y1, zg, i_glass)
        if n_panes > 1 and mullion_w > 1e-4:
            for k in range(1, n_panes):
                mx = in_x0 + k * (in_x1 - in_x0) / n_panes
                _add_bm_box(
                    bm,
                    mx - 0.5 * mullion_w,
                    in_y0,
                    zg - 0.04,
                    mx + 0.5 * mullion_w,
                    in_y1,
                    zg + 0.05,
                    i_mull,
                )

    return _finalize_module_object(name, col, bm, mats_list)


def _build_opaque_module(
    name: str,
    col: bpy.types.Collection,
    *,
    w: float,
    h: float,
    z_front: float,
    z_back: float,
    frame_mat: bpy.types.Material,
    back_mat: bpy.types.Material | None = None,
    inset_margin: float = 0.14,
    panel_z: float | None = None,
    front_slab_z: float | None = None,
) -> bpy.types.Object:
    """Build a watertight opaque maquette panel module with shadow reveal relief."""
    back_mat = back_mat or frame_mat
    mats_list: list[bpy.types.Material] = [frame_mat]
    if back_mat is not frame_mat:
        mats_list.append(back_mat)
        i_back = 1
    else:
        i_back = 0

    bm = bmesh.new()
    hw = 0.5 * w
    pz = (z_front + 0.08) if panel_z is None else panel_z
    _add_bm_box(bm, -hw, 0.0, z_back - 0.08, hw, h, z_front - 0.04, i_back)
    m = min(inset_margin, 0.35 * hw, 0.35 * h)
    _add_bm_box(bm, -hw + m, m, z_front - 0.06, hw - m, h - m, pz, 0)
    if front_slab_z is not None and front_slab_z > max(z_front, pz) + 1e-4:
        _add_bm_box(bm, -hw, 0.0, z_front, hw, 0.28, front_slab_z, 0)
    return _finalize_module_object(name, col, bm, mats_list)


def _populate_facade_collections(
    assets_col: bpy.types.Collection, mats: dict[str, bpy.types.Material]
) -> None:
    def _fresh(name: str) -> bpy.types.Collection:
        c = _ensure_collection(name, assets_col)
        c.hide_viewport = True
        c.hide_render = True
        _clear_collection_objects(c)
        return c

    # HERO_ShadowBox (reference maquette: cantilevered white museum-board boxes + blush podium)
    c_sb = _fresh("HERO_ShadowBox")
    _build_framed_module(
        "00_Ground_BlushPlinth",
        c_sb,
        w=3.0,
        h=4.2,
        z_front=0.26,
        z_back=-0.75,
        frame_mat=mats["Mat_Hero_Accent"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.28,
        t_bot=0.06,
        t_top=0.40,
    )
    _build_framed_module(
        "01_Ground_1m_Blush",
        c_sb,
        w=1.0,
        h=4.2,
        z_front=0.22,
        z_back=-0.65,
        frame_mat=mats["Mat_Hero_Accent"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.10,
        t_bot=0.06,
        t_top=0.40,
    )
    _build_framed_module(
        "02_Narrow_1m_GlazedFrame",
        c_sb,
        w=1.0,
        h=3.0,
        z_front=0.28,
        z_back=-0.95,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.09,
        t_bot=0.14,
        t_top=0.14,
    )
    _build_framed_module(
        "03_Box_DeepCantilever",
        c_sb,
        w=3.0,
        h=3.0,
        z_front=0.72,
        z_back=-1.05,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.14,
        t_bot=0.14,
        t_top=0.14,
        center_pier=0.08,
    )
    _build_framed_module(
        "04_Box_MidSplit",
        c_sb,
        w=3.0,
        h=3.0,
        z_front=0.42,
        z_back=-1.10,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.14,
        t_bot=0.14,
        t_top=0.14,
        split_x=0.62,
    )
    _build_framed_module(
        "05_Box_FlushLoggia",
        c_sb,
        w=3.0,
        h=3.0,
        z_front=0.14,
        z_back=-1.15,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.16,
        t_bot=0.14,
        t_top=0.14,
        center_pier=0.08,
    )
    _build_opaque_module(
        "06_Box_OpaqueRelief",
        c_sb,
        w=3.0,
        h=3.0,
        z_front=0.22,
        z_back=-0.42,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        inset_margin=0.14,
        panel_z=0.34,
    )

    # HERO_RibbonLouver (continuous cantilevered slabs + crisp vertical louver rhythm)
    c_rl = _fresh("HERO_RibbonLouver")
    _build_framed_module(
        "00_Ground_BlushArcade",
        c_rl,
        w=3.0,
        h=4.2,
        z_front=0.22,
        z_back=-0.58,
        frame_mat=mats["Mat_Hero_Accent"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        mullion_mat=mats["Mat_Dark"],
        t_side=0.30,
        t_bot=0.05,
        t_top=0.38,
        z_glass=-0.34,
        n_panes=2,
    )
    _build_framed_module(
        "01_Ground_1m_Blush",
        c_rl,
        w=1.0,
        h=4.2,
        z_front=0.20,
        z_back=-0.40,
        frame_mat=mats["Mat_Hero_Accent"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.10,
        t_bot=0.05,
        t_top=0.38,
        z_glass=-0.22,
    )
    _build_framed_module(
        "02_Narrow_1m_Ribbon",
        c_rl,
        w=1.0,
        h=3.0,
        z_front=0.08,
        z_back=-0.32,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.07,
        t_bot=0.24,
        t_top=0.24,
        z_glass=-0.16,
        louvers=1,
        louver_z_front=0.24,
        front_slab_z=0.38,
    )
    _build_framed_module(
        "03_Wide_RibbonLouvers",
        c_rl,
        w=3.0,
        h=3.0,
        z_front=0.08,
        z_back=-0.34,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        t_side=0.08,
        t_bot=0.24,
        t_top=0.24,
        z_glass=-0.18,
        louvers=4,
        louver_z_front=0.24,
        front_slab_z=0.38,
    )
    _build_framed_module(
        "04_Wide_RibbonBalcony",
        c_rl,
        w=3.0,
        h=3.0,
        z_front=0.08,
        z_back=-0.42,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        glass_mat=mats["Glass"],
        mullion_mat=mats["Mat_Dark"],
        t_side=0.10,
        t_bot=0.26,
        t_top=0.24,
        z_glass=-0.22,
        n_panes=2,
        front_slab_z=0.50,
    )
    _build_opaque_module(
        "05_Wide_OpaqueSlab",
        c_rl,
        w=3.0,
        h=3.0,
        z_front=0.08,
        z_back=-0.28,
        frame_mat=mats["Mat_Hero_White"],
        back_mat=mats["Mat_Hero_Slab"],
        inset_margin=0.10,
        panel_z=0.16,
        front_slab_z=0.38,
    )

    # Context Height-Tiered Monotone Collections (ALL using ONLY Mat_Context_Maquette)
    m_ctx = mats["Mat_Context_Maquette"]

    c_ls = _fresh("CTX_Low_Storefront")
    _build_framed_module(
        "00_Ground_Wide",
        c_ls,
        w=2.8,
        h=4.2,
        z_front=0.0,
        z_back=-0.38,
        frame_mat=m_ctx,
        t_side=0.26,
        t_bot=0.02,
        t_top=0.36,
    )
    _build_framed_module(
        "01_Upper_Wide",
        c_ls,
        w=2.8,
        h=3.6,
        z_front=0.0,
        z_back=-0.28,
        frame_mat=m_ctx,
        t_side=0.24,
        t_bot=0.78,
        t_top=0.36,
    )

    c_lp = _fresh("CTX_Low_Punched")
    _build_framed_module(
        "00_Ground_Tall",
        c_lp,
        w=2.4,
        h=4.2,
        z_front=0.0,
        z_back=-0.34,
        frame_mat=m_ctx,
        t_side=0.30,
        t_bot=0.02,
        t_top=0.40,
    )
    _build_framed_module(
        "01_Upper_Square",
        c_lp,
        w=2.4,
        h=3.4,
        z_front=0.0,
        z_back=-0.28,
        frame_mat=m_ctx,
        t_side=0.38,
        t_bot=0.84,
        t_top=0.44,
    )
    _build_framed_module(
        "02_Upper_SquareDeep",
        c_lp,
        w=2.4,
        h=3.4,
        z_front=0.0,
        z_back=-0.34,
        frame_mat=m_ctx,
        t_side=0.34,
        t_bot=0.76,
        t_top=0.38,
    )

    c_mp = _fresh("CTX_Mid_PunchedGrid")
    _build_framed_module(
        "00_Ground_Arcade",
        c_mp,
        w=2.5,
        h=4.8,
        z_front=0.04,
        z_back=-0.46,
        frame_mat=m_ctx,
        t_side=0.28,
        t_bot=0.02,
        t_top=0.44,
    )
    _build_framed_module(
        "01_Upper_Punched",
        c_mp,
        w=2.5,
        h=3.6,
        z_front=0.0,
        z_back=-0.36,
        frame_mat=m_ctx,
        t_side=0.38,
        t_bot=0.82,
        t_top=0.42,
    )
    _build_framed_module(
        "02_Upper_PunchedAlt",
        c_mp,
        w=2.5,
        h=3.6,
        z_front=0.0,
        z_back=-0.32,
        frame_mat=m_ctx,
        t_side=0.44,
        t_bot=0.68,
        t_top=0.38,
    )

    c_mt = _fresh("CTX_Mid_PairedBays")
    _build_framed_module(
        "00_Ground_Double",
        c_mt,
        w=2.8,
        h=4.8,
        z_front=0.04,
        z_back=-0.42,
        frame_mat=m_ctx,
        t_side=0.26,
        t_bot=0.02,
        t_top=0.42,
        center_pier=0.24,
    )
    _build_framed_module(
        "01_Upper_Twin",
        c_mt,
        w=2.8,
        h=3.6,
        z_front=0.0,
        z_back=-0.34,
        frame_mat=m_ctx,
        t_side=0.32,
        t_bot=0.76,
        t_top=0.40,
        center_pier=0.22,
    )
    _build_framed_module(
        "02_Upper_TwinAlt",
        c_mt,
        w=2.8,
        h=3.6,
        z_front=0.04,
        z_back=-0.38,
        frame_mat=m_ctx,
        t_side=0.36,
        t_bot=0.68,
        t_top=0.36,
        center_pier=0.22,
    )

    c_mh = _fresh("CTX_Mid_HorizontalBand")
    _build_framed_module(
        "00_Ground_Plinth",
        c_mh,
        w=2.6,
        h=4.8,
        z_front=0.06,
        z_back=-0.42,
        frame_mat=m_ctx,
        t_side=0.24,
        t_bot=0.02,
        t_top=0.40,
    )
    _build_framed_module(
        "01_Upper_Ribbon",
        c_mh,
        w=2.6,
        h=3.6,
        z_front=0.08,
        z_back=-0.28,
        frame_mat=m_ctx,
        t_side=0.02,
        t_bot=0.88,
        t_top=0.38,
    )

    c_hf = _fresh("CTX_High_FlutedTower")
    _build_framed_module(
        "00_Ground_Lobby",
        c_hf,
        w=2.4,
        h=5.2,
        z_front=0.08,
        z_back=-0.48,
        frame_mat=m_ctx,
        t_side=0.26,
        t_bot=0.02,
        t_top=0.46,
        center_pier=0.20,
    )
    _build_framed_module(
        "01_Upper_Fluted",
        c_hf,
        w=2.4,
        h=3.6,
        z_front=0.0,
        z_back=-0.26,
        frame_mat=m_ctx,
        t_side=0.14,
        t_bot=0.42,
        t_top=0.24,
        louvers=3,
        louver_z_front=0.18,
    )

    c_hg = _fresh("CTX_High_GridTower")
    _build_framed_module(
        "00_Ground_Lobby",
        c_hg,
        w=2.4,
        h=5.2,
        z_front=0.08,
        z_back=-0.46,
        frame_mat=m_ctx,
        t_side=0.26,
        t_bot=0.02,
        t_top=0.44,
    )
    _build_framed_module(
        "01_Upper_FineGrid",
        c_hg,
        w=2.4,
        h=3.6,
        z_front=0.08,
        z_back=-0.26,
        frame_mat=m_ctx,
        t_side=0.20,
        t_bot=0.70,
        t_top=0.30,
        center_pier=0.18,
    )
    _build_framed_module(
        "02_Upper_FineGridAlt",
        c_hg,
        w=2.4,
        h=3.6,
        z_front=0.06,
        z_back=-0.30,
        frame_mat=m_ctx,
        t_side=0.24,
        t_bot=0.62,
        t_top=0.28,
        center_pier=0.20,
    )


def _sync_humans_collection(
    assets_col: bpy.types.Collection,
    human_mat: bpy.types.Material,
) -> bpy.types.Collection:
    c_humans = _ensure_collection("Maquette_Humans", assets_col)
    c_humans.hide_viewport = True
    c_humans.hide_render = True
    for obj in list(c_humans.objects):
        c_humans.objects.unlink(obj)

    human_objs = [bpy.data.objects.get(f"Prop_Human_{i:02d}") for i in range(6)]
    human_objs = [o for o in human_objs if o is not None]
    if not human_objs:
        for fallback in ("Prop_ScaleFigure_A", "Prop_ScaleFigure_B"):
            if (obj := bpy.data.objects.get(fallback)) is not None:
                human_objs.append(obj)
    for obj in human_objs:
        c_humans.objects.link(obj)
        if obj.type == "MESH" and obj.data is not None:
            obj.data.materials.clear()
            obj.data.materials.append(human_mat)
            obj.data.use_fake_user = True
    return c_humans


def _populate_prop_collection(
    assets_col: bpy.types.Collection, mats: dict[str, bpy.types.Material]
) -> bpy.types.Collection:
    c_props = _ensure_collection("Maquette_Props", assets_col)
    c_props.hide_viewport = True
    c_props.hide_render = True
    has_baked = (
        bool(c_props.get("baked_external_props"))
        or bpy.data.objects.get("Prop_Car_Taxi") is not None
    )
    if has_baked:
        if bpy.data.objects.get("Prop_Tree_Maquette") is None:
            bm = bmesh.new()
            _add_bm_cylinder(bm, 0.0, 0.0, 0.0, 2.4, 0.14, 8, 0)
            _add_bm_cylinder(bm, 0.0, 0.0, 2.2, 5.4, 1.35, 10, 1)
            _finalize_module_object(
                "Prop_Tree_Maquette",
                c_props,
                bm,
                [mats["Mat_Prop_Grey"], mats["Mat_Green_Maquette"]],
            )
        for obj in c_props.objects:
            if obj.data is not None:
                obj.data.use_fake_user = True
        return _sync_humans_collection(assets_col, mats["Humans"])

    _clear_collection_objects(c_props)

    # Slender 5.4 m architectural street lamp
    bm = bmesh.new()
    _add_bm_cylinder(bm, 0.0, 0.0, 0.0, 0.40, 0.14, 8, 0)
    _add_bm_cylinder(bm, 0.0, 0.0, 0.40, 5.4, 0.08, 8, 0)
    _add_bm_box(bm, -0.05, 0.0, 5.28, 0.05, 1.05, 5.40, 0)
    _add_bm_box(bm, -0.12, 0.78, 5.18, 0.12, 1.15, 5.34, 0)
    _finalize_module_object("Prop_StreetLamp", c_props, bm, [mats["Mat_Dark"]])

    # Clean 1.75 m maquette scale figures (A standing, B slightly slimmer)
    for name, w_s in (("Prop_ScaleFigure_A", 1.0), ("Prop_ScaleFigure_B", 0.92)):
        bm = bmesh.new()
        _add_bm_cylinder(bm, -0.07 * w_s, 0.0, 0.0, 0.82, 0.065 * w_s, 6, 0)
        _add_bm_cylinder(bm, 0.07 * w_s, 0.0, 0.0, 0.82, 0.065 * w_s, 6, 0)
        _add_bm_box(bm, -0.19 * w_s, -0.10 * w_s, 0.82, 0.19 * w_s, 0.10 * w_s, 1.48, 0)
        _add_bm_cylinder(bm, 0.0, 0.0, 1.50, 1.75, 0.10 * w_s, 8, 0)
        _finalize_module_object(name, c_props, bm, [mats["Humans"]])

    # Clean low-poly maquette sedans (~4.2 m x 1.74 m x 1.38 m, wheels at Z=0)
    for name, body_mat in (
        ("Prop_Car_White", mats["Mat_Prop_White"]),
        ("Prop_Car_Grey", mats["Mat_Prop_Grey"]),
        ("Prop_Car_Taxi", mats["Mat_Car_Accent"]),
    ):
        bm = bmesh.new()
        _add_bm_box(bm, -2.10, -0.87, 0.22, 2.10, 0.87, 0.78, 0)
        _add_bm_box(bm, -1.05, -0.74, 0.78, 0.95, 0.74, 1.36, 0)
        for wx in (-1.28, 1.28):
            for wy in (-0.89, 0.73):
                _add_bm_box(bm, wx - 0.28, wy, 0.0, wx + 0.28, wy + 0.16, 0.48, 1)
        _finalize_module_object(name, c_props, bm, [body_mat, mats["Mat_Dark"]])

    # Clean minimalist maquette tree
    bm = bmesh.new()
    _add_bm_cylinder(bm, 0.0, 0.0, 0.0, 2.4, 0.14, 8, 0)
    _add_bm_cylinder(bm, 0.0, 0.0, 2.2, 5.4, 1.35, 10, 1)
    _finalize_module_object(
        "Prop_Tree_Maquette",
        c_props,
        bm,
        [mats["Mat_Prop_Grey"], mats["Mat_Green_Maquette"]],
    )
    return _sync_humans_collection(assets_col, mats["Humans"])


def _build_scatter_humans_on_paths_group(
    humans_col: bpy.types.Collection,
) -> bpy.types.NodeTree:
    """Scatter maquette scale humans along pedestrian spline curves aligned to walkway direction."""
    ng = _recreate_geo_group("Scatter Humans On Paths")
    ng.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    ng.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )

    s_col = ng.interface.new_socket(
        "Humans Collection", in_out="INPUT", socket_type="NodeSocketCollection"
    )
    s_col.default_value = humans_col

    for name, dv in (
        ("Spacing", 4.2),
        ("Keep Chance", 0.30),
        ("Along Jitter", 1.65),
        ("Lateral Jitter", 0.32),
        ("Flip Direction", 0.5),
    ):
        s = ng.interface.new_socket(name, in_out="INPUT", socket_type="NodeSocketFloat")
        s.default_value = dv

    s_seed = ng.interface.new_socket(
        "Seed", in_out="INPUT", socket_type="NodeSocketInt"
    )
    s_seed.default_value = 0

    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")

    resamp = ng.nodes.new("GeometryNodeResampleCurve")
    _in_sock(resamp, "Mode").default_value = "Length"
    ng.links.new(_out_sock(gin, "Geometry"), _in_sock(resamp, "Curve"))
    ng.links.new(_out_sock(gin, "Spacing"), _in_sock(resamp, "Length"))

    sp_param = ng.nodes.new("GeometryNodeSplineParameter")
    cap = ng.nodes.new("GeometryNodeCaptureAttribute")
    cap.domain = "POINT"
    cap.capture_items.new("FLOAT", name="SplineFactor")
    ng.links.new(_out_sock(resamp, "Curve"), _in_sock(cap, "Geometry"))
    ng.links.new(_out_sock(sp_param, "Factor"), _in_sock(cap, "SplineFactor"))

    ctp = ng.nodes.new("GeometryNodeCurveToPoints")
    ctp.mode = "EVALUATED"
    ng.links.new(_out_sock(cap, "Geometry"), _in_sock(ctp, "Curve"))

    cross = ng.nodes.new("ShaderNodeVectorMath")
    cross.operation = "CROSS_PRODUCT"
    ng.links.new(_out_sock(ctp, "Tangent"), cross.inputs[0])
    cross.inputs[1].default_value = (0.0, 0.0, 1.0)

    norm_lat = ng.nodes.new("ShaderNodeVectorMath")
    norm_lat.operation = "NORMALIZE"
    ng.links.new(_out_sock(cross, "Vector"), norm_lat.inputs[0])

    seed_17 = ng.nodes.new("ShaderNodeMath")
    seed_17.operation = "ADD"
    ng.links.new(_out_sock(gin, "Seed"), seed_17.inputs[0])
    seed_17.inputs[1].default_value = 17.0

    neg_jit = ng.nodes.new("ShaderNodeMath")
    neg_jit.operation = "MULTIPLY"
    ng.links.new(_out_sock(gin, "Lateral Jitter"), neg_jit.inputs[0])
    neg_jit.inputs[1].default_value = -1.0

    rand_lat = ng.nodes.new("FunctionNodeRandomValue")
    rand_lat.data_type = "FLOAT"
    ng.links.new(_out_sock(neg_jit, "Value"), _in_sock(rand_lat, "Min"))
    ng.links.new(_out_sock(gin, "Lateral Jitter"), _in_sock(rand_lat, "Max"))
    ng.links.new(_out_sock(seed_17, "Value"), _in_sock(rand_lat, "Seed"))

    lat_vec = ng.nodes.new("ShaderNodeVectorMath")
    lat_vec.operation = "SCALE"
    ng.links.new(_out_sock(norm_lat, "Vector"), _in_sock(lat_vec, "Vector"))
    ng.links.new(_out_sock(rand_lat, "Value"), _in_sock(lat_vec, "Scale"))

    # Taper longitudinal slide to 0 at clipped spline endpoints: 4 * f * (1 - f)
    one_m_f = ng.nodes.new("ShaderNodeMath")
    one_m_f.operation = "SUBTRACT"
    one_m_f.inputs[0].default_value = 1.0
    ng.links.new(_out_sock(cap, "SplineFactor"), one_m_f.inputs[1])

    f_prod = ng.nodes.new("ShaderNodeMath")
    f_prod.operation = "MULTIPLY"
    ng.links.new(_out_sock(cap, "SplineFactor"), f_prod.inputs[0])
    ng.links.new(_out_sock(one_m_f, "Value"), f_prod.inputs[1])

    end_w = ng.nodes.new("ShaderNodeMath")
    end_w.operation = "MULTIPLY"
    ng.links.new(_out_sock(f_prod, "Value"), end_w.inputs[0])
    end_w.inputs[1].default_value = 4.0

    seed_31 = ng.nodes.new("ShaderNodeMath")
    seed_31.operation = "ADD"
    ng.links.new(_out_sock(gin, "Seed"), seed_31.inputs[0])
    seed_31.inputs[1].default_value = 31.0

    neg_along = ng.nodes.new("ShaderNodeMath")
    neg_along.operation = "MULTIPLY"
    ng.links.new(_out_sock(gin, "Along Jitter"), neg_along.inputs[0])
    neg_along.inputs[1].default_value = -1.0

    rand_along = ng.nodes.new("FunctionNodeRandomValue")
    rand_along.data_type = "FLOAT"
    ng.links.new(_out_sock(neg_along, "Value"), _in_sock(rand_along, "Min"))
    ng.links.new(_out_sock(gin, "Along Jitter"), _in_sock(rand_along, "Max"))
    ng.links.new(_out_sock(seed_31, "Value"), _in_sock(rand_along, "Seed"))

    along_mag = ng.nodes.new("ShaderNodeMath")
    along_mag.operation = "MULTIPLY"
    ng.links.new(_out_sock(rand_along, "Value"), along_mag.inputs[0])
    ng.links.new(_out_sock(end_w, "Value"), along_mag.inputs[1])

    along_vec = ng.nodes.new("ShaderNodeVectorMath")
    along_vec.operation = "SCALE"
    ng.links.new(_out_sock(ctp, "Tangent"), _in_sock(along_vec, "Vector"))
    ng.links.new(_out_sock(along_mag, "Value"), _in_sock(along_vec, "Scale"))

    tot_off = ng.nodes.new("ShaderNodeVectorMath")
    tot_off.operation = "ADD"
    ng.links.new(_out_sock(lat_vec, "Vector"), tot_off.inputs[0])
    ng.links.new(_out_sock(along_vec, "Vector"), tot_off.inputs[1])

    set_pos = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(_out_sock(ctp, "Points"), _in_sock(set_pos, "Geometry"))
    ng.links.new(_out_sock(tot_off, "Vector"), _in_sock(set_pos, "Offset"))

    align = ng.nodes.new("FunctionNodeAlignRotationToVector")
    align.axis = "Y"
    align.pivot_axis = "Z"
    ng.links.new(_out_sock(ctp, "Tangent"), _in_sock(align, "Vector"))

    seed_7 = ng.nodes.new("ShaderNodeMath")
    seed_7.operation = "ADD"
    ng.links.new(_out_sock(gin, "Seed"), seed_7.inputs[0])
    seed_7.inputs[1].default_value = 7.0

    rand_flip = ng.nodes.new("FunctionNodeRandomValue")
    rand_flip.data_type = "BOOLEAN"
    ng.links.new(_out_sock(gin, "Flip Direction"), _in_sock(rand_flip, "Probability"))
    ng.links.new(_out_sock(seed_7, "Value"), _in_sock(rand_flip, "Seed"))

    sw_pi = ng.nodes.new("GeometryNodeSwitch")
    sw_pi.input_type = "FLOAT"
    ng.links.new(_out_sock(rand_flip, "Value"), _in_sock(sw_pi, "Switch"))
    _in_sock(sw_pi, "False").default_value = 0.0
    _in_sock(sw_pi, "True").default_value = 3.14159265

    seed_23 = ng.nodes.new("ShaderNodeMath")
    seed_23.operation = "ADD"
    ng.links.new(_out_sock(gin, "Seed"), seed_23.inputs[0])
    seed_23.inputs[1].default_value = 23.0

    rand_yaw = ng.nodes.new("FunctionNodeRandomValue")
    rand_yaw.data_type = "FLOAT"
    _in_sock(rand_yaw, "Min").default_value = -0.14
    _in_sock(rand_yaw, "Max").default_value = 0.14
    ng.links.new(_out_sock(seed_23, "Value"), _in_sock(rand_yaw, "Seed"))

    tot_yaw = ng.nodes.new("ShaderNodeMath")
    tot_yaw.operation = "ADD"
    ng.links.new(_out_sock(sw_pi, "Output"), tot_yaw.inputs[0])
    ng.links.new(_out_sock(rand_yaw, "Value"), tot_yaw.inputs[1])

    comb_xyz = ng.nodes.new("ShaderNodeCombineXYZ")
    ng.links.new(_out_sock(tot_yaw, "Value"), _in_sock(comb_xyz, "Z"))

    rot = ng.nodes.new("FunctionNodeRotateRotation")
    rot.rotation_space = "LOCAL"
    ng.links.new(_out_sock(align, "Rotation"), _in_sock(rot, "Rotation"))
    ng.links.new(_out_sock(comb_xyz, "Vector"), _in_sock(rot, "Rotate By"))

    col_info = ng.nodes.new("GeometryNodeCollectionInfo")
    col_info.transform_space = "RELATIVE"
    _in_sock(col_info, "Separate Children").default_value = True
    _in_sock(col_info, "Reset Children").default_value = True
    ng.links.new(_out_sock(gin, "Humans Collection"), _in_sock(col_info, "Collection"))

    rand_idx = ng.nodes.new("FunctionNodeRandomValue")
    rand_idx.data_type = "INT"
    _in_sock(rand_idx, "Min").default_value = 0
    _in_sock(rand_idx, "Max").default_value = 100
    ng.links.new(_out_sock(gin, "Seed"), _in_sock(rand_idx, "Seed"))

    seed_41 = ng.nodes.new("ShaderNodeMath")
    seed_41.operation = "ADD"
    ng.links.new(_out_sock(gin, "Seed"), seed_41.inputs[0])
    seed_41.inputs[1].default_value = 41.0

    rand_keep = ng.nodes.new("FunctionNodeRandomValue")
    rand_keep.data_type = "BOOLEAN"
    ng.links.new(_out_sock(gin, "Keep Chance"), _in_sock(rand_keep, "Probability"))
    ng.links.new(_out_sock(seed_41, "Value"), _in_sock(rand_keep, "Seed"))

    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(_out_sock(set_pos, "Geometry"), _in_sock(inst, "Points"))
    ng.links.new(_out_sock(rand_keep, "Value"), _in_sock(inst, "Selection"))
    ng.links.new(_out_sock(col_info, "Instances"), _in_sock(inst, "Instance"))
    _in_sock(inst, "Pick Instance").default_value = True
    ng.links.new(_out_sock(rand_idx, "Value"), _in_sock(inst, "Instance Index"))
    ng.links.new(_out_sock(rot, "Rotation"), _in_sock(inst, "Rotation"))

    ng.links.new(_out_sock(inst, "Instances"), _in_sock(gout, "Geometry"))
    return ng


def configure_studio_world(
    hdris_dir: Path | None = None,
    *,
    hdri_name: str = "studio_small_08_4k.hdr",
    rotation_z: float = 0.85,
    strength: float = 1.0,
    light_col: bpy.types.Collection | None = None,
) -> None:
    """Configure EEVEE raytraced studio HDRI lighting directly matching the maquette look."""
    _ = light_col
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("Maquette_Studio_World")
        scene.world = world
    if world.node_tree is None:
        world.use_nodes = True

    nt = world.node_tree
    nt.nodes.clear()

    base_dir = hdris_dir or (
        Path(__file__).resolve().parent.parent / "blender" / "hdris"
    )
    hdri_file = (base_dir / hdri_name).resolve()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength

    if hdri_file.is_file():
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, rotation_z)
        nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

        env_tex = nt.nodes.new("ShaderNodeTexEnvironment")
        img = bpy.data.images.load(str(hdri_file), check_existing=True)
        if (
            bpy.data.filepath
            and Path(bpy.data.filepath).resolve().parent
            == hdri_file.parent.parent.resolve()
        ):
            img.filepath = f"//hdris/{hdri_name}"
        env_tex.image = img
        nt.links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
        nt.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (0.86, 0.85, 0.83, 1.0)

    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    if (old_key := bpy.data.objects.get("Studio_Softbox_Key")) is not None:
        key_data = old_key.data if old_key.type == "LIGHT" else None
        bpy.data.objects.remove(old_key, do_unlink=True)
        if key_data is not None and key_data.users == 0:
            bpy.data.lights.remove(key_data)

    scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True
    if hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 64
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - High Contrast"
    scene.view_settings.exposure = 0.0


def ensure_maquette_library() -> dict:
    """Create or refresh all maquette materials, node groups, facade modules, and props."""
    scene_col = bpy.context.scene.collection
    assets_col = _ensure_collection("00_Maquette_Assets", scene_col)
    assets_col.hide_viewport = True
    assets_col.hide_render = True

    mats = _ensure_materials()
    _cleanup_duplicates_and_unused_geonodes(mats)
    _build_panelize_building_group()
    _build_assign_panel_id_group()
    _build_hero_curtain_wall_group(mats)

    _populate_facade_collections(assets_col, mats)
    humans_col = _populate_prop_collection(assets_col, mats)
    _build_scatter_humans_on_paths_group(humans_col)

    _cleanup_duplicates_and_unused_geonodes(mats)
    return {"materials": mats}


def main() -> None:
    ensure_maquette_library()
    configure_studio_world()
    if bpy.data.filepath:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
        print(f"Populated maquette library into {bpy.data.filepath}")


if __name__ == "__main__":
    main()
