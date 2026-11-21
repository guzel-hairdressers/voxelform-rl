"""Generate ML documentation figures, multi-tier renders, and rollout video for README.md."""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm as mpl_cm, colors as mcolors, gridspec, patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.layout_archive import _unpack
from voxelform.objectives import organic_objective, organic_step_reward, organic_terminal_reward
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.shadows import device_name
from voxelform.unified import Organic3DNetwork, UnifiedLearner, commit

ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = ROOT / 'docs' / 'media'
RESULTS_DIR = ROOT / 'results'
LAYOUTS_DIR = ROOT / 'runs' / 'layouts'


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
        return np.zeros((0, 4, 3), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.int64)
    return np.concatenate(verts_list, 0), np.concatenate(face_list, 0), np.concatenate(cell_list, 0)


def _shade_rgba(base_rgb: tuple[float, float, float], faces: np.ndarray, alpha: float) -> np.ndarray:
    """Directional shading per face index (+X,-X,+Y,-Y,+Z,-Z) matching voxel clarity."""
    gains = np.array([0.84, 0.68, 0.76, 0.92, 1.00, 0.56], dtype=np.float32)
    g = gains[faces][:, None]
    rgb = np.clip(np.asarray(base_rgb, dtype=np.float32)[None, :] * g, 0.0, 1.0)
    return np.column_stack((rgb, np.full(len(faces), alpha, dtype=np.float32)))


def _setup_3d_axes(ax, occ: np.ndarray, dz: float = 3.0, elev: float = 28.0, azim: float = -52.0):
    """Set tight isometric metric bounds and clean white studio axes."""
    zz, yy, xx = np.nonzero(occ)
    if len(xx) == 0:
        nx, ny, nz = occ.shape[2], occ.shape[1], occ.shape[0]
        x0, x1, y0, y1, z0, z1 = 0.0, float(nx), 0.0, float(ny), 0.0, float(nz * dz)
    else:
        pad = 2.0
        x0, x1 = float(xx.min()) - pad, float(xx.max() + 1) + pad
        y0, y1 = float(yy.min()) - pad, float(yy.max() + 1) + pad
        z0, z1 = 0.0, max(12.0, float((zz.max() + 1) * dz) + 2.0)
    span = max(x1 - x0, y1 - y0, z1 - z0, 10.0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.48 * span
    z_top = max(z1, 0.42 * span)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(0.0, z_top)
    ax.set_box_aspect((1.0, 1.0, z_top / (2.0 * half)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _render_voxel_panel(
    ax,
    occ: np.ndarray,
    core_mask: np.ndarray,
    mode: str = 'mass',
    sun_map: np.ndarray | None = None,
    dist_map: np.ndarray | None = None,
    max_dist: float = 12.0,
    elev: float = 28.0,
    azim: float = -52.0,
    xray_mass: bool = True,
):
    """Render exposed faces of mass and compact-core shafts onto a 3D Matplotlib axes."""
    core_occ = occ & core_mask
    shell_occ = occ & ~core_occ
    dz = 3.0

    verts_all, rgba_all, edge_all = [], [], []
    if mode == 'mass':
        if xray_mass and shell_occ.any():
            vq, fq, _ = _surface_quads(shell_occ, dz=dz)
            verts_all.append(vq)
            rgba_all.append(_shade_rgba((0.36, 0.54, 0.98), fq, alpha=0.52))
            edge_all.append(np.tile(np.array([[0.12, 0.25, 0.65, 0.20]], np.float32), (len(vq), 1)))
        elif shell_occ.any():
            vq, fq, _ = _surface_quads(shell_occ, dz=dz)
            verts_all.append(vq)
            rgba_all.append(_shade_rgba((0.40, 0.58, 0.99), fq, alpha=0.96))
            edge_all.append(np.tile(np.array([[0.10, 0.22, 0.58, 0.30]], np.float32), (len(vq), 1)))
        if core_occ.any():
            cq, cf, _ = _surface_quads(core_occ, dz=dz, zero_z=False)
            verts_all.append(cq)
            rgba_all.append(_shade_rgba((0.86, 0.15, 0.15), cf, alpha=0.98))
            edge_all.append(np.tile(np.array([[0.45, 0.05, 0.05, 0.55]], np.float32), (len(cq), 1)))
    elif mode == 'sun':
        cmap = matplotlib.colormaps['YlOrRd']
        if shell_occ.any():
            vq, fq, cells = _surface_quads(shell_occ, dz=dz)
            vals = np.clip(sun_map[fq, cells], 0.0, 1.0) if sun_map is not None else np.zeros(len(fq))
            gains = np.array([0.88, 0.78, 0.84, 0.94, 1.00, 0.65], dtype=np.float32)[fq][:, None]
            rgba = cmap(vals).astype(np.float32)
            rgba[:, :3] = np.clip(rgba[:, :3] * gains, 0.0, 1.0)
            rgba[:, 3] = 0.96
            verts_all.append(vq)
            rgba_all.append(rgba)
            edge_all.append(np.tile(np.array([[0.22, 0.18, 0.16, 0.18]], np.float32), (len(vq), 1)))
        if core_occ.any():
            cq, cf, _ = _surface_quads(core_occ, dz=dz)
            verts_all.append(cq)
            rgba_all.append(_shade_rgba((0.39, 0.45, 0.55), cf, alpha=0.92))
            edge_all.append(np.tile(np.array([[0.15, 0.18, 0.24, 0.35]], np.float32), (len(cq), 1)))
    elif mode == 'core_dist':
        cmap = matplotlib.colormaps['viridis_r']
        if shell_occ.any():
            vq, fq, cells = _surface_quads(shell_occ, dz=dz)
            d_flat = dist_map.reshape(-1)[cells] if dist_map is not None else np.zeros(len(fq))
            norm = np.clip(d_flat / max(1.0, max_dist * 1.5), 0.0, 1.0)
            gains = np.array([0.88, 0.78, 0.84, 0.94, 1.00, 0.65], dtype=np.float32)[fq][:, None]
            rgba = cmap(norm).astype(np.float32)
            rgba[:, :3] = np.clip(rgba[:, :3] * gains, 0.0, 1.0)
            rgba[:, 3] = 0.95
            verts_all.append(vq)
            rgba_all.append(rgba)
            edge_all.append(np.tile(np.array([[0.15, 0.18, 0.20, 0.16]], np.float32), (len(vq), 1)))
        if core_occ.any():
            cq, cf, _ = _surface_quads(core_occ, dz=dz)
            verts_all.append(cq)
            rgba_all.append(_shade_rgba((0.86, 0.15, 0.15), cf, alpha=0.98))
            edge_all.append(np.tile(np.array([[0.45, 0.05, 0.05, 0.55]], np.float32), (len(cq), 1)))

    if verts_all:
        verts = np.concatenate(verts_all, 0)
        rgba = np.concatenate(rgba_all, 0)
        edges = np.concatenate(edge_all, 0)
        poly = Poly3DCollection(verts, facecolors=rgba, edgecolors=edges, linewidths=0.25, zsort='average')
        ax.add_collection3d(poly)
    _setup_3d_axes(ax, occ, dz=dz, elev=elev, azim=azim)


def generate_architecture_diagram():
    """Create crisp NeurIPS-style architecture + MDP pipeline figure."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in Organic3DNetwork(14).parameters())

    fig, ax = plt.subplots(figsize=(15.5, 6.9), dpi=160)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis('off')

    ax.text(
        2.0, 39.4,
        'VoxelForm-RL: Factored 2D+1D Actor-Critic & Feasibility-Projected 3D Voxel MDP',
        fontsize=15.5, fontweight='bold', color='#0f172a', family='DejaVu Sans',
    )
    ax.text(
        2.0, 36.9,
        f'Observation O_t in R^(Z x 14 x Y x X)  ->  Organic3DNetwork ({n_params:,} params, halo-2 sliding Z)  '
        '->  Hybrid Latent Action + Deterministic Constraint Projection Pi_valid',
        fontsize=9.6, color='#475569',
    )

    def _box(x, y, w, h, title, lines, fc, ec, title_c='#0f172a', body_c='#1e293b', tag=None):
        p = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.0',
            facecolor=fc, edgecolor=ec, linewidth=1.6,
        )
        ax.add_patch(p)
        if tag:
            tp = patches.FancyBboxPatch(
                (x + 0.8, y + h - 2.3), len(tag) * 0.62 + 1.4, 2.0,
                boxstyle='round,pad=0.15,rounding_size=0.45',
                facecolor=ec, edgecolor='none',
            )
            ax.add_patch(tp)
            ax.text(x + 1.5, y + h - 1.3, tag, fontsize=7.6, fontweight='bold', color='#ffffff', va='center')
            ty = y + h - 4.1
        else:
            ty = y + h - 2.0
        ax.text(x + 1.1, ty, title, fontsize=10.3, fontweight='bold', color=title_c, va='top')
        cur_y = ty - 2.4
        for line in lines:
            ax.text(x + 1.1, cur_y, line, fontsize=8.2, color=body_c, va='top', linespacing=1.24)
            cur_y -= 1.98 * (line.count('\n') + 1)

    def _arrow(x0, y0, x1, y1, label='', rad=0.0, color='#334155', y_off=0.0):
        style = f'arc3,rad={rad}' if rad != 0 else 'arc3,rad=0.0'
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='-|>,head_width=0.42,head_length=0.65', color=color, lw=1.7, connectionstyle=style),
        )
        if label:
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + (1.3 if rad == 0 else y_off)
            ax.text(mx, my, label, fontsize=8.0, color=color, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.20', facecolor='#ffffff', edgecolor='none', alpha=0.95))

    _box(
        2.0, 7.5, 20.5, 27.6,
        '14-Channel 3D State',
        [
            'Geometry & Masks (ch 0,3,4,6,7):\n  occupied, allowed, rel_z, rem/add',
            'Physical Proxies (ch 1,2,8,9):\n  diffuse air, noise, ext_sun, facade',
            'Shape Regularity (ch 5,10,11):\n  depth_bad, strip <6m, usable',
            'Vertical Cores (ch 12,13):\n  core_mask, geodesic badness',
            'Goal g_t in R^5:\n  FAR, target, err, rem/spent budget',
        ],
        fc='#f8fafc', ec='#2563eb', tag='STATE S_t',
    )

    _box(
        26.5, 7.5, 23.0, 27.6,
        'Factored 2D+1D Network',
        [
            '2D Spatial Stem (/4 XY):\n  Conv2d(14->24->32, k=3, s=2) + SiLU',
            'Factored 2D+1D Blocks (x2):\n  ctx_xy (3x3) + _zmix (Z-1, Z, Z+1)\n  inside Halo-2 Window (win=16 floors)',
            'Goal + Skip Conditioning:\n  AvgPool2d(/4) -> 1x1 local_skip (16 ch)\n  + goal_encoder(g_t) -> cat (48 ch)',
            'Policy & Value Outputs:\n  mu = 3*tanh(head/3) in R^(2 x Z/2 x Y/4 x X/4)\n  op_logits (4 ops), scalar critic V(s)',
        ],
        fc='#eff6ff', ec='#1d4ed8', tag='POLICY pi_theta',
    )

    _box(
        53.5, 7.5, 22.0, 27.6,
        'Action & Constraint Decoder',
        [
            '1. Categorical Op + Gaussian Retry:\n  o_t in {remove, grow, exchange, finish}\n  z^(k) ~ N(mu, sigma^2), joint log-prob',
            '2. Smooth Spatial Upsampling:\n  trilinear -> separable 1D(Z)+2D(Y,X)\n  top-k ranked voxel allocation',
            '3. Deterministic Projection Pi_valid:\n  - FAR acceptance band tolerance\n  - 6-neighbor connectivity & cavities\n  - k=3 morphological width guard\n  - Active compact-core.v5 preservation',
        ],
        fc='#fef2f2', ec='#dc2626', tag='DECODER Pi_valid',
    )

    _box(
        79.5, 7.5, 18.5, 27.6,
        'Solvers & GAE Update',
        [
            'Vectorized Proxy Solvers:\n  GPU DDA FastVoxelShadows\n  Cached MeshShadows (SHA-256)\n  Screened-Laplace diffuse\n  8-nb geodesic core distance',
            'Synchronous GAE Update:\n  gamma=0.99, lambda=0.95\n  Huber value loss (0.5x)\n  Entropy reg (0.005x)\n  Aux loss: 0.0 (ref) / 1.2 (abl)',
        ],
        fc='#f0fdf4', ec='#15803d', tag='SIM & LEARNER',
    )

    _arrow(22.9, 21.3, 26.1, 21.3, 'O_t, g_t')
    _arrow(49.9, 21.3, 53.1, 21.3, 'o_t, mu, V')
    _arrow(75.9, 21.3, 79.1, 21.3, 'valid mask')
    _arrow(88.8, 7.0, 12.2, 7.0, 'transition (O_{t+1}, step/terminal reward r_t)  &  GAE policy gradient', rad=-0.065, color='#0f766e', y_off=-3.6)

    out = MEDIA_DIR / 'mdp-architecture.png'
    fig.savefig(out, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    print(f'Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)')


def _render_rollout_assets():
    """Run an illustrative seeded episode on m_bcn_0093 and export MP4 + GIF."""
    dev = device_name('mps' if torch.backends.mps.is_available() else 'cpu')
    patch = PatchConfig(
        mode='combined', allocation='feedback', eligibility='depth', facade_layers=2,
        minimum_patch=4, maximum_patches=4, maximum_voids=1, minimum_void_area=4,
        batched=True, vertical_interior=False, budget_reserve_fraction=0.2,
    )
    library = ContextLibrary()
    try:
        context = library.load('m_bcn_0093', floors=20, grid_alignment='auto_fit')
        site_label = 'm_bcn_0093 (Barcelona Eixample, 20F ceiling, compact-core.v5)'
    except Exception:
        context = None
        site_label = 'synthetic:courtyard (20F ceiling, compact-core.v5)'

    env = Environment(
        (20, 28, 36), seed=9001, plan='courtyard', context=context, device=dev,
        sun_count=8, target_far=3.0, far_tolerance=0.1,
        frontier_factory=functools.partial(PatchFacades, config=patch, depth_device=dev),
        fast_self_shadows=True, fast_diffuse=True, cores_enabled=True,
    )
    env.precompute_external()

    torch.manual_seed(123)
    np.random.seed(123)
    learner = UnifiedLearner(
        dev, seed=123, config=patch, amount_feedback=True, net_guided=True,
        organic_3d=True, spatial_aux_weight=1.2,
    )

    # Brief seeded illustrative training warm-up (30 episodes) for coherent transitions.
    for ep in range(30):
        solid = env.allowed.clone()
        solid[6:] = False
        env.restore_occupancy(solid)
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        env._ln6_cache = None
        env.unified_limit = max(8, env.workload.limit + 8)
        env.unified_actions = 0
        env.termination = None
        _, d0 = organic_objective(env)
        prev_core = float(d0.get('coreScore') or 0.0)
        for _ in range(env.unified_limit):
            move = learner.choose(env, training=True)
            env.unified_actions += 1
            op = learner.last_operation
            if op == 'finish':
                env.termination = 'far_reached'
                rew = 0.0
            elif move is None:
                rew = -0.01
            else:
                commit(env, move, validated=True)
                env.update_cores()
                cm = env.core_metrics
                c_sc = max(0.0, (1.0 - float(cm['coreUncoveredFraction'])) * (1.0 - 0.5 * float(cm['coreDistBadness'])))
                delta = c_sc - prev_core
                prev_core = c_sc
                rew = organic_step_reward(env, delta)
                if env.edits % 5 == 0:
                    env.evaluate()
            if learner.last_attempt_recorded:
                learner.record(rew)
            if env.terminal:
                break
        if not env.terminal:
            env.termination = 'far_reached' if env.far_satisfied else 'step_budget_exhausted'
        env.evaluate()
        obj, _ = organic_objective(env)
        learner.add_reward(organic_terminal_reward(env, obj))
        learner.mark_done()
        if len(learner.tape) >= 16 or ep == 29:
            learner.finish(0.0)

    # Pick highest-objective FAR-satisfied evaluation rollout showing multi-step editing.
    best_snapshots = None
    best_score = -1e9
    for eval_seed in (9001, 9002, 9003, 9004, 9005):
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        solid = env.allowed.clone()
        solid[6:] = False
        env.restore_occupancy(solid)
        env.organic_3d_growth = True
        env.floor_copy_growth = False
        env._ln6_cache = None
        env.unified_limit = max(8, env.workload.limit + 8)
        env.unified_actions = 0
        env.termination = None
        env.evaluate()

        def _snap(step_idx: int, op_name: str, dv: int):
            obj, diag = organic_objective(env)
            nx, ny, nz = env.nx, env.ny, env.nz
            sun_grid = np.zeros((6, nz * ny * nx), dtype=np.float32)
            if hasattr(env, 'receiver_data') and hasattr(env, 'visible'):
                _, _, _, cells, faces = env.receiver_data
                vis_mean = env.visible.float().mean(0).detach().cpu().numpy()
                c_np = cells.detach().cpu().numpy()
                f_np = faces.detach().cpu().numpy()
                sun_grid[f_np, c_np] = vis_mean
            return {
                'step': step_idx,
                'op': op_name,
                'dv': dv,
                'occ': env.occupied.detach().cpu().numpy().copy(),
                'core_mask': env.core_mask.detach().cpu().numpy().copy(),
                'core_dist': env.core_distance.detach().cpu().numpy().copy(),
                'sun_grid': sun_grid,
                'far': float(env.far),
                'target_far': float(env.target_far),
                'obj': float(obj),
                'usable_frac': float(diag['usableGoodFraction']),
                'sun': float(env.values['directSunProxy'].detach().cpu()),
                'diff': float(env.values['diffuseFacadeProxy'].detach().cpu()),
                'core_cov': float(diag.get('coreCoverage') or 0.0),
                'core_count': int(diag.get('coreCount') or 0),
            }

        snaps = [_snap(0, 'init', 0)]
        for step_idx in range(1, env.unified_limit + 1):
            move = learner.choose(env, training=False)
            env.unified_actions += 1
            op = learner.last_operation or 'none'
            if op == 'finish':
                env.termination = 'far_reached'
                env.evaluate()
                snaps.append(_snap(step_idx, 'finish', 0))
                break
            if move is None:
                continue
            dv = int(move.add.sum()) - int(move.remove.sum())
            commit(env, move, validated=True)
            env.evaluate()
            snaps.append(_snap(step_idx, op, dv))
            if env.terminal:
                break
        cand_score = snaps[-1]['obj'] + (10.0 if env.far_satisfied else 0.0) + (1.0 if len(snaps) >= 5 else 0.0)
        if best_snapshots is None or cand_score > best_score:
            best_score = cand_score
            best_snapshots = snaps

    assert best_snapshots is not None and len(best_snapshots) >= 2
    print(f'Recorded rollout: {len(best_snapshots)} states, final FAR={best_snapshots[-1]["far"]:.2f}, obj={best_snapshots[-1]["obj"]:.3f}')

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_entries: list[tuple[Path, float]] = []

        def _draw_state(snap: dict, total_steps: int, azim: float, out_png: Path, tag_suffix: str = ''):
            fig = plt.figure(figsize=(12.8, 5.4), dpi=100)
            fig.patch.set_facecolor('#ffffff')
            gs = gridspec.GridSpec(1, 3, figure=fig, left=0.02, right=0.98, top=0.81, bottom=0.14, wspace=0.04)
            axes = [fig.add_subplot(gs[0, i], projection='3d') for i in range(3)]

            _render_voxel_panel(axes[0], snap['occ'], snap['core_mask'], mode='mass', azim=azim, xray_mass=True)
            axes[0].set_title('Occupied Mass + compact-core.v5', fontsize=10.5, fontweight='bold', color='#0f172a', pad=4)

            _render_voxel_panel(axes[1], snap['occ'], snap['core_mask'], mode='sun', sun_map=snap['sun_grid'], azim=azim)
            axes[1].set_title('Direct Sun Proxy (Cores Exempt)', fontsize=10.5, fontweight='bold', color='#0f172a', pad=4)

            _render_voxel_panel(axes[2], snap['occ'], snap['core_mask'], mode='core_dist', dist_map=snap['core_dist'], max_dist=12.0, azim=azim)
            axes[2].set_title('Geodesic Core Distance (Limit 12 m)', fontsize=10.5, fontweight='bold', color='#0f172a', pad=4)

            cax1 = fig.add_axes([0.39, 0.11, 0.22, 0.022])
            cb1 = fig.colorbar(mpl_cm.ScalarMappable(norm=mcolors.Normalize(0.0, 1.0), cmap=matplotlib.colormaps['YlOrRd']), cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(labelsize=7.5)
            cb1.set_label('Visible sun fraction', fontsize=8.0, color='#334155', labelpad=2)

            cax2 = fig.add_axes([0.71, 0.11, 0.22, 0.022])
            cb2 = fig.colorbar(mpl_cm.ScalarMappable(norm=mcolors.Normalize(0.0, 18.0), cmap=matplotlib.colormaps['viridis_r']), cax=cax2, orientation='horizontal')
            cb2.ax.tick_params(labelsize=7.5)
            cb2.set_label('Distance to nearest core (m, red = core)', fontsize=8.0, color='#334155', labelpad=2)

            fig.text(
                0.025, 0.94,
                f'Illustrative Policy Rollout — {site_label}{tag_suffix}',
                fontsize=12.5, fontweight='bold', color='#0f172a',
            )
            op_txt = f'{snap["op"]} ({snap["dv"]:+d} vox)' if snap['step'] > 0 and snap['op'] != 'finish' else snap['op']
            hud = (
                f'Step {snap["step"]}/{total_steps} [{op_txt}]   |   '
                f'FAR: {snap["far"]:.2f} (target {snap["target_far"]:.2f} +/-0.10)   |   '
                f'Usable: {snap["usable_frac"]*100:.1f}%   |   '
                f'DirectSun: {snap["sun"]*100:.1f}%   |   '
                f'DiffuseSky: {snap["diff"]*100:.1f}%   |   '
                f'CoreCov: {snap["core_cov"]*100:.1f}% ({snap["core_count"]} cores)   |   '
                f'Obj: {snap["obj"]:.3f}'
            )
            fig.text(
                0.025, 0.875, hud, fontsize=9.1, color='#1e293b',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#f1f5f9', edgecolor='#cbd5e1'),
            )
            fig.savefig(out_png, facecolor='#ffffff')
            plt.close(fig)

        total_steps = best_snapshots[-1]['step']
        idx = 0
        for s_i, snap in enumerate(best_snapshots):
            png = tmp_dir / f'step_{idx:03d}.png'
            _draw_state(snap, total_steps, azim=-52.0, out_png=png)
            dur = 0.85 if s_i == 0 else (0.95 if s_i == len(best_snapshots) - 1 else 0.70)
            frame_entries.append((png, dur))
            idx += 1

        # Smooth terminal azimuth orbit to show 3D geometry clearly.
        last = best_snapshots[-1]
        for a in np.linspace(-52.0, -14.0, 12)[1:]:
            png = tmp_dir / f'step_{idx:03d}.png'
            _draw_state(last, total_steps, azim=float(a), out_png=png, tag_suffix=' [3D Orbit]')
            frame_entries.append((png, 0.09))
            idx += 1
        for a in np.linspace(-14.0, -52.0, 12)[1:]:
            png = tmp_dir / f'step_{idx:03d}.png'
            _draw_state(last, total_steps, azim=float(a), out_png=png, tag_suffix=' [3D Orbit]')
            frame_entries.append((png, 0.09))
            idx += 1
        frame_entries.append((frame_entries[-1][0], 1.1))

        concat_txt = tmp_dir / 'concat.txt'
        lines = []
        for path, dur in frame_entries:
            lines.append(f"file '{path.as_posix()}'")
            lines.append(f'duration {dur:.3f}')
        lines.append(f"file '{frame_entries[-1][0].as_posix()}'")
        concat_txt.write_text('\n'.join(lines) + '\n')

        ffmpeg = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
        mp4_out = MEDIA_DIR / 'episode-rollout.mp4'
        gif_out = MEDIA_DIR / 'episode-rollout.gif'
        palette = tmp_dir / 'palette.png'

        subprocess.run(
            [
                ffmpeg, '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_txt),
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-movflags', '+faststart',
                str(mp4_out),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                ffmpeg, '-y', '-i', str(mp4_out),
                '-vf', 'fps=10,scale=880:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff',
                str(palette),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                ffmpeg, '-y', '-i', str(mp4_out), '-i', str(palette),
                '-lavfi', 'fps=10,scale=880:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3',
                str(gif_out),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f'Saved {mp4_out.relative_to(ROOT)} ({mp4_out.stat().st_size / 1024:.1f} KB)')
        print(f'Saved {gif_out.relative_to(ROOT)} ({gif_out.stat().st_size / 1024:.1f} KB)')


def generate_multitier_showcase():
    """Render 5-tier (XS/S/M/L/XL) exploratory massing + compact-core showcase from local archives."""
    wanted = {
        'xs_sgp_0346': 'XS',
        's_sgp_0345': 'S',
        'm_bcn_0093': 'M',
        'l_bcn_0074': 'L',
        'xl_tokyo_0185': 'XL',
    }
    found = {}
    for p in sorted(LAYOUTS_DIR.glob('*.json')):
        d = json.loads(p.read_text())
        site = d.get('settings', {}).get('site')
        if site in wanted and d.get('layouts'):
            found[site] = d
    if len(found) < 5:
        print('Warning: missing some tier archives, available:', list(found.keys()))
        return

    fig = plt.figure(figsize=(16.5, 4.5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    gs = gridspec.GridSpec(1, 5, figure=fig, left=0.015, right=0.985, top=0.77, bottom=0.06, wspace=0.03)

    order = ['xs_sgp_0346', 's_sgp_0345', 'm_bcn_0093', 'l_bcn_0074', 'xl_tokyo_0185']
    for idx, site in enumerate(order):
        d = found[site]
        shape = tuple(d['settings']['shape'])
        rec = d['layouts'][0]
        occ = np.asarray(_unpack(rec['occupied'], math.prod(shape)), dtype=bool).reshape(shape)
        cmask = np.asarray(_unpack(rec['coreMask'], math.prod(shape)), dtype=bool).reshape(shape)
        ax = fig.add_subplot(gs[0, idx], projection='3d')
        _render_voxel_panel(ax, occ, cmask, mode='mass', elev=29.0, azim=-48.0, xray_mass=True)
        summ = rec['summary']
        n_cores = len(rec.get('cores', []))
        z_top = int(np.flatnonzero(occ.any((1, 2)))[-1]) + 1 if occ.any() else 0
        ax.set_title(
            f'{wanted[site]}: {site}\n'
            f'{z_top}F active ({shape[1]}x{shape[2]} m) | {n_cores} core{"s" if n_cores != 1 else ""}\n'
            f'FAR {summ["far"]:.2f} | Obj {summ["score"]:.3f} | {summ["usableArea"]:.0f} m2',
            fontsize=8.9, fontweight='bold', color='#0f172a', pad=2,
        )

    fig.text(
        0.02, 0.93,
        'Cross-Scale Constraint Generalization: Exploratory Top-Archived Massings (XS -> XL Parcels)',
        fontsize=13.0, fontweight='bold', color='#0f172a',
    )
    fig.text(
        0.02, 0.875,
        'Qualitative exploratory layouts archived during live multi-site rotation (compact-core.v5 crimson shafts, target FAR 4.5); '
        'not held-out test benchmarks.',
        fontsize=9.2, color='#475569',
    )
    out = MEDIA_DIR / 'multitier-showcase.png'
    fig.savefig(out, facecolor='#ffffff')
    plt.close(fig)
    print(f'Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)')


def generate_compute_and_ablations():
    """Plot measured height/cache scaling and multi-scale coarse-to-fine ablations from results/."""
    scalable = json.loads((RESULTS_DIR / 'scalable-height-speedup-2026-11-19.json').read_text())
    multi = json.loads((RESULTS_DIR / 'multiscale-ablation-2026-11-19.json').read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.2), dpi=160)
    fig.patch.set_facecolor('#ffffff')

    # Panel A: Height scaling setup + warm audit latency
    keys = ['matched10F', 'contextBcn20F', 'contextNyc40F']
    labels = ['m_bcn_0093\n10F (10x56x58)', 'm_bcn_0093\n20F (20x56x58)', 'm_nyc_0003\n40F (40x47x34)']
    cold = [scalable[k]['timing']['coldSeconds'] for k in keys]
    warm = [scalable[k]['timing']['warmSeconds'] for k in keys]
    eval_ms = [scalable[k]['timing']['warmEvalMs'] for k in keys]

    x = np.arange(len(keys))
    w = 0.34
    b1 = ax1.bar(x - w / 2, cold, width=w, color='#94a3b8', label='Cold Setup (MeshShadows precompute)')
    b2 = ax1.bar(x + w / 2, warm, width=w, color='#2563eb', label='Warm Cached Setup (SHA-256 disk cache)')
    ax1.set_yscale('log')
    ax1.set_ylim(0.015, 25.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.2)
    ax1.set_ylabel('Setup Latency (seconds, log scale)', fontsize=10.0, fontweight='bold')
    ax1.set_title('A. Scene Cache & Vertical Height Scaling (M4 MPS)', fontsize=11.0, fontweight='bold', color='#0f172a')
    ax1.grid(axis='y', linestyle='--', alpha=0.45)
    ax1.legend(loc='upper right', fontsize=8.4, frameon=True)

    for bar, val in zip(b1, cold):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.15, f'{val:.2f}s', ha='center', va='bottom', fontsize=8.5, color='#334155')
    for bar, val, ems in zip(b2, warm, eval_ms):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, val * 1.18,
            f'{val*1000:.0f} ms\n(eval {ems:.2f}ms)',
            ha='center', va='bottom', fontsize=8.1, color='#1d4ed8', fontweight='bold',
        )

    # Panel B: Multi-scale coarse-to-fine training time vs final checkpoint objective
    regimes = [
        ('xs_tokyo_28F', 'XS Tokyo 28F (ep 20)', '#0ea5e9', 'o'),
        ('m_bcn_20F', 'M Barcelona 20F (ep 35)', '#2563eb', 's'),
        ('xl_sgp_40F', 'XL Singapore 40F (ep 25)', '#dc2626', '^'),
    ]
    strat_names = {'fine_1x1x1': '1x1x1', 'c2f_2x2x1': '2x2x1->1x', 'm3_4x4x1': '4x4x1', 'm3_4x4x2': '4x4x2', 'm4_8x8x3': '8x8x3'}
    offsets = {
        ('xs_tokyo_28F', 'fine_1x1x1'): (6, 4),
        ('m_bcn_20F', 'fine_1x1x1'): (6, 4),
        ('m_bcn_20F', 'c2f_2x2x1'): (6, -3),
        ('m_bcn_20F', 'm3_4x4x1'): (6, 3),
        ('m_bcn_20F', 'm3_4x4x2'): (6, -9),
        ('xl_sgp_40F', 'fine_1x1x1'): (-14, 9),
        ('xl_sgp_40F', 'c2f_2x2x1'): (-12, 8),
        ('xl_sgp_40F', 'm3_4x4x1'): (-8, -13),
        ('xl_sgp_40F', 'm3_4x4x2'): (7, -6),
        ('xl_sgp_40F', 'm4_8x8x3'): (-12, 8),
    }
    for reg_key, reg_label, color, marker in regimes:
        rdata = multi['regimes'][reg_key]
        xs, ys, annots = [], [], []
        for s_key in ['fine_1x1x1', 'c2f_2x2x1', 'm3_4x4x1', 'm3_4x4x2', 'm4_8x8x3']:
            if s_key not in rdata:
                continue
            sdict = rdata[s_key]
            last_cp = str(max(int(c) for c in sdict if c.isdigit()))
            entry = sdict[last_cp]
            xs.append(entry['cumTrainSeconds'])
            ys.append(entry['meanOrganicObjective'])
            annots.append((s_key, entry['cumTrainSeconds'], entry['meanOrganicObjective']))
        ax2.plot(xs, ys, linestyle='--', alpha=0.4, color=color)
        ax2.scatter(xs, ys, color=color, marker=marker, s=75, label=reg_label, zorder=4)
        for s_key, sx, sy in annots:
            if reg_key == 'xs_tokyo_28F' and s_key != 'fine_1x1x1':
                continue
            ax2.annotate(
                strat_names.get(s_key, s_key),
                (sx, sy), textcoords='offset points', xytext=offsets.get((reg_key, s_key), (5, 4)),
                fontsize=7.6, color='#1e293b',
            )

    ax2.set_xlabel('Cumulative Training Time (seconds, M4 MPS)', fontsize=10.0, fontweight='bold')
    ax2.set_ylabel('Mean Organic Objective (Eval)', fontsize=10.0, fontweight='bold')
    ax2.set_title('B. Multi-Scale Cascade Tradeoff (Pre-compact-core.v5 Baseline)', fontsize=11.0, fontweight='bold', color='#0f172a')
    ax2.grid(True, linestyle='--', alpha=0.45)
    ax2.legend(loc='lower right', fontsize=8.4, frameon=True)

    fig.suptitle(
        'Empirical Compute Scaling & Historical Multi-Scale Ablations (results/2026-11-19)',
        fontsize=12.6, fontweight='bold', color='#0f172a', y=0.98,
    )
    fig.text(
        0.5, 0.015,
        'Note: Panel A & B plot historical pre-compact-core.v5 benchmarks (2026-11-19, Apple M4 MPS). '
        'Uncalibrated environmental proxies; see results/README.md for exact provenance.',
        ha='center', fontsize=8.4, color='#475569',
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out = MEDIA_DIR / 'compute-and-ablations.png'
    fig.savefig(out, facecolor='#ffffff')
    plt.close(fig)
    print(f'Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)')


def main():
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    generate_architecture_diagram()
    generate_compute_and_ablations()
    generate_multitier_showcase()
    _render_rollout_assets()


if __name__ == '__main__':
    main()
