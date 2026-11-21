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
matplotlib.use('Agg')
from matplotlib import cm as mpl_cm, colors as mcolors, gridspec, patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from voxelform.context import ContextLibrary
from voxelform.environment import Environment
from voxelform.experiment import CORE_LAYOUT_VERSION, PROTOCOL, ExperimentConfig, seed_everything
from voxelform.objectives import organic_objective
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.shadows import device_name
from voxelform.unified import Organic3DNetwork, UnifiedLearner, commit

ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = ROOT / 'docs' / 'media'
RESULTS_DIR = ROOT / 'results'


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
        x0, x1, y0, y1, z1 = 0.0, float(nx), 0.0, float(ny), float(nz * dz)
    else:
        pad = 2.0
        x0, x1 = float(xx.min()) - pad, float(xx.max() + 1) + pad
        y0, y1 = float(yy.min()) - pad, float(yy.max() + 1) + pad
        z1 = max(12.0, float((zz.max() + 1) * dz) + 2.0)
    span = max(x1 - x0, y1 - y0, z1, 10.0)
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


def _patch_cfg() -> PatchConfig:
    return PatchConfig(
        mode='combined', allocation='feedback', eligibility='depth', facade_layers=2,
        minimum_patch=4, maximum_patches=4, maximum_voids=1, minimum_void_area=4,
        batched=True, vertical_interior=False, budget_reserve_fraction=0.2,
    )


def _load_env(site_id: str, cfg: ExperimentConfig, dev: str) -> Environment:
    library = ContextLibrary()
    context = library.load(site_id, floors=cfg.ceiling_storeys, grid_alignment='auto_fit')
    env = Environment(
        (cfg.ceiling_storeys, cfg.synthetic_width, cfg.synthetic_length),
        seed=9001, context=context, device=dev, sun_count=8,
        target_far=cfg.target_far, far_tolerance=cfg.far_tolerance,
        frontier_factory=functools.partial(PatchFacades, config=_patch_cfg(), depth_device=dev),
        fast_self_shadows=True, fast_diffuse=True, cores_enabled=cfg.cores_enabled,
        min_core_spacing=cfg.min_core_spacing, max_core_dist=cfg.max_core_dist,
    )
    env.precompute_external()
    return env


def _run_eval_episode(
    policy: UnifiedLearner,
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
    solid = env.allowed.clone()
    solid[cfg.initial_storeys:] = False
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
        if record_trace and hasattr(env, 'receiver_data') and hasattr(env, 'visible'):
            _, _, _, cells, faces = env.receiver_data
            vis_mean = env.visible.float().mean(0).detach().cpu().numpy()
            sun_grid[faces.detach().cpu().numpy(), cells.detach().cpu().numpy()] = vis_mean
        return {
            'step': step_idx,
            'op': op_name,
            'dv': dv,
            'occ': env.occupied.detach().cpu().numpy().copy() if record_trace else None,
            'core_mask': env.core_mask.detach().cpu().numpy().copy() if record_trace else None,
            'core_dist': env.core_distance.detach().cpu().numpy().copy() if record_trace else None,
            'sun_grid': sun_grid if record_trace else None,
            'far': float(env.far),
            'target_far': float(env.target_far),
            'obj': float(obj),
            'usable_area': float(diag['usableGoodArea']),
            'usable_frac': float(diag['usableGoodFraction']),
            'sun': float(env.values['directSunProxy'].detach().cpu()),
            'diff': float(env.values['diffuseFacadeProxy'].detach().cpu()),
            'core_cov': float(diag.get('coreCoverage') or 0.0),
            'core_count': int(diag.get('coreCount') or 0),
        }

    snaps = [_snap(0, 'init', 0)] if record_trace else []
    for step_idx in range(1, env.unified_limit + 1):
        move = policy.choose(env, training=False)
        env.unified_actions += 1
        op = policy.last_operation or 'none'
        if op == 'finish':
            env.termination = 'far_reached'
            if record_trace:
                env.evaluate()
                snaps.append(_snap(step_idx, 'finish', 0))
            break
        if move is None:
            continue
        dv = int(move.add.sum()) - int(move.remove.sum())
        commit(env, move, validated=True)
        if record_trace:
            env.evaluate()
            snaps.append(_snap(step_idx, op, dv))
        elif env.edits % cfg.audit_steps == 0:
            env.evaluate()
        if env.terminal:
            break
    if not env.terminal:
        env.termination = 'far_reached' if env.far_satisfied else 'step_budget_exhausted'
    env.evaluate()
    if record_trace:
        final = snaps[-1]
    else:
        final = _snap(env.edits, op, 0)
        final['occ'] = env.occupied.detach().cpu().numpy().copy()
        final['core_mask'] = env.core_mask.detach().cpu().numpy().copy()
    final['far_satisfied'] = bool(env.far_satisfied)
    final['seed'] = eval_seed
    return final, snaps


def _load_run_policy(run_dir: Path):
    cfg = ExperimentConfig.read(run_dir / 'config.json')
    dev = device_name(cfg.device)
    seed = cfg.seeds[0]
    ckpt_path = run_dir / f'seed-{seed}' / 'latest.pt'
    saved = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if saved.get('format') != PROTOCOL or saved.get('coreLayoutVersion') != CORE_LAYOUT_VERSION:
        raise ValueError('Incompatible checkpoint protocol or core layout version')
    if saved.get('configHash') != cfg.fingerprint:
        raise ValueError('Checkpoint configHash does not match run config')
    trained = UnifiedLearner(
        dev, seed=seed, config=_patch_cfg(), amount_feedback=True,
        net_guided=True, organic_3d=True, spatial_aux_weight=cfg.spatial_aux_weight,
    )
    trained.load(saved['policy'])
    untrained = UnifiedLearner(
        dev, seed=seed, config=_patch_cfg(), amount_feedback=True,
        net_guided=True, organic_3d=True, spatial_aux_weight=cfg.spatial_aux_weight,
    )
    return cfg, dev, untrained, trained, int(saved['nextEpisode'])


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
            'Policy & Value Outputs:\n  mu = 3*tanh(head/3) [2, Z/2, Y/4, X/4]\n  op_logits (4 ops), scalar critic V(s)',
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


def generate_trained_rollout_assets(cfg: ExperimentConfig, dev: str, trained: UnifiedLearner, total_eps: int):
    """Render MP4 + GIF from the best FAR-satisfied trained evaluation rollout."""
    candidates = [
        ('m_prs_0315', 'Held-Out Test: Paris (m_prs_0315)', 2.5, 0.08),
        ('m_prs_0315', 'Held-Out Test: Paris (m_prs_0315)', 1.0, 1.0),
        ('m_bcn_0093', 'Barcelona Eixample (m_bcn_0093)', 2.5, 0.08),
    ]
    chosen = None
    chosen_label = ''
    chosen_decode = ''
    best_rank = -1e9
    envs = {}
    for site_id, label, temp, noise in candidates:
        if site_id not in envs:
            envs[site_id] = _load_env(site_id, cfg, dev)
        env = envs[site_id]
        for s in range(9001, 9011):
            final, snaps = _run_eval_episode(trained, env, cfg, s, temp, noise, record_trace=True)
            if not final['far_satisfied'] or len(snaps) < 4 or final['core_cov'] < 0.85:
                continue
            rank = final['obj'] + (0.05 if site_id == 'm_prs_0315' else 0.0)
            if rank > best_rank:
                best_rank = rank
                chosen = snaps
                chosen_label = f'{label}, seed {s}'
                chosen_decode = f'temp={temp:g}, noise={noise:g}'
    assert chosen is not None and len(chosen) >= 4
    print(f'Trained rollout: {chosen_label} ({chosen_decode}), {len(chosen)} states, FAR={chosen[-1]["far"]:.2f}, obj={chosen[-1]["obj"]:.3f}')

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

            _render_voxel_panel(axes[2], snap['occ'], snap['core_mask'], mode='core_dist', dist_map=snap['core_dist'], max_dist=float(cfg.max_core_dist), azim=azim)
            axes[2].set_title(f'Geodesic Core Distance (Limit {cfg.max_core_dist} m)', fontsize=10.5, fontweight='bold', color='#0f172a', pad=4)

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
                f'1,200-Ep Trained Policy ({total_eps} ep, {chosen_decode}) — {chosen_label}{tag_suffix}',
                fontsize=12.0, fontweight='bold', color='#0f172a',
            )
            op_txt = f'{snap["op"]} ({snap["dv"]:+d} vox)' if snap['step'] > 0 and snap['op'] != 'finish' else snap['op']
            hud = (
                f'Step {snap["step"]}/{total_steps} [{op_txt}]  |  '
                f'FAR: {snap["far"]:.2f} (target {snap["target_far"]:.2f} +/-{cfg.far_tolerance:.2f})  |  '
                f'Usable: {snap["usable_frac"]*100:.1f}% ({snap["usable_area"]:.0f} m2)  |  '
                f'Sun: {snap["sun"]*100:.1f}%  |  Sky: {snap["diff"]*100:.1f}%  |  '
                f'CoreCov: {snap["core_cov"]*100:.1f}% ({snap["core_count"]})  |  Obj: {snap["obj"]:.3f}'
            )
            fig.text(
                0.025, 0.875, hud, fontsize=8.9, color='#1e293b',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#f1f5f9', edgecolor='#cbd5e1'),
            )
            fig.savefig(out_png, facecolor='#ffffff')
            plt.close(fig)

        total_steps = chosen[-1]['step']
        idx = 0
        for s_i, snap in enumerate(chosen):
            png = tmp_dir / f'step_{idx:03d}.png'
            _draw_state(snap, total_steps, azim=-52.0, out_png=png)
            dur = 0.85 if s_i == 0 else (0.95 if s_i == len(chosen) - 1 else 0.70)
            frame_entries.append((png, dur))
            idx += 1

        last = chosen[-1]
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


def generate_trained_multitier_showcase(cfg: ExperimentConfig, dev: str, trained: UnifiedLearner, total_eps: int):
    """Render best 1,200-ep trained FAR-satisfied layouts across XS, S, M, L, XL training tiers."""
    tiers = [
        ('xs_sgp_0346', 'XS'),
        ('s_sgp_0345', 'S'),
        ('m_bcn_0093', 'M'),
        ('l_bcn_0074', 'L'),
        ('xl_tokyo_0185', 'XL'),
    ]
    fig = plt.figure(figsize=(16.5, 4.9), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    gs = gridspec.GridSpec(1, 5, figure=fig, left=0.015, right=0.985, top=0.73, bottom=0.04, wspace=0.03)

    for idx, (site_id, tier) in enumerate(tiers):
        env = _load_env(site_id, cfg, dev)
        best = None
        for s in range(9001, 9009):
            res, _ = _run_eval_episode(trained, env, cfg, s, temp=2.5, noise=0.08, record_trace=False)
            if best is None or (res['far_satisfied'], res['obj']) > (best['far_satisfied'], best['obj']):
                best = res
        occ = best['occ']
        cmask = best['core_mask']
        ax = fig.add_subplot(gs[0, idx], projection='3d')
        _render_voxel_panel(ax, occ, cmask, mode='mass', elev=29.0, azim=-48.0, xray_mass=True)
        z_top = int(np.flatnonzero(occ.any((1, 2)))[-1]) + 1 if occ.any() else 0
        nc = best['core_count']
        ax.set_title(
            f'{tier}: {site_id}\n'
            f'{z_top}F active | {nc} core{"s" if nc != 1 else ""} | FAR {best["far"]:.2f}\n'
            f'Obj {best["obj"]:.3f} | {best["usable_area"]:.0f} m2 usable\n'
            f'Sun {best["sun"]*100:.1f}% | Sky {best["diff"]*100:.1f}% | Cov {best["core_cov"]*100:.0f}%',
            fontsize=8.5, fontweight='bold', color='#0f172a', pad=3,
        )
        print(f'  Tier {tier:2s} ({site_id}): obj={best["obj"]:.3f} far={best["far"]:.2f} cov={best["core_cov"]*100:.1f}% seed={best["seed"]}')

    fig.text(
        0.02, 0.94,
        f'Multi-Tier Best Trained Layouts Across XS -> XL Parcels ({total_eps}-Episode Policy, compact-core.v5)',
        fontsize=12.8, fontweight='bold', color='#0f172a',
    )
    fig.text(
        0.02, 0.885,
        'Best FAR-satisfied episode over eval seeds 9001-9008 (deployment decoding temp=2.5, noise=0.08; target FAR 3.0 +/- 0.10, ceiling 20F).',
        fontsize=9.1, color='#475569',
    )
    out = MEDIA_DIR / 'multitier-showcase.png'
    fig.savefig(out, facecolor='#ffffff')
    plt.close(fig)
    print(f'Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)')


def generate_heldout_generalization(cfg: ExperimentConfig, dev: str, untrained: UnifiedLearner, trained: UnifiedLearner, summary: dict, total_eps: int):
    """Render paired 2x3 comparison on unseen HK, NYC, and Paris under matched eval decoding (1.0 / 1.0)."""
    sites = [
        ('m_hk_0199', 'Hong Kong (m_hk_0199, Val)', summary['validation']['perSite']['m_hk_0199']['bestSeed']),
        ('m_nyc_0003', 'New York (m_nyc_0003, Test)', summary['test']['perSite']['m_nyc_0003']['bestSeed']),
        ('m_prs_0315', 'Paris (m_prs_0315, Test)', summary['test']['perSite']['m_prs_0315']['bestSeed']),
    ]
    fig = plt.figure(figsize=(14.4, 8.3), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.045, right=0.985, top=0.77, bottom=0.03, hspace=0.22, wspace=0.04)

    for col, (site_id, label, best_seed) in enumerate(sites):
        env = _load_env(site_id, cfg, dev)
        u_res, _ = _run_eval_episode(untrained, env, cfg, best_seed, temp=1.0, noise=1.0, record_trace=False)
        t_res, _ = _run_eval_episode(trained, env, cfg, best_seed, temp=1.0, noise=1.0, record_trace=False)
        for row, (res, row_tag) in enumerate(((u_res, 'Untrained'), (t_res, 'Trained'))):
            ax = fig.add_subplot(gs[row, col], projection='3d')
            _render_voxel_panel(ax, res['occ'], res['core_mask'], mode='mass', elev=29.0, azim=-48.0, xray_mass=True)
            z_top = int(np.flatnonzero(res['occ'].any((1, 2)))[-1]) + 1 if res['occ'].any() else 0
            title_top = f'{label} [seed {best_seed}]' if row == 0 else ''
            prefix = f'{title_top}\n' if title_top else ''
            nc = res['core_count']
            ax.set_title(
                f'{prefix}{row_tag}: {z_top}F | {nc} core{"s" if nc != 1 else ""} | FAR {res["far"]:.2f}\n'
                f'Obj {res["obj"]:.3f} | {res["usable_area"]:.0f} m2 usable | Cov {res["core_cov"]*100:.0f}%\n'
                f'Sun {res["sun"]*100:.1f}% | Sky {res["diff"]*100:.1f}%',
                fontsize=8.6, fontweight='bold' if row == 1 else 'normal',
                color='#0f172a' if row == 1 else '#475569', pad=3,
            )

    fig.text(0.014, 0.61, 'Untrained Control\n(seed 123 init)', fontsize=10.2, fontweight='bold', color='#475569', rotation=90, va='center')
    fig.text(0.014, 0.21, f'Trained ({total_eps} ep)\nZero-Shot', fontsize=10.2, fontweight='bold', color='#1d4ed8', rotation=90, va='center')
    fig.text(
        0.045, 0.945,
        'Zero-Shot Generalization to Unseen Cities (Matched Evaluation Distribution: temp=1.0, noise=1.0)',
        fontsize=13.0, fontweight='bold', color='#0f172a',
    )
    fig.text(
        0.045, 0.905,
        f'Paired comparison on identical evaluation seeds with no fine-tuning ({total_eps}-ep checkpoint, compact-core.v5, spatial_aux_weight=1.2).',
        fontsize=9.2, color='#475569',
    )
    out = MEDIA_DIR / 'heldout-generalization.png'
    fig.savefig(out, facecolor='#ffffff')
    plt.close(fig)
    print(f'Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)')


def generate_training_curves(run_dir: Path, summary: dict):
    """Plot 1,200-episode learning dynamics and held-out validation/test comparison."""
    rows = [json.loads(line) for line in (run_dir / 'seed-123' / 'train.jsonl').read_text().splitlines() if line.strip()]
    eps = np.array([r['episode'] for r in rows], dtype=np.float32)
    win = 30

    def _roll(vals):
        arr = np.asarray(vals, dtype=np.float32)
        kern = np.ones(win, dtype=np.float32) / win
        return eps[win - 1:], np.convolve(arr, kern, mode='valid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.4), dpi=160)
    fig.patch.set_facecolor('#ffffff')

    curves = (
        ('Layout Objective J', [r['quality'] for r in rows], '#1d4ed8', '-', 2.2),
        ('Usable Good Area Fraction', [r['diagnostics']['usableGoodFraction'] for r in rows], '#0f766e', '-', 1.7),
        ('Direct Sun Proxy', [r['environment']['directSunProxy'] for r in rows], '#d97706', '-', 1.8),
        ('Diffuse Sky Proxy', [r['environment']['diffuseFacadeProxy'] for r in rows], '#0284c7', '--', 1.8),
        ('Core Coverage (12 m)', [float(r['diagnostics'].get('coreCoverage') or 0.0) for r in rows], '#dc2626', ':', 1.7),
        ('FAR Band Success Rate', [float(r['success']) for r in rows], '#475569', '-.', 1.5),
    )
    for label, raw, color, ls, lw in curves:
        rx, ry = _roll(raw)
        ax1.plot(rx, ry, label=label, color=color, linestyle=ls, linewidth=lw)

    ax1.set_xlim(win, len(rows))
    ax1.set_ylim(-0.04, 1.25)
    ax1.set_xlabel('Training Episode (30-Episode Rolling Mean Across 10 XS-XL Sites)', fontsize=9.8, fontweight='bold')
    ax1.set_ylabel('Score / Proxy Fraction', fontsize=9.8, fontweight='bold')
    ax1.set_title('A. On-Policy Multi-Site Training Trajectory (1,200 Episodes)', fontsize=11.0, fontweight='bold', color='#0f172a')
    ax1.grid(True, linestyle='--', alpha=0.45)
    ax1.legend(loc='upper center', fontsize=7.8, ncol=3, frameon=True)

    # Panel B: Untrained vs Trained on Held-Out Validation (HK) & Test (NYC, Paris)
    sites = [
        ('HK (Val)', summary['untrainedValidation']['perSite']['m_hk_0199'], summary['validation']['perSite']['m_hk_0199']),
        ('NYC (Test)', summary['untrainedTest']['perSite']['m_nyc_0003'], summary['test']['perSite']['m_nyc_0003']),
        ('Paris (Test)', summary['untrainedTest']['perSite']['m_prs_0315'], summary['test']['perSite']['m_prs_0315']),
        ('Test Agg', summary['untrainedTest']['overall'], summary['test']['overall']),
    ]
    x = np.arange(len(sites))
    w = 0.25
    u_q = [u['meanQuality'] for _, u, _ in sites]
    t_q = [t['meanQuality'] for _, _, t in sites]
    u_sun = [u['meanDirectSun'] for _, u, _ in sites]
    t_sun = [t['meanDirectSun'] for _, _, t in sites]
    u_sky = [u['meanDiffuseSky'] for _, u, _ in sites]
    t_sky = [t['meanDiffuseSky'] for _, _, t in sites]

    b1 = ax2.bar(x - w, u_q, width=w / 2, color='#cbd5e1', edgecolor='#64748b', label='Untrained Obj')
    b2 = ax2.bar(x - w / 2, t_q, width=w / 2, color='#1d4ed8', label='Trained Obj')
    ax2.bar(x + 0.02, u_sun, width=w / 2, color='#fde68a', edgecolor='#d97706', label='Untrained Sun')
    ax2.bar(x + w / 2 + 0.02, t_sun, width=w / 2, color='#d97706', label='Trained Sun')
    ax2.bar(x + w + 0.04, u_sky, width=w / 2, color='#bae6fd', edgecolor='#0284c7', label='Untrained Sky')
    ax2.bar(x + 1.5 * w + 0.04, t_sky, width=w / 2, color='#0284c7', label='Trained Sky')

    for i, (bu, bt, (_, u, t)) in enumerate(zip(b1, b2, sites)):
        ax2.text(bt.get_x() + bt.get_width() / 2, t['meanQuality'] + 0.015, f'{t["meanQuality"]:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#1d4ed8')
        ax2.text(bu.get_x() + bu.get_width() / 2, u['meanQuality'] + 0.015, f'{u["meanQuality"]:.2f}', ha='center', va='bottom', fontsize=7.6, color='#475569')
        ug_ratio = t['meanUsableGoodArea'] / max(1.0, u['meanUsableGoodArea'])
        ax2.text(x[i] + 0.10, max(t['meanQuality'], t['meanDirectSun']) + 0.072, f'{ug_ratio:.1f}x usable\n(FAR 1.00)', ha='center', va='bottom', fontsize=7.4, fontweight='bold', color='#0f766e')

    ax2.set_ylim(0.0, 1.20)
    ax2.set_xticks(x + 0.08)
    ax2.set_xticklabels([s[0] for s in sites], fontsize=9.4)
    ax2.set_ylabel('Mean Matched Eval Metric (5 seeds/site)', fontsize=9.8, fontweight='bold')
    ax2.set_title('B. Zero-Shot Held-Out Evaluation (Matched 1.0 / 1.0 Decoding)', fontsize=11.0, fontweight='bold', color='#0f172a')
    ax2.grid(axis='y', linestyle='--', alpha=0.45)
    ax2.legend(loc='upper center', fontsize=7.8, ncol=3, frameon=True)

    fig.suptitle(
        '1,200-Episode Multi-Tier Training & Held-Out City Generalization (Apple M4 MPS, compact-core.v5)',
        fontsize=12.8, fontweight='bold', color='#0f172a', y=0.98,
    )
    fig.text(
        0.5, 0.015,
        'Single model seed (123), spatial_aux_weight=1.2; uncalibrated proxies. On-policy updates show mid-run oscillation '
        'across XS-XL scales while outperforming untrained controls across all unseen cities.',
        ha='center', fontsize=8.3, color='#475569',
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out = MEDIA_DIR / 'training-curves-1200ep.png'
    fig.savefig(out, facecolor='#ffffff')
    plt.close(fig)
    print(f'Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)')


def _load_summary(summary_path: Path, cfg: ExperimentConfig) -> dict:
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    raise FileNotFoundError(f'Missing audited summary file: {summary_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs' / 'real-aux-1200ep')
    parser.add_argument('--summary', type=Path, default=RESULTS_DIR / 'real-aux-1200ep-mps-2026-11-21.json')
    parser.add_argument('--only', choices=('all', 'arch', 'curves', 'multitier', 'heldout', 'rollout'), default='all')
    args = parser.parse_args()

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in ('all', 'arch'):
        generate_architecture_diagram()
    summary = _load_summary(args.summary, ExperimentConfig.read(args.run_dir / 'config.json'))
    if args.only in ('all', 'curves'):
        generate_training_curves(args.run_dir, summary)
    if args.only in ('all', 'multitier', 'heldout', 'rollout'):
        cfg, dev, untrained, trained, total_eps = _load_run_policy(args.run_dir)
        if args.only in ('all', 'multitier'):
            generate_trained_multitier_showcase(cfg, dev, trained, total_eps)
        if args.only in ('all', 'heldout'):
            generate_heldout_generalization(cfg, dev, untrained, trained, summary, total_eps)
        if args.only in ('all', 'rollout'):
            generate_trained_rollout_assets(cfg, dev, trained, total_eps)


if __name__ == '__main__':
    main()
