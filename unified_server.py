"""Unified VoxelForm Live Preview Server.

Runs the unified action space (neural remove / roof grow / exchange / finish)
with vertical headroom (10 storeys envelope, 6 storeys initial mass)
on an independent port without modifying the VoxelForm repository.
"""
import functools
import heapq
import io
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# Add VoxelForm-RL to Python path
VOXELFORM_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VOXELFORM_ROOT))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.ndimage import gaussian_filter
from shapely import Polygon, constrained_delaunay_triangles
from shapely.geometry import LineString, MultiPolygon, box
from shapely.ops import unary_union

from voxelform.context import ContextLibrary, public_scene
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.session import DENSITY_PRESETS
from voxelform.shadows import device_name, sync
from voxelform.training import depth_potential
from voxelform.unified import UnifiedLearner, commit, OPERATIONS

torch.set_num_threads(max(1, int(os.getenv('VOXELFORM_THREADS', '1'))))

class UnifiedSession:
    def __init__(self, library=None, device=None, initial_storeys=6, ceiling_storeys=10):
        self.library = library or ContextLibrary()
        self.lock = threading.RLock()
        self.running = False
        self.single = False
        self.training = False
        self.ready = False
        self.error = None
        self.initial_storeys = initial_storeys
        self.ceiling_storeys = ceiling_storeys

        self.active = 0.0
        self.started = None
        self.episodes = 0
        self.steps = 0
        self.revision = 0
        self.pending_seed = 123
        self.plan = 'rectangle'
        self.site_id = None
        self.recent = []
        self.edit_trace = []
        self.stats = dict(updates=0, directMaskUpdates=0, importedUpdates=0, loss=0.0, gradientNorm=0.0)
        self.rollout_steps = 32
        self.audit_steps = 8
        self.force_operation = None

        chosen_device = device or os.getenv('VOXELFORM_DEVICE', 'auto')
        self.device = device_name(chosen_device)
        self.target_far = 3.0
        self.density_preset = 'medium'
        self.episode_recorded = False
        self.action_seconds = 0.0
        self.learning_seconds = 0.0

        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def clock(self):
        return self.active + (time.perf_counter() - self.started if self.started is not None else 0.0)

    def pause(self):
        if self.started is not None:
            self.active += time.perf_counter() - self.started
            self.started = None
        self.running = False
        self.single = False
        self.force_operation = None

    def build(self):
        catalog = self.library.catalog()
        if not catalog:
            raise ValueError('No cached urban scenes found. Ensure Module-Lab is accessible.')
        initial_id = os.getenv('VOXELFORM_SITE_ID', 'm_nyc_0003')
        initial = next((r for r in catalog if r['site_id'] == initial_id), catalog[0])
        self.site_id = initial['site_id']

        context = self.library.load(self.site_id, floors=self.ceiling_storeys, grid_alignment='auto_fit')
        self.config = PatchConfig(
            mode='combined',
            allocation='feedback',
            eligibility='depth',
            facade_layers=2,
            minimum_patch=4,
            maximum_patches=4,
            maximum_voids=1,
            minimum_void_area=4,
            batched=True,
            vertical_interior=True,
            budget_reserve_fraction=0.2,
        )
        factory = functools.partial(PatchFacades, config=self.config, depth_device=self.device if self.device != 'cpu' else None)
        self.env = Environment(
            shape=context['shape'],
            context=context,
            device=self.device,
            far_tolerance=0.1,
            frontier_factory=factory,
        )

        solid = self.env.allowed.clone()
        solid[self.initial_storeys:] = False
        self.env.restore_occupancy(solid)
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None

        self.policy = UnifiedLearner(
            device=self.device,
            seed=self.pending_seed,
            config=self.config,
            amount_feedback=True,
            net_guided=False,
        )

        # Check for pre-trained unified checkpoint
        ckpt_path = VOXELFORM_ROOT / 'results/checkpoints/unified-2026-11-10/unified-123.pt'
        if ckpt_path.is_file():
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                if 'policy' in ckpt:
                    self.policy.load(ckpt['policy'])
                    self.policy.net_guided = False
                    self.policy.amount_feedback = True
                print(f'Loaded pre-trained checkpoint from {ckpt_path.name} (updates: {self.policy.updates})')
            except Exception as e:
                print(f'Note: Could not load checkpoint {ckpt_path}: {e}')

        self.stats = dict(
            updates=self.policy.updates,
            directMaskUpdates=self.policy.direct_updates,
            importedUpdates=0,
            loss=0.0,
            gradientNorm=0.0,
        )
        self.ready = True
        self.revision += 1

    def next_episode(self):
        self.env.reset()
        solid = self.env.allowed.clone()
        solid[self.initial_storeys:] = False
        self.env.restore_occupancy(solid)
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None
        self.steps = 0
        self.policy.clear()
        self.episode_recorded = False
        self.edit_trace = []
        self.revision += 1

    def start(self, training=True, single=False, force_operation=None):
        new_training = bool(training)
        if new_training != self.training:
            self.policy.clear()
        self.training = new_training
        self.force_operation = force_operation if force_operation in ('remove', 'grow', 'exchange') else None
        if single:
            self.pause()
            self.force_operation = force_operation if force_operation in ('remove', 'grow', 'exchange') else None
            self.single = True
        else:
            self.running = True
            if self.started is None:
                self.started = time.perf_counter()

    def configure_far(self, target, preset='custom', tolerance=None):
        self.env.validate_far(target)
        if preset not in (*DENSITY_PRESETS, 'custom'):
            raise ValueError('Unknown density preset')
        if preset != 'custom' and float(target) != DENSITY_PRESETS[preset]:
            raise ValueError('Density preset does not match target FAR')
        tolerance = self.env.far_tolerance if tolerance is None else self.env.validate_tolerance(tolerance)
        self.pause()
        self.env.configure_far(target, tolerance)
        self.target_far = self.env.target_far
        self.density_preset = preset
        self.steps = 0
        self.episode_recorded = False
        self.edit_trace = []
        self.policy.clear()
        self.revision += 1

    def change_site(self, source, site_id=None, plan='rectangle'):
        if source not in ('cached_osm', 'synthetic'):
            raise ValueError('Unknown context source')
        if plan not in ('rectangle', 'triangle', 'courtyard', 'branch'):
            raise ValueError('Unknown synthetic plot')
        self.pause()
        context = self.library.load(site_id or self.site_id, floors=self.ceiling_storeys, grid_alignment=(self.env.context or {}).get('gridAlignment', 'auto_fit')) if source == 'cached_osm' else None
        factory = functools.partial(PatchFacades, config=self.config, depth_device=self.device if self.device != 'cpu' else None)
        shape = context['shape'] if context else (self.ceiling_storeys, 24, 32)
        replacement = Environment(
            shape=shape,
            seed=self.pending_seed + 1,
            device=self.device,
            plan=plan,
            context=context,
            target_far=self.target_far,
            frontier_factory=factory,
            far_tolerance=self.env.far_tolerance,
        )
        self.env = replacement
        solid = self.env.allowed.clone()
        solid[self.initial_storeys:] = False
        self.env.restore_occupancy(solid)
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None

        self.pending_seed += 1
        self.plan = plan
        self.site_id = context['site_id'] if context else None
        self.steps = 0
        self.episode_recorded = False
        self.edit_trace = []
        self.policy.clear()
        self.revision += 1

    def configure_grid(self, alignment):
        if self.env.context is None:
            raise ValueError('Grid alignment requires a fetched plot')
        context = self.library.load(self.site_id, floors=self.ceiling_storeys, grid_alignment=alignment)
        if context['gridAlignment'] == self.env.context.get('gridAlignment', 'world'):
            return
        self.pause()
        factory = functools.partial(PatchFacades, config=self.config, depth_device=self.device if self.device != 'cpu' else None)
        replacement = Environment(
            shape=context['shape'],
            seed=self.pending_seed,
            device=self.device,
            context=context,
            target_far=self.target_far,
            frontier_factory=factory,
            far_tolerance=self.env.far_tolerance,
        )
        self.env = replacement
        solid = self.env.allowed.clone()
        solid[self.initial_storeys:] = False
        self.env.restore_occupancy(solid)
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None

        self.steps = 0
        self.episode_recorded = False
        self.edit_trace = []
        self.policy.clear()
        self.revision += 1

    def audit_environment(self):
        before = float(self.env.quality.detach().cpu())
        if self.env.last_audit_edit != self.env.edits:
            self.env.evaluate()
        if self.training and self.policy.tape:
            bonus = 5.0 * (float(self.env.quality.detach().cpu()) - before)
            self.policy.add_reward(bonus)

    def learn_rollout(self):
        if self.training and self.policy.tape:
            stamp = time.perf_counter()
            self.stats = self.policy.finish(self.policy.fields(self.env)[1].detach())
            sync(self.device)
            self.learning_seconds = time.perf_counter() - stamp

    def finish_episode(self):
        if self.episode_recorded:
            return
        self.audit_environment()
        self.learn_rollout()
        self.episodes += 1
        self.episode_recorded = True
        self.recent.append(
            dict(
                episode=self.episodes,
                siteId=self.site_id,
                seed=self.pending_seed,
                training=self.training,
                densityPreset=self.density_preset,
                success=self.env.termination == 'far_reached',
                **self.env.metrics(),
            )
        )
        self.recent = self.recent[-100:]
        if not self.training or self.env.termination != 'far_reached' or self.env.edits == 0:
            self.pause()

    def advance_one(self):
        if not (self.running or self.single):
            return
        if self.env.terminal:
            if not self.episode_recorded:
                self.finish_episode()
                self.single = False
                self.revision += 1
                return
            self.next_episode()
            if self.env.terminal:
                self.finish_episode()
                self.single = False
                self.revision += 1
                return

        stamp = time.perf_counter()
        force_op = self.force_operation if not self.training else None
        move = self.policy.choose(self.env, training=self.training, force_operation=force_op)
        sync(self.device)
        self.action_seconds = time.perf_counter() - stamp

        operation = getattr(self.policy, 'last_operation', None)
        if operation == 'finish':
            self.env.termination = 'far_reached'
            if self.training and self.policy.tape:
                self.policy.record(0.1)
            self.finish_episode()
            self.single = False
            self.revision += 1
            return

        if move is None:
            self.env.termination = self.policy.last_rejection or 'mask_generation_failed'
            if self.training and self.policy.tape:
                self.policy.record(-0.03)
            self.finish_episode()
            self.single = False
            self.revision += 1
            return

        before = abs(self.env.count - self.env.goal_count)
        depth = depth_potential(self.env)
        far_before = self.env.far

        commit(self.env, move)
        self.steps = self.env.edits

        if self.policy.amount_feedback and self.policy.last_generated_count:
            self.env.workload.observe(
                int(move.remove.sum() + move.add.sum()),
                self.policy.last_generated_count,
            )

        if self.training:
            initial = max(1, abs(self.env.start_count - self.env.goal_count))
            limit = getattr(self.env, 'unified_limit', 16)
            reward = 0.2 * (before - abs(self.env.count - self.env.goal_count)) / initial + depth_potential(self.env) - depth - 0.01 / limit
            self.policy.record(reward)

        cells = np.argwhere(move.remove | move.add)
        if len(cells) > 0:
            low = cells.min(axis=0)
            high = cells.max(axis=0)
            op_label = 'exchange' if (move.remove.any() and move.add.any()) else 'grow' if move.add.any() else 'subtract'
            self.edit_trace.append(
                dict(
                    step=self.steps,
                    operation=op_label,
                    changedVoxels=int(move.remove.sum() + move.add.sum()),
                    facadeSnapshot=move.version,
                    maskSource='unified_' + str(operation),
                    generatedVoxels=self.policy.last_generated_count,
                    projectionRounds=1,
                    bounds=[low[::-1].tolist(), high[::-1].tolist()],
                    mask=cells[:, ::-1].tolist(),
                    facadeOnly=False,
                    surfaceLayers=None,
                    rawStructure={},
                    appliedStructure={},
                    farBefore=far_before,
                    farAfter=self.env.far,
                )
            )
            self.edit_trace = self.edit_trace[-64:]

        limit = getattr(self.env, 'unified_limit', 16)
        if self.steps >= limit and not self.env.far_satisfied:
            self.env.termination = 'step_budget_exhausted'

        if self.env.terminal:
            self.finish_episode()
        else:
            if self.steps % self.audit_steps == 0:
                self.audit_environment()
            if self.steps % self.rollout_steps == 0:
                self.learn_rollout()

        self.single = False
        self.revision += 1

    def loop(self):
        try:
            if not self.ready:
                with self.lock:
                    self.build()
            while True:
                with self.lock:
                    self.advance_one()
                time.sleep(0.002 if self.running else 0.01)
        except Exception:
            with self.lock:
                self.error = traceback.format_exc()
                self.pause()

    def state(self):
        with self.lock:
            if not self.ready:
                return dict(ready=False, error=self.error, device=self.device)
            e = self.env
            metrics = e.metrics()
            return dict(
                ready=True,
                error=self.error,
                running=self.running,
                training=self.training,
                activeSeconds=self.clock(),
                episodes=self.episodes,
                step=self.steps,
                maxSteps=getattr(e, 'unified_limit', 16),
                rolloutSteps=self.rollout_steps,
                environmentAuditSteps=self.audit_steps,
                episodeComplete=e.terminal,
                episodeRecorded=self.episode_recorded,
                termination=e.termination,
                settings=dict(
                    targetFar=self.target_far,
                    farTolerance=e.far_tolerance,
                    gridAlignment=(e.context or {}).get('gridAlignment', 'auto_fit'),
                    densityPreset=self.density_preset,
                    program='residential',
                    densityPresets=DENSITY_PRESETS,
                    maximumFar=e.capacity / e.plot_area,
                    maskMode='combined',
                    deepInterior=True,
                    minimumPatch=self.policy.config.minimum_patch,
                    maximumInteriorVoids=self.policy.config.maximum_voids,
                    minimumVoidArea=self.policy.config.minimum_void_area,
                    maximumEditVoxels=self.policy.config.maximum_edit_voxels,
                    editAllocation=self.policy.config.allocation,
                ),
                seed=self.pending_seed,
                plan=self.plan,
                siteId=self.site_id,
                revision=self.revision,
                metrics=metrics,
                learning=self.stats,
                policyMigration='Unified action space: neural remove/grow/exchange/finish (roof growth enabled)',
                maskGenerator='unified_roof_growth',
                maskStructure=dict(lastOperation=getattr(self.policy, 'last_operation', None)),
                maskProfile=dict(
                    samplingSeconds=0.0,
                    transferSeconds=0.0,
                    geometrySeconds=getattr(self.policy, 'generation_seconds', 0.0),
                ),
                autosave=dict(lastSaved=None, error=None),
                neuralGenerationSeconds=getattr(self.policy, 'generation_seconds', 0.0),
                maskGenerationAttempts=1,
                editTrace=list(self.edit_trace),
                context=public_scene(e.context) if e.context is not None else None,
                actionSeconds=self.action_seconds,
                learningSeconds=self.learning_seconds,
                shape=list(e.shape),
                occupied=e.occupied.to('cpu', torch.uint8).flatten().tolist(),
                allowed=e.allowed.to('cpu', torch.uint8).flatten().tolist(),
                facadeDepth=e.depth.detach().cpu().flatten().tolist(),
                depthBadness=e.depth_badness.detach().cpu().flatten().tolist(),
                light=e.light.detach().cpu().flatten().tolist(),
                noise=e.noise.detach().cpu().flatten().tolist(),
                usable=e.usable.to('cpu', torch.uint8).flatten().tolist(),
                boxes=e.boxes.tolist(),
                recent=self.recent,
            )

# =============================================================================
# General-Purpose Live Devoxelizer (Coplanar 2.5D + Non-Flat Taubin / DC+QEM)
# =============================================================================

_DEVOX_CACHE: dict[tuple[int, str], dict] = {}
_DEVOX_LOCK = threading.Lock()


def _collapse_collinear(coords: list[tuple[float, float]], eps: float = 1e-6) -> list[tuple[float, float]]:
    """Remove redundant collinear vertices from a closed 2D ring."""
    pts = list(coords[:-1]) if len(coords) > 1 and np.allclose(coords[0], coords[-1]) else list(coords)
    if len(pts) < 3:
        return pts
    out = []
    n = len(pts)
    for i in range(n):
        ax, ay = pts[(i - 1) % n]
        bx, by = pts[i]
        cx, cy = pts[(i + 1) % n]
        ux, uy = bx - ax, by - ay
        vx, vy = cx - bx, cy - by
        if math.hypot(ux, uy) < eps:
            continue
        cross = ux * vy - uy * vx
        if abs(cross) > eps * max(1.0, math.hypot(ux, uy) * math.hypot(vx, vy)):
            out.append((float(bx), float(by)))
    return out if len(out) >= 3 else pts


def _fit_line_2d(points: np.ndarray, snap_deg: float = 4.5) -> tuple[float, float, float]:
    """Fit unit-normal 2D line (nx, ny, d) with right-hand outward orientation."""
    p0, p1 = points[0], points[-1]
    base_dir = p1 - p0
    if len(points) >= 3:
        centered = points - points.mean(axis=0)
        cov = centered.T @ centered
        vals, vecs = np.linalg.eigh(cov)
        tangent = vecs[:, int(np.argmax(vals))]
        if np.dot(tangent, base_dir) < 0:
            tangent = -tangent
    else:
        norm = float(np.linalg.norm(base_dir)) or 1.0
        tangent = base_dir / norm

    ang = math.atan2(float(tangent[1]), float(tangent[0]))
    step = math.radians(15.0)
    nearest = round(ang / step) * step
    if abs(ang - nearest) <= math.radians(snap_deg):
        tangent = np.array([math.cos(nearest), math.sin(nearest)], dtype=np.float64)

    nx, ny = float(tangent[1]), -float(tangent[0])
    nlen = math.hypot(nx, ny) or 1.0
    nx, ny = nx / nlen, ny / nlen
    d = float(np.mean(points[:, 0] * nx + points[:, 1] * ny))
    return nx, ny, d


def _intersect_lines(l0: tuple[float, float, float], l1: tuple[float, float, float], fallback: tuple[float, float]) -> tuple[float, float]:
    """Intersect two 2D lines (nx, ny, d)."""
    nx0, ny0, d0 = l0
    nx1, ny1, d1 = l1
    det = nx0 * ny1 - ny0 * nx1
    if abs(det) < 1e-4:
        return fallback
    x = (d0 * ny1 - ny0 * d1) / det
    y = (nx0 * d1 - d0 * nx1) / det
    if math.hypot(x - fallback[0], y - fallback[1]) > 4.0:
        return fallback
    return (float(x), float(y))


def _regularize_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Regularize a grid staircase loop into crisp rectilinear anchors + sloped segments."""
    pts = _collapse_collinear(coords)
    n = len(pts)
    if n < 4:
        return pts

    lengths = [math.hypot(pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1]) for i in range(n)]
    is_anchor = []
    for i in range(n):
        L = lengths[i]
        prev_L = lengths[(i - 1) % n]
        next_L = lengths[(i + 1) % n]
        anchor = (L >= 2.95) or (L >= 1.95 and prev_L >= 1.95 and next_L >= 1.95)
        is_anchor.append(anchor)

    if all(is_anchor):
        return pts

    start_idx = next((i for i, a in enumerate(is_anchor) if a), None)
    if start_idx is None:
        simplified = LineString(pts + [pts[0]]).simplify(0.85, preserve_topology=True)
        sc = list(simplified.coords)[:-1]
        return sc if len(sc) >= 3 else pts

    ordered_edges = [(start_idx + k) % n for k in range(n)]
    lines_with_pts: list[tuple[tuple[float, float, float], tuple[float, float], tuple[float, float]]] = []
    k = 0
    while k < n:
        ei = ordered_edges[k]
        if is_anchor[ei]:
            p0 = pts[ei]
            p1 = pts[(ei + 1) % n]
            arr = np.array([p0, p1], dtype=np.float64)
            lines_with_pts.append((_fit_line_2d(arr, snap_deg=2.0), p0, p1))
            k += 1
            continue

        chain_pts = [pts[ei]]
        while k < n and not is_anchor[ordered_edges[k]]:
            ej = ordered_edges[k]
            chain_pts.append(pts[(ej + 1) % n])
            k += 1

        if len(chain_pts) <= 2:
            arr = np.array(chain_pts, dtype=np.float64)
            lines_with_pts.append((_fit_line_2d(arr, snap_deg=4.5), chain_pts[0], chain_pts[-1]))
        else:
            # Midpoint staircase smoothing + RDP breakpoint splitting
            mids = [chain_pts[0]]
            for j in range(1, len(chain_pts) - 1):
                mids.append((
                    0.5 * (chain_pts[j][0] + chain_pts[j + 1][0]),
                    0.5 * (chain_pts[j][1] + chain_pts[j + 1][1]),
                ))
            mids.append(chain_pts[-1])
            simp = list(LineString(mids).simplify(0.85).coords)
            # Split chain vertices at simplified corner indices
            split_idx = [0]
            for sp in simp[1:-1]:
                dists = [math.hypot(cp[0] - sp[0], cp[1] - sp[1]) for cp in chain_pts]
                best_j = int(np.argmin(dists))
                if best_j > split_idx[-1] and best_j < len(chain_pts) - 1:
                    split_idx.append(best_j)
            split_idx.append(len(chain_pts) - 1)
            for si in range(len(split_idx) - 1):
                a_i, b_i = split_idx[si], split_idx[si + 1]
                sub = np.array(chain_pts[a_i:b_i + 1], dtype=np.float64)
                if len(sub) >= 3:
                    # Drop the extreme corners when they belong to orthogonal anchor turns
                    core = sub[1:-1] if len(sub) >= 4 else sub
                    line = _fit_line_2d(core, snap_deg=4.5)
                else:
                    line = _fit_line_2d(sub, snap_deg=4.5)
                lines_with_pts.append((line, tuple(sub[0]), tuple(sub[-1])))

    m = len(lines_with_pts)
    if m < 3:
        return pts
    out = []
    for i in range(m):
        l_prev = lines_with_pts[(i - 1) % m][0]
        l_curr, corner_fb, _ = lines_with_pts[i]
        out.append(_intersect_lines(l_prev, l_curr, corner_fb))
    return _collapse_collinear(out)


def _ensure_ccw(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return closed/open 2D ring ordered strictly counter-clockwise (positive signed area)."""
    pts = list(coords[:-1]) if len(coords) > 1 and np.allclose(coords[0], coords[-1]) else list(coords)
    if len(pts) < 3:
        return pts
    area2 = 0.0
    m = len(pts)
    for i in range(m):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % m]
        area2 += x0 * y1 - x1 * y0
    if area2 < 0.0:
        pts.reverse()
    return pts


def _regularize_polygon(raw_geom) -> list[Polygon]:
    """Convert raw Shapely union geometry into cleaned list of strictly CCW-exterior Polygons."""
    if raw_geom is None or raw_geom.is_empty:
        return []
    polys = [raw_geom] if raw_geom.geom_type == 'Polygon' else [g for g in raw_geom.geoms if g.geom_type == 'Polygon']
    out = []
    for poly in polys:
        if poly.area < 0.5:
            continue
        ext = _ensure_ccw(_regularize_ring(_ensure_ccw(list(poly.exterior.coords))))
        holes = []
        for interior in poly.interiors:
            hr = _ensure_ccw(_regularize_ring(_ensure_ccw(list(interior.coords))))
            if len(hr) >= 3:
                holes.append(hr)
        cand = Polygon(ext, holes)
        if not cand.is_valid:
            cand = cand.buffer(0)
        if cand.is_empty:
            out.append(poly)
        elif cand.geom_type == 'Polygon':
            out.append(cand)
        else:
            for g in cand.geoms:
                if g.geom_type == 'Polygon' and g.area >= 0.5:
                    out.append(g)
    return out


def _align_vertical_support_planes(floor_polys: list, tol_d: float = 0.35, cos_tol: float = 0.996) -> list:
    """Snap near-coplanar walls across storeys so shared facades sit on exact common planes."""
    # Only operate cleanly when each non-empty floor is a single simple Polygon with exterior ring
    edge_records = []
    rings_by_z: dict[int, list[tuple[float, float]]] = {}
    holes_by_z: dict[int, list] = {}
    for z, geoms in enumerate(floor_polys):
        if len(geoms) != 1:
            continue
        poly = geoms[0]
        pts = _ensure_ccw(_collapse_collinear(list(poly.exterior.coords)))
        if len(pts) < 3:
            continue
        rings_by_z[z] = pts
        holes_by_z[z] = [list(h.coords) for h in poly.interiors]
        m = len(pts)
        for i in range(m):
            p0, p1 = pts[i], pts[(i + 1) % m]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx, dy)
            if L < 0.25:
                continue
            nx, ny = dy / L, -dx / L
            d = nx * 0.5 * (p0[0] + p1[0]) + ny * 0.5 * (p0[1] + p1[1])
            edge_records.append(dict(z=z, ei=i, nx=nx, ny=ny, d=d, length=L))

    if not edge_records:
        return floor_polys

    # Cluster nearly identical planes across consecutive/nearby floors
    used = [False] * len(edge_records)
    snapped_lines: dict[tuple[int, int], tuple[float, float, float]] = {}
    for i, rec in enumerate(edge_records):
        if used[i]:
            continue
        cluster = [rec]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, cand in enumerate(edge_records):
                if used[j]:
                    continue
                if any(
                    abs(cand['z'] - c['z']) <= 2
                    and (cand['nx'] * c['nx'] + cand['ny'] * c['ny']) >= cos_tol
                    and abs(cand['d'] - c['d']) <= tol_d
                    for c in cluster
                ):
                    used[j] = True
                    cluster.append(cand)
                    changed = True

        # Choose longest/median support plane within cluster
        best = max(cluster, key=lambda c: c['length'])
        nx_c, ny_c = best['nx'], best['ny']
        if len(cluster) >= 2:
            ds = sorted(c['d'] for c in cluster)
            # Prefer exact half/integer grid plane if close to longest anchor wall
            d_rep = best['d'] if best['length'] >= 3.0 else ds[len(ds) // 2]
        else:
            d_rep = best['d']
        for c in cluster:
            snapped_lines[(c['z'], c['ei'])] = (nx_c, ny_c, d_rep)

    aligned = list(floor_polys)
    for z, pts in rings_by_z.items():
        m = len(pts)
        lines = []
        for i in range(m):
            if (z, i) in snapped_lines:
                lines.append(snapped_lines[(z, i)])
            else:
                p0, p1 = pts[i], pts[(i + 1) % m]
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                L = math.hypot(dx, dy) or 1.0
                nx, ny = dy / L, -dx / L
                lines.append((nx, ny, nx * p0[0] + ny * p0[1]))
        new_pts = [_intersect_lines(lines[(i - 1) % m], lines[i], pts[i]) for i in range(m)]
        cand = Polygon(new_pts, holes_by_z[z])
        if not cand.is_valid:
            cand = cand.buffer(0)
        orig_area = floor_polys[z][0].area
        if not cand.is_empty and cand.geom_type == 'Polygon' and abs(cand.area - orig_area) <= max(2.0, 0.08 * orig_area):
            aligned[z] = [cand]
    return aligned


def _triangulate_cap(geom, z_val: float, upward: bool, add_tri_fn) -> None:
    """Triangulate any 2D Shapely Polygon/MultiPolygon cleanly via constrained Delaunay."""
    if geom is None or geom.is_empty:
        return
    polys = [geom] if geom.geom_type == 'Polygon' else [g for g in geom.geoms if g.geom_type == 'Polygon']
    for poly in polys:
        if poly.area < 1e-4:
            continue
        try:
            cdt = constrained_delaunay_triangles(poly)
            tris = list(cdt.geoms)
        except Exception:
            continue
        for t in tris:
            if t.area < 1e-5:
                continue
            c = list(t.exterior.coords)[:3]
            if len(c) < 3:
                continue
            (x0, y0), (x1, y1), (x2, y2) = c[0], c[1], c[2]
            cross_z = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if abs(cross_z) < 1e-7:
                continue
            if (cross_z > 0) == upward:
                add_tri_fn((x0, y0, z_val), (x1, y1, z_val), (x2, y2, z_val))
            else:
                add_tri_fn((x0, y0, z_val), (x2, y2, z_val), (x1, y1, z_val))


def _finalize_mesh(verts: list[tuple[float, float, float]], faces: list[list[int]], crease_deg: float, non_flat: bool, method: str, start_time: float) -> dict:
    """Compute crease edges, clean topology, and package grid-unit mesh JSON."""
    if not faces:
        return dict(vertices=[], faces=[], creases=[], nonFlat=non_flat, stats=dict(method=method, vertices=0, triangles=0, volume_m3=0.0, elapsed_ms=0.0))

    v_arr = np.asarray(verts, dtype=np.float64)
    f_arr = np.asarray(faces, dtype=np.int32)

    p0 = v_arr[f_arr[:, 0]]
    p1 = v_arr[f_arr[:, 1]]
    p2 = v_arr[f_arr[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    fn = np.cross(e1, e2)
    lens = np.linalg.norm(fn, axis=1, keepdims=True)
    valid = (lens[:, 0] > 1e-8)
    f_arr = f_arr[valid]
    fn = fn[valid] / lens[valid]

    # Signed volume in world m³ (x: 1m, y: 1m, z_floor: 3m -> *3.0)
    vol_grid = float(np.sum(np.einsum('ij,ij->i', v_arr[f_arr[:, 0]], np.cross(v_arr[f_arr[:, 1]], v_arr[f_arr[:, 2]]))) / 6.0)
    vol_m3 = round(abs(vol_grid) * 3.0, 1)

    cos_thr = math.cos(math.radians(crease_deg))
    edge_map: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(f_arr.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append(fi)

    creases = []
    for (u, v), flist in edge_map.items():
        if len(flist) != 2:
            creases.append([u, v])
        else:
            dot = float(np.dot(fn[flist[0]], fn[flist[1]]))
            if dot < cos_thr:
                creases.append([u, v])

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
    return dict(
        vertices=np.round(v_arr, 4).tolist(),
        faces=f_arr.tolist(),
        creases=creases,
        nonFlat=non_flat,
        stats=dict(
            method=method,
            vertices=int(len(v_arr)),
            triangles=int(len(f_arr)),
            creaseEdges=int(len(creases)),
            volume_m3=vol_m3,
            elapsed_ms=elapsed_ms,
        ),
    )


def _extract_floor_polygons(occ_np: np.ndarray) -> list[list[Polygon]]:
    """Build vertical-plane-aligned regularized 2D Polygon lists for each storey."""
    nz = occ_np.shape[0]
    raw_by_z: list[list[Polygon]] = []
    for z in range(nz):
        pts = np.argwhere(occ_np[z])
        if len(pts) == 0:
            raw_by_z.append([])
            continue
        raw_geom = unary_union([box(float(x), float(y), float(x + 1), float(y + 1)) for y, x in pts])
        raw_by_z.append(_regularize_polygon(raw_geom))
    return _align_vertical_support_planes(raw_by_z)


def _match_loft_pair(pts_a: list[tuple[float, float]], pts_b: list[tuple[float, float]], max_shift: float = 2.6) -> list[tuple[float, float]] | None:
    """Align same-count CCW vertex rings between consecutive floors if drift is gradual."""
    n = len(pts_a)
    if len(pts_b) != n or n < 3:
        return None
    best_shift = None
    best_rms = float('inf')
    best_max = float('inf')
    for shift in range(n):
        sq = 0.0
        mx = 0.0
        for i in range(n):
            dx = pts_a[i][0] - pts_b[(i + shift) % n][0]
            dy = pts_a[i][1] - pts_b[(i + shift) % n][1]
            d = math.hypot(dx, dy)
            if d > mx:
                mx = d
            sq += dx * dx + dy * dy
        rms = math.sqrt(sq / n)
        if rms < best_rms:
            best_rms = rms
            best_max = mx
            best_shift = shift
    if best_shift is None or best_max > max_shift:
        return None
    return [pts_b[(i + best_shift) % n] for i in range(n)]


def _build_live_25d(occ_np: np.ndarray, loft_slopes: bool, method: str, t0: float) -> dict:
    """Conformal watertight 2.5D shell (either coplanar extruded slabs or correlated slope loft)."""
    nz = occ_np.shape[0]
    floor_polys = _extract_floor_polygons(occ_np)

    vmap: dict[tuple[float, float, float], int] = {}
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []

    def vid(x: float, y: float, z: float) -> int:
        key = (round(float(x), 4), round(float(y), 4), round(float(z), 4))
        idx = vmap.get(key)
        if idx is None:
            idx = len(verts)
            vmap[key] = idx
            verts.append(key)
        return idx

    def add_tri(a, b, c):
        ia, ib, ic = vid(*a), vid(*b), vid(*c)
        if ia != ib and ib != ic and ic != ia:
            faces.append([ia, ib, ic])

    def emit_ring_quad(p0, p1, z0, q0, q1, z1, is_hole: bool):
        if is_hole:
            a, b, c, d = (p1[0], p1[1], z0), (p0[0], p0[1], z0), (q0[0], q0[1], z1), (q1[0], q1[1], z1)
        else:
            a, b, c, d = (p0[0], p0[1], z0), (p1[0], p1[1], z0), (q1[0], q1[1], z1), (q0[0], q0[1], z1)
        add_tri(a, b, c)
        add_tri(a, c, d)

    def emit_vertical_ring(ring_pts, z_bot, z_top, is_hole: bool):
        pts = _ensure_ccw(_collapse_collinear(list(ring_pts)))
        m = len(pts)
        for i in range(m):
            emit_ring_quad(pts[i], pts[(i + 1) % m], z_bot, pts[i], pts[(i + 1) % m], z_top, is_hole)

    union_by_z = [unary_union(geoms) if geoms else None for geoms in floor_polys]

    if not loft_slopes:
        # Crisp conformal vertical tiers with zero internal diaphragms
        for z in range(nz + 1):
            below = union_by_z[z - 1] if z > 0 else None
            above = union_by_z[z] if z < nz else None
            if below is not None:
                diff_up = below if above is None else below.difference(above)
                _triangulate_cap(diff_up, float(z), True, add_tri)
            if above is not None:
                diff_dn = above if below is None else above.difference(below)
                _triangulate_cap(diff_dn, float(z), False, add_tri)

        for z in range(nz):
            for poly in floor_polys[z]:
                emit_vertical_ring(list(poly.exterior.coords), float(z), float(z + 1), False)
                for hole in poly.interiors:
                    emit_vertical_ring(list(hole.coords), float(z), float(z + 1), True)
        return _finalize_mesh(verts, faces, crease_deg=3.5, non_flat=False, method=method, start_time=t0)

    # Correlated Support-Plane Loft: smoothly connect gradual multi-floor slopes + cap setbacks
    # Identify loftable floor interfaces z (where floor z-1 top blends smoothly to floor z top)
    active_zs = [z for z in range(nz) if floor_polys[z]]
    if not active_zs:
        return _finalize_mesh([], [], crease_deg=3.5, non_flat=False, method=method, start_time=t0)

    # Group contiguous simple floors into correlated runs when topology matches smoothly
    # Each storey z spans [z, z+1]; if storey z+1 has matching ring & gradual shift, linearly blend z+0.5..z+1.5
    matched_to_next: dict[int, list[tuple[float, float]]] = {}
    for z in range(nz - 1):
        if len(floor_polys[z]) == 1 and len(floor_polys[z + 1]) == 1:
            pa, pb = floor_polys[z][0], floor_polys[z + 1][0]
            if not pa.interiors and not pb.interiors:
                ra = _ensure_ccw(_collapse_collinear(list(pa.exterior.coords)))
                rb = _ensure_ccw(_collapse_collinear(list(pb.exterior.coords)))
                aligned_b = _match_loft_pair(ra, rb, max_shift=2.6)
                if aligned_b is not None:
                    matched_to_next[z] = aligned_b

    # Compute mid-plane contour at integer elevation z for smooth continuous slopes
    # For contiguous sloped runs where vertices move between floors, average adjacent ring corners at interior z
    bottom_caps = []
    top_caps = []
    for z in range(nz):
        if not floor_polys[z]:
            continue
        poly_z = floor_polys[z][0] if len(floor_polys[z]) == 1 else None
        is_smooth_bot = (z - 1) in matched_to_next
        is_smooth_top = z in matched_to_next

        # Bottom horizontal cap if NOT smoothly bridged from floor z-1
        if not is_smooth_bot:
            below = union_by_z[z - 1] if z > 0 else None
            curr = union_by_z[z]
            diff_dn = curr if below is None else curr.difference(below)
            bottom_caps.append((z, diff_dn))

        # Top horizontal cap if NOT smoothly bridging to floor z+1
        if not is_smooth_top:
            above = union_by_z[z + 1] if z + 1 < nz else None
            curr = union_by_z[z]
            diff_up = curr if above is None else curr.difference(above)
            top_caps.append((z + 1, diff_up))

        if z in matched_to_next and poly_z is not None:
            ra = _ensure_ccw(_collapse_collinear(list(poly_z.exterior.coords)))
            rb = matched_to_next[z]
            z_low = float(z)
            z_high = float(z + 1)
            if (z - 1) in matched_to_next:
                prev_b = _ensure_ccw(_collapse_collinear(list(floor_polys[z - 1][0].exterior.coords)))
                prev_aligned = _match_loft_pair(ra, prev_b, max_shift=2.6)
                pts_bot = [((ra[i][0] + prev_aligned[i][0]) * 0.5, (ra[i][1] + prev_aligned[i][1]) * 0.5) for i in range(len(ra))] if prev_aligned else ra
            else:
                pts_bot = ra
            pts_top = [((ra[i][0] + rb[i][0]) * 0.5, (ra[i][1] + rb[i][1]) * 0.5) for i in range(len(ra))]
            m = len(ra)
            for i in range(m):
                emit_ring_quad(pts_bot[i], pts_bot[(i + 1) % m], z_low, pts_top[i], pts_top[(i + 1) % m], z_high, False)
        elif (z - 1) in matched_to_next and poly_z is not None:
            ra = _ensure_ccw(_collapse_collinear(list(poly_z.exterior.coords)))
            prev_b = _ensure_ccw(_collapse_collinear(list(floor_polys[z - 1][0].exterior.coords)))
            prev_aligned = _match_loft_pair(ra, prev_b, max_shift=2.6)
            pts_bot = [((ra[i][0] + prev_aligned[i][0]) * 0.5, (ra[i][1] + prev_aligned[i][1]) * 0.5) for i in range(len(ra))] if prev_aligned else ra
            m = len(ra)
            for i in range(m):
                emit_ring_quad(pts_bot[i], pts_bot[(i + 1) % m], float(z), ra[i], ra[(i + 1) % m], float(z + 1), False)
        else:
            for poly in floor_polys[z]:
                emit_vertical_ring(list(poly.exterior.coords), float(z), float(z + 1), False)
                for hole in poly.interiors:
                    emit_vertical_ring(list(hole.coords), float(z), float(z + 1), True)

    for z_val, geom in bottom_caps:
        _triangulate_cap(geom, float(z_val), False, add_tri)
    for z_val, geom in top_caps:
        _triangulate_cap(geom, float(z_val), True, add_tri)

    return _finalize_mesh(verts, faces, crease_deg=3.5, non_flat=False, method=method, start_time=t0)


def _extract_exposed_iso_mesh(occ_iso: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast vectorized extraction of watertight boundary mesh from 1x1x1 isotropic voxel grid."""
    nz, ny, nx = occ_iso.shape
    padded = np.zeros((nz + 2, ny + 2, nx + 2), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = occ_iso
    c = padded[1:-1, 1:-1, 1:-1]

    tris_v = []

    def add_face_dir(mask, offsets):
        zs, ys, xs = np.nonzero(mask)
        if len(zs) == 0:
            return
        pts = np.stack([xs, ys, zs], axis=1).astype(np.int32)
        q = pts[:, None, :] + np.asarray(offsets, dtype=np.int32)[None, :, :]
        tris_v.append(q[:, [0, 1, 2], :])
        tris_v.append(q[:, [0, 2, 3], :])

    add_face_dir(c & ~padded[:-2, 1:-1, 1:-1], [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])
    add_face_dir(c & ~padded[2:, 1:-1, 1:-1], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    add_face_dir(c & ~padded[1:-1, :-2, 1:-1], [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])
    add_face_dir(c & ~padded[1:-1, 2:, 1:-1], [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)])
    add_face_dir(c & ~padded[1:-1, 1:-1, :-2], [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)])
    add_face_dir(c & ~padded[1:-1, 1:-1, 2:], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)])

    if not tris_v:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    all_t = np.concatenate(tris_v, axis=0)  # (T, 3, 3)
    flat = all_t.reshape(-1, 3)
    keys = flat[:, 0] * 1_000_000 + flat[:, 1] * 1_000 + flat[:, 2]
    _, uniq_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    verts = flat[uniq_idx].astype(np.float64)
    faces = inv.reshape(-1, 3).astype(np.int32)
    return verts, faces


def _build_live_taubin(occ_np: np.ndarray, t0: float) -> dict:
    """Non-flat isotropic Taubin lambda/mu curved architectural shell."""
    iso = np.repeat(occ_np, 3, axis=0)
    verts, faces = _extract_exposed_iso_mesh(iso)
    if len(verts) == 0:
        return _finalize_mesh([], [], crease_deg=22.0, non_flat=True, method='taubin_curved', start_time=t0)

    # Undirected mesh edges for Laplacian smoothing
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    all_e = np.vstack([e0, e1, e2])
    all_e = np.sort(all_e, axis=1)
    edges = np.unique(all_e, axis=0)
    u, v = edges[:, 0], edges[:, 1]

    deg = np.zeros(len(verts), dtype=np.float64)
    np.add.at(deg, u, 1.0)
    np.add.at(deg, v, 1.0)
    deg = np.maximum(deg, 1.0)[:, None]
    pinned = verts[:, 2] <= 0.15

    def lap_step(pos: np.ndarray, factor: float) -> np.ndarray:
        acc = np.zeros_like(pos)
        np.add.at(acc, u, pos[v])
        np.add.at(acc, v, pos[u])
        avg = acc / deg
        nxt = pos + factor * (avg - pos)
        nxt[pinned, 2] = 0.0
        return nxt

    for _ in range(10):
        verts = lap_step(verts, 0.50)
        verts = lap_step(verts, -0.53)

    # Convert isotropic 1m Z back to VoxelForm floor units (3m/storey)
    verts[:, 2] /= 3.0
    return _finalize_mesh(verts.tolist(), faces.tolist(), crease_deg=22.0, non_flat=True, method='taubin_curved', start_time=t0)


def _qem_simplify(verts: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Compact pure-Python/NumPy Quadric Error Metric mesh decimation."""
    n_v = len(verts)
    if len(faces) <= target_faces:
        return verts, faces

    Q = np.zeros((n_v, 4, 4), dtype=np.float64)
    for a, b, c in faces:
        p0, p1, p2 = verts[a], verts[b], verts[c]
        n = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(n))
        if ln < 1e-10:
            continue
        n /= ln
        d = -float(np.dot(n, p0))
        k = np.array([n[0], n[1], n[2], d], dtype=np.float64)
        kp = np.outer(k, k)
        Q[a] += kp
        Q[b] += kp
        Q[c] += kp

    # Adjacency + active face set
    v_faces: list[set[int]] = [set() for _ in range(n_v)]
    active_f = np.ones(len(faces), dtype=bool)
    f_arr = faces.copy()
    v_alive = np.ones(n_v, dtype=bool)
    edges_set: set[tuple[int, int]] = set()
    for fi, (a, b, c) in enumerate(f_arr):
        v_faces[a].add(fi)
        v_faces[b].add(fi)
        v_faces[c].add(fi)
        edges_set.add((min(a, b), max(a, b)))
        edges_set.add((min(b, c), max(b, c)))
        edges_set.add((min(c, a), max(c, a)))

    def eval_pair(u: int, v: int) -> tuple[float, np.ndarray]:
        Qs = Q[u] + Q[v]
        A = np.array([
            [Qs[0, 0], Qs[0, 1], Qs[0, 2], Qs[0, 3]],
            [Qs[0, 1], Qs[1, 1], Qs[1, 2], Qs[1, 3]],
            [Qs[0, 2], Qs[1, 2], Qs[2, 2], Qs[2, 3]],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        pos = None
        if abs(np.linalg.det(A)) > 1e-8:
            try:
                sol = np.linalg.solve(A, np.array([0.0, 0.0, 0.0, 1.0]))
                p_cand = sol[:3]
                if np.linalg.norm(p_cand - 0.5 * (verts[u] + verts[v])) < 2.2:
                    pos = p_cand
            except np.linalg.LinAlgError:
                pass
        if pos is None:
            pos = 0.5 * (verts[u] + verts[v])
        vh = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)
        cost = float(vh @ Qs @ vh)
        return max(0.0, cost), pos

    ver = [0] * n_v
    heap = []
    for u, v in edges_set:
        c, pos = eval_pair(u, v)
        heapq.heappush(heap, (c, u, v, ver[u], ver[v], float(pos[0]), float(pos[1]), float(pos[2])))

    rem_faces = int(len(f_arr))
    pos_arr = verts.copy()
    while rem_faces > target_faces and heap:
        cost, u, v, vu, vv, px, py, pz = heapq.heappop(heap)
        if not v_alive[u] or not v_alive[v] or vu != ver[u] or vv != ver[v] or u == v:
            continue
        pos_arr[u] = [px, py, pz]
        Q[u] = Q[u] + Q[v]
        v_alive[v] = False
        ver[u] += 1

        affected = list(v_faces[v])
        for fi in affected:
            if not active_f[fi]:
                continue
            a, b, c = int(f_arr[fi, 0]), int(f_arr[fi, 1]), int(f_arr[fi, 2])
            if a == v:
                a = u
            if b == v:
                b = u
            if c == v:
                c = u
            if a == b or b == c or c == a:
                active_f[fi] = False
                rem_faces -= 1
                for w in (int(f_arr[fi, 0]), int(f_arr[fi, 1]), int(f_arr[fi, 2])):
                    v_faces[w].discard(fi)
            else:
                f_arr[fi] = [a, b, c]
                v_faces[u].add(fi)
        v_faces[v].clear()

        nbrs: set[int] = set()
        for fi in v_faces[u]:
            if active_f[fi]:
                for w in f_arr[fi]:
                    wi = int(w)
                    if wi != u and v_alive[wi]:
                        nbrs.add(wi)
        for w in nbrs:
            a, b = (u, w) if u < w else (w, u)
            c, npos = eval_pair(a, b)
            heapq.heappush(heap, (c, a, b, ver[a], ver[b], float(npos[0]), float(npos[1]), float(npos[2])))

    kept_f = f_arr[active_f]
    used_ids, inv = np.unique(kept_f.reshape(-1), return_inverse=True)
    return pos_arr[used_ids], inv.reshape(-1, 3).astype(np.int32)


def _build_live_dc_qem(occ_np: np.ndarray, t0: float) -> dict:
    """SDF Surface-Nets / Dual Contouring + QEM simplification."""
    iso = np.repeat(occ_np, 3, axis=0)
    nz, ny, nx = iso.shape
    pad = np.zeros((nz + 4, ny + 4, nx + 4), dtype=np.float64)
    pad[2:-2, 2:-2, 2:-2] = iso.astype(np.float64)
    sdf = gaussian_filter(pad, sigma=0.72) - 0.48
    # Keep bottom ground slab sharp at z_pad = 2 (world z = 0)
    sdf[:2, :, :] = -0.5

    # Active cells touching any sign-changing corner
    active_c = (
        (sdf[:-1, :-1, :-1] > 0) | (sdf[:-1, :-1, 1:] > 0) |
        (sdf[:-1, 1:, :-1] > 0) | (sdf[:-1, 1:, 1:] > 0) |
        (sdf[1:, :-1, :-1] > 0) | (sdf[1:, :-1, 1:] > 0) |
        (sdf[1:, 1:, :-1] > 0) | (sdf[1:, 1:, 1:] > 0)
    ) & (
        (sdf[:-1, :-1, :-1] <= 0) | (sdf[:-1, :-1, 1:] <= 0) |
        (sdf[:-1, 1:, :-1] <= 0) | (sdf[:-1, 1:, 1:] <= 0) |
        (sdf[1:, :-1, :-1] <= 0) | (sdf[1:, :-1, 1:] <= 0) |
        (sdf[1:, 1:, :-1] <= 0) | (sdf[1:, 1:, 1:] <= 0)
    )

    czs, cys, cxs = np.nonzero(active_c)
    if len(czs) == 0:
        return _finalize_mesh([], [], crease_deg=3.5, non_flat=False, method='dc_qem', start_time=t0)

    cell_id = np.full(active_c.shape, -1, dtype=np.int32)
    cell_id[czs, cys, cxs] = np.arange(len(czs), dtype=np.int32)

    # Surface Nets vertex placement: average zero-crossings on active cell's 12 edges
    acc = np.zeros((len(czs), 3), dtype=np.float64)
    cnt = np.zeros(len(czs), dtype=np.float64)
    corners = [
        ((0, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 0, 0), (1, 0, 1)), ((1, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 0), (1, 1, 0)), ((1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)), ((0, 1, 1), (1, 1, 1)),
    ]
    for (dz0, dy0, dx0), (dz1, dy1, dx1) in corners:
        va = sdf[czs + dz0, cys + dy0, cxs + dx0]
        vb = sdf[czs + dz1, cys + dy1, cxs + dx1]
        chg = (va > 0) != (vb > 0)
        if not np.any(chg):
            continue
        t = va[chg] / np.where(np.abs(va[chg] - vb[chg]) > 1e-8, va[chg] - vb[chg], 1.0)
        t = np.clip(t, 0.0, 1.0)
        acc[chg, 0] += cxs[chg] + dx0 + t * (dx1 - dx0)
        acc[chg, 1] += cys[chg] + dy0 + t * (dy1 - dy0)
        acc[chg, 2] += czs[chg] + dz0 + t * (dz1 - dz0)
        cnt[chg] += 1.0

    verts = acc / np.maximum(cnt[:, None], 1.0) - np.array([2.0, 2.0, 2.0])
    verts[verts[:, 2] <= 0.25, 2] = 0.0

    tris = []
    # Emit quads for sign-changing grid edges along X, Y, Z
    # X-edge between (z, y, x) and (z, y, x+1) shared by 4 cells (z-1..z, y-1..y, x)
    ex = ((sdf[1:-1, 1:-1, :-1] > 0) != (sdf[1:-1, 1:-1, 1:] > 0))
    ezs, eys, exs = np.nonzero(ex)
    ezs += 1; eys += 1
    for z, y, x in zip(ezs.tolist(), eys.tolist(), exs.tolist()):
        if z - 1 < 0 or y - 1 < 0 or z >= cell_id.shape[0] or y >= cell_id.shape[1] or x >= cell_id.shape[2]:
            continue
        c00, c10, c11, c01 = int(cell_id[z-1, y-1, x]), int(cell_id[z-1, y, x]), int(cell_id[z, y, x]), int(cell_id[z, y-1, x])
        if min(c00, c10, c11, c01) < 0:
            continue
        if sdf[z, y, x] > 0:
            tris.extend([[c00, c10, c11], [c00, c11, c01]])
        else:
            tris.extend([[c00, c11, c10], [c00, c01, c11]])

    ey = ((sdf[1:-1, :-1, 1:-1] > 0) != (sdf[1:-1, 1:, 1:-1] > 0))
    ezs, eys, exs = np.nonzero(ey)
    ezs += 1; exs += 1
    for z, y, x in zip(ezs.tolist(), eys.tolist(), exs.tolist()):
        if z - 1 < 0 or x - 1 < 0 or z >= cell_id.shape[0] or x >= cell_id.shape[2] or y >= cell_id.shape[1]:
            continue
        c00, c10, c11, c01 = int(cell_id[z-1, y, x-1]), int(cell_id[z, y, x-1]), int(cell_id[z, y, x]), int(cell_id[z-1, y, x])
        if min(c00, c10, c11, c01) < 0:
            continue
        if sdf[z, y, x] > 0:
            tris.extend([[c00, c10, c11], [c00, c11, c01]])
        else:
            tris.extend([[c00, c11, c10], [c00, c01, c11]])

    ez = ((sdf[:-1, 1:-1, 1:-1] > 0) != (sdf[1:, 1:-1, 1:-1] > 0))
    ezs, eys, exs = np.nonzero(ez)
    eys += 1; exs += 1
    for z, y, x in zip(ezs.tolist(), eys.tolist(), exs.tolist()):
        if y - 1 < 0 or x - 1 < 0 or y >= cell_id.shape[1] or x >= cell_id.shape[2] or z >= cell_id.shape[0]:
            continue
        c00, c10, c11, c01 = int(cell_id[z, y-1, x-1]), int(cell_id[z, y-1, x]), int(cell_id[z, y, x]), int(cell_id[z, y, x-1])
        if min(c00, c10, c11, c01) < 0:
            continue
        if sdf[z, y, x] > 0:
            tris.extend([[c00, c10, c11], [c00, c11, c01]])
        else:
            tris.extend([[c00, c11, c10], [c00, c01, c11]])

    if not tris:
        return _finalize_mesh([], [], crease_deg=3.5, non_flat=False, method='dc_qem', start_time=t0)

    faces = np.asarray(tris, dtype=np.int32)
    target = min(360, max(120, len(faces) // 8))
    verts_s, faces_s = _qem_simplify(verts, faces, target_faces=target)
    verts_s[verts_s[:, 2] <= 0.15, 2] = 0.0
    verts_s[:, 2] /= 3.0
    return _finalize_mesh(verts_s.tolist(), faces_s.tolist(), crease_deg=3.5, non_flat=False, method='dc_qem', start_time=t0)


def _fit_tier_vsa_planes(rings: list[list[tuple[float, float]]], z_start: int, z_end: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Fit global 3D plane equations across multi-storey tier rings and evaluate at tier ends."""
    k_cnt = len(rings)
    if k_cnt <= 1:
        return rings[0], rings[-1]

    m = len(rings[0])
    zs = np.array([z_start + k + 0.5 for k in range(k_cnt)], dtype=np.float64)
    lines_bot = []
    lines_top = []
    for ei in range(m):
        nx_acc, ny_acc = 0.0, 0.0
        for k in range(k_cnt):
            p0, p1 = rings[k][ei], rings[k][(ei + 1) % m]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx, dy)
            if L > 1e-6:
                nx_acc += dy
                ny_acc -= dx
        nlen = math.hypot(nx_acc, ny_acc) or 1.0
        nx, ny = nx_acc / nlen, ny_acc / nlen

        ds = []
        for k in range(k_cnt):
            p0, p1 = rings[k][ei], rings[k][(ei + 1) % m]
            mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
            ds.append(nx * mx + ny * my)
        d_arr = np.asarray(ds, dtype=np.float64)

        if float(d_arr.max() - d_arr.min()) < 0.35:
            d_med = float(np.median(d_arr))
            lines_bot.append((nx, ny, d_med))
            lines_top.append((nx, ny, d_med))
        else:
            A = np.column_stack([np.ones(k_cnt), zs])
            d0, alpha = np.linalg.lstsq(A, d_arr, rcond=None)[0]
            lines_bot.append((nx, ny, float(d0 + alpha * z_start)))
            lines_top.append((nx, ny, float(d0 + alpha * z_end)))

    ring_bot = _ensure_ccw([_intersect_lines(lines_bot[(i - 1) % m], lines_bot[i], rings[0][i]) for i in range(m)])
    ring_top = _ensure_ccw([_intersect_lines(lines_top[(i - 1) % m], lines_top[i], rings[-1][i]) for i in range(m)])

    pb = Polygon(ring_bot)
    pt = Polygon(ring_top)
    ref_b = Polygon(rings[0]).area
    ref_t = Polygon(rings[-1]).area
    if not pb.is_valid or pb.is_empty or abs(pb.area - ref_b) > max(2.5, 0.15 * ref_b):
        ring_bot = rings[0]
    if not pt.is_valid or pt.is_empty or abs(pt.area - ref_t) > max(2.5, 0.15 * ref_t):
        ring_top = rings[-1]
    return ring_bot, ring_top


def _build_live_vsa_planes(occ_np: np.ndarray, t0: float) -> dict:
    """Mode 3: Multi-storey 3D Plane Arrangement (VSA) merging planar tiers across floors."""
    nz = occ_np.shape[0]
    floor_polys = _extract_floor_polygons(occ_np)

    vmap: dict[tuple[float, float, float], int] = {}
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []

    def vid(x: float, y: float, z: float) -> int:
        key = (round(float(x), 4), round(float(y), 4), round(float(z), 4))
        idx = vmap.get(key)
        if idx is None:
            idx = len(verts)
            vmap[key] = idx
            verts.append(key)
        return idx

    def add_tri(a, b, c):
        ia, ib, ic = vid(*a), vid(*b), vid(*c)
        if ia != ib and ib != ic and ic != ia:
            faces.append([ia, ib, ic])

    def emit_ring_quad(p0, p1, z0, q0, q1, z1, is_hole: bool):
        if is_hole:
            a, b, c, d = (p1[0], p1[1], z0), (p0[0], p0[1], z0), (q0[0], q0[1], z1), (q1[0], q1[1], z1)
        else:
            a, b, c, d = (p0[0], p0[1], z0), (p1[0], p1[1], z0), (q1[0], q1[1], z1), (q0[0], q0[1], z1)
        add_tri(a, b, c)
        add_tri(a, c, d)

    # Group contiguous storeys into maximal VSA tiers: (z_start, z_end, polys_or_fitted_pair)
    tiers: list[tuple[int, int, Polygon, Polygon]] = []
    extra_slabs: list[tuple[int, Polygon]] = []
    z = 0
    while z < nz:
        geoms = floor_polys[z]
        if not geoms:
            z += 1
            continue
        if len(geoms) != 1 or geoms[0].interiors:
            for g in geoms:
                extra_slabs.append((z, g))
            z += 1
            continue

        r0 = _ensure_ccw(_collapse_collinear(list(geoms[0].exterior.coords)))
        aligned_rings = [r0]
        z_end = z + 1
        while z_end < nz and len(floor_polys[z_end]) == 1 and not floor_polys[z_end][0].interiors:
            rc = _ensure_ccw(_collapse_collinear(list(floor_polys[z_end][0].exterior.coords)))
            matched = _match_loft_pair(aligned_rings[-1], rc, max_shift=2.6)
            if matched is None:
                break
            aligned_rings.append(matched)
            z_end += 1

        ring_bot, ring_top = _fit_tier_vsa_planes(aligned_rings, z, z_end)
        tiers.append((z, z_end, Polygon(ring_bot), Polygon(ring_top)))
        z = z_end

    # Track horizontal unions by elevation z for conformal zero-diaphragm capping
    below_top: dict[int, list[Polygon]] = {}
    above_bot: dict[int, list[Polygon]] = {}

    for z0, z1, p_bot, p_top in tiers:
        above_bot.setdefault(z0, []).append(p_bot)
        below_top.setdefault(z1, []).append(p_top)
        rb = _ensure_ccw(list(p_bot.exterior.coords))
        rt = _ensure_ccw(list(p_top.exterior.coords))
        if len(rb) == len(rt):
            m = len(rb)
            for i in range(m):
                emit_ring_quad(rb[i], rb[(i + 1) % m], float(z0), rt[i], rt[(i + 1) % m], float(z1), False)
        else:
            # Fallback vertical extrusions per edge if polygon sanitization altered vertex count
            for i in range(len(rb)):
                emit_ring_quad(rb[i], rb[(i + 1) % len(rb)], float(z0), rb[i], rb[(i + 1) % len(rb)], float(z1), False)

    for z_s, g in extra_slabs:
        above_bot.setdefault(z_s, []).append(g)
        below_top.setdefault(z_s + 1, []).append(g)
        ext = _ensure_ccw(list(g.exterior.coords))
        for i in range(len(ext)):
            emit_ring_quad(ext[i], ext[(i + 1) % len(ext)], float(z_s), ext[i], ext[(i + 1) % len(ext)], float(z_s + 1), False)
        for hole in g.interiors:
            hr = _ensure_ccw(list(hole.coords))
            for i in range(len(hr)):
                emit_ring_quad(hr[i], hr[(i + 1) % len(hr)], float(z_s), hr[i], hr[(i + 1) % len(hr)], float(z_s + 1), True)

    all_zs = sorted(set(below_top.keys()) | set(above_bot.keys()))
    for zv in all_zs:
        b_geom = unary_union(below_top[zv]) if zv in below_top else None
        a_geom = unary_union(above_bot[zv]) if zv in above_bot else None
        if b_geom is not None:
            diff_up = b_geom if a_geom is None else b_geom.difference(a_geom)
            _triangulate_cap(diff_up, float(zv), True, add_tri)
        if a_geom is not None:
            diff_dn = a_geom if b_geom is None else a_geom.difference(b_geom)
            _triangulate_cap(diff_dn, float(zv), False, add_tri)

    return _finalize_mesh(verts, faces, crease_deg=3.5, non_flat=False, method='vsa_planes', start_time=t0)


def devoxelize_occupancy(occ_np: np.ndarray, revision: int, method: str = 'correlated_loft') -> dict:
    """Devoxelize binary occupancy (NZ, NY, NX) for modes 1, 2, 3 with caching."""
    valid_methods = {'correlated_loft', 'extruded_slabs', 'vsa_planes'}
    if method not in valid_methods:
        method = 'correlated_loft'

    cache_key = (int(revision), method)
    with _DEVOX_LOCK:
        cached = _DEVOX_CACHE.get(cache_key)
        if cached is not None:
            return cached

    t0 = time.perf_counter()
    if method == 'correlated_loft':
        res = _build_live_25d(occ_np, loft_slopes=True, method=method, t0=t0)
    elif method == 'extruded_slabs':
        res = _build_live_25d(occ_np, loft_slopes=False, method=method, t0=t0)
    else:
        res = _build_live_vsa_planes(occ_np, t0=t0)

    with _DEVOX_LOCK:
        if len(_DEVOX_CACHE) > 24:
            _DEVOX_CACHE.clear()
        _DEVOX_CACHE[cache_key] = res
    return res


# Create FastAPI app
library = ContextLibrary()
session = UnifiedSession(library=library)
app = FastAPI(title='VoxelForm RL Unified')

_DEVOX_VIEWER_EXT = """
// --- Live Devoxelization Extension (Unified Server) ---
let devoxEnabled = false;
let currentDevoxMesh = null;
let lastDevoxState = null;
let lastDevoxView = 'solid';
let devoxGroup = null;
let devoxMeshObj = null;
let devoxLinesObj = null;

function rebuildDevoxSceneObjects() {
  if (!mass || !mass.parent) return;
  if (!devoxGroup) {
    devoxGroup = new THREE.Group();
    devoxGroup.name = 'devoxGroup';
    mass.parent.add(devoxGroup);
  }
  if (devoxMeshObj) {
    devoxGroup.remove(devoxMeshObj);
    devoxMeshObj.geometry.dispose();
    if (devoxMeshObj.material && devoxMeshObj.material.dispose) devoxMeshObj.material.dispose();
    devoxMeshObj = null;
  }
  if (devoxLinesObj) {
    devoxGroup.remove(devoxLinesObj);
    devoxLinesObj.geometry.dispose();
    if (devoxLinesObj.material && devoxLinesObj.material.dispose) devoxLinesObj.material.dispose();
    devoxLinesObj = null;
  }

  const active = Boolean(devoxEnabled && currentDevoxMesh && currentDevoxMesh.faces && currentDevoxMesh.faces.length && lastDevoxView === 'solid' && lastDevoxState);
  mass.visible = !active;
  if (outline) outline.visible = !active;
  devoxGroup.visible = active;
  if (!active) {
    if (typeof render === 'function') render();
    else if (typeof requestRender === 'function') requestRender();
    else window.dispatchEvent(new Event('resize'));
    return;
  }

  const verts = currentDevoxMesh.vertices;
  const faces = currentDevoxMesh.faces;
  const creases = currentDevoxMesh.creases || [];
  const offset = sceneOffset(lastDevoxState);
  const angle = lastDevoxState.context?.gridAngle || 0;
  const displayLift = Math.max(0.2, ...(lastDevoxState.context?.site?.vertices || []).map(v => v[2])) + 0.03;

  const worldVerts = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    const [gx, gy, gz] = verts[i];
    const [wx, wy] = gridToWorld(gx, gy, offset, angle);
    worldVerts[i * 3] = wx;
    worldVerts[i * 3 + 1] = gz * 3.0 + displayLift;
    worldVerts[i * 3 + 2] = -wy;
  }

  let geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(worldVerts, 3));
  const idx = [];
  for (let i = 0; i < faces.length; i++) {
    idx.push(faces[i][0], faces[i][1], faces[i][2]);
  }
  geo.setIndex(idx);

  if (currentDevoxMesh.nonFlat) {
    geo.computeVertexNormals();
  } else {
    geo = geo.toNonIndexed();
    geo.computeVertexNormals();
  }

  // Match VoxelForm signature lightBlue ink shading when using vertexColors or uniform
  const useVC = Boolean(mass.material && mass.material.vertexColors);
  const posAttr = geo.getAttribute('position');
  if (useVC && posAttr) {
    const cols = new Float32Array(posAttr.count * 3);
    for (let i = 0; i < posAttr.count; i++) {
      cols[i * 3] = lightBlue.r;
      cols[i * 3 + 1] = lightBlue.g;
      cols[i * 3 + 2] = lightBlue.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  }

  const mat = mass.material ? mass.material.clone() : new THREE.MeshLambertMaterial({ color: lightBlue });
  devoxMeshObj = new THREE.Mesh(geo, mat);
  devoxMeshObj.castShadow = true;
  devoxMeshObj.receiveShadow = true;
  devoxGroup.add(devoxMeshObj);

  if (creases.length) {
    const linePos = new Float32Array(creases.length * 6);
    for (let i = 0; i < creases.length; i++) {
      const [u, v] = creases[i];
      linePos[i * 6 + 0] = worldVerts[u * 3 + 0];
      linePos[i * 6 + 1] = worldVerts[u * 3 + 1] + 0.015;
      linePos[i * 6 + 2] = worldVerts[u * 3 + 2];
      linePos[i * 6 + 3] = worldVerts[v * 3 + 0];
      linePos[i * 6 + 4] = worldVerts[v * 3 + 1] + 0.015;
      linePos[i * 6 + 5] = worldVerts[v * 3 + 2];
    }
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    const lineMat = new THREE.LineBasicMaterial({ color: outlineBlue, transparent: true, opacity: 0.9 });
    devoxLinesObj = new THREE.LineSegments(lineGeo, lineMat);
    devoxGroup.add(devoxLinesObj);
  }
  if (typeof render === 'function') render();
  else if (typeof requestRender === 'function') requestRender();
  else window.dispatchEvent(new Event('resize'));
}

window.addEventListener('message', event => {
  if (event.source !== window.parent || event.origin !== location.origin) return;
  const msg = event.data;
  if (!msg) return;
  if (['voxel_initialize', 'voxel_update', 'voxel_display'].includes(msg.type)) {
    if (msg.state) lastDevoxState = msg.state;
    if (msg.view) lastDevoxView = msg.view;
    rebuildDevoxSceneObjects();
  } else if (msg.type === 'voxel_devox') {
    devoxEnabled = Boolean(msg.enabled);
    currentDevoxMesh = msg.mesh || null;
    rebuildDevoxSceneObjects();
  }
});
"""

@app.get('/assets/viewer.js')
def custom_viewer_js():
    base_js = (VOXELFORM_ROOT / 'public/viewer.js').read_text()
    return Response(base_js + '\n' + _DEVOX_VIEWER_EXT, media_type='application/javascript')

app.mount('/assets', StaticFiles(directory=VOXELFORM_ROOT / 'public'), name='assets')

viewer_public = library.root / 'frontend/public'
if not (viewer_public / 'scripts/urban_viewer.js').is_file():
    viewer_public = library.root / 'public'
for directory in ('scripts', 'styles', 'vendor'):
    app.mount(f'/shared-viewer/{directory}', StaticFiles(directory=viewer_public / directory), name=f'shared-viewer-{directory}')

@app.get('/')
def home():
    html_path = VOXELFORM_ROOT / 'public/index.html'
    content = html_path.read_text()
    
    # Inject indicator and unified action selector without touching the repository file
    badge = '<span class="badge" style="background:#5e35b1;color:#fff;margin-left:6px;font-size:11px;">Unified Action Space (Roof Growth)</span>'
    content = content.replace('<span class="badge">Experimental proxies</span>', '<span class="badge">Experimental proxies</span>' + badge)
    
    # Inject Operation dropdown next to One Mask button
    op_select = (
        '<button id="step">One mask</button>'
        '<select id="unifiedOp" style="background:#1e232a;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 6px;font-size:12px;margin-left:4px;cursor:pointer;" title="Unified Action Space Operation">'
        '<option value="auto">Auto (Neural)</option>'
        '<option value="exchange">Exchange (Roof + Carve)</option>'
        '<option value="grow">Grow (Roof Up)</option>'
        '<option value="remove">Remove (Carve)</option>'
        '</select>'
    )
    content = content.replace('<button id="step">One mask</button>', op_select)

    # Inject Live Devoxelization panel controls (Modes 0–3, OFF by default, end-of-single-generation only)
    devox_controls = (
        '<div style="margin:10px 0 6px;padding:8px 10px;border:1px solid #30363d;border-radius:6px;background:rgba(15,23,42,0.45);">'
        '<label class="checkbox" style="margin-bottom:6px;display:flex;align-items:center;gap:6px;">'
        '<input id="devoxToggle" type="checkbox" />'
        '<span>Devoxelize after single generation</span>'
        '</label>'
        '<label for="devoxAlgo" style="display:block;font-size:11px;opacity:0.8;margin-bottom:4px;">Devoxelization mode (0–3)</label>'
        '<select id="devoxAlgo" style="width:100%;background:#1e232a;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:4px 6px;font-size:12px;cursor:pointer;">'
        '<option value="off">0. Raw Voxel Grid (Devox Off)</option>'
        '<option value="correlated_loft" selected>1. Correlated Support-Plane Loft (Default)</option>'
        '<option value="extruded_slabs">2. 2.5D Extruded Floor Slabs</option>'
        '<option value="vsa_planes">3. 3D Plane Arrangement (VSA)</option>'
        '</select>'
        '<div id="devoxMeta" style="margin-top:6px;font-size:11px;color:#8b949e;display:flex;justify-content:space-between;align-items:center;gap:6px;">'
        '<span>0. Devox OFF (raw voxels)</span>'
        '<a href="/devoxelize" target="_blank" style="color:#58a6ff;text-decoration:none;">Compare 0–3 Lab ↗</a>'
        '</div>'
        '</div>'
        '<div id="layerLegend"'
    )
    content = content.replace('<div id="layerLegend"', devox_controls)

    # Intercept control/state fetches and gate live devoxelization strictly to finished single generation
    inject_script = """
    <script>
    (function(){
      let latestRev = -1;
      let latestState = null;
      let appliedKey = '';
      let pendingTimer = null;
      const linkHtml = '<a href="/devoxelize" target="_blank" style="color:#58a6ff;text-decoration:none;">Compare 0–3 Lab ↗</a>';

      function clearDevoxViewport(statusText) {
        const vp = document.getElementById('viewport');
        const metaEl = document.getElementById('devoxMeta');
        if (vp && vp.contentWindow) {
          vp.contentWindow.postMessage({ type: 'voxel_devox', enabled: false, mesh: null }, location.origin);
        }
        if (metaEl && statusText) {
          metaEl.innerHTML = `<span>${statusText}</span>` + linkHtml;
        }
      }

      async function syncDevox(force) {
        const toggle = document.getElementById('devoxToggle');
        const algoSel = document.getElementById('devoxAlgo');
        const metaEl = document.getElementById('devoxMeta');
        const vp = document.getElementById('viewport');
        if (!toggle || !algoSel || !vp || !vp.contentWindow) return;

        const method = algoSel.value || 'correlated_loft';
        const enabled = Boolean(toggle.checked && method !== 'off');

        if (!enabled) {
          appliedKey = 'off';
          clearDevoxViewport('0. Devox OFF (raw voxels)');
          return;
        }

        const st = latestState;
        const canDevox = Boolean(st && st.ready && !st.running && !st.training && st.episodeComplete);
        if (!canDevox) {
          appliedKey = 'waiting';
          let msg = 'Armed · devoxelizes when single generation finishes';
          if (st && st.training) msg = 'Skipped during training (raw voxels)';
          else if (st && st.running) msg = 'Generating… (will devoxelize at end)';
          clearDevoxViewport(msg);
          return;
        }

        const key = `${latestRev}:${method}`;
        if (!force && key === appliedKey) return;
        try {
          if (metaEl) metaEl.innerHTML = '<span>Devoxelizing finished massing…</span>' + linkHtml;
          const resp = await origFetch(`/api/devoxelize_live?method=${encodeURIComponent(method)}&revision=${latestRev}`);
          if (!resp.ok) {
            clearDevoxViewport('Armed · devoxelizes when single generation finishes');
            return;
          }
          const mesh = await resp.json();
          const stNow = latestState;
          if (!toggle.checked || algoSel.value !== method || !stNow || stNow.running || stNow.training || !stNow.episodeComplete) {
            return;
          }
          appliedKey = `${mesh.revision ?? latestRev}:${method}`;
          vp.contentWindow.postMessage({ type: 'voxel_devox', enabled: true, mesh }, location.origin);
          const s = mesh.stats || {};
          if (metaEl) metaEl.innerHTML = `<span>${s.triangles ?? 0} tris · ${s.volume_m3 ?? 0} m³ (${s.elapsed_ms ?? 0}ms)</span>` + linkHtml;
        } catch(e) {}
      }

      document.getElementById('devoxToggle')?.addEventListener('change', () => {
        const toggle = document.getElementById('devoxToggle');
        const algoSel = document.getElementById('devoxAlgo');
        if (toggle && algoSel && toggle.checked && algoSel.value === 'off') {
          algoSel.value = 'correlated_loft';
        }
        syncDevox(true);
      });
      document.getElementById('devoxAlgo')?.addEventListener('change', () => {
        const toggle = document.getElementById('devoxToggle');
        const algoSel = document.getElementById('devoxAlgo');
        if (toggle && algoSel) {
          toggle.checked = (algoSel.value !== 'off');
        }
        syncDevox(true);
      });
      document.getElementById('viewport')?.addEventListener('load', () => {
        syncDevox(true);
      });

      const origFetch = window.fetch;
      window.fetch = function(url, opts) {
        if (typeof url === 'string' && url.endsWith('/api/control') && opts && opts.body) {
          appliedKey = '';
          clearDevoxViewport(document.getElementById('devoxToggle')?.checked ? 'Updating… (raw voxels)' : '0. Devox OFF (raw voxels)');
          try {
            const body = JSON.parse(opts.body);
            const op = document.getElementById('unifiedOp')?.value;
            if (op && op !== 'auto') {
              body.forceOperation = op;
            }
            opts.body = JSON.stringify(body);
          } catch(e) {}
        }
        const prom = origFetch.apply(this, arguments);
        if (typeof url === 'string' && url.includes('/api/state')) {
          prom.then(r => r.clone().json()).then(st => {
            if (st && typeof st.revision === 'number') {
              const prevComplete = Boolean(latestState && latestState.episodeComplete);
              const prevRunning = Boolean(latestState && latestState.running);
              const prevTraining = Boolean(latestState && latestState.training);
              latestState = st;
              const changed = (
                st.revision !== latestRev ||
                Boolean(st.episodeComplete) !== prevComplete ||
                Boolean(st.running) !== prevRunning ||
                Boolean(st.training) !== prevTraining
              );
              latestRev = st.revision;
              if (changed) {
                clearTimeout(pendingTimer);
                pendingTimer = setTimeout(() => syncDevox(false), 25);
              }
            }
          }).catch(() => {});
        }
        return prom;
      };
    })();
    </script>
    </body>
    """
    content = content.replace('</body>', inject_script)
    return HTMLResponse(content)

@app.get('/api/devoxelize_live')
def devoxelize_live(method: str = 'correlated_loft', force: bool = False):
    with session.lock:
        if not session.ready or session.env is None:
            raise HTTPException(409, 'Environment is still preparing')
        if not force and (session.running or session.training or not session.env.terminal):
            raise HTTPException(409, 'Devoxelization only runs at the end of a single generation')
        occ_np = session.env.occupied.to('cpu', torch.bool).numpy().copy()
        rev = int(session.revision)
    res = devoxelize_occupancy(occ_np, rev, method=method)
    return dict(revision=rev, **res)

@app.get('/devoxelize')
def devoxelize():
    html_file = VOXELFORM_ROOT / 'public/devoxelize.html'
    return FileResponse(html_file)

@app.get('/api/state')
def state():
    return session.state()

@app.get('/api/sites')
def sites():
    return [
        dict(
            siteId=r['site_id'],
            city=r.get('city_name', 'Urban site'),
            cityCode=r.get('city_code', ''),
            tier=r.get('area_tier', ''),
            area=r.get('site_area_m2'),
        )
        for r in library.catalog()
    ]

@app.post('/api/control')
async def control(request: Request):
    data = await request.json()
    command = data.get('command')
    force_op = data.get('forceOperation')
    with session.lock:
        if not session.ready:
            raise HTTPException(409, 'Environment is still preparing')
        try:
            if command == 'run':
                session.start(training=bool(data.get('training', True)))
            elif command == 'pause':
                session.pause()
            elif command == 'step':
                session.start(training=bool(data.get('training', False)), single=True, force_operation=force_op)
            elif command == 'masks':
                pass # Unified action space handles masks
            elif command == 'grid':
                session.configure_grid(data.get('alignment', 'world'))
            elif command == 'reset':
                session.pause()
                session.next_episode()
            elif command == 'settings':
                session.configure_far(data.get('targetFar'), data.get('densityPreset', 'custom'), data.get('farTolerance'))
            elif command == 'site':
                session.change_site(data.get('source', 'cached_osm'), data.get('siteId'), data.get('plan', 'rectangle'))
            else:
                raise ValueError('Unknown command')
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
    return dict(ok=True)

@app.get('/api/policy')
def save_policy():
    with session.lock:
        if not session.ready:
            raise HTTPException(409, 'Not ready')
        buffer = io.BytesIO()
        torch.save(session.policy.state(), buffer)
    return Response(
        buffer.getvalue(),
        media_type='application/octet-stream',
        headers={'Content-Disposition': 'attachment; filename="voxelform-unified-policy.pt"'},
    )

if __name__ == '__main__':
    port = int(os.getenv('VOXELFORM_PORT', '8012'))
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='info')
