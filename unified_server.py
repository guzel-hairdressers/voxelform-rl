"""Unified VoxelForm Live Preview Server.

Runs the organic 3D volumetric action space (neural remove / grow / exchange / finish)
with vertical headroom (10 storeys envelope, 6 storeys initial mass)
as the active VoxelForm application.
"""
import functools
import io
import json
import math
import os
import pickle
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
from fastapi.responses import HTMLResponse, Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voxelform.context import ContextLibrary, public_scene
from voxelform.cores import DEFAULT_MAX_CORE_DIST, DEFAULT_MIN_CORE_SPACING, core_templates_summary
from voxelform.core_plans import plan_library, render_plan_svg, CORE_LAYOUT_VERSION
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.session import DENSITY_PRESETS
from voxelform.shadows import device_name, sync
from voxelform.training import depth_potential
from voxelform.unified import (
    OPERATIONS,
    UnifiedLearner,
    _allowed_long_narrow,
    clean_floor_tensor,
    commit,
    long_narrow_mask,
    min_room_width,
)


from voxelform.objectives import organic_objective, organic_step_reward, organic_terminal_reward

def derive_site_storeys(record=None, buildings=None):
    """Derive default initial and ceiling storeys from surrounding context height."""
    avg_h = record.get('avg_height_m') if record else None
    max_h = record.get('max_height_m') if record else None
    if (avg_h is None or max_h is None) and buildings:
        hs = [max(v[2] for v in b['vertices']) for b in buildings if b.get('vertices')]
        if hs:
            if avg_h is None:
                avg_h = float(np.median(hs))
            if max_h is None:
                max_h = float(np.max(hs))
    avg_h = 18.0 if avg_h is None else float(avg_h)
    max_h = 30.0 if max_h is None else float(max_h)
    initial = max(4, min(12, int(round(avg_h / 3.0))))
    ceiling = max(initial + 10, min(40, int(math.ceil(max_h / 3.0)) + 6))
    return initial, ceiling

torch.set_num_threads(max(1, int(os.getenv('VOXELFORM_THREADS', '1'))))

class UnifiedSession:
    def __init__(self, library=None, device=None, initial_storeys=None, ceiling_storeys=None, autostart=True):
        self.library = library or ContextLibrary()
        self.lock = threading.RLock()
        self.running = False
        self.single = False
        self.training = False
        self.ready = False
        self.error = None
        self._fixed_initial = int(initial_storeys) if initial_storeys is not None else None
        self._fixed_ceiling = int(ceiling_storeys) if ceiling_storeys is not None else None
        self.initial_storeys = 6 if self._fixed_initial is None else self._fixed_initial
        self.ceiling_storeys = 10 if self._fixed_ceiling is None else self._fixed_ceiling

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
        self.rollout_steps = 16
        self.audit_steps = 5
        self.force_operation = None
        self.scene_cache_dir = VOXELFORM_ROOT / 'results' / 'scene_cache'
        self._ctx_cache = {}
        self._init_snap = None
        self._state_cache = {}
        self._env_static_cache = None
        self.episode_started = time.perf_counter()
        self.op_counts = {k: 0 for k in OPERATIONS}
        self.last_objective = 0.0
        self.last_diag = {}

        chosen_device = device or os.getenv('VOXELFORM_DEVICE', 'auto')
        self.device = device_name(chosen_device)
        self.target_far = 3.0
        self.density_preset = 'medium'
        self.cores_enabled = True
        self.min_core_spacing = DEFAULT_MIN_CORE_SPACING
        self.max_core_dist = DEFAULT_MAX_CORE_DIST
        self.episode_recorded = False
        self.action_seconds = 0.0
        self.learning_seconds = 0.0

        self.worker = threading.Thread(target=self.loop, daemon=True)
        if autostart:self.worker.start()

    def clock(self):
        return self.active + (time.perf_counter() - self.started if self.started is not None else 0.0)

    def pause(self):
        if self.started is not None:
            self.active += time.perf_counter() - self.started
            self.started = None
        self.running = False
        self.single = False
        self.force_operation = None

    def _load_context(self, site_id, grid_alignment='auto_fit', ceiling_storeys=None):
        ceil_s = int(self.ceiling_storeys if ceiling_storeys is None else ceiling_storeys)
        key = (str(site_id), ceil_s, str(grid_alignment))
        record = next((r for r in self.library.catalog() if r['site_id'] == site_id), None)
        src_path = self.library._scene_path(record) if record is not None else None
        sig = None
        if src_path is not None and src_path.is_file():
            st = src_path.stat()
            sig = (int(st.st_mtime_ns), int(st.st_size))
        cached = self._ctx_cache.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        safe_align = str(grid_alignment).replace(':', '_')
        cache_file = self.scene_cache_dir / f'ctx_{site_id}_{ceil_s}_{safe_align}.pkl'
        if sig is not None and cache_file.is_file():
            try:
                with cache_file.open('rb') as fh:
                    disk_sig, ctx = pickle.load(fh)
                if disk_sig == sig and isinstance(ctx, dict):
                    self._ctx_cache[key] = (sig, ctx)
                    return ctx
            except Exception:
                pass
        ctx = self.library.load(site_id, floors=ceil_s, grid_alignment=grid_alignment)
        self._ctx_cache[key] = (sig, ctx)
        if sig is not None:
            try:
                self.scene_cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_name(f'{cache_file.name}.{os.getpid()}.tmp')
                with tmp.open('wb') as fh:
                    pickle.dump((sig, ctx), fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, cache_file)
            except Exception:
                pass
        return ctx

    def _install_env(self, context, seed, plan, far_tolerance=0.1, target_far=None):
        factory = functools.partial(PatchFacades, config=self.config, depth_device=self.device if self.device != 'cpu' else None)
        shape = context['shape'] if context is not None else (int(self.ceiling_storeys), 24, 32)
        desired_far = float(self.target_far if target_far is None else target_far)
        if context is not None:
            max_far = int(context['allowed'].sum()) * int(shape[0]) / float(context['siteArea'])
            if desired_far > max_far:
                desired_far = max(0.1, math.floor(max_far * 100.0) / 100.0)
                self.density_preset = 'custom'
        try:
            replacement = Environment(
                shape=shape,
                seed=seed,
                device=self.device,
                plan=plan,
                context=context,
                target_far=desired_far,
                frontier_factory=factory,
                far_tolerance=far_tolerance,
                scene_cache_dir=self.scene_cache_dir if context is not None else None,
                fast_self_shadows=True,
                fast_diffuse=True,
                cores_enabled=self.cores_enabled,
                min_core_spacing=self.min_core_spacing,
                max_core_dist=self.max_core_dist,
            )
        except ValueError:
            desired_far = max(0.1, float(shape[0]) * 0.5)
            self.density_preset = 'custom'
            replacement = Environment(
                shape=shape,
                seed=seed,
                device=self.device,
                plan=plan,
                context=context,
                target_far=desired_far,
                frontier_factory=factory,
                far_tolerance=far_tolerance,
                scene_cache_dir=self.scene_cache_dir if context is not None else None,
                fast_self_shadows=True,
                fast_diffuse=True,
                cores_enabled=self.cores_enabled,
                min_core_spacing=self.min_core_spacing,
                max_core_dist=self.max_core_dist,
            )
        self.env = replacement
        self.target_far = self.env.target_far
        self.env.precompute_external()
        solid = self.env.allowed.clone()
        solid[int(self.initial_storeys):] = False
        self.env.restore_occupancy(solid)
        self.env.copy_labels = np.zeros(self.env.shape, dtype=np.int64)
        self.env._ln6_cache = None
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None
        pol = getattr(self, 'policy', None)
        self.env.organic_3d_growth = getattr(pol, 'organic_3d', True)
        self.env.floor_copy_growth = getattr(pol, 'floor_copy', False)
        self.last_objective, self.last_diag = organic_objective(self.env)

    def build(self):
        catalog = self.library.catalog()
        if not catalog:
            raise ValueError('No cached urban scenes found. Ensure Module-Lab is accessible.')
        initial_id = os.getenv('VOXELFORM_SITE_ID', 'm_bcn_0093')
        initial = next((r for r in catalog if r['site_id'] == initial_id), catalog[0])
        self.site_id = initial['site_id']
        d_init, d_ceil = derive_site_storeys(initial)
        if self._fixed_initial is None:
            self.initial_storeys = d_init
        if self._fixed_ceiling is None:
            self.ceiling_storeys = d_ceil

        context = self._load_context(self.site_id, grid_alignment='auto_fit', ceiling_storeys=self.ceiling_storeys)
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
            vertical_interior=False,
            budget_reserve_fraction=0.2,
        )
        self.policy = UnifiedLearner(
            device=self.device,
            seed=self.pending_seed,
            config=self.config,
            amount_feedback=True,
            net_guided=True,
            floor_copy=False,
            organic_3d=True,
        )
        self._install_env(context, seed=self.pending_seed, plan=self.plan, far_tolerance=0.1)

        # New action semantics start fresh. Explicit imports must match them.
        self.stats = dict(
            updates=self.policy.updates,
            directMaskUpdates=self.policy.direct_updates,
            importedUpdates=0,
            loss=0.0,
            gradientNorm=0.0,
        )
        self.episode_started = time.perf_counter()
        self._capture_initial_snapshot()
        self._state_cache.clear()
        self._env_static_cache = None
        self.ready = True
        self.revision += 1

    def _snap_key(self):
        e = self.env
        return (
            id(e),
            self.site_id,
            tuple(e.shape),
            float(self.target_far),
            float(e.far_tolerance),
            self.config,
            int(self.initial_storeys),
            self.plan,
            bool(self.cores_enabled),
            int(self.min_core_spacing),
            int(self.max_core_dist),
        )

    def _capture_initial_snapshot(self):
        e = self.env; f = e.frontier
        if e.edits != 0 or not hasattr(f, 'facade_band'):
            self._init_snap = None
            return
        self._init_snap = (
            self._snap_key(),
            dict(
                occupied=e.occupied.clone(),
                count=int(e.count),
                start_count=int(e.start_count),
                light=e.light.clone(),
                facade_light_faces=e.facade_light_faces.clone(),
                facade_light=e.facade_light.clone(),
                usable=e.usable.clone(),
                visible=e.visible.clone(),
                receiver_data=e.receiver_data,
                quality=e.quality.clone(),
                values={k: v.clone() for k, v in e.values.items()},
                depth=e.depth.clone(),
                depth_badness=e.depth_badness.clone(),
                cores=tuple(e.cores),
                core_mask=e.core_mask.clone(),
                core_distance=e.core_distance.clone(),
                core_dist_badness=e.core_dist_badness.clone(),
                core_metrics=dict(e.core_metrics),
                core_seconds=float(e.core_seconds),
                last_receiver_count=int(e.last_receiver_count),
                last_new_receivers=int(e.last_new_receivers),
                last_eval_seconds=float(e.last_eval_seconds),
                depth_seconds=float(e.depth_seconds),
                last_objective=float(self.last_objective),
                last_diag=dict(self.last_diag),
                frontier=dict(
                    occupied=f.occupied.copy(),
                    exterior_air=f.exterior_air.copy(),
                    remove_faces=f.remove_faces.copy(),
                    add_faces=f.add_faces.copy(),
                    remove_skin=f.remove_skin.copy(),
                    add_skin=f.add_skin.copy(),
                    components=int(f.components),
                    count=int(f.count),
                    depth_tensor=f.depth_tensor.clone() if f.depth_tensor is not None else None,
                    depth_numpy=f.depth_numpy.copy() if f.depth_numpy is not None else None,
                    old_voids=[(lab.copy(), list(ids), sz.copy()) for lab, ids, sz in f.old_voids] if f.old_voids is not None else None,
                    facade_band=f.facade_band.copy(),
                    remove_weights=f.remove_weights.copy() if f.remove_weights is not None else None,
                    _remove_eligible=f._remove_eligible.copy(),
                    facade_seconds=float(f.facade_seconds),
                ),
            ),
        )

    def _restore_initial_snapshot(self):
        if self._init_snap is None or self._init_snap[0] != self._snap_key():
            return False
        from voxelform.workload import reset_budget
        snap = self._init_snap[1]
        e = self.env; f = e.frontier; fs = snap['frontier']
        e.occupied = snap['occupied'].clone()
        e.count = snap['count']; e.start_count = snap['start_count']
        e.edits = 0; e.voxels_edited = 0; e.last_changed_voxels = 0; e.last_operation = None
        e.light = snap['light'].clone()
        e.facade_light_faces = snap['facade_light_faces'].clone()
        e.facade_light = snap['facade_light'].clone()
        e.usable = snap['usable'].clone()
        e.visible = snap['visible'].clone()
        e.receiver_data = snap['receiver_data']
        e.quality = snap['quality'].clone()
        e.values = {k: v.clone() for k, v in snap['values'].items()}
        e.depth = snap['depth'].clone()
        e.depth_badness = snap['depth_badness'].clone()
        e.cores = tuple(snap['cores'])
        e.core_mask = snap['core_mask'].clone()
        e.core_distance = snap['core_distance'].clone()
        e.core_dist_badness = snap['core_dist_badness'].clone()
        e.core_metrics = dict(snap['core_metrics'])
        e.core_seconds = float(snap['core_seconds'])
        e.depth_edit = 0; e.last_audit_edit = 0
        e.last_receiver_count = snap['last_receiver_count']
        e.last_new_receivers = snap['last_new_receivers']
        e.last_eval_seconds = snap['last_eval_seconds']
        e.depth_seconds = snap['depth_seconds']
        f.occupied = fs['occupied'].copy()
        f.exterior_air = fs['exterior_air'].copy()
        f.remove_faces = fs['remove_faces'].copy()
        f.add_faces = fs['add_faces'].copy()
        f.remove_skin = fs['remove_skin'].copy()
        f.add_skin = fs['add_skin'].copy()
        f.components = fs['components']; f.count = fs['count']
        f.depth_tensor = fs['depth_tensor'].clone() if fs['depth_tensor'] is not None else None
        f.depth_numpy = fs['depth_numpy'].copy() if fs['depth_numpy'] is not None else None
        f.old_voids = [(lab.copy(), list(ids), sz.copy()) for lab, ids, sz in fs['old_voids']] if fs['old_voids'] is not None else None
        f.facade_band = fs['facade_band'].copy()
        f.remove_weights = fs['remove_weights'].copy() if fs['remove_weights'] is not None else None
        f._remove_eligible = fs['_remove_eligible'].copy()
        f._weight_tensors = {}
        if hasattr(f, '_prev_occ'):
            f._prev_occ = f.occupied.copy()
            f._prev_air = f.exterior_air.copy()
        f.version += 1
        f.facade_seconds = fs['facade_seconds']; f.projection_seconds = 0.; f.commit_seconds = 0.
        reset_budget(e)
        self.last_objective = snap['last_objective']
        self.last_diag = dict(snap['last_diag'])
        return True

    def configure_cores(self, enabled=None, min_spacing=None, max_dist=None):
        if enabled is not None:
            self.cores_enabled = bool(enabled)
        if min_spacing is not None:
            val_s = int(min_spacing)
            if val_s < 0 or val_s > 60:
                raise ValueError('minCoreSpacing must be between 0 and 60')
            self.min_core_spacing = val_s
        if max_dist is not None:
            val_d = int(max_dist)
            if val_d < 2 or val_d > 80:
                raise ValueError('maxCoreDist must be between 2 and 80')
            self.max_core_dist = val_d
        if self.env is not None:
            self.env.configure_cores(
                cores_enabled=self.cores_enabled,
                min_core_spacing=self.min_core_spacing,
                max_core_dist=self.max_core_dist,
            )
            if getattr(self.policy, 'organic_3d', False):
                self.last_objective, self.last_diag = organic_objective(self.env)
        self._init_snap = None
        self._capture_initial_snapshot()
        self._state_cache.clear()
        self.revision += 1

    def next_episode(self, clear_tape=True):
        if not self._restore_initial_snapshot():
            solid = self.env.allowed.clone()
            solid[self.initial_storeys:] = False
            self.env.restore_occupancy(solid)
        self.env.copy_labels=np.zeros(self.env.shape,dtype=np.int64)
        self.env._ln6_cache = None
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None
        self.env.organic_3d_growth = getattr(getattr(self, 'policy', None), 'organic_3d', False)
        self.env.floor_copy_growth = getattr(getattr(self, 'policy', None), 'floor_copy', False)
        if self._init_snap is None or self._init_snap[0] != self._snap_key():
            self.last_objective, self.last_diag = organic_objective(self.env)
            self._capture_initial_snapshot()
        self.steps = 0
        self.episode_started = time.perf_counter()
        if clear_tape:
            self.policy.clear()
        self.episode_recorded = False
        self.edit_trace = []
        self._state_cache.clear()
        self.revision += 1

    def start(self, training=True, single=False, force_operation=None):
        if force_operation is not None:
            if force_operation not in ('remove','grow','exchange'):raise ValueError('Unknown probe action')
            if not bool(self.policy.legal_operations(self.env,net_guided=False)[OPERATIONS.index(force_operation)]):raise ValueError('That action is unavailable for the current mass or height ceiling')
        new_training = bool(training)
        if new_training != self.training:
            self.policy.clear()
        self.training = new_training
        self.force_operation = force_operation if force_operation in ('remove', 'grow', 'exchange') else None
        if self.steps == 0 and getattr(self.env, 'unified_actions', 0) == 0:
            self.episode_started = time.perf_counter()
        if single:
            self.pause()
            self.force_operation = force_operation if force_operation in ('remove', 'grow', 'exchange') else None
            self.single = True
        else:
            self.running = True
            if self.started is None:
                self.started = time.perf_counter()
        self._state_cache.clear()

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
        self.env.unified_actions = 0
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.termination = None
        self.last_objective, self.last_diag = organic_objective(self.env)
        self.steps = 0
        self.episode_started = time.perf_counter()
        self.episode_recorded = False
        self.edit_trace = []
        self.op_counts = {k: 0 for k in OPERATIONS}
        self._capture_initial_snapshot()
        self._state_cache.clear()
        self.policy.clear()
        self.revision += 1

    def change_site(self, source, site_id=None, plan='rectangle'):
        if source not in ('cached_osm', 'synthetic'):
            raise ValueError('Unknown context source')
        if plan not in ('rectangle', 'triangle', 'courtyard', 'branch'):
            raise ValueError('Unknown synthetic plot')
        self.pause()
        if source == 'cached_osm':
            target_id = site_id or self.site_id
            record = next((r for r in self.library.catalog() if r['site_id'] == target_id), None)
            d_init, d_ceil = derive_site_storeys(record)
            if self._fixed_initial is None:
                self.initial_storeys = d_init
            if self._fixed_ceiling is None:
                self.ceiling_storeys = d_ceil
            context = self._load_context(target_id, grid_alignment=(self.env.context or {}).get('gridAlignment', 'auto_fit'), ceiling_storeys=self.ceiling_storeys)
        else:
            if self._fixed_initial is None:
                self.initial_storeys = 6
            if self._fixed_ceiling is None:
                self.ceiling_storeys = 10
            context = None

        self.pending_seed += 1
        self.plan = plan
        self.site_id = context['site_id'] if context else None
        self._install_env(context, seed=self.pending_seed, plan=plan, far_tolerance=self.env.far_tolerance)

        self.steps = 0
        self.episode_started = time.perf_counter()
        self.episode_recorded = False
        self.edit_trace = []
        self.op_counts = {k: 0 for k in OPERATIONS}
        self._init_snap = None
        self._env_static_cache = None
        self._state_cache.clear()
        self._capture_initial_snapshot()
        self.policy.clear()
        self.revision += 1

    def configure_grid(self, alignment):
        if self.env.context is None:
            raise ValueError('Grid alignment requires a fetched plot')
        context = self._load_context(self.site_id, grid_alignment=alignment, ceiling_storeys=self.ceiling_storeys)
        if context['gridAlignment'] == self.env.context.get('gridAlignment', 'world'):
            return
        self.pause()
        self._install_env(context, seed=self.pending_seed, plan=self.plan, far_tolerance=self.env.far_tolerance)

        self.steps = 0
        self.episode_started = time.perf_counter()
        self.episode_recorded = False
        self.edit_trace = []
        self.op_counts = {k: 0 for k in OPERATIONS}
        self._init_snap = None
        self._env_static_cache = None
        self._state_cache.clear()
        self._capture_initial_snapshot()
        self.policy.clear()
        self.revision += 1

    def configure_envelope(self, initial_storeys=None, ceiling_storeys=None, auto=False):
        self.pause()
        if auto:
            if self.env.context is not None:
                record = next((r for r in self.library.catalog() if r['site_id'] == self.site_id), None)
                d_init, d_ceil = derive_site_storeys(record, self.env.context.get('buildings'))
            else:
                d_init, d_ceil = 6, 10
            self._fixed_initial = None
            self._fixed_ceiling = None
            self.initial_storeys = d_init
            self.ceiling_storeys = d_ceil
        else:
            if initial_storeys is None or ceiling_storeys is None or isinstance(initial_storeys, bool) or isinstance(ceiling_storeys, bool):
                raise ValueError('Initial and ceiling storeys must be integers')
            init_s = int(initial_storeys)
            ceil_s = int(ceiling_storeys)
            if init_s < 1 or ceil_s < 2 or ceil_s > 60 or init_s > ceil_s:
                raise ValueError('Require 1 <= initialStoreys <= ceilingStoreys <= 60 (minimum ceiling 2)')
            self._fixed_initial = init_s
            self._fixed_ceiling = ceil_s
            self.initial_storeys = init_s
            self.ceiling_storeys = ceil_s
        context = (
            self._load_context(self.site_id, grid_alignment=(self.env.context or {}).get('gridAlignment', 'auto_fit'), ceiling_storeys=self.ceiling_storeys)
            if self.env.context is not None else None
        )
        self._install_env(context, seed=self.pending_seed, plan=self.plan, far_tolerance=self.env.far_tolerance)
        self.steps = 0
        self.episode_started = time.perf_counter()
        self.episode_recorded = False
        self.edit_trace = []
        self.op_counts = {k: 0 for k in OPERATIONS}
        self._init_snap = None
        self._env_static_cache = None
        self._state_cache.clear()
        self._capture_initial_snapshot()
        self.policy.clear()
        self.revision += 1

    def reset_weights(self):
        self.pause()
        self.pending_seed += 1
        old = self.policy
        self.policy = UnifiedLearner(
            device=self.device,
            seed=self.pending_seed,
            config=self.config,
            amount_feedback=getattr(old, 'amount_feedback', True),
            repair_remove=getattr(old, 'repair_remove', True),
            net_guided=getattr(old, 'net_guided', True),
            floor_copy=getattr(old, 'floor_copy', False),
            organic_3d=getattr(old, 'organic_3d', True),
        )
        self.stats = dict(
            updates=0,
            directMaskUpdates=0,
            importedUpdates=0,
            loss=0.0,
            gradientNorm=0.0,
        )
        self.episodes = 0
        self.active = 0.0
        self.started = None
        self.recent = []
        self.op_counts = {k: 0 for k in OPERATIONS}
        self.action_seconds = 0.0
        self.learning_seconds = 0.0
        self.next_episode(clear_tape=True)
        self.revision += 1

    def configure_masks(self,mode,deep,voids,cap,allocation):
        from dataclasses import replace
        if mode not in ('combined','correlated'):raise ValueError('Growing preview uses correlated fields')
        if allocation!='feedback':raise ValueError('Growing preview uses workload feedback')
        if int(voids)<0 or int(cap)<0:raise ValueError('Limits must be nonnegative')
        config=replace(self.config,mode=mode,eligibility='depth' if deep else 'facade',maximum_voids=int(voids),maximum_edit_voxels=int(cap))
        self.pause();self.policy.clear()
        self.config=self.policy.config=config
        self.env.frontier.config=config;self.env.frontier.refresh()
        self.env.termination=None;self.env.unified_actions=0
        self.env.unified_limit=max(8,self.env.workload.limit+8)
        self._init_snap=None;self._capture_initial_snapshot();self._state_cache.clear()
        self.episode_recorded=False;self.revision+=1

    def audit_environment(self):
        if getattr(self.policy, 'organic_3d', False):
            if self.env.last_audit_edit != self.env.edits:
                self.env.evaluate()
            self.last_objective, self.last_diag = organic_objective(self.env)
            return
        before = float(self.env.quality.detach().cpu())
        if self.env.last_audit_edit != self.env.edits:
            self.env.evaluate()
        if self.training and self.policy.tape:
            bonus = 5.0 * (float(self.env.quality.detach().cpu()) - before)
            self.policy.add_reward(bonus)

    def learn_rollout(self):
        if self.training and self.policy.tape:
            stamp = time.perf_counter()
            self.stats = self.policy.finish(0. if self.env.terminal else self.policy.fields(self.env)[1].detach())
            sync(self.device)
            self.learning_seconds = time.perf_counter() - stamp

    def finish_episode(self):
        if self.episode_recorded:
            return
        is_organic = getattr(self.policy, 'organic_3d', False)
        if is_organic:
            self.audit_environment()
            if self.training and self.policy.tape:
                self.policy.add_reward(organic_terminal_reward(self.env,self.last_objective))
                self.policy.mark_done()
                if len(self.policy.tape) >= self.rollout_steps:
                    self.learn_rollout()
        else:
            if self.training and self.policy.tape and self.env.termination != 'far_reached':
                self.policy.add_reward(-0.5 * abs(self.env.count - self.env.goal_count) / max(1, abs(self.env.start_count - self.env.goal_count)))
            self.audit_environment()
            self.learn_rollout()
        self.episodes += 1
        self.episode_recorded = True
        row = dict(
            episode=self.episodes,
            siteId=self.site_id,
            plan=self.plan,
            seed=self.pending_seed,
            training=self.training,
            densityPreset=self.density_preset,
            success=self.env.termination == 'far_reached',
            episodeSeconds=max(0.001, time.perf_counter() - getattr(self, 'episode_started', time.perf_counter())),
            **self.env.metrics(),
        )
        if is_organic and self.last_diag:
            row.update(self.last_diag)
        self.recent.append(row)
        self.recent = self.recent[-200:]
        if not self.training:
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
            self.next_episode(clear_tape=not self.training)
            if self.env.terminal:
                self.finish_episode()
                self.single = False
                self.revision += 1
                return

        stamp = time.perf_counter()
        force_op = self.force_operation if not self.training else None
        move = self.policy.choose(self.env, training=self.training, force_operation=force_op)
        self.env.unified_actions += 1
        if not self.training:
            sync(self.device)
        self.action_seconds = time.perf_counter() - stamp

        is_organic = getattr(self.policy, 'organic_3d', False)
        operation = getattr(self.policy, 'last_operation', None)
        if operation in self.op_counts:
            self.op_counts[operation] += 1
        if operation == 'finish':
            self.env.termination = 'far_reached'
            if self.training and self.policy.tape:
                self.policy.record(0.0 if is_organic else 0.1)
            self.finish_episode()
            self.single = False
            self.revision += 1
            return

        if move is None:
            if self.policy.amount_feedback and self.policy.last_generated_count:
                self.env.workload.observe(0, self.policy.last_generated_count)
            if self.training and self.policy.tape:
                self.policy.record(-0.02 if is_organic else -0.03)
            if self.env.unified_actions >= self.env.unified_limit:
                self.env.termination = 'far_reached' if self.env.far_satisfied else 'step_budget_exhausted'
                self.finish_episode()
            elif not is_organic and self.training and len(self.policy.tape) >= self.rollout_steps:
                self.learn_rollout()
            self.single = False
            self.revision += 1
            return

        before = abs(self.env.count - self.env.goal_count)
        depth = depth_potential(self.env)
        far_before = self.env.far
        core_shaping = bool(is_organic and self.training and getattr(self.env, 'cores_enabled', False))
        prev_core_score = float(self.last_diag.get('coreScore') or 0.0) if core_shaping else 0.0

        commit(self.env, move, validated=True)
        self.steps = self.env.edits

        if self.policy.amount_feedback and self.policy.last_generated_count:
            self.env.workload.observe(
                int(move.remove.sum() + move.add.sum()),
                self.policy.last_generated_count,
            )

        core_delta = 0.0
        if core_shaping:
            self.env.update_cores()
            cur_cov = max(0.0, 1.0 - float(self.env.core_metrics.get('coreUncoveredFraction', 0.0)))
            cur_bad = float(self.env.core_metrics.get('coreDistBadness', 0.0))
            cur_core_score = max(0.0, cur_cov * (1.0 - 0.5 * cur_bad))
            core_delta = cur_core_score - prev_core_score
            self.last_diag['coreScore'] = cur_core_score

        if self.training:
            initial = max(1, abs(self.env.start_count - self.env.goal_count))
            limit = getattr(self.env, 'unified_limit', 16)
            if is_organic:
                reward = organic_step_reward(self.env,core_delta)
            else:
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
                    attempts=self.env.unified_actions,
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
        if self.env.unified_actions >= limit:
            self.env.termination = 'far_reached' if self.env.far_satisfied else 'step_budget_exhausted'

        if self.env.terminal:
            self.finish_episode()
        else:
            if self.steps % self.audit_steps == 0:
                self.audit_environment()
            elif not self.training:
                self.env.update_cores()
                if is_organic:
                    self.last_objective, self.last_diag = organic_objective(self.env)
            if not is_organic and self.training and len(self.policy.tape) >= self.rollout_steps:
                self.learn_rollout()

        self.single = False
        self.revision += 1

    def loop(self):
        try:
            if not self.ready:
                with self.lock:
                    self.build()
            while True:
                burst = 8 if (self.running and self.training) else 1
                with self.lock:
                    for _ in range(burst):
                        if not (self.running or self.single):
                            break
                        before_ep = self.episodes
                        self.advance_one()
                        if self.episodes != before_ep or not (self.running and self.training):
                            break
                time.sleep(0.001 if self.running else 0.01)
        except Exception:
            with self.lock:
                self.error = traceback.format_exc()
                self.pause()

    def _dashboard_summary(self):
        rows = [r for r in self.recent if r.get('siteId') == self.site_id and r.get('plan', self.plan) == self.plan] if self.recent else []
        if not rows:
            rows = list(self.recent)
        objs = [r['organicObjective'] for r in rows if r.get('organicObjective') is not None]
        ep_secs = [r['episodeSeconds'] for r in rows[-10:] if r.get('episodeSeconds')]
        mean_sec = float(np.mean(ep_secs)) if ep_secs else None
        best_row = max(rows, key=lambda r: r.get('organicObjective', -1e9)) if objs else None
        return dict(
            baselineObjective=float(np.mean(objs[:5])) if objs else float(self.last_objective),
            recentObjective=float(np.mean(objs[-5:])) if objs else float(self.last_objective),
            bestObjective=float(best_row['organicObjective']) if best_row else float(self.last_objective),
            bestEpisode=int(best_row['episode']) if best_row else None,
            bestUsableGoodArea=float(best_row.get('usableGoodArea', 0.0)) if best_row else float(self.last_diag.get('usableGoodArea', 0.0)),
            bestDirectSun=float(best_row.get('directSunProxy', 0.0)) if best_row else None,
            meanEpisodeSeconds=mean_sec,
            episodesPerMinute=(60.0 / mean_sec) if (mean_sec and mean_sec > 0) else None,
            opCounts=dict(self.op_counts),
        )

    def state_json(self, view='all'):
        with self.lock:
            if not self.ready:
                return json.dumps(dict(ready=False, error=self.error, device=self.device), separators=(',', ':')).encode('utf-8')
            req_view = view if view in ('all', 'solid', 'usable', 'light', 'depthBadness', 'noise', 'coreDistance') else 'all'
            key = (self.revision, self.running, self.training, self.episodes, int(self.stats.get('updates', 0)), req_view)
            cached = self._state_cache.get(req_view)
            if cached is not None and cached[0] == key:
                return cached[1]
            encoded = json.dumps(self.state(view=req_view), separators=(',', ':')).encode('utf-8')
            self._state_cache[req_view] = (key, encoded)
            return encoded

    def state(self, view='all'):
        with self.lock:
            if not self.ready:
                return dict(ready=False, error=self.error, device=self.device)
            e = self.env
            metrics = e.metrics()
            metrics.update(stepBudget=e.unified_limit, attempts=e.unified_actions)
            is_organic = getattr(self.policy, 'organic_3d', False)
            if is_organic and self.last_diag:
                metrics.update(self.last_diag)
            req_view = view if view in ('all', 'solid', 'usable', 'light', 'depthBadness', 'noise', 'coreDistance') else 'all'
            static = self._env_static_cache
            if static is None or static[0] != id(e):
                static = (
                    id(e),
                    e.allowed.to('cpu', torch.uint8).flatten().tolist(),
                    e.boxes.tolist(),
                    public_scene(e.context) if e.context is not None else None,
                )
                self._env_static_cache = static
            _, allowed_list, boxes_list, ctx_pub = static
            payload = dict(
                ready=True,
                coreLayoutVersion=CORE_LAYOUT_VERSION,
                error=self.error,
                running=self.running,
                training=self.training,
                activeSeconds=self.clock(),
                episodes=self.episodes,
                step=self.steps,
                attempts=e.unified_actions,
                lastRejection=self.policy.last_rejection,
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
                    growthMode='organic_3d' if is_organic else 'copy_top_floor',
                    initialStoreys=self.initial_storeys,
                    ceilingStoreys=self.ceiling_storeys,
                    contextStoreys=dict(avg=max(1, int(round(e.context_avg_height / 3.0))), max=max(1, int(math.ceil(e.context_max_height / 3.0)))),
                    densityPresets=DENSITY_PRESETS,
                    maximumFar=e.capacity / e.plot_area,
                    maskMode='combined',
                    deepInterior=self.config.eligibility=='depth',
                    minimumPatch=self.policy.config.minimum_patch,
                    maximumInteriorVoids=self.policy.config.maximum_voids,
                    minimumVoidArea=self.policy.config.minimum_void_area,
                    maximumEditVoxels=self.policy.config.maximum_edit_voxels,
                    editAllocation=self.policy.config.allocation,
                    coresEnabled=self.cores_enabled,
                    minCoreSpacing=self.min_core_spacing,
                    maxCoreDist=self.max_core_dist,
                    coreTemplates=core_templates_summary(),
                ),
                seed=self.pending_seed,
                plan=self.plan,
                siteId=self.site_id,
                revision=self.revision,
                metrics=metrics,
                learning=self.stats,
                policyMigration='Organic 3D volumetric editing · cantilevers unrestricted · vectorized floor usability guard' if is_organic else 'Whole-floor growth · paired carving/growth available from the first edit · column support enforced',
                maskGenerator='unified_organic_3d' if is_organic else 'unified_floor_copy',
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
                context=ctx_pub,
                actionSeconds=self.action_seconds,
                learningSeconds=self.learning_seconds,
                shape=list(e.shape),
                occupied=e.occupied.to('cpu', torch.uint8).flatten().tolist(),
                cores=[c.to_dict() for c in e.cores],
                coreMask=e.core_mask.to('cpu', torch.uint8).flatten().tolist(),
                allowed=allowed_list,
                boxes=boxes_list,
                recent=self.recent,
                dashboard=self._dashboard_summary(),
            )
            if req_view == 'all':
                payload['facadeDepth'] = e.depth.detach().cpu().flatten().tolist()
                payload['depthBadness'] = e.depth_badness.detach().cpu().flatten().tolist()
                payload['light'] = e.light.detach().cpu().flatten().tolist()
                payload['noise'] = e.noise.detach().cpu().flatten().tolist()
                payload['usable'] = e.usable.to('cpu', torch.uint8).flatten().tolist()
                payload['coreDistance'] = e.core_distance.detach().cpu().flatten().tolist()
                payload['coreDistBadness'] = e.core_dist_badness.detach().cpu().flatten().tolist()
            elif req_view == 'usable':
                payload['usable'] = e.usable.to('cpu', torch.uint8).flatten().tolist()
            elif req_view == 'light':
                payload['light'] = e.light.detach().cpu().flatten().tolist()
            elif req_view == 'depthBadness':
                payload['depthBadness'] = e.depth_badness.detach().cpu().flatten().tolist()
            elif req_view == 'noise':
                payload['noise'] = e.noise.detach().cpu().flatten().tolist()
            elif req_view == 'coreDistance':
                payload['coreDistance'] = e.core_distance.detach().cpu().flatten().tolist()
                payload['coreDistBadness'] = e.core_dist_badness.detach().cpu().flatten().tolist()
            return payload

# =============================================================================
# General-Purpose Live Devoxelizer (Coplanar 2.5D + Non-Flat Taubin / DC+QEM)
# =============================================================================

from voxelform.devoxelize import devoxelize_occupancy

# Create FastAPI app
library = ContextLibrary()
session = UnifiedSession(library=library, autostart=False)
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app):
    session.worker.start()
    yield
app = FastAPI(lifespan=lifespan, title='VoxelForm RL Unified')


@app.get('/assets/viewer.js')
def custom_viewer_js():
    base_js = (VOXELFORM_ROOT / 'public/viewer.js').read_text()
    return Response(base_js + '\n' + (VOXELFORM_ROOT / 'public/devox-viewer.js').read_text(), media_type='application/javascript')

app.mount('/assets', StaticFiles(directory=VOXELFORM_ROOT / 'public'), name='assets')

viewer_public = library.root / 'frontend/public'
if not (viewer_public / 'scripts/urban_viewer.js').is_file():
    viewer_public = library.root / 'public'
for directory in ('scripts', 'styles', 'vendor'):
    app.mount(f'/shared-viewer/{directory}', StaticFiles(directory=viewer_public / directory, check_dir=False), name=f'shared-viewer-{directory}')

@app.get('/')
def home():
    html_path = VOXELFORM_ROOT / 'public/index.html'
    content = html_path.read_text()
    
    # Inject indicator and unified action selector without touching the repository file
    badge = '<span class="badge" style="background:#5e35b1;color:#fff;margin-left:6px;font-size:11px;">Organic 3D</span>'
    content = content.replace('<span class="badge">Experimental proxies</span>', '<span class="badge">Experimental proxies</span>' + badge)
    
    # Inject Operation dropdown next to One Mask button
    op_select = (
        '<button id="step">One mask</button>'
        '<select id="unifiedOp" style="background:#fff;color:#18315a;border:1px solid #cbd6e8;border-radius:4px;padding:3px 6px;font-size:12px;margin-left:4px;cursor:pointer;" title="One-mask action · probes can bypass the FAR direction">'
        '<option value="auto">Auto · learned action</option>'
        '<option value="exchange">Grow + carve (exchange)</option>'
        '<option value="grow">Grow 3D · probe</option>'
        '<option value="remove">Carve · probe</option>'
        '</select>'
    )
    content = content.replace('<button id="step">One mask</button>', op_select)

    # Inject Live Devoxelization panel controls (Modes 0–3, OFF by default, end-of-single-generation only)
    devox_controls = (
        '<div style="margin:10px 0 6px;padding:8px 10px;border:1px solid #dbe4f4;border-radius:6px;background:#f4f7ff;">'
        '<label class="checkbox" style="margin-bottom:6px;display:flex;align-items:center;gap:6px;">'
        '<input id="devoxToggle" type="checkbox" />'
        '<span>Devoxelize after single generation</span>'
        '</label>'
        '<label for="devoxAlgo" style="display:block;font-size:11px;opacity:0.8;margin-bottom:4px;">Devoxelization mode (0–3)</label>'
        '<select id="devoxAlgo" style="width:100%;background:#fff;color:#18315a;border:1px solid #cbd6e8;border-radius:4px;padding:4px 6px;font-size:12px;cursor:pointer;">'
        '<option value="off">0. Raw Voxel Grid (Devox Off)</option>'
        '<option value="correlated_loft" selected>1. Correlated Support-Plane Loft (Default)</option>'
        '<option value="extruded_slabs">2. 2.5D Extruded Floor Slabs</option>'
        '<option value="vsa_planes">3. 3D Plane Arrangement (VSA)</option>'
        '</select>'
        '<div id="devoxMeta" style="margin-top:6px;font-size:11px;color:#63718b;display:flex;justify-content:space-between;align-items:center;gap:6px;">'
        '<span>0. Devox OFF (raw voxels)</span>'
        '<a href="/devoxelize" target="_blank" style="color:#1645ff;text-decoration:none;">Compare 0–3 Lab ↗</a>'
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
      const linkHtml = '<a href="/devoxelize" target="_blank" style="color:#1645ff;text-decoration:none;">Compare 0–3 Lab ↗</a>';

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
          if (mesh.revision !== latestRev || !toggle.checked || algoSel.value !== method || !stNow || stNow.running || stNow.training || !stNow.episodeComplete) {
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
def devoxelize_live(method: str = 'correlated_loft', revision: int | None = None):
    with session.lock:
        if not session.ready or session.env is None:
            raise HTTPException(409, 'Environment is still preparing')
        if session.running or session.training or not session.env.terminal:
            raise HTTPException(409, 'Devoxelization only runs at the end of a single generation')
        if revision is not None and revision != session.revision:raise HTTPException(409, 'Preview changed')
        if method not in ('correlated_loft','extruded_slabs','vsa_planes'):raise HTTPException(400, 'Unknown devoxelization method')
        occ_np = session.env.occupied.to('cpu', torch.bool).numpy().copy()
        rev = int(session.revision)
    res = devoxelize_occupancy(occ_np, rev, method=method)
    with session.lock:
        if session.revision != rev or session.running or session.training:raise HTTPException(409, 'Preview changed during reconstruction')
    return dict(revision=rev, **res)

@app.get('/devoxelize')
def devoxelize():
    html_file = VOXELFORM_ROOT / 'public/devoxelize.html'
    return FileResponse(html_file)


@app.get('/api/core-plans')
def core_plan_data():
    return JSONResponse(dict(version=CORE_LAYOUT_VERSION,plans=plan_library()),headers={'Cache-Control':'no-store'})

@app.get('/api/core-plans/{tier}.svg')
def core_plan_svg(tier: str):
    plan=next((p for p in plan_library() if p['tier']==tier),None)
    if plan is None:raise HTTPException(404,'Unknown core tier')
    return Response(render_plan_svg(plan),media_type='image/svg+xml',headers={'Cache-Control':'no-store'})

@app.get('/core-plans')
def core_plans():
    html_file = VOXELFORM_ROOT / 'public/core-plans.html'
    return FileResponse(html_file,headers={'Cache-Control':'no-store'})

@app.get('/api/state')
def state(view: str = 'all'):
    return Response(content=session.state_json(view=view), media_type='application/json')

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
                session.configure_masks(data.get('mode','combined'), data.get('deepInterior',True), data.get('maximumInteriorVoids',1), data.get('maximumEditVoxels',0), data.get('editAllocation','feedback'))
            elif command == 'grid':
                session.configure_grid(data.get('alignment', 'world'))
            elif command == 'reset':
                session.pause()
                session.next_episode()
            elif command == 'reset_weights':
                session.reset_weights()
            elif command == 'envelope':
                session.configure_envelope(data.get('initialStoreys'), data.get('ceilingStoreys'), auto=bool(data.get('auto', False)))
            elif command == 'cores':
                session.configure_cores(enabled=data.get('coresEnabled'), min_spacing=data.get('minCoreSpacing'), max_dist=data.get('maxCoreDist'))
            elif command == 'settings':
                session.configure_far(data.get('targetFar'), data.get('densityPreset', 'custom'), data.get('farTolerance'))
            elif command == 'site':
                session.change_site(data.get('source', 'cached_osm'), data.get('siteId'), data.get('plan', 'rectangle'))
            else:
                raise ValueError('Unknown command')
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
    return dict(ok=True)

@app.post('/api/policy')
async def load_policy(request: Request):
    payload=await request.body()
    with session.lock:
        if not session.ready:raise HTTPException(409,'Not ready')
        try:
            try:saved=torch.load(io.BytesIO(payload),map_location=session.device,weights_only=True)
            except Exception as exc:raise ValueError('Could not read this policy file') from exc
            if not isinstance(saved,dict):raise ValueError('Invalid policy file')
            if not (saved.get('organic3d') or saved.get('floorCopy')) or not saved.get('netGuided'):raise ValueError('Import an organic 3D or whole-floor unified policy saved from this preview')
            replacement=UnifiedLearner(session.device)
            replacement.load(saved)
            session.pause();session.policy=replacement;session.config=replacement.config
            session.env.organic_3d_growth=replacement.organic_3d
            session.env.floor_copy_growth=replacement.floor_copy
            session.env.frontier.config=replacement.config;session.env.frontier.refresh()
            session._init_snap=None;session._capture_initial_snapshot();session._state_cache.clear()
            session.policy.clear();session.revision+=1
            session.stats.update(updates=replacement.updates,directMaskUpdates=replacement.direct_updates)
        except (ValueError,KeyError,RuntimeError,TypeError,pickle.UnpicklingError,EOFError) as exc:raise HTTPException(400,str(exc)) from exc
    return dict(ok=True)

@app.get('/api/policy')
def save_policy():
    with session.lock:
        if not session.ready:
            raise HTTPException(409, 'Not ready')
        # Include learning from a paused partial rollout in a manual export.
        session.learn_rollout()
        buffer = io.BytesIO()
        torch.save(session.policy.state(), buffer)
    return Response(
        buffer.getvalue(),
        media_type='application/octet-stream',
        headers={'Content-Disposition': 'attachment; filename="voxelform-unified-policy.pt"'},
    )

if __name__ == '__main__':
    port = int(os.getenv('VOXELFORM_PORT', '8011'))
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='info')
