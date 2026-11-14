"""Unified VoxelForm Live Preview Server.

Runs the unified action space (neural remove / roof grow / exchange / finish)
with vertical headroom (10 storeys envelope, 6 storeys initial mass)
as the active VoxelForm application. Growth copies a complete top floor.
"""
import functools
import io
import json
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
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

from voxelform.context import ContextLibrary, public_scene
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.session import DENSITY_PRESETS
from voxelform.shadows import device_name, sync
from voxelform.training import depth_potential
from voxelform.unified import UnifiedLearner, commit, OPERATIONS

torch.set_num_threads(max(1, int(os.getenv('VOXELFORM_THREADS', '1'))))

class UnifiedSession:
    def __init__(self, library=None, device=None, initial_storeys=6, ceiling_storeys=10, autostart=True):
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
        self.env.copy_labels=np.zeros(self.env.shape,dtype=np.int64)
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None
        self.env.precompute_external()

        self.policy = UnifiedLearner(
            device=self.device,
            seed=self.pending_seed,
            config=self.config,
            amount_feedback=True,
            net_guided=True,
            floor_copy=True,
        )

        # New action semantics start fresh. Explicit imports must match them.
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
        self.env.copy_labels=np.zeros(self.env.shape,dtype=np.int64)
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None
        self.steps = 0
        self.policy.clear()
        self.episode_recorded = False
        self.edit_trace = []
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
        self.env.unified_actions = 0
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.termination = None
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
        self.env.precompute_external()
        solid = self.env.allowed.clone()
        solid[self.initial_storeys:] = False
        self.env.restore_occupancy(solid)
        self.env.copy_labels=np.zeros(self.env.shape,dtype=np.int64)
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
        self.env.precompute_external()
        solid = self.env.allowed.clone()
        solid[self.initial_storeys:] = False
        self.env.restore_occupancy(solid)
        self.env.copy_labels=np.zeros(self.env.shape,dtype=np.int64)
        self.env.unified_limit = max(8, self.env.workload.limit + 8)
        self.env.unified_actions = 0
        self.env.termination = None

        self.steps = 0
        self.episode_recorded = False
        self.edit_trace = []
        self.policy.clear()
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
        self.episode_recorded=False;self.revision+=1

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
            self.stats = self.policy.finish(0. if self.env.terminal else self.policy.fields(self.env)[1].detach())
            sync(self.device)
            self.learning_seconds = time.perf_counter() - stamp

    def finish_episode(self):
        if self.episode_recorded:
            return
        if self.training and self.policy.tape and self.env.termination!='far_reached':
            self.policy.add_reward(-0.5*abs(self.env.count-self.env.goal_count)/max(1,abs(self.env.start_count-self.env.goal_count)))
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
        self.env.unified_actions += 1
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
            if self.policy.amount_feedback and self.policy.last_generated_count:
                self.env.workload.observe(0, self.policy.last_generated_count)
            if self.training and self.policy.tape:self.policy.record(-0.03)
            if self.env.unified_actions >= self.env.unified_limit:
                self.env.termination = 'far_reached' if self.env.far_satisfied else 'step_budget_exhausted'
                self.finish_episode()
            elif self.training and len(self.policy.tape)>=self.rollout_steps:self.learn_rollout()
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
            metrics.update(stepBudget=e.unified_limit, attempts=e.unified_actions)
            return dict(
                ready=True,
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
                    growthMode='copy_top_floor',
                    initialStoreys=self.initial_storeys,
                    ceilingStoreys=self.ceiling_storeys,
                    densityPresets=DENSITY_PRESETS,
                    maximumFar=e.capacity / e.plot_area,
                    maskMode='combined',
                    deepInterior=self.config.eligibility=='depth',
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
                policyMigration='Whole-floor growth · paired carving/growth available from the first edit · column support enforced',
                maskGenerator='unified_floor_copy',
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
    app.mount(f'/shared-viewer/{directory}', StaticFiles(directory=viewer_public / directory), name=f'shared-viewer-{directory}')

@app.get('/')
def home():
    html_path = VOXELFORM_ROOT / 'public/index.html'
    content = html_path.read_text()
    
    # Inject indicator and unified action selector without touching the repository file
    badge = '<span class="badge" style="background:#5e35b1;color:#fff;margin-left:6px;font-size:11px;">Whole-floor growth</span>'
    content = content.replace('<span class="badge">Experimental proxies</span>', '<span class="badge">Experimental proxies</span>' + badge)
    
    # Inject Operation dropdown next to One Mask button
    op_select = (
        '<button id="step">One mask</button>'
        '<select id="unifiedOp" style="background:#fff;color:#18315a;border:1px solid #cbd6e8;border-radius:4px;padding:3px 6px;font-size:12px;margin-left:4px;cursor:pointer;" title="One-mask action · probes can bypass the FAR direction">'
        '<option value="auto">Auto · learned action</option>'
        '<option value="exchange">Copy floor + carve</option>'
        '<option value="grow">Copy top floor · probe</option>'
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
                session.configure_masks(data.get('mode','combined'), data.get('deepInterior',True), data.get('maximumInteriorVoids',1), data.get('maximumEditVoxels',0), data.get('editAllocation','feedback'))
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

@app.post('/api/policy')
async def load_policy(request: Request):
    payload=await request.body()
    with session.lock:
        if not session.ready:raise HTTPException(409,'Not ready')
        try:
            try:saved=torch.load(io.BytesIO(payload),map_location=session.device,weights_only=True)
            except Exception as exc:raise ValueError('Could not read this policy file') from exc
            if not isinstance(saved,dict):raise ValueError('Invalid policy file')
            if not saved.get('floorCopy') or not saved.get('netGuided'):raise ValueError('Import a whole-floor growing policy saved from this preview')
            replacement=UnifiedLearner(session.device)
            replacement.load(saved)
            session.pause();session.policy=replacement;session.config=replacement.config
            session.env.frontier.config=replacement.config;session.env.frontier.refresh()
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
