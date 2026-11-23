"""Headless, auditable train/evaluate entry point for the organic RL task."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import shutil
import statistics
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy import stats as scipy_stats

from voxelform.coarse_to_fine import (
    CoarseEnv,
    CoarseToFineLearner,
    CoarseToFineNetwork,
    run_cascade_episode,
    tier_cascade_blocks,
)
from voxelform.context import ContextLibrary
from voxelform.cores import solve_cores
from voxelform.environment import Environment
from voxelform.experiment import (
    ROOT,
    ExperimentConfig,
    archive_source,
    read_checkpoint,
    restore_rng,
    restore_training_checkpoint,
    rng_state,
    runtime_manifest,
    save_checkpoint,
    seed_everything,
    source_manifest,
    write_json,
)
from voxelform.objectives import (
    organic_objective,
    organic_step_reward,
    organic_terminal_reward,
)
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.rollout import finish_episode_rollout
from voxelform.shadows import device_name, sync
from voxelform.unified import UnifiedLearner, commit


class Scenes:
    def __init__(self, config):
        self.config = config
        self.device = device_name(config.device)
        self.envs = {}
        self.manifest = {}
        self.setup = []
        self._coarse = {}
        self.patch = PatchConfig(
            mode="combined",
            allocation="feedback",
            eligibility="depth",
            facade_layers=2,
            minimum_patch=4,
            maximum_patches=4,
            maximum_voids=1,
            minimum_void_area=4,
            batched=True,
            vertical_interior=False,
            budget_reserve_fraction=0.2,
        )

    def get(self, site):
        if site in self.envs:
            return self.envs[site]
        t = time.perf_counter()
        c = self.config
        context = None
        plan = "rectangle"
        scene_seed = int.from_bytes(
            hashlib.sha256(site.encode()).digest()[:4], "little"
        ) % (2**31)
        if site.startswith("synthetic:"):
            plan = site.split(":", 1)[1]
            source_hash = hashlib.sha256(
                f"{site}:{scene_seed}:{c.synthetic_width}:{c.synthetic_length}".encode()
            ).hexdigest()
        else:
            library = ContextLibrary()
            record = next((r for r in library.catalog() if r["site_id"] == site), None)
            if record is None:
                raise ValueError(
                    f"Cached scene unavailable: {site}; see docs/DATA_CARD.md"
                )
            source_hash = hashlib.sha256(
                library._scene_path(record).read_bytes()
            ).hexdigest()
            context = library.load(
                site, floors=c.ceiling_storeys, grid_alignment="auto_fit"
            )
        env = Environment(
            (c.ceiling_storeys, c.synthetic_width, c.synthetic_length),
            seed=scene_seed,
            plan=plan,
            context=context,
            device=self.device,
            sun_count=8,
            target_far=c.target_far,
            far_tolerance=c.far_tolerance,
            frontier_factory=functools.partial(
                PatchFacades, config=self.patch, depth_device=self.device
            ),
            fast_self_shadows=True,
            fast_diffuse=True,
            cores_enabled=c.cores_enabled,
            min_core_spacing=c.min_core_spacing,
            max_core_dist=c.max_core_dist,
        )
        env.precompute_external()
        self.envs[site] = env
        self.manifest[site] = dict(
            sourceSHA256=source_hash,
            sceneSeed=scene_seed,
            shape=list(env.shape),
            parcelArea=env.plot_area,
            allowedSHA256=hashlib.sha256(
                np.packbits(env.allowed.cpu().numpy()).tobytes()
            ).hexdigest(),
            contextBuildings=len(context["buildings"]) if context else 16,
            gridAngle=(context or {}).get("gridAngle", 0.0),
            center=[context.get("lat"), context.get("lon")] if context else None,
            contextRadius=(context or {}).get("radius", 100.0),
        )
        self.setup.append(dict(site=site, seconds=time.perf_counter() - t))
        return env

    def coarse_envs(self, site):
        if self.config.multiscale_schedule == "none":
            return ()
        if site not in self._coarse:
            env = self.get(site)
            self._coarse[site] = tuple(
                CoarseEnv(env, blk)
                for blk in tier_cascade_blocks(env, self.config.multiscale_schedule)
            )
        return self._coarse[site]

    def prepare(self):
        groups = (
            self.config.train_sites,
            self.config.validation_sites,
            self.config.test_sites,
        )
        for group in groups:
            for site in group:
                self.get(site)
        # Reject cross-split context circles that overlap when georeferencing is
        # available. Distinct site identifiers alone do not prevent spatial leakage.
        for i, group in enumerate(groups):
            for later in groups[i + 1 :]:
                for left in group:
                    for right in later:
                        a, b = self.manifest[left], self.manifest[right]
                        if (
                            not a["center"]
                            or not b["center"]
                            or any(v is None for v in (*a["center"], *b["center"]))
                        ):
                            continue
                        la, lo = map(math.radians, a["center"])
                        lb, lp = map(math.radians, b["center"])
                        q = (
                            math.sin((lb - la) / 2) ** 2
                            + math.cos(la) * math.cos(lb) * math.sin((lp - lo) / 2) ** 2
                        )
                        distance = 2 * 6371000 * math.asin(min(1.0, math.sqrt(q)))
                        if distance <= float(a["contextRadius"]) + float(
                            b["contextRadius"]
                        ):
                            raise ValueError(
                                f"Spatial context leakage between {left} and {right}"
                            )


def make_policy(config, device, seed):
    patch = PatchConfig(
        mode="combined",
        allocation="feedback",
        eligibility="depth",
        facade_layers=2,
        minimum_patch=4,
        maximum_patches=4,
        maximum_voids=1,
        minimum_void_area=4,
        batched=True,
        vertical_interior=False,
        budget_reserve_fraction=0.2,
    )
    if config.multiscale_schedule != "none":
        learner = CoarseToFineLearner(
            device,
            seed,
            block=(1, 2, 2),
            fine_config=patch,
            input_channels=14 if config.cores_enabled else 12,
            spatial_aux_weight=config.spatial_aux_weight,
            evaluation_temperature=config.evaluation_temperature,
            evaluation_noise_scale=config.evaluation_noise_scale,
            lr=1.2e-3,
            training_objective=config.training_objective,
            decoder_retry_schedule=config.decoder_retry_schedule,
            spatial_aux_mode=config.spatial_aux_mode,
            retry_logp_norm=config.retry_logp_norm,
            continuous_entropy_coef=config.continuous_entropy_coef,
        )
        if config.teacher_checkpoints:
            teachers: dict[str, CoarseToFineNetwork] = {}
            for spec in config.teacher_checkpoints:
                if "=" in spec:
                    tier_key, raw_path = spec.split("=", 1)
                else:
                    tier_key, raw_path = "ALL", spec
                ckpt_path = Path(raw_path)
                if not ckpt_path.is_file():
                    ckpt_path = ROOT / raw_path
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                policy_state = ckpt["policy"] if "policy" in ckpt else ckpt
                in_ch = int(
                    policy_state.get(
                        "inputChannels", 14 if config.cores_enabled else 12
                    )
                )
                net = CoarseToFineNetwork(in_ch).to(device)
                net.load_state_dict(policy_state["model"])
                teachers[tier_key] = net
            learner.set_teachers(teachers)
        return learner
    p = UnifiedLearner(
        device,
        seed,
        patch,
        amount_feedback=True,
        net_guided=True,
        organic_3d=True,
        spatial_aux_weight=config.spatial_aux_weight,
    )
    p.evaluation_temperature = config.evaluation_temperature
    p.evaluation_noise_scale = config.evaluation_noise_scale
    return p


def _reference_eval(env):
    # A common fixed referee makes cores-on/off and spacing ablations
    # comparable without showing extra channels to the acting policy.
    res = solve_cores(env.occupied, min_core_spacing=10, max_core_dist=12)
    referee = SimpleNamespace(**env.__dict__)
    referee.cores_enabled = True
    referee.max_core_dist = 12
    referee.core_mask = res.core_mask
    referee.core_distance = res.core_distance
    referee.core_dist_badness = res.core_dist_badness
    referee.core_metrics = res.metrics
    direct, diffuse = env.daylight_proxies(res.core_mask)
    referee.values = dict(env.values, directSunProxy=direct, diffuseFacadeProxy=diffuse)
    score, diagnostics = organic_objective(referee)
    return dict(
        quality=score, diagnostics=diagnostics, minCoreSpacing=10, maxCoreDist=12
    )


def episode(policy, env, config, training, coarse_envs=()):
    objective = functools.partial(organic_objective, profile=config.reward_profile)
    if isinstance(policy, CoarseToFineLearner):
        raw = run_cascade_episode(
            env,
            coarse_envs,
            policy,
            config.initial_storeys,
            objective,
            training=training,
            rollout_steps=config.rollout_steps,
            audit_steps=config.audit_steps,
            step_reward_fn=organic_step_reward,
            terminal_reward_fn=organic_terminal_reward,
            reward_profile=config.reward_profile,
            potential_shaping_gamma=config.potential_shaping_gamma,
        )
        return dict(
            reference=_reference_eval(env) if not training else None,
            success=raw["success"],
            termination=raw["termination"],
            far=raw["far"],
            farError=raw["farError"],
            steps=raw["steps"],
            attempts=raw["attempts"],
            budget=raw["budget"],
            seconds=raw["seconds"],
            reward=raw["reward"],
            quality=float(raw["rawObjective"]),
            environment={k: float(v.detach().cpu()) for k, v in env.values.items()},
            diagnostics=raw["diagnostics"],
            learning=raw["learning"],
            trace=raw["trace"],
            stageSteps=raw["stageSteps"],
        )
    solid = env.allowed.clone()
    solid[config.initial_storeys :] = False
    env.restore_occupancy(solid)
    env.organic_3d_growth = True
    env.floor_copy_growth = False
    env._ln6_cache = None
    env.unified_limit = max(8, env.workload.limit + 8)
    env.unified_actions = 0
    env.termination = None
    start = time.perf_counter()
    trace = []
    reward_sum = 0.0
    tape_start = len(policy.tape)
    _, diag = objective(env)
    previous_core = float(diag.get("coreScore") or 0.0)
    for _ in range(env.unified_limit):
        move = policy.choose(env, training=training)
        env.unified_actions += 1
        op = policy.last_operation
        if op == "finish":
            env.termination = "far_reached"
            reward = 0.0
        elif move is None:
            reward = -0.01
        else:
            commit(env, move, validated=True)
            delta = 0.0
            if training and config.cores_enabled:
                env.update_cores()
                m = env.core_metrics
                score = max(
                    0.0,
                    (1.0 - float(m["coreUncoveredFraction"]))
                    * (1.0 - 0.5 * float(m["coreDistBadness"])),
                )
                delta = score - previous_core
                previous_core = score
            reward = organic_step_reward(env, delta)
            if env.edits % config.audit_steps == 0:
                env.evaluate()
        if training and policy.last_attempt_recorded:
            policy.record(reward)
        reward_sum += reward
        trace.append(
            dict(
                attempt=env.unified_actions,
                operation=op,
                accepted=move is not None,
                removed=int(move.remove.sum()) if move else 0,
                added=int(move.add.sum()) if move else 0,
                far=env.far,
            )
        )
        if env.terminal:
            break
    if not env.terminal:
        env.termination = (
            "far_reached" if env.far_satisfied else "step_budget_exhausted"
        )
    env.evaluate()
    obj, diag = objective(env)
    terminal = organic_terminal_reward(env, obj)
    reward_sum += terminal
    learning = None
    if training:
        learning = finish_episode_rollout(
            policy, tape_start, terminal, config.rollout_steps
        )
    reference = _reference_eval(env) if not training else None
    sync(env.device)
    return dict(
        reference=reference,
        success=env.termination == "far_reached",
        termination=env.termination,
        far=env.far,
        farError=env.far_error,
        steps=env.edits,
        attempts=env.unified_actions,
        budget=env.unified_limit,
        seconds=time.perf_counter() - start,
        reward=reward_sum,
        quality=float(obj),
        environment={k: float(v.detach().cpu()) for k, v in env.values.items()},
        diagnostics=diag,
        learning=learning,
        trace=trace,
    )


def evaluate(policy, scenes, config, split):
    """Common random eval seeds; no optimizer step or perturbation of training RNG."""
    state = rng_state()
    rows = []
    try:
        for site in getattr(config, split + "_sites"):
            env = scenes.get(site)
            for seed in config.evaluation_seeds:
                seed_everything(seed, config.strict_determinism)
                rows.append(
                    dict(
                        split=split,
                        site=site,
                        evaluationSeed=seed,
                        **episode(
                            policy,
                            env,
                            config,
                            False,
                            coarse_envs=scenes.coarse_envs(site),
                        ),
                    )
                )
    finally:
        restore_rng(state)
    return rows


def spatial_aux_weight_at(config: ExperimentConfig, episode_index: int) -> float:
    if config.spatial_aux_schedule == "constant":
        return config.spatial_aux_weight
    if config.spatial_aux_schedule == "annealed_floor":
        warmup = max(1, config.episodes // 4)
        progress = min(
            1.0,
            max(0.0, (episode_index - warmup + 1) / max(1, config.episodes - warmup)),
        )
        floor = 0.15 * config.spatial_aux_weight
        return floor + (config.spatial_aux_weight - floor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    warmup = max(1, config.episodes // 5)
    progress = max(0.0, (episode_index - warmup + 1) / max(1, config.episodes - warmup))
    return config.spatial_aux_weight * 0.5 * (1.0 + math.cos(math.pi * progress))


def summarize(rows):
    # Include unsuccessful episodes in every reported denominator/mean.
    if not rows:
        return dict(count=0)
    return dict(
        count=len(rows),
        farSuccessRate=statistics.mean(float(r["success"]) for r in rows),
        meanQuality=statistics.mean(r["quality"] for r in rows),
        meanReferenceQuality=statistics.mean(r["reference"]["quality"] for r in rows),
        meanSeconds=statistics.mean(r["seconds"] for r in rows),
        meanUsableGoodArea=statistics.mean(
            r["diagnostics"]["usableGoodArea"] for r in rows
        ),
        meanDirectSun=statistics.mean(r["environment"]["directSunProxy"] for r in rows),
        meanDiffuseSky=statistics.mean(
            r["environment"]["diffuseFacadeProxy"] for r in rows
        ),
    )


def per_site_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_site: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_site.setdefault(row["site"], []).append(row)
    out = {}
    for site, srows in by_site.items():
        summary = summarize(srows)
        best = max(srows, key=lambda r: (bool(r["success"]), r["quality"]))
        summary["bestSeed"] = best["evaluationSeed"]
        summary["bestQuality"] = best["quality"]
        summary["bestDirectSun"] = best["environment"]["directSunProxy"]
        summary["bestDiffuseSky"] = best["environment"]["diffuseFacadeProxy"]
        out[site] = summary
    return out


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("validation", "test", "finalValidation", "finalTest"):
        rows = [s[split] for s in summaries if split in s]
        if not rows:
            continue
        stats = {}
        for key in (
            "farSuccessRate",
            "meanQuality",
            "meanReferenceQuality",
            "meanUsableGoodArea",
            "meanDirectSun",
            "meanDiffuseSky",
        ):
            vals = [r[key] for r in rows]
            mean_val = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            ci = None
            t_ci = None
            if sd is not None and len(vals) >= 2:
                half = (
                    float(scipy_stats.t.ppf(0.975, df=len(vals) - 1))
                    * sd
                    / math.sqrt(len(vals))
                )
                t_ci = [mean_val - half, mean_val + half]
            if len(vals) >= 3:
                rng = np.random.default_rng(2026)
                boots = rng.choice(vals, size=(5000, len(vals))).mean(1)
                ci = np.quantile(boots, [0.025, 0.975]).tolist()
            stats[key] = dict(
                mean=mean_val,
                stdAcrossModelSeeds=sd,
                tCi95AcrossModelSeeds=t_ci,
                bootstrap95AcrossModelSeeds=ci,
            )
        out[split] = dict(modelSeeds=len(rows), metrics=stats)
    eps = [
        s["bestValidationEpisode"] for s in summaries if "bestValidationEpisode" in s
    ]
    if eps:
        out["bestValidationEpisodes"] = eps
    return out


def _check_existing_run(out, config):
    manifest = json.loads((out / "manifest.json").read_text())
    recorded = ExperimentConfig.read(out / "config.json")
    if (
        recorded.fingerprint != config.fingerprint
        or manifest["configHash"] != config.fingerprint
    ):
        raise ValueError(
            "The run configuration and model-seed roster are fixed at launch"
        )
    if manifest["source"] != source_manifest():
        raise ValueError("Source changed; start a new experiment directory")
    return manifest


def train(args):
    config = ExperimentConfig.read(args.config)
    torch.set_num_threads(config.threads)
    out = Path(args.output)
    only_seed = getattr(args, "only_seed", None)
    finalize_only = bool(getattr(args, "finalize_only", False))

    if args.resume and (only_seed is not None or finalize_only):
        raise ValueError("Resume one checkpoint into a new output directory")

    if finalize_only:
        _check_existing_run(out, config)
        summaries = []
        for s in config.seeds:
            sp = out / f"seed-{s}" / "summary.json"
            if not sp.is_file():
                raise FileNotFoundError(f"Missing seed summary: {sp}")
            result = json.loads(sp.read_text())
            if (
                result["modelSeed"] != s
                or result["completedEpisodes"] != config.episodes
            ):
                raise ValueError(f"Seed {s} has not completed the configured budget")
            summaries.append(result)
        payload = dict(
            seeds=summaries,
            aggregate=aggregate(summaries),
            note="All failures retained. Confidence intervals resample model seeds, not correlated episodes.",
        )
        write_json(out / "summary.json", payload)
        print(json.dumps(payload), flush=True)
        return

    if only_seed is not None:
        if only_seed not in config.seeds:
            raise ValueError(
                f"--only-seed {only_seed} is not in config.seeds={config.seeds}"
            )
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=False)

    seed_everything(config.seeds[0], config.strict_determinism)
    manifest_path = out / "manifest.json"
    if only_seed is not None and (
        manifest_path.exists() or (out / "config.json").exists()
    ):
        manifest = _check_existing_run(out, config)
    else:
        manifest = runtime_manifest()
        manifest["configHash"] = config.fingerprint
        manifest["resumeFrom"] = (
            str(Path(args.resume).resolve()) if args.resume else None
        )
        write_json(out / "config.json", config.to_dict())
        write_json(manifest_path, manifest)
        archive_source(out / "source.tar.gz", manifest)

    scenes = Scenes(config)
    scenes.prepare()
    scene_file = out / "scenes.json"
    if scene_file.is_file():
        if json.loads(scene_file.read_text()) != scenes.manifest:
            raise ValueError("Scene content changed; start a new experiment directory")
    else:
        write_json(scene_file, scenes.manifest)
        write_json(out / "setup.json", scenes.setup)

    saved = read_checkpoint(args.resume, config) if args.resume else None
    if saved and saved["scenes"] != scenes.manifest:
        raise ValueError("Scene content changed since checkpoint")
    seeds_to_run = (
        [saved["seed"]]
        if saved
        else ([only_seed] if only_seed is not None else config.seeds)
    )
    summaries = []
    val_summary_keys = (
        "count",
        "farSuccessRate",
        "meanQuality",
        "meanReferenceQuality",
        "meanSeconds",
        "meanUsableGoodArea",
        "meanDirectSun",
        "meanDiffuseSky",
    )

    quality_key = (
        "meanQuality"
        if config.reward_profile in ("default", "dense_potential")
        else "meanReferenceQuality"
    )

    def _val_rank(vsum):
        return (round(float(vsum["farSuccessRate"]), 6), float(vsum[quality_key]))

    for seed in seeds_to_run:
        seed_everything(seed, config.strict_determinism)
        policy = make_policy(config, scenes.device, seed)
        folder = out / f"seed-{seed}"
        folder.mkdir()
        start = 0
        best_val_key = None
        best_val_episode = 0
        best_val_rows = None
        best_val_summary = None
        val_history = []
        if saved:
            restore_training_checkpoint(policy, saved)
            start = int(saved["nextEpisode"])
            resume_dir = Path(args.resume).parent
            hist_path = resume_dir / "validation-history.jsonl"
            if hist_path.is_file():
                val_history = [
                    json.loads(line)
                    for line in hist_path.read_text().splitlines()
                    if line.strip() and int(json.loads(line)["episode"]) <= start
                ]
                if val_history:
                    best_row = max(
                        val_history,
                        key=lambda r: (*_val_rank(r), -int(r["episode"])),
                    )
                    best_val_key = _val_rank(best_row)
                    best_val_episode = int(best_row["episode"])
                    best_val_summary = {
                        k: best_row[k] for k in val_summary_keys if k in best_row
                    }
                    checkpoint = resume_dir / f"checkpoint-{best_val_episode:06d}.pt"
                    if not checkpoint.is_file():
                        checkpoint = resume_dir / "best.pt"
                    if (
                        read_checkpoint(checkpoint, config)["nextEpisode"]
                        != best_val_episode
                    ):
                        raise ValueError(
                            "Best checkpoint does not match recorded validation history"
                        )
                    shutil.copy2(checkpoint, folder / "best.pt")
        else:
            baseline = evaluate(policy, scenes, config, "validation")
            write_json(folder / "untrained-validation.json", baseline)
        t = time.perf_counter()
        completed = start
        final_val_rows = None
        final_val_summary = None
        with (
            (folder / "train.jsonl").open("x") as ledger,
            (folder / "validation-history.jsonl").open("x") as val_ledger,
        ):
            for prev_val in val_history:
                val_ledger.write(json.dumps(prev_val, allow_nan=False) + "\n")
            if val_history:
                val_ledger.flush()
            base_lr = float(
                getattr(
                    policy,
                    "lr",
                    1.2e-3 if isinstance(policy, CoarseToFineLearner) else 1e-3,
                )
            )
            min_lr = min(2e-4, base_lr)
            for i in range(start, config.episodes):
                if config.spatial_aux_schedule != "constant":
                    policy.spatial_aux_weight = spatial_aux_weight_at(config, i)
                if config.lr_schedule == "cosine":
                    frac = min(1.0, i / max(1, config.episodes - 1))
                    lr = min_lr + 0.5 * (base_lr - min_lr) * (
                        1.0 + math.cos(math.pi * frac)
                    )
                    for g in policy.optimizer.param_groups:
                        g["lr"] = lr
                site = config.train_sites[i % len(config.train_sites)]
                row = dict(
                    episode=i + 1,
                    modelSeed=seed,
                    site=site,
                    spatialAuxWeight=policy.spatial_aux_weight,
                    **episode(
                        policy,
                        scenes.get(site),
                        config,
                        True,
                        coarse_envs=scenes.coarse_envs(site),
                    ),
                )
                ledger.write(json.dumps(row, allow_nan=False) + "\n")
                ledger.flush()
                completed = i + 1
                stopping = (
                    args.stop_after is not None and completed - start >= args.stop_after
                )
                at_budget = completed == config.episodes
                # A shorter last batch is part of the configured training budget,
                # independent of periodic snapshots, validation and interruptions.
                if at_budget and policy.tape:
                    policy.finish(0.0)
                validation_due = (
                    completed % config.checkpoint_interval == 0 or at_budget
                )
                if validation_due or stopping:
                    if source_manifest() != manifest["source"]:
                        raise RuntimeError(
                            "Source changed during training; preserve this run and start a new experiment"
                        )
                    ckpt_path = folder / f"checkpoint-{completed:06d}.pt"
                    save_checkpoint(
                        ckpt_path,
                        policy,
                        config,
                        seed,
                        completed,
                        scenes.manifest,
                    )
                if validation_due:
                    val_rows = evaluate(policy, scenes, config, "validation")
                    val_sum = summarize(val_rows)
                    val_entry = dict(
                        episode=completed,
                        updates=policy.updates,
                        **val_sum,
                    )
                    val_history.append(val_entry)
                    val_ledger.write(json.dumps(val_entry, allow_nan=False) + "\n")
                    val_ledger.flush()
                    if completed == config.episodes:
                        final_val_rows = val_rows
                        final_val_summary = val_sum
                    vk = _val_rank(val_sum)
                    if best_val_key is None or vk > best_val_key:
                        best_val_key = vk
                        best_val_episode = completed
                        best_val_rows = val_rows
                        best_val_summary = val_sum
                        shutil.copy2(ckpt_path, folder / "best.pt")
                        write_json(folder / "validation.json", val_rows)
                if validation_due or stopping:
                    shutil.copy2(ckpt_path, folder / "latest.pt")
                if stopping:
                    break
        summary = dict(
            modelSeed=seed,
            completedEpisodes=completed,
            trainingSeconds=time.perf_counter() - t,
            updates=policy.updates,
        )
        if val_history:
            summary["bestValidationEpisode"] = best_val_episode
            summary["validationHistory"] = val_history
        if completed == config.episodes:
            if final_val_rows is None:
                final_val_rows = evaluate(policy, scenes, config, "validation")
                final_val_summary = summarize(final_val_rows)
            write_json(folder / "validation-final.json", final_val_rows)
            summary["finalValidation"] = final_val_summary
            if config.test_sites:
                random_policy = make_policy(config, scenes.device, seed)
                baseline = evaluate(random_policy, scenes, config, "test")
                write_json(folder / "untrained-test.json", baseline)
                summary["untrainedTest"] = summarize(baseline)
                test_final_rows = evaluate(policy, scenes, config, "test")
                write_json(folder / "test-final.json", test_final_rows)
                summary["finalTest"] = summarize(test_final_rows)

            if best_val_episode == completed:
                best_val_rows = final_val_rows
            else:
                best_ckpt = read_checkpoint(folder / "best.pt", config)
                policy.load(best_ckpt["policy"])
                if best_val_rows is None:
                    best_val_rows = evaluate(policy, scenes, config, "validation")
                    best_val_summary = summarize(best_val_rows)
            write_json(folder / "validation.json", best_val_rows)
            summary["validation"] = best_val_summary
            if config.test_sites:
                test_best_rows = (
                    test_final_rows
                    if best_val_episode == completed
                    else evaluate(policy, scenes, config, "test")
                )
                write_json(folder / "test.json", test_best_rows)
                summary["test"] = summarize(test_best_rows)
        summaries.append(summary)
        write_json(folder / "summary.json", summary)
        print(json.dumps(summary), flush=True)

    if (
        only_seed is None
        and tuple(seeds_to_run) == config.seeds
        and all(row["completedEpisodes"] == config.episodes for row in summaries)
    ):
        write_json(
            out / "summary.json",
            dict(
                seeds=summaries,
                aggregate=aggregate(summaries),
                note="All failures retained. Confidence intervals resample model seeds, not correlated episodes. Small runs are pipeline checks, not convergence evidence.",
            ),
        )


def evaluate_heuristic(args):
    config = ExperimentConfig.read(args.config)
    torch.set_num_threads(config.threads)
    seed_everything(config.seeds[0], config.strict_determinism)
    manifest = runtime_manifest()
    manifest["configHash"] = config.fingerprint
    scenes = Scenes(config)
    scenes.prepare()
    policy = CoarseToFineLearner(
        scenes.device,
        seed=config.seeds[0],
        block=(1, 2, 2),
        fine_config=scenes.patch,
        input_channels=14 if config.cores_enabled else 12,
        spatial_aux_weight=0.0,
        evaluation_temperature=config.evaluation_temperature,
        evaluation_noise_scale=0.0,
        heuristic_only=True,
        decoder_retry_schedule=config.decoder_retry_schedule,
    )
    path = Path(args.output)
    if path.exists():
        raise ValueError("Refuse to overwrite existing evidence")
    val_rows = evaluate(policy, scenes, config, "validation")
    test_rows = evaluate(policy, scenes, config, "test")
    payload = dict(
        baseline="greedy_physics_heuristic",
        config=config.to_dict(),
        runtime=manifest,
        validation=dict(
            summary=summarize(val_rows),
            perSite=per_site_summary(val_rows),
            episodes=val_rows,
        ),
        test=dict(
            summary=summarize(test_rows),
            perSite=per_site_summary(test_rows),
            episodes=test_rows,
        ),
        note="Non-learned greedy baseline ranking frontier edits with compute_spatial_desirability. The first proposal is deterministic; rejected proposals can trigger Gaussian retries. Uses the same cascade stages and feasibility projector as the learned policy.",
    )
    write_json(path, payload)
    print(path.resolve())


def evaluate_checkpoint(args):
    saved = read_checkpoint(args.checkpoint)
    config = ExperimentConfig(
        **{
            k: v
            for k, v in saved["config"].items()
            if k in {f.name for f in fields(ExperimentConfig)}
        }
    )
    torch.set_num_threads(config.threads)
    seed_everything(saved["seed"], config.strict_determinism)
    scenes = Scenes(config)
    scenes.prepare()
    if saved["scenes"] != scenes.manifest:
        raise ValueError("Evaluation scene content changed")
    policy = make_policy(config, scenes.device, saved["seed"])
    policy.load(saved["policy"])
    path = Path(args.output)
    if path.exists():
        raise ValueError("Refuse to overwrite existing evidence")
    rows = evaluate(policy, scenes, config, args.split)
    write_json(
        path,
        dict(
            summary=summarize(rows),
            episodes=rows,
            checkpointSHA256=hashlib.sha256(
                Path(args.checkpoint).read_bytes()
            ).hexdigest(),
        ),
    )
    print(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--output", required=True)
    t.add_argument(
        "--resume",
        help="Resume one model seed into a NEW run directory at an episode boundary",
    )
    t.add_argument(
        "--stop-after",
        type=int,
        help="Cleanly stop after N additional episodes, saving a resumable checkpoint",
    )
    t.add_argument(
        "--only-seed",
        type=int,
        help="Train/evaluate a single seed from config.seeds inside --output",
    )
    t.add_argument(
        "--finalize-only",
        action="store_true",
        help="Aggregate completed per-seed summaries inside --output into summary.json",
    )
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", choices=["validation", "test"], default="test")
    e.add_argument("--output", required=True)
    h = sub.add_parser("heuristic")
    h.add_argument("--config", required=True)
    h.add_argument("--output", required=True)
    args = parser.parse_args()
    if getattr(args, "stop_after", None) is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if args.command == "train":
        train(args)
    elif args.command == "heuristic":
        evaluate_heuristic(args)
    else:
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
