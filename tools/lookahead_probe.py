"""Compare one- and two-edit decisions in a controlled equal-FAR planning case."""

from __future__ import annotations

import argparse
import hashlib
from functools import partial
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from tools.research import make_policy
from voxelform.coarse_to_fine import CoarseToFineLearner
from voxelform.environment import Environment
from voxelform.experiment import (
    ExperimentConfig,
    runtime_manifest,
    seed_everything,
    write_json,
)
from voxelform.objectives import organic_objective
from voxelform.patch_policy import PatchConfig, PatchFacades
from voxelform.unified import commit, project_organic_3d

SHAPE = (8, 16, 22)
SCENE_SEED = 9951
POLICY_SEEDS = (9301, 9302, 9303, 9304, 9305)
ROOFS = ((3, 7, 2, 6), (9, 13, 2, 6), (3, 7, 15, 19), (9, 13, 15, 19))
TRANSFERS = tuple(permutations(range(len(ROOFS)), 2))


def fixture(cores_enabled: bool = False):
    patch = PatchConfig(
        mode="combined",
        allocation="feedback",
        eligibility="depth",
        facade_layers=2,
        minimum_patch=4,
        maximum_patches=4,
        vertical_interior=False,
        batched=True,
    )
    env = Environment(
        SHAPE,
        seed=SCENE_SEED,
        device="cpu",
        sun_count=8,
        cores_enabled=cores_enabled,
        frontier_factory=partial(PatchFacades, config=patch, depth_device="cpu"),
        fast_self_shadows=True,
        fast_diffuse=True,
        target_far=3.0,
        far_tolerance=0.1,
    )
    env.precompute_external()
    env.organic_3d_growth = True
    start = np.zeros(SHAPE, dtype=bool)
    start[:3] = True
    start[2, :2, :8] = False
    start[2, :8, 20:22] = False
    start[3:5, 3:7, 2:6] = True
    env.restore_occupancy(start)
    if env.cores_enabled:
        env.update_cores()
    return env, patch, start


def measure(env):
    return dict(
        quality=organic_objective(env)[0],
        far=env.far,
        storeys=int(np.flatnonzero(env.frontier.occupied.any((1, 2)))[-1]) + 1,
        occupied=env.count,
    )


def replay_edit(env, occupancy, action):
    remove_roof, grow_roof = action
    ya, yb, xa, xb = ROOFS[remove_roof]
    a, b, c, d = ROOFS[grow_roof]
    top_remove = int(np.flatnonzero(occupancy[:, ya, xa])[-1])
    top_add = int(np.flatnonzero(occupancy[:, a, c])[-1]) + 1
    if top_remove < 3:
        return None
    masks = np.zeros((2, *SHAPE), dtype=bool)
    masks[0, top_remove, ya:yb, xa:xb] = True
    masks[1, top_add, a:b, c:d] = True
    if (masks[0] & ~occupancy).any():
        return None
    env.restore_occupancy(occupancy)
    if env.cores_enabled:
        env.update_cores()
    move = project_organic_3d(env, masks, masks.astype(np.float32), "exchange", True)
    if move is None or not (
        np.array_equal(move.remove, masks[0]) and np.array_equal(move.add, masks[1])
    ):
        return None
    commit(env, move, audit=True, validated=True)
    if env.cores_enabled:
        env.update_cores()
    if env.count != int(occupancy.sum()):
        raise RuntimeError("The controlled case must preserve exact volume")
    return env.frontier.occupied.copy(), measure(env)


def planning_comparison(env, start):
    def _choose_terminal(row):
        return row["final"]["quality"]

    env.restore_occupancy(start)
    if env.cores_enabled:
        env.update_cores()
    try:
        initial = measure(env)
        routes = []
        first_moves = {}
        for first in TRANSFERS:
            step = replay_edit(env, start, first)
            if step is None:
                continue
            occupancy, first_metrics = step
            first_moves[first] = first_metrics
            for second in TRANSFERS:
                step = replay_edit(env, occupancy, second)
                if step is not None:
                    routes.append(
                        dict(
                            actions=(first, second),
                            intermediate=first_metrics,
                            final=step[1],
                        )
                    )
        # Give the one-step oracle its best continuation and allow it to stop early.
        choose_first = max(
            first_moves, key=lambda action: first_moves[action]["quality"]
        )
        myopic = max(
            (r for r in routes if r["actions"][0] == choose_first), key=_choose_terminal
        )
        lookahead = max(routes, key=_choose_terminal)
        one_step_best = max(
            initial["quality"],
            myopic["intermediate"]["quality"],
            myopic["final"]["quality"],
        )
        if not (
            lookahead["intermediate"]["quality"] < initial["quality"]
            and lookahead["final"]["quality"] > one_step_best
        ):
            raise RuntimeError(
                "Expected the planned path to trade intermediate for terminal quality"
            )
        return dict(
            initial=initial,
            validFirstEdits=len(first_moves),
            validTwoEditRoutes=len(routes),
            oneStepOracle=myopic,
            oneStepBestWithStop=one_step_best,
            twoStepOracle=lookahead,
        )
    finally:
        env.restore_occupancy(start)
        if env.cores_enabled:
            env.update_cores()


def policy_rollout(env, start, policy, seed):
    seed_everything(seed, strict=False)
    env.restore_occupancy(start)
    if env.cores_enabled:
        env.update_cores()
    env.termination = None
    env.unified_actions = 0
    env.unified_limit = max(8, env.workload.limit + 8)
    rows = []
    for _ in range(2):
        move = policy.choose_stage(env, is_coarse=False, training=False)
        env.unified_actions += 1
        operation = policy.last_operation
        if move is not None:
            commit(env, move, audit=True, validated=True)
            if env.cores_enabled:
                env.update_cores()
            count = int(move.remove.sum() + move.add.sum())
            env.workload.observe(count, count)
        rows.append(
            dict(operation=operation, accepted=move is not None, **measure(env))
        )
        if operation == "finish":
            break
    return dict(evaluationSeed=seed, decisions=rows, committedEdits=env.edits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="Finished research_suite output")
    parser.add_argument(
        "--cores",
        action="store_true",
        help="Evaluate policies with cores enabled (in-distribution for 14-channel models)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Give a new output path")
    torch.set_num_threads(1)
    oracle_env, patch, start = fixture(cores_enabled=False)
    eval_env, _, _ = (
        fixture(cores_enabled=True) if args.cores else (oracle_env, patch, start)
    )
    payload = dict(
        sceneSeed=SCENE_SEED,
        runtime=runtime_manifest(),
        coresEnabledForPolicyProbe=bool(args.cores),
        fixture=(
            "Exploratory synthetic witness chosen in development. "
            + (
                "Policy probe runs with cores enabled (14 channels in-distribution) while the scripted oracle runs without cores."
                if args.cores
                else "Cores are disabled to isolate mass redistribution."
            )
        ),
        oracleProtocol="Transfer 16-cell roof patches among four roof positions. The score lookahead compares up to two exchanges, all checked by the actual geometry projector at identical FAR.",
        oracle=planning_comparison(oracle_env, start),
    )
    if args.study is not None:
        greedy_cfg_path = args.study / "greedy.json"
        if not greedy_cfg_path.is_file():
            first_cfg = sorted(args.study.glob("confirm-*/config.json"))
            if not first_cfg:
                parser.error(
                    f"Missing greedy.json or confirm-*/config.json in {args.study}"
                )
            greedy_cfg_path = first_cfg[0]
        cfg = ExperimentConfig.read(greedy_cfg_path)
        greedy = CoarseToFineLearner(
            "cpu",
            cfg.seeds[0],
            fine_config=patch,
            input_channels=14,
            heuristic_only=True,
            evaluation_noise_scale=0.0,
            decoder_retry_schedule=cfg.decoder_retry_schedule,
        )
        payload["policyProtocol"] = (
            "Free masks on the same initial state, up to two decisions. Finish is legal. No action template is supplied."
        )
        payload["greedy"] = [
            policy_rollout(eval_env, start, greedy, seed) for seed in POLICY_SEEDS
        ]
        learned = {}
        for arm in sorted(args.study.glob("confirm-*")):
            if not arm.is_dir():
                continue
            cfg = ExperimentConfig.read(arm / "config.json")
            learned[arm.name] = []
            for model_seed in cfg.seeds:
                for ckpt_name in ("best.pt", "latest.pt"):
                    path = arm / f"seed-{model_seed}/{ckpt_name}"
                    if not path.is_file():
                        continue
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    policy = make_policy(cfg, "cpu", model_seed)
                    policy.load(saved["policy"])
                    learned[arm.name].append(
                        dict(
                            modelSeed=model_seed,
                            checkpoint=ckpt_name,
                            checkpointSHA256=hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest(),
                            episodes=[
                                policy_rollout(eval_env, start, policy, seed)
                                for seed in POLICY_SEEDS
                            ],
                        )
                    )
        payload["learned"] = learned
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
