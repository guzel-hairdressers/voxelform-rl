"""Run the frozen decoder, supervision, and reward comparisons for the L/XL study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from tools.research import Scenes, evaluate, make_policy, summarize
from voxelform.decoder_diagnostics import DecoderDiagnostics
from voxelform.experiment import ExperimentConfig, runtime_manifest, write_json

SCREEN_SEEDS = (4111, 4112)
SCREEN_EVALUATION_SEEDS = (9101, 9102, 9103)
SCREEN_EPISODES = 30
ARMS = {
    "fixed": {},
    "annealed": {"spatial_aux_schedule": "annealed"},
    "scalar": {"spatial_aux_weight": 0.0},
    "imitation": {"spatial_aux_weight": 0.0, "training_objective": "imitation"},
    "coverage-only": {"reward_profile": "coverage_only"},
    "daylight-low": {"reward_profile": "daylight_low"},
}
RL_ARMS = ("fixed", "annealed", "scalar")
DECODER_MIN_RELATIVE_REDUCTION = 0.10
DECODER_MAX_QUALITY_LOSS = 0.02


def read_json(path):
    return json.loads(path.read_text())


def decoder_screen(config, path, output):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if set(config.validation_sites) & checkpoint["scenes"].keys():
        raise ValueError(
            "Decoder probe requires validation plots new to the checkpoint"
        )
    torch.set_num_threads(config.threads)
    scenes = Scenes(config)
    results = {}
    for schedule in ("legacy", "adaptive"):
        trial = replace(config, decoder_retry_schedule=schedule)
        policy = make_policy(trial, scenes.device, checkpoint["seed"])
        policy.load(checkpoint["policy"])
        policy.decoder_diagnostics = DecoderDiagnostics()
        rows = evaluate(policy, scenes, trial, "validation")
        attempts = policy.decoder_diagnostics.attempts
        rejected = sum(a.accepted_digest is None for a in attempts)
        metrics = summarize(rows)
        metrics.update(
            draws=len(attempts),
            rejectedDraws=rejected,
            rejectionsPerCommittedEdit=rejected / max(1, sum(r["steps"] for r in rows)),
            meanProjectionMs=1000
            * statistics.mean(a.projection_seconds for a in attempts),
            meanProposalMs=1000 * statistics.mean(a.proposal_seconds for a in attempts),
        )
        results[schedule] = metrics
        write_json(
            output / f"decoder-{schedule}.json", {"episodes": rows, "metrics": metrics}
        )
    baseline, adaptive = results["legacy"], results["adaptive"]
    improved = (
        baseline["rejectionsPerCommittedEdit"] > 0
        and adaptive["rejectionsPerCommittedEdit"]
        <= (1 - DECODER_MIN_RELATIVE_REDUCTION) * baseline["rejectionsPerCommittedEdit"]
        and adaptive["farSuccessRate"] >= baseline["farSuccessRate"]
        and adaptive["meanReferenceQuality"]
        >= baseline["meanReferenceQuality"] - DECODER_MAX_QUALITY_LOSS
    )
    return {"selected": "adaptive" if improved else "legacy", "metrics": results}


def run_research(command, config, output, name):
    config_path = output / f"{name}.json"
    write_json(config_path, config.to_dict())
    destination = (
        output / name if command == "train" else output / f"{name}-result.json"
    )
    with (output / f"{name}.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.research",
                command,
                "--config",
                str(config_path),
                "--output",
                str(destination),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return read_json(
        destination / "summary.json" if command == "train" else destination
    )


def comparison_rows(output, name, config, split):
    return [
        read_json(output / name / f"seed-{seed}" / f"{split}.json")
        for seed in config.seeds
    ]


def site_metrics(seeds):
    return {
        site: {
            "pooled": summarize(
                [r for rows in seeds for r in rows if r["site"] == site]
            ),
            "meanEdits": statistics.mean(
                r["steps"] for rows in seeds for r in rows if r["site"] == site
            ),
            "meanAttempts": statistics.mean(
                r["attempts"] for rows in seeds for r in rows if r["site"] == site
            ),
            "meanCoreCoverage": statistics.mean(
                r["diagnostics"]["coreCoverage"]
                for rows in seeds
                for r in rows
                if r["site"] == site
            ),
            "meanDepthBadFraction": statistics.mean(
                r["diagnostics"]["depthBadFraction"]
                for rows in seeds
                for r in rows
                if r["site"] == site
            ),
            "referenceStdAcrossModelSeeds": (
                statistics.stdev(
                    statistics.mean(
                        r["reference"]["quality"] for r in rows if r["site"] == site
                    )
                    for rows in seeds
                )
                if len(seeds) > 1
                else None
            ),
        }
        for site in sorted({r["site"] for rows in seeds for r in rows})
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = ExperimentConfig.read(args.config)
    if base.multiscale_schedule != "tier_adaptive" or not base.test_sites:
        parser.error("The benchmark requires a tier-adaptive config with a test split")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    screen = replace(
        base,
        seeds=SCREEN_SEEDS,
        test_sites=(),
        evaluation_seeds=SCREEN_EVALUATION_SEEDS,
        episodes=SCREEN_EPISODES,
        checkpoint_interval=SCREEN_EPISODES,
    )
    if set(screen.seeds + screen.evaluation_seeds) & set(
        base.seeds + base.evaluation_seeds
    ):
        parser.error("Use independent screen and confirmation seeds")
    write_json(
        output / "plan.json",
        {
            "runtime": runtime_manifest(),
            "contextRoot": os.getenv("VOXELFORM_CONTEXT_ROOT"),
            "checkpointSHA256": hashlib.sha256(
                args.decoder_checkpoint.read_bytes()
            ).hexdigest(),
            "screen": screen.to_dict(),
            "screenArms": ARMS,
            "confirmation": base.to_dict(),
            "decoderGate": {
                "minimumRelativeRejectionReduction": DECODER_MIN_RELATIVE_REDUCTION,
                "maximumReferenceQualityLoss": DECODER_MAX_QUALITY_LOSS,
                "FARGuard": "validation FAR success must equal or exceed legacy",
            },
            "selection": "Validation FAR success, then common reference quality; ties favor fixed, annealed, scalar in that order. Confirm this arm and imitation at the frozen final episode budget.",
            "reporting": "Show all model seeds, all attempts, per-site and pooled reference quality and FAR success. Greedy uses the same decoder, site split and evaluation seeds.",
        },
    )
    decoder = decoder_screen(screen, args.decoder_checkpoint, output)
    write_json(output / "decoder-screen.json", decoder)
    print("Decoder:", decoder["selected"], flush=True)
    screen = replace(screen, decoder_retry_schedule=decoder["selected"])
    screening = {}
    for arm, overrides in ARMS.items():
        trial = replace(screen, **overrides)
        name = f"screen-{arm}"
        result = run_research("train", trial, output, name)
        screening[arm] = result | {
            "perSite": site_metrics(comparison_rows(output, name, trial, "validation"))
        }
        write_json(output / "screening.json", screening)
        scores = result["aggregate"]["validation"]["metrics"]
        print(
            arm,
            scores["meanReferenceQuality"]["mean"],
            scores["farSuccessRate"]["mean"],
            flush=True,
        )

    def rank(arm):
        scores = screening[arm]["aggregate"]["validation"]["metrics"]
        return (
            scores["farSuccessRate"]["mean"],
            scores["meanReferenceQuality"]["mean"],
        )

    promoted = max(RL_ARMS, key=rank)
    final = replace(base, decoder_retry_schedule=decoder["selected"])
    learner_configs = {
        arm: replace(final, **ARMS[arm]) for arm in (promoted, "imitation")
    }
    write_json(
        output / "frozen-selection.json",
        {
            "promoted": promoted,
            "learners": {k: v.to_dict() for k, v in learner_configs.items()},
            "greedy": final.to_dict(),
        },
    )
    print("Confirmation promoted:", promoted, flush=True)
    final_results = {}
    for arm, trial in learner_configs.items():
        name = f"confirm-{arm}"
        result = run_research("train", trial, output, name)
        final_results[arm] = result | {
            "perSiteTest": site_metrics(comparison_rows(output, name, trial, "test"))
        }
        write_json(output / "confirmation.json", final_results)
        print("Confirmed", arm, flush=True)
    greedy = run_research("heuristic", final, output, "greedy")
    final_results["greedy"] = {
        split: {
            "summary": greedy[split]["summary"],
            "perSite": site_metrics([greedy[split]["episodes"]]),
        }
        for split in ("validation", "test")
    }
    write_json(output / "confirmation.json", final_results)
    print("Finished:", output, flush=True)


if __name__ == "__main__":
    main()
