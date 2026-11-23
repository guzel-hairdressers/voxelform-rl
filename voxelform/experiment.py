"""Validated experiment configuration and reproducibility artifacts.

No server, renderer or external dataset is required for synthetic experiments.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .core_plans import CORE_LAYOUT_VERSION

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "voxelform.research.v2"


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (123, 456, 789)
    train_sites: tuple[str, ...] = ("synthetic:rectangle",)
    validation_sites: tuple[str, ...] = ("synthetic:courtyard",)
    test_sites: tuple[str, ...] = ("synthetic:branch",)
    evaluation_seeds: tuple[int, ...] = (9001, 9002, 9003)
    episodes: int = 100
    initial_storeys: int = 4
    ceiling_storeys: int = 8
    synthetic_width: int = 24
    synthetic_length: int = 24
    target_far: float = 3.0
    far_tolerance: float = 0.1
    device: str = "cpu"
    threads: int = 1
    cores_enabled: bool = True
    min_core_spacing: int = 10
    max_core_dist: int = 12
    spatial_aux_weight: float = 0.0
    rollout_steps: int = 16
    audit_steps: int = 5
    checkpoint_interval: int = 10
    strict_determinism: bool = False
    evaluation_temperature: float = 1.0
    evaluation_noise_scale: float = 1.0
    multiscale_schedule: str = "none"
    lr_schedule: str = "constant"
    training_objective: str = "actor_critic"
    spatial_aux_schedule: str = "constant"
    decoder_retry_schedule: str = "legacy"
    reward_profile: str = "default"
    teacher_checkpoints: tuple[str, ...] = ()
    spatial_aux_mode: str = "reinforce"
    retry_logp_norm: str = "sum"
    potential_shaping_gamma: float | None = None
    continuous_entropy_coef: float = 0.0

    def __post_init__(self):
        for key in (
            "seeds",
            "evaluation_seeds",
            "train_sites",
            "validation_sites",
            "test_sites",
        ):
            v = getattr(self, key)
            if not isinstance(v, (list, tuple)) or (not v and key != "test_sites"):
                raise ValueError(f"{key} must be a nonempty sequence")
            object.__setattr__(self, key, tuple(v))
            if len(set(v)) != len(v):
                raise ValueError(f"Duplicate values in {key}")
        for key in (
            "episodes",
            "initial_storeys",
            "ceiling_storeys",
            "synthetic_width",
            "synthetic_length",
            "threads",
            "max_core_dist",
            "rollout_steps",
            "audit_steps",
            "checkpoint_interval",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.initial_storeys > self.ceiling_storeys or self.ceiling_storeys > 60:
            raise ValueError("Require 1 <= initial_storeys <= ceiling_storeys <= 60")
        if self.synthetic_width < 8 or self.synthetic_length < 8:
            raise ValueError("Synthetic extents must be at least eight cells")
        if type(self.min_core_spacing) is not int or self.min_core_spacing < 0:
            raise ValueError("Invalid core spacing")
        for k in ("cores_enabled", "strict_determinism"):
            if type(getattr(self, k)) is not bool:
                raise ValueError(f"{k} must be boolean")
        for key in ("seeds", "evaluation_seeds"):
            if any(
                type(s) is not int or s < 0 or s >= 2**31 for s in getattr(self, key)
            ):
                raise ValueError("Seeds must be integers in [0, 2**31)")
        if set(self.seeds) & set(self.evaluation_seeds):
            raise ValueError("Training and evaluation seed sets overlap")
        splits = [
            set(self.train_sites),
            set(self.validation_sites),
            set(self.test_sites),
        ]
        if any(a & b for i, a in enumerate(splits) for b in splits[i + 1 :]):
            raise ValueError("Train/validation/test sites overlap")
        for group in splits:
            for site in group:
                if not isinstance(site, str) or not site:
                    raise ValueError("Invalid site ID")
                if site.startswith("synthetic:") and site.split(":", 1)[1] not in (
                    "rectangle",
                    "triangle",
                    "courtyard",
                    "branch",
                ):
                    raise ValueError("Unknown synthetic geometry")
        for key in (
            "target_far",
            "far_tolerance",
            "spatial_aux_weight",
            "evaluation_temperature",
            "evaluation_noise_scale",
        ):
            x = getattr(self, key)
            if type(x) not in (int, float) or not np.isfinite(x) or x < 0:
                raise ValueError(f"Invalid {key}")
        if (
            self.target_far <= 0
            or self.far_tolerance >= self.target_far
            or self.target_far + self.far_tolerance > self.ceiling_storeys
        ):
            raise ValueError("Invalid FAR range")
        if self.evaluation_temperature < 0:
            raise ValueError("Evaluation temperature must be non-negative")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Unknown compute device")
        if self.multiscale_schedule not in (
            "none",
            "tier_adaptive",
            "tier_meso2",
            "tier_macro4",
            "tier_adaptive_xl8",
        ):
            raise ValueError("Unknown multiscale schedule")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("Unknown learning-rate schedule")
        if self.training_objective not in (
            "actor_critic",
            "imitation",
            "on_policy_distill",
        ):
            raise ValueError("Unknown training objective")
        if self.spatial_aux_schedule not in (
            "constant",
            "annealed",
            "annealed_floor",
        ):
            raise ValueError("Unknown spatial auxiliary schedule")
        if self.decoder_retry_schedule not in ("legacy", "adaptive", "learned_cap"):
            raise ValueError("Unknown decoder retry schedule")
        if self.reward_profile not in (
            "default",
            "coverage_only",
            "daylight_low",
            "dense_potential",
        ):
            raise ValueError("Unknown reward profile")
        tc = self.teacher_checkpoints
        if not isinstance(tc, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in tc
        ):
            raise ValueError(
                "teacher_checkpoints must be a sequence of nonempty strings"
            )
        object.__setattr__(self, "teacher_checkpoints", tuple(tc))
        if self.spatial_aux_mode not in ("reinforce", "regression"):
            raise ValueError("Unknown spatial auxiliary mode")
        if self.retry_logp_norm not in ("sum", "mean"):
            raise ValueError("Unknown retry log-probability normalization")
        if self.potential_shaping_gamma is not None:
            if (
                type(self.potential_shaping_gamma) not in (int, float)
                or not np.isfinite(self.potential_shaping_gamma)
                or not (0.0 < float(self.potential_shaping_gamma) <= 1.0)
            ):
                raise ValueError("Invalid potential_shaping_gamma")
            object.__setattr__(
                self, "potential_shaping_gamma", float(self.potential_shaping_gamma)
            )
        if (
            type(self.continuous_entropy_coef) not in (int, float)
            or not np.isfinite(self.continuous_entropy_coef)
            or float(self.continuous_entropy_coef) < 0.0
        ):
            raise ValueError("Invalid continuous_entropy_coef")
        object.__setattr__(
            self, "continuous_entropy_coef", float(self.continuous_entropy_coef)
        )
        if self.training_objective == "imitation" and self.spatial_aux_weight:
            raise ValueError("Imitation uses its own supervised loss")
        if self.multiscale_schedule == "none" and (
            self.training_objective in ("imitation", "on_policy_distill")
            or self.decoder_retry_schedule != "legacy"
            or self.spatial_aux_schedule != "constant"
            or self.spatial_aux_mode != "reinforce"
            or self.retry_logp_norm != "sum"
            or self.potential_shaping_gamma is not None
            or self.continuous_entropy_coef != 0.0
            or bool(self.teacher_checkpoints)
        ):
            raise ValueError(
                "Imitation, distillation, adaptive retries, shaping and annealing require multiscale"
            )

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Configuration must be an object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        d = asdict(self)
        if not self.teacher_checkpoints:
            d.pop("teacher_checkpoints", None)
        if self.spatial_aux_mode == "reinforce":
            d.pop("spatial_aux_mode", None)
        if self.retry_logp_norm == "sum":
            d.pop("retry_logp_norm", None)
        if self.potential_shaping_gamma is None:
            d.pop("potential_shaping_gamma", None)
        if self.continuous_entropy_coef == 0.0:
            d.pop("continuous_entropy_coef", None)
        return d

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()


def write_json(path, data):
    """Durable atomic write; refuse nonfinite metrics rather than hiding failures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def seed_everything(seed, strict=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(strict)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = strict


def rng_state():
    n = np.random.get_state()
    state = dict(
        python=random.getstate(),
        numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
        torch=torch.get_rng_state(),
    )
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def source_manifest(root: Path = ROOT) -> dict[str, str]:
    paths: list[Path] = []
    for folder in ("voxelform", "tools", "configs"):
        paths.extend(
            p for p in (root / folder).rglob("*") if p.suffix in (".py", ".json")
        )
    paths.extend(
        p
        for p in (root / "pyproject.toml", root / "uv.lock", root / "unified_server.py")
        if p.is_file()
    )
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if "__pycache__" not in p.parts
    }


def runtime_manifest():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    packages: dict[str, str | None] = {}
    for pkg in ("torch", "numpy", "scipy", "shapely", "fastapi", "uvicorn"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = None
    return dict(
        protocol=PROTOCOL,
        createdAt=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        packages=packages,
        gitCommit=git("rev-parse", "HEAD"),
        gitDirty=bool(git("status", "--porcelain")),
        source=source_manifest(),
        coreLayoutVersion=CORE_LAYOUT_VERSION,
        cuda=torch.cuda.is_available(),
        mps=torch.backends.mps.is_available(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )


def archive_source(path, manifest):
    with tarfile.open(path, "w:gz") as tar:
        for name in manifest["source"]:
            tar.add(ROOT / name, arcname=name)


def _move_rollout(value, device):
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("Research rollouts must store detached tensors")
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_rollout(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_rollout(item, device) for item in value)
    if isinstance(value, list):
        return [_move_rollout(item, device) for item in value]
    if value is None or type(value) in (int, float, bool, str):
        return value
    raise TypeError(f"Unsupported rollout value: {type(value).__name__}")


def _validate_episode_boundary(tape):
    if tape and (
        not all(isinstance(entry, dict) and "done" in entry for entry in tape)
        or tape[-1]["done"] is not True
    ):
        raise ValueError(
            "Save pending rollout transitions at a completed episode boundary"
        )


def restore_training_checkpoint(policy, saved):
    """Restore optimizer and completed-episode transitions without taking an update."""
    tape = saved["pendingRollout"]
    _validate_episode_boundary(tape)
    policy.load(saved["policy"])
    policy.tape = _move_rollout(tape, policy.device)
    restore_rng(saved["rng"])


def save_checkpoint(path, policy, config, seed, next_episode, scene_manifest):
    _validate_episode_boundary(policy.tape)
    saved = dict(
        format=PROTOCOL,
        coreLayoutVersion=CORE_LAYOUT_VERSION,
        config=config.to_dict(),
        configHash=config.fingerprint,
        seed=seed,
        nextEpisode=next_episode,
        policy=policy.state(),
        pendingRollout=_move_rollout(policy.tape, "cpu"),
        rng=rng_state(),
        scenes=scene_manifest,
        source=source_manifest(),
    )
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(saved, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_checkpoint(path, config=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != PROTOCOL:
        raise ValueError(
            "This runner requires a research v2 checkpoint. Use the recorded source revision for earlier checkpoints."
        )
    if saved["coreLayoutVersion"] != CORE_LAYOUT_VERSION:
        raise ValueError(
            "Core library changed; this is a transfer experiment, not an exact resume"
        )
    if config is not None and saved["configHash"] != config.fingerprint:
        raise ValueError("Resume configuration differs")
    if saved["source"] != source_manifest():
        raise ValueError("Source changed; refuse to label this an exact resume")
    return saved
