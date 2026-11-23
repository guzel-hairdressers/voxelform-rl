"""Headless training reward experiment; production session reward is unchanged."""

import torch
from torch import nn

from .patch_policy import PatchConfig, PatchLearner
from .workload import action_reward


def step_reward(env, before, mode="current"):
    if mode == "current":
        return action_reward(env, before)
    if mode != "quality":
        raise ValueError("Unknown reward mode")
    initial = max(1, env.workload.required)
    remaining = abs(env.count - env.goal_count)
    progress = (before * env.plot_area - remaining) / initial
    reward = 0.1 * progress - 0.01 / max(1, env.workload.limit)
    if env.termination == "far_reached":
        reward += 0.1
    elif env.terminal:
        reward -= 2.0 * remaining / initial
    return reward


def audit_reward(before, after, mode="current"):
    if mode not in ("current", "quality"):
        raise ValueError("Unknown reward mode")
    return (0.1 if mode == "current" else 5.0) * (after - before)


def depth_potential(env):
    # Already-computed exact same-floor field. Fixed denominator avoids changing
    # the reward scale as volume changes. No extra environmental simulation.
    return -float(env.depth_badness.sum().detach().cpu()) / env.capacity


def easy_probability(mode, fraction):
    if mode == "mixed":
        return 0.5
    if mode == "curriculum":
        return 0.8 - 0.6 * max(0.0, min(1.0, fraction))
    raise ValueError("Unknown training schedule")


class EnvironmentLearner(PatchLearner):
    """Experimental informed actor; zero new weights preserve initialization.

    Adds depth, current edit eligibility, fixed direct-sun exposure and a
    separate diffuse facade-sky channel sampled from neighbouring air. Direct
    sun never substitutes for diffuse light. Audits retain their current cadence.
    Separate checkpoint format prevents accidental loading into the live actor.
    """

    def __init__(self, device="cpu", seed=123, config=None, include_facade_sky=True):
        super().__init__(device, seed, config or PatchConfig(allocation="feedback"))
        self.include_facade_sky = bool(include_facade_sky)
        if self.config.allocation != "feedback":
            raise ValueError("Informed pilot requires adaptive edit amounts")
        old = self.model.encoder[0]
        assert isinstance(old, nn.Conv2d) and old.bias is not None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expanded = nn.Conv2d(
                9 if self.include_facade_sky else 8, 16, 3, padding=1
            ).to(device)
        assert expanded.bias is not None
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :5].copy_(old.weight)
            expanded.bias.copy_(old.bias)
        self.model.encoder[0] = expanded
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def informed_observation(self, env):
        if not hasattr(env, "external_potential"):
            env.precompute_external()
            normals = env.suns.new_tensor(
                ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            )
            weights = (env.suns @ normals.T).clamp_min(0) * env.suns.new_tensor(
                (3.0, 3.0, 3.0, 3.0, 1.0, 1.0)
            )[None]
            visible = env.static_receivers.values.reshape(len(env.suns), 6, *env.shape)
            env.external_potential = (
                visible.float() * weights[:, :, None, None, None]
            ).sum((0, 1)) / weights.sum().clamp_min(1)
        operation = -1 if env.count > env.goal_count else 1
        eligible = torch.as_tensor(
            env.frontier.eligible(operation), dtype=torch.float32, device=self.device
        )
        channels: tuple[torch.Tensor, ...] = (
            env.observation(include_depth=True),
            eligible[:, None],
            env.external_potential[:, None],
        )
        if self.include_facade_sky:
            channels = (*channels, env.facade_light[:, None])
        return torch.cat(channels, dim=1)

    def fields(self, env):
        self._env = env
        return self.model(
            self.informed_observation(env),
            torch.tensor(env.goal_features, dtype=torch.float32, device=self.device),
        )

    def state(self):
        state = super().state()
        state["format"] = (
            "voxelform.environment_patch.v2"
            if self.include_facade_sky
            else "voxelform.environment_patch.v1"
        )
        return state

    def load(self, state):
        if state.get("format") not in (
            "voxelform.environment_patch.v1",
            "voxelform.environment_patch.v2",
        ):
            raise ValueError("Expected an experimental environment-input checkpoint")
        replacement = EnvironmentLearner(
            str(self.device),
            config=PatchConfig(**state["config"]),
            include_facade_sky=state["format"].endswith("v2"),
        )
        replacement.model.load_state_dict(state["model"])
        replacement.optimizer.load_state_dict(state["optimizer"])
        for key, source in [
            ("updates", "updates"),
            ("direct_updates", "directMaskUpdates"),
            ("imported_updates", "importedUpdates"),
        ]:
            setattr(replacement, key, int(state[source]))
        replacement.migration = state.get("migration")
        self.__dict__.update(replacement.__dict__)
