"""Shared episode boundaries and generalized advantage estimation."""

from collections.abc import Sequence, Sized
from typing import Any, Protocol, TypedDict

import torch


class AdvantageTransition(TypedDict):
    value: torch.Tensor
    reward: torch.Tensor | float | None
    done: bool


class EpisodeLearner(Protocol):
    @property
    def tape(self) -> Sized: ...

    def add_reward(self, reward: float) -> None: ...

    def mark_done(self) -> None: ...

    def finish(self, bootstrap: float = 0.0) -> dict[str, Any]: ...


def finish_episode_rollout(
    learner: EpisodeLearner,
    tape_start: int,
    terminal_reward: float,
    rollout_steps: int,
) -> dict[str, Any] | None:
    """Close the current episode if it contributed transitions to the batch."""
    if len(learner.tape) <= tape_start:
        return None
    learner.add_reward(terminal_reward)
    learner.mark_done()
    if len(learner.tape) >= rollout_steps:
        return learner.finish(0.0)
    return None


def generalized_advantages(
    tape: Sequence[AdvantageTransition],
    bootstrap: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack([entry["value"] for entry in tape])
    next_value = torch.as_tensor(bootstrap, dtype=torch.float32, device=device).detach()
    gae = torch.zeros((), dtype=torch.float32, device=device)
    zeros = torch.zeros_like(gae)
    reversed_advantages = []
    for index in range(len(tape) - 1, -1, -1):
        entry = tape[index]
        reward = entry["reward"]
        reward_tensor = (
            zeros
            if reward is None
            else torch.as_tensor(reward, dtype=torch.float32, device=device).detach()
        )
        mask = 0.0 if entry["done"] else 1.0
        delta = reward_tensor + 0.99 * next_value * mask - values[index]
        gae = delta + (0.99 * 0.95) * mask * gae
        reversed_advantages.append(gae)
        next_value = values[index]
    advantage = torch.stack(list(reversed(reversed_advantages)))
    returns = advantage + values
    if len(advantage) > 1:
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-5)
    return advantage, returns
