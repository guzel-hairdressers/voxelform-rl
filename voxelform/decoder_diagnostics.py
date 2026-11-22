"""Opt-in measurements at the cascade decoder's feasibility boundary."""

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .redistribution import Exchange


def mask_digest(*masks: NDArray[np.bool_]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for mask in masks:
        digest.update(np.packbits(mask))
    return digest.hexdigest()


@dataclass(frozen=True)
class DecoderAttempt:
    decision: int
    scale: int
    operation: str
    retry: int
    requested_voxels: int
    proposed_voxels: int
    kept_voxels: int
    added_by_projection: int
    proposal_digest: str
    accepted_digest: str | None
    proposal_seconds: float
    projection_seconds: float


class DecoderDiagnostics:
    def __init__(self) -> None:
        self.attempts: list[DecoderAttempt] = []

    def record(
        self,
        *,
        scale: int,
        operation: str,
        retry: int,
        requested: int,
        masks: NDArray[np.bool_],
        proposed: int,
        move: Exchange | None,
        proposal_seconds: float,
        projection_seconds: float,
    ) -> None:
        previous = self.attempts[-1].decision if self.attempts else 0
        decision = previous + int(retry == 0)
        kept = added = 0
        accepted_digest = None
        if move is not None:
            kept = int((move.remove & masks[0]).sum() + (move.add & masks[1]).sum())
            added = int(move.remove.sum() + move.add.sum()) - kept
            accepted_digest = mask_digest(move.remove, move.add)
        self.attempts.append(
            DecoderAttempt(
                decision,
                scale,
                operation,
                retry,
                requested,
                proposed,
                kept,
                added,
                mask_digest(masks[0], masks[1]),
                accepted_digest,
                proposal_seconds,
                projection_seconds,
            )
        )
