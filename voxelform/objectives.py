"""Shared live/headless task objective. These are proxies, not measured outcomes."""

from .cores import DEFAULT_MAX_CORE_DIST
from .unified import (
    _allowed_long_narrow,
    clean_floor_tensor,
    long_narrow_mask,
    min_room_width,
)


def organic_objective(env, *, profile="default"):
    """Scalar research objective over uncalibrated environment proxies."""
    count = max(1.0, float(env.occupied.sum().detach().cpu()))
    mw = min_room_width(env)
    usable_mask = env.usable
    if env.plot_area >= 400:
        allowed_ln6 = _allowed_long_narrow(env, 6, 2)
        ln6 = long_narrow_mask(env.occupied, 6, 2) & ~allowed_ln6
        ln6_frac = float(ln6.sum().detach().cpu()) / count
        usable_mask = usable_mask & ~ln6
        slim_voxels = int(
            (
                (env.occupied & ~clean_floor_tensor(env.occupied, mw))
                & clean_floor_tensor(env.allowed, mw)
            )
            .sum()
            .detach()
            .cpu()
        )
    else:
        ln6_frac = 0.0
        slim_voxels = 0
    core_enabled = bool(
        getattr(env, "cores_enabled", False) and hasattr(env, "core_mask")
    )
    core_m = getattr(env, "core_metrics", None) or {}
    if core_enabled:
        core_ok = (~env.core_mask) & (
            env.core_distance
            <= float(getattr(env, "max_core_dist", DEFAULT_MAX_CORE_DIST))
        )
        usable_good = float((usable_mask & core_ok).sum().detach().cpu()) / count
        core_uncov = float(core_m.get("coreUncoveredFraction", 0.0))
        core_bad = float(core_m.get("coreDistBadness", 0.0))
        core_coverage = max(0.0, 1.0 - core_uncov)
        core_score = max(0.0, core_coverage * (1.0 - 0.5 * core_bad))
        core_score_term = 0.20 * core_score
    else:
        usable_good = float(usable_mask.sum().detach().cpu()) / count
        core_uncov = 0.0
        core_bad = 0.0
        core_coverage = None
        core_score = None
        core_score_term = 0.0
    cantilever_voxels = int(
        (env.occupied[1:] & ~env.occupied[:-1]).sum().detach().cpu()
    )
    sun = float(env.values["directSunProxy"].detach().cpu())
    diff = float(env.values["diffuseFacadeProxy"].detach().cpu())
    roof = float(env.values["roofOpportunity"].detach().cpu())
    deep = float(env.depth_badness.sum().detach().cpu()) / count
    noise = (
        float(env.values["noisePrior"].detach().cpu()) if env.context is None else 0.0
    )
    obj = (
        0.65 * usable_good
        + 0.55 * sun
        + 0.45 * diff
        + 0.10 * roof
        + core_score_term
        - 0.20 * deep
        - 0.55 * ln6_frac
        - 0.08 * noise
        - 0.45 * core_uncov
        - 0.35 * core_bad
    )
    if profile == "daylight_low":
        obj -= 0.25 * (0.55 * sun + 0.45 * diff)
    elif profile == "coverage_only" and core_enabled:
        obj += 0.20 - 0.20 * core_uncov + 0.35 * core_bad - core_score_term
    diag = dict(
        organicObjective=obj,
        usableGoodFraction=usable_good,
        usableGoodArea=usable_good * count,
        depthBadFraction=deep,
        longNarrowFraction=ln6_frac,
        cantileverVoxels=cantilever_voxels,
        slimVoxels=slim_voxels,
        coresEnabled=core_enabled,
        coreCoverage=core_coverage,
        coreScore=core_score,
        coreCount=int(core_m.get("coreCount", 0)) if core_enabled else 0,
        coreRatio=float(core_m.get("coreRatio", 0.0)) if core_enabled else 0.0,
        coreDistanceMean=float(core_m.get("coreDistanceMean", 0.0))
        if core_enabled
        else 0.0,
        coreDistanceMax=float(core_m.get("coreDistanceMax", 0.0))
        if core_enabled
        else 0.0,
        coreUncoveredFraction=core_uncov,
        coreUnreachableFraction=float(core_m.get("coreUnreachableFraction", 0.0))
        if core_enabled
        else 0.0,
        coreDistBadness=core_bad,
        coreMinSpacingActual=float(core_m.get("coreMinSpacingActual", 0.0))
        if core_enabled
        else 0.0,
    )
    return obj, diag


def organic_step_reward(env, core_delta=0.0):
    return (-0.005 if env.far_satisfied else 0.0) + 0.50 * core_delta


def coarse_potential(ce) -> float:
    """Fast multi-objective potential on a CoarseEnv after evaluate_fast()."""
    count = max(1.0, float(ce.occupied.sum().detach().cpu()))
    core_enabled = bool(
        getattr(ce, "cores_enabled", False) and hasattr(ce, "core_mask")
    )
    if core_enabled:
        core_ok = (~ce.core_mask) & (
            ce.core_distance
            <= float(getattr(ce, "max_core_dist", DEFAULT_MAX_CORE_DIST))
        )
        usable_good = float((ce.usable & core_ok).sum().detach().cpu()) / count
        core_term = 0.20 * float(getattr(ce, "core_score", 0.0))
    else:
        usable_good = float(ce.usable.sum().detach().cpu()) / count
        core_term = 0.0
    occ_f = ce.occupied.float()
    sun_proxy = float((ce.external_potential * occ_f).sum().detach().cpu()) / count
    diff_proxy = float((ce.facade_light * occ_f).sum().detach().cpu()) / count
    deep = float(ce.depth_badness.sum().detach().cpu()) / count
    far_prog = (
        0.0
        if ce.near_far
        else -0.35
        * abs(ce.count_fine - ce.goal_count_fine)
        / max(1.0, abs(ce.start_count_fine - ce.goal_count_fine))
    )
    return (
        0.65 * usable_good
        + 0.55 * sun_proxy
        + 0.45 * diff_proxy
        + core_term
        - 0.20 * deep
        + far_prog
    )


def fine_potential(
    env, objective_value: float, start_count: int | None = None
) -> float:
    """Potential combining fine organic_objective and normalized FAR progress."""
    s_cnt = env.start_count if start_count is None else start_count
    far_prog = (
        0.0
        if env.far_satisfied
        else -0.35
        * abs(env.count - env.goal_count)
        / max(1.0, abs(s_cnt - env.goal_count))
    )
    return float(objective_value) + far_prog


def organic_terminal_reward(env, objective):
    penalty = (
        0.0
        if env.far_satisfied
        else -(
            0.35
            + 0.85
            * abs(env.count - env.goal_count)
            / max(1, abs(env.start_count - env.goal_count))
        )
    )
    return 5.0 * (objective + penalty)
