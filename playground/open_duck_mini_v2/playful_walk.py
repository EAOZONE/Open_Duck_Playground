"""Shared cadence helpers for the playful high-clearance walking policy."""

SKIP_COMMAND_INDEX = 6
SKIP_CUE_MAGNITUDE = 0.5
SKIP_EVERY_GAIT_CYCLES = 3
NORMAL_FOOT_HEIGHT = 0.045
SKIP_FOOT_HEIGHT = 0.065


def skip_side_for_cycle(cycle, xp, cadence: int = SKIP_EVERY_GAIT_CYCLES):
    """Returns -1 normally, then alternating left/right accent-foot indices."""
    active = xp.mod(cycle + 1, cadence) == 0
    side = xp.mod(xp.floor_divide(cycle, cadence), 2)
    return xp.where(active, side, -1)


def skip_cue_for_cycle(cycle, xp, cadence: int = SKIP_EVERY_GAIT_CYCLES):
    """Encodes no accent as 0, left as +0.5, and right as -0.5."""
    side = skip_side_for_cycle(cycle, xp, cadence)
    return xp.where(
        side == 0,
        SKIP_CUE_MAGNITUDE,
        xp.where(side == 1, -SKIP_CUE_MAGNITUDE, 0.0),
    )


def desired_foot_heights(
    cycle,
    xp,
    normal_height: float = NORMAL_FOOT_HEIGHT,
    skip_height: float = SKIP_FOOT_HEIGHT,
    cadence: int = SKIP_EVERY_GAIT_CYCLES,
):
    """Returns per-foot swing-height targets for the current gait cycle."""
    side = skip_side_for_cycle(cycle, xp, cadence)
    return xp.where(
        xp.arange(2) == side,
        skip_height,
        normal_height,
    )
