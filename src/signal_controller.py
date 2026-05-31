"""
signal_controller.py - Adaptive green time allocation and emergency priority logic.

Design:
- allocate_green_times()       proportional PCU-based allocation
- apply_emergency_priority()   override if EMV is detected
- static_vs_adaptive_comparison()  show difference vs fixed timer
"""

from src.pcu_logic import calculate_pcu_score

# Green time (seconds) given unconditionally to a lane with an emergency vehicle
EMERGENCY_PRIORITY_GREEN = 60

# Minimum green time any lane receives (prevents starvation)
DEFAULT_MIN_GREEN = 15

# Maximum green time any single lane can get per cycle
DEFAULT_MAX_GREEN = 60


def allocate_green_times(
    lanes: list,
    total_cycle_time: int = 120,
    min_green: int = DEFAULT_MIN_GREEN,
    max_green: int = DEFAULT_MAX_GREEN,
) -> list:
    """
    Proportionally allocate green signal time based on each lane's PCU score.

    Higher PCU score = more traffic = more green time.
    Each lane is clamped between min_green and max_green to prevent extremes.

    Args:
        lanes:            list of lane result dicts (must contain 'pcu_score')
        total_cycle_time: total signal cycle length in seconds (default 120s)
        min_green:        minimum green time any lane gets (seconds)
        max_green:        maximum green time any lane can get (seconds)

    Returns:
        List of lane dicts with 'green_time' and 'allocation_reason' added.
    """
    lanes = [dict(lane) for lane in lanes]  # work on copies

    pcu_scores = [lane.get("pcu_score", 0) for lane in lanes]
    total_pcu = sum(pcu_scores)

    if total_pcu == 0:
        # No traffic detected anywhere - split equally
        equal_time = total_cycle_time // len(lanes)
        for lane in lanes:
            lane["green_time"] = max(min_green, min(max_green, equal_time))
            lane["allocation_reason"] = "equal split (no traffic detected)"
        return lanes

    for lane in lanes:
        proportion = lane.get("pcu_score", 0) / total_pcu
        raw_time = proportion * total_cycle_time
        clamped = max(min_green, min(max_green, int(raw_time)))
        lane["green_time"] = clamped
        lane["allocation_reason"] = f"proportional PCU ({lane.get('pcu_score', 0):.1f} PCU)"

    return lanes


def apply_emergency_priority(lanes: list) -> list:
    """
    Override green time allocation if an emergency vehicle (EMV) is detected.

    The lane with the EMV gets EMERGENCY_PRIORITY_GREEN seconds.
    All other lanes get the minimum green time so the intersection clears fast.

    Args:
        lanes: list of lane dicts (must have 'detected_emergency', 'pcu_score')

    Returns:
        List of lane dicts with updated 'green_time' and 'priority_reason'.
    """
    lanes = [dict(lane) for lane in lanes]

    emv_detected = any(lane.get("detected_emergency", False) for lane in lanes)

    if not emv_detected:
        # No EMV - clear the priority_reason field and return unchanged
        for lane in lanes:
            lane["priority_reason"] = ""
        return lanes

    for lane in lanes:
        if lane.get("detected_emergency", False):
            lane["green_time"] = EMERGENCY_PRIORITY_GREEN
            lane["priority_reason"] = "EMV DETECTED - emergency priority override"
        else:
            lane["green_time"] = DEFAULT_MIN_GREEN
            lane["priority_reason"] = f"reduced to {DEFAULT_MIN_GREEN}s while EMV clears"

    return lanes


def static_vs_adaptive_comparison(lanes: list, static_green: int = 30) -> list:
    """
    Annotate each lane with static vs adaptive green time comparison.

    Args:
        lanes:        list of lane dicts with 'green_time' already set
        static_green: fixed green time in a traditional signal (seconds)

    Returns:
        List of lane dicts with 'static_green' and 'green_time_diff' added.
        Positive diff means adaptive gives MORE green time than fixed.
    """
    lanes = [dict(lane) for lane in lanes]

    for lane in lanes:
        adaptive = lane.get("green_time", static_green)
        lane["static_green"] = static_green
        lane["green_time_diff"] = adaptive - static_green

    return lanes
