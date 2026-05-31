"""
environment.py - Estimated environmental and economic impact of adaptive signal timing.

IMPORTANT: All values are estimates for comparative analysis only.
They are NOT exact field measurements.

Assumptions used:
  - Idle fuel consumption: 0.6 L/hour per vehicle = 0.000166 L/second
  - CO2 emitted per liter of petrol burned: ~2300 g/L
  - Petrol price in Pakistan (2024 average): ~270 PKR/L

Methodology:
  For each lane, if adaptive green time > static green time, vehicles in
  that lane experience less idle waiting. We estimate how much fuel (and
  therefore CO2 and money) is saved by that reduction in idle time.
"""

# Physical / economic constants
IDLE_FUEL_PER_SECOND = 0.000166   # liters per second per vehicle (0.6 L/hr ÷ 3600)
CO2_PER_LITER_PETROL = 2300       # grams of CO2 per liter of petrol burned
PETROL_PRICE_PKR = 270            # PKR per liter (update as needed)


def estimate_environmental_impact(lanes: list, static_green: int = 30) -> dict:
    """
    Estimate fuel, CO2, and cost savings from adaptive vs fixed signal timing.

    Logic:
        For each lane where adaptive_green > static_green,
        vehicles saved (adaptive - static) seconds of idle engine time.
        We multiply by vehicle count and the per-second idle fuel rate.

    Args:
        lanes:        list of lane dicts with 'total_vehicles' and 'green_time'
        static_green: baseline fixed green time per lane (seconds)

    Returns:
        dict with:
            estimated_wait_reduction_sec  - total vehicle-seconds of wait saved
            fuel_saved_liters             - estimated fuel saved
            co2_saved_g                   - estimated CO2 not emitted (grams)
            pkr_saved                     - estimated money saved (PKR)
            total_vehicles_analyzed       - total vehicles across all lanes
            note                          - disclaimer string
    """
    total_wait_reduction = 0.0
    total_vehicles = 0

    for lane in lanes:
        vehicles = lane.get("total_vehicles", 0)
        adaptive = lane.get("green_time", static_green)

        # Positive difference means vehicles waited less with adaptive timing
        wait_reduction_per_vehicle = max(0, adaptive - static_green)
        total_wait_reduction += wait_reduction_per_vehicle * vehicles
        total_vehicles += vehicles

    fuel_saved = total_wait_reduction * IDLE_FUEL_PER_SECOND
    co2_saved = fuel_saved * CO2_PER_LITER_PETROL
    pkr_saved = fuel_saved * PETROL_PRICE_PKR

    return {
        "estimated_wait_reduction_sec": round(total_wait_reduction, 1),
        "fuel_saved_liters": round(fuel_saved, 4),
        "co2_saved_g": round(co2_saved, 1),
        "pkr_saved": round(pkr_saved, 2),
        "total_vehicles_analyzed": total_vehicles,
        "note": (
            "Estimates based on 0.6 L/hr idle consumption, 2300 g CO2/L, "
            "270 PKR/L. For comparative analysis only — not exact field measurements."
        ),
    }
