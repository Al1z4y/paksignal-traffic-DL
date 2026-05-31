"""
pcu_logic.py - Passenger Car Unit (PCU) weight definitions and scoring.

PCU is a standard traffic engineering concept that converts mixed vehicle
types into a common unit (equivalent passenger cars) so different vehicle
sizes can be compared on an equal footing.

References: IRC:106-1990 (Indian/Pakistani highway capacity guidelines)
"""

# Standard PCU weights for Pakistani urban traffic
PCU_WEIGHTS = {
    "bike":     0.5,   # motorcycles take half the road space of a car
    "car":      1.0,   # baseline unit
    "rickshaw": 1.5,   # three-wheelers are wider and slower than cars
    "htv":      3.0,   # heavy transport vehicles (buses, trucks, coasters)
    "emv":      2.0,   # emergency vehicles (handled separately for priority)
}

# Fallback weight for any class not in the map
DEFAULT_PCU_WEIGHT = 1.0


def calculate_pcu_score(counts: dict) -> float:
    """
    Calculate the total PCU-weighted traffic load for a lane.

    Args:
        counts: dict mapping class name -> vehicle count
                e.g. {"bike": 5, "car": 3, "htv": 1}

    Returns:
        Total PCU score as a float.
        e.g. 5*0.5 + 3*1.0 + 1*3.0 = 8.5
    """
    total = 0.0
    for cls, count in counts.items():
        weight = PCU_WEIGHTS.get(cls, DEFAULT_PCU_WEIGHT)
        total += weight * count
    return round(total, 2)


def get_pcu_weight(class_name: str) -> float:
    """Return the PCU weight for a single vehicle class."""
    return PCU_WEIGHTS.get(class_name, DEFAULT_PCU_WEIGHT)
