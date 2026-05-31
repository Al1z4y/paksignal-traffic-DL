"""
utils.py - Shared helper utilities for PakSignal.

Handles class name normalization so the system works even if
the Roboflow dataset uses slightly different names across versions.
"""

import os
import yaml


# Maps raw Roboflow class names -> standard PakSignal names
CLASS_NORMALIZATION_MAP = {
    # bikes / motorcycles
    "bike": "bike",
    "motorcycle": "bike",
    "motorbike": "bike",
    "motor bike": "bike",
    # cars
    "car": "car",
    # rickshaws (common spelling variations in Pakistani datasets)
    "rickshaw": "rickshaw",
    "rikshaw": "rickshaw",
    "ricksha": "rickshaw",
    "auto": "rickshaw",
    "autorickshaw": "rickshaw",
    # heavy transport vehicles
    "htv": "htv",
    "bus": "htv",
    "truck": "htv",
    "minibus": "htv",
    "coaster": "htv",
    "van": "htv",
    # emergency vehicles
    "emv": "emv",
    "emergency": "emv",
    "ambulance": "emv",
    "police": "emv",
    "fire truck": "emv",
    "firetruck": "emv",
}


def normalize_class_name(name: str) -> str:
    """Normalize a raw dataset class name to a standard PakSignal class name."""
    cleaned = name.lower().strip()
    return CLASS_NORMALIZATION_MAP.get(cleaned, cleaned)


def load_class_names_from_yaml(yaml_path: str) -> list:
    """
    Load and normalize class names from a YOLOv8 data.yaml file.

    Returns:
        List of normalized class name strings.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_names = cfg.get("names", [])
    return [normalize_class_name(n) for n in raw_names]


def check_model_exists(model_path: str) -> bool:
    """Return True if the model weights file exists."""
    return os.path.isfile(model_path)


def check_dataset_structure(dataset_root: str) -> dict:
    """
    Verify that a YOLOv8 dataset has the expected folder structure.

    Returns a dict with each expected path and whether it exists.
    """
    expected = {
        "train/images": os.path.join(dataset_root, "train", "images"),
        "train/labels": os.path.join(dataset_root, "train", "labels"),
        "valid/images": os.path.join(dataset_root, "valid", "images"),
        "valid/labels": os.path.join(dataset_root, "valid", "labels"),
        "test/images":  os.path.join(dataset_root, "test",  "images"),
        "test/labels":  os.path.join(dataset_root, "test",  "labels"),
        "data.yaml":    os.path.join(dataset_root, "data.yaml"),
    }
    results = {}
    for key, path in expected.items():
        exists = os.path.exists(path)
        count = len(os.listdir(path)) if exists and os.path.isdir(path) else None
        results[key] = {"path": path, "exists": exists, "file_count": count}
    return results
