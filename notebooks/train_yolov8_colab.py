# =============================================================================
# PakSignal - YOLOv8 Training Script
# Compatible with Google Colab (paste each # %% section into a new cell)
# Dataset: density-traffic-controller-v1 v8 (Roboflow)
# Classes: bike, car, emv, htv, rickshaw
# =============================================================================

# %% ── Cell 1: Install dependencies ─────────────────────────────────────────
# Run FIRST. Restart runtime if Colab asks you to after installation.

# pip install -q installs without verbose output
# ultralytics includes YOLOv8 + all required torch dependencies

get_ipython().system('pip install ultralytics -q')
get_ipython().system('pip install roboflow -q')
get_ipython().system('pip install opencv-python-headless -q')  # headless for Colab

print("✅ Dependencies installed.")
print("   If Colab shows a 'Restart Runtime' button, click it before continuing.")

# %% ── Cell 2: Check GPU and imports ─────────────────────────────────────────

import os
import shutil
import glob
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from IPython.display import Image as IPImage, display

print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = "0"   # YOLO device string for GPU 0
else:
    print("⚠️  No GPU found. Go to: Runtime → Change runtime type → T4 GPU")
    DEVICE = "cpu"

# %% ── Cell 3: Mount Google Drive ────────────────────────────────────────────
# Mount Drive so you can save best.pt there permanently.
# Skip this cell if you don't want to use Drive.

from google.colab import drive
drive.mount("/content/drive")
print("✅ Google Drive mounted at /content/drive")

# %% ── Cell 4: Download dataset from Roboflow ────────────────────────────────
#
# Option A (recommended): Roboflow API - fills in your key below
# Option B: If you already downloaded the zip manually, see the comment below
#
# To get your API key:
#   1. Go to roboflow.com and log in
#   2. Click your profile picture → Settings → Roboflow API
#   3. Copy the Private API Key

from roboflow import Roboflow

RF_API_KEY = "YOUR_ROBOFLOW_API_KEY"    # <── paste your key here

rf = Roboflow(api_key=RF_API_KEY)
project = rf.workspace("fypszabist-i9gqf").project("density-traffic-controller-v1")
dataset = project.version(8).download("yolov8")

DATASET_PATH = dataset.location
print(f"✅ Dataset downloaded to: {DATASET_PATH}")

# ── Option B: manual upload ──────────────────────────────────────────────────
# If you downloaded the zip from Roboflow manually and uploaded it to Colab:
#
#   from google.colab import files
#   uploaded = files.upload()               # pick your dataset.zip
#   !unzip -q dataset.zip -d /content/dataset
#   DATASET_PATH = "/content/dataset"

# %% ── Cell 5: Verify dataset structure ──────────────────────────────────────

# Update DATASET_PATH if Cell 4 set it differently
# DATASET_PATH = "/content/density-traffic-controller-v1-8"  # uncomment if needed

YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")

print("=" * 55)
print("DATASET STRUCTURE CHECK")
print("=" * 55)

required_dirs = [
    ("train/images", os.path.join(DATASET_PATH, "train", "images")),
    ("train/labels", os.path.join(DATASET_PATH, "train", "labels")),
    ("valid/images", os.path.join(DATASET_PATH, "valid", "images")),
    ("valid/labels", os.path.join(DATASET_PATH, "valid", "labels")),
    ("test/images",  os.path.join(DATASET_PATH, "test",  "images")),
    ("test/labels",  os.path.join(DATASET_PATH, "test",  "labels")),
]

for label, path in required_dirs:
    if os.path.isdir(path):
        n = len(os.listdir(path))
        print(f"  ✅ {label:<18} ({n} files)")
    else:
        print(f"  ❌ MISSING: {label}")

if os.path.isfile(YAML_PATH):
    with open(YAML_PATH) as f:
        data_cfg = yaml.safe_load(f)
    nc    = data_cfg.get("nc", "?")
    names = data_cfg.get("names", [])
    print(f"\n📋 Classes ({nc}): {names}")
else:
    print(f"\n❌ data.yaml not found at {YAML_PATH}")

# %% ── Cell 6: Train YOLOv8n ─────────────────────────────────────────────────
#
# YOLOv8n = nano model — fastest to train, good starting point
# epochs=50, patience=10 means training stops if no improvement for 10 epochs
# imgsz=640 is the standard YOLOv8 input size
# batch=16 works on T4 GPU; drop to 8 if you get CUDA out-of-memory

PROJECT_NAME = "PakSignal"
RUN_NAME     = "yolov8n_pak_v1"

print("🚀 Starting YOLOv8n training...")
print(f"   Dataset:  {YAML_PATH}")
print(f"   Device:   {DEVICE}")
print(f"   Output:   {PROJECT_NAME}/{RUN_NAME}/")
print()

model = YOLO("yolov8n.pt")  # downloads pretrained COCO weights (~6 MB)

results = model.train(
    data=YAML_PATH,
    epochs=50,
    imgsz=640,
    batch=16,          # try 8 if you get CUDA OOM
    patience=10,       # early stopping
    project=PROJECT_NAME,
    name=RUN_NAME,
    device=DEVICE,
    workers=2,
    exist_ok=True,
    verbose=True,
)

print("\n✅ Training complete!")

# %% ── Cell 7: Paths to results ───────────────────────────────────────────────

RUN_DIR  = f"{PROJECT_NAME}/{RUN_NAME}"
BEST_PT  = f"{RUN_DIR}/weights/best.pt"
LAST_PT  = f"{RUN_DIR}/weights/last.pt"

print("=" * 55)
print("SAVED ARTIFACTS")
print("=" * 55)

artifacts = {
    "best.pt":                BEST_PT,
    "last.pt":                LAST_PT,
    "results.png":            f"{RUN_DIR}/results.png",
    "confusion_matrix.png":   f"{RUN_DIR}/confusion_matrix.png",
    "PR_curve.png":           f"{RUN_DIR}/PR_curve.png",
    "F1_curve.png":           f"{RUN_DIR}/F1_curve.png",
}

for name, path in artifacts.items():
    tick = "✅" if os.path.exists(path) else "❌"
    print(f"  {tick} {name:<28} {path}")

# %% ── Cell 8: Show training plots ───────────────────────────────────────────

for plot_file in ["results.png", "confusion_matrix.png", "PR_curve.png"]:
    path = f"{RUN_DIR}/{plot_file}"
    if os.path.exists(path):
        print(f"\n── {plot_file} ──")
        display(IPImage(path))
    else:
        print(f"⚠️  {plot_file} not found")

# %% ── Cell 9: Validate on test split ────────────────────────────────────────

print("🧪 Running validation on TEST split...")

best_model = YOLO(BEST_PT)
metrics    = best_model.val(data=YAML_PATH, split="test")

print("\n📊 Test Metrics:")
print(f"  mAP50:     {metrics.box.map50:.4f}")
print(f"  mAP50-95:  {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall:    {metrics.box.mr:.4f}")

# %% ── Cell 10: Predict on sample test images ────────────────────────────────

sample_images = glob.glob(f"{DATASET_PATH}/test/images/*.jpg")[:5]
print(f"Running prediction on {len(sample_images)} sample images...\n")

for img_path in sample_images:
    best_model.predict(
        img_path,
        conf=0.35,
        save=True,
        project="sample_predictions",
        name="test_run",
        exist_ok=True,
    )
    print(f"  → Predicted: {os.path.basename(img_path)}")

# Show saved predictions
pred_imgs = sorted(glob.glob("sample_predictions/test_run/*.jpg"))[:5]
for p in pred_imgs:
    display(IPImage(p, width=640))

# %% ── Cell 11: Save best.pt to Google Drive ─────────────────────────────────

DRIVE_MODEL_DIR = "/content/drive/MyDrive/PakSignal/models"
os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)

drive_dest = os.path.join(DRIVE_MODEL_DIR, "best.pt")
shutil.copy(BEST_PT, drive_dest)
print(f"✅ best.pt saved to Drive: {drive_dest}")

# Zip the entire run for archiving
zip_name = "/content/PakSignal_run.zip"
get_ipython().system(f'zip -r "{zip_name}" "{PROJECT_NAME}/{RUN_NAME}"')
shutil.copy(zip_name, "/content/drive/MyDrive/PakSignal/")
print(f"✅ Full run zipped and copied to Drive.")

# %% ── Cell 12: Download best.pt directly (if not using Drive) ───────────────

from google.colab import files
files.download(BEST_PT)
print("✅ Download triggered. Check your browser's download bar.")

# =============================================================================
# After downloading best.pt:
#   1. Put it in:   PakSignal/models/best.pt
#   2. Run app:     streamlit run app.py
#   3. Upload feed images/videos in the dashboard
# =============================================================================
