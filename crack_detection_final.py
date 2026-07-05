#!/usr/bin/env python
"""
YOLOv5 Crack‑Detection (webcam) — **fully fixed** for
• PyTorch ≥ 2.6 ‘weights‑only’ safe‑loading
• recent YOLOv5 code that removed the `grid` attribute
• Windows or Raspberry Pi (CPU‑only)

How this version differs:
1. **Uses DetectMultiBackend** from YOLOv5 instead of `torch.hub`.
   – avoids the AutoShape/`grid` crash entirely.
2. Adds `DetectionModel` to PyTorch’s safe globals once, so checkpoints load.
3. Minimal external deps; installs the YOLOv5 repo from GitHub if missing.
4. Saves annotated frames that contain ≥1 crack detection.
"""

from __future__ import annotations

import subprocess
import sys
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
MODEL_PATH   = Path(r"C:/Users/kanis/runs/detect/train/weights/best.pt")
CONF_THRES   = 0.50           # confidence threshold for NMS
IMG_SIZE     = 640            # inference resolution (square)
SAVE_DIR     = Path("detected_cracks")  # folder for snapshots
SAVE_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────
# AUTO‑INSTALL REQUIRED PACKAGES (once)
# ────────────────────────────────────────────────────────────
REQUIRED: Dict[str, str] = {
    "torch"          : "torch",
    "opencv-python" : "cv2",
    "numpy"         : "numpy",
    "ultralytics"   : "ultralytics",      # provides DetectionModel class
    # we install the repo below for yolov5 utils
}


def ensure_packages() -> None:
    """Install any missing packages listed in REQUIRED."""
    for pip_name, mod_name in REQUIRED.items():
        try:
            importlib.import_module(mod_name)
        except ImportError:
            print(f"⏳ Installing {pip_name} …")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_packages()

# Install yolov5 repo (brings utils/, models/, etc.) if not present
try:
    import yolov5  # type: ignore  # noqa: F401
except ImportError:
    print("⏳ Installing yolov5 repo …")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/ultralytics/yolov5.git",
        ]
    )

# ────────────────────────────────────────────────────────────
# NOW IMPORT LIBS THAT MIGHT HAVE JUST BEEN INSTALLED
# ────────────────────────────────────────────────────────────
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

# yolov5 internals
from yolov5.models.common import DetectMultiBackend  # noqa: E402
from yolov5.utils.general import (
    non_max_suppression,
    scale_coords,
    check_img_size,
)
from yolov5.utils.torch_utils import select_device  # noqa: E402
from yolov5.utils.augmentations import letterbox  # noqa: E402

# ────────────────────────────────────────────────────────────
# SAFE‑LOAD PATCH FOR PYTORCH ≥ 2.6
# ────────────────────────────────────────────────────────────
torch.serialization.add_safe_globals(
    {"ultralytics.nn.tasks.DetectionModel": DetectionModel}
)


# ────────────────────────────────────────────────────────────
# LOAD MODEL VIA DetectMultiBackend (NO AutoShape, NO grid crash)
# ────────────────────────────────────────────────────────────

def load_model() -> DetectMultiBackend | None:
    if not MODEL_PATH.is_file():
        print(f"❌ Could not find weights: {MODEL_PATH}")
        return None

    device = select_device("cpu")
    try:
        model = DetectMultiBackend(MODEL_PATH, device=device)
        model.conf = CONF_THRES
        print("✅ Model loaded →", MODEL_PATH.name)
        return model
    except Exception as err:  # noqa: BLE001
        print(f"❌ Model load failed: {err}")
        return None


# ────────────────────────────────────────────────────────────
# MAIN DETECTION LOOP
# ────────────────────────────────────────────────────────────

def run_webcam(model: DetectMultiBackend) -> None:
    stride, names = model.stride, model.names
    imgsz = check_img_size(IMG_SIZE, s=stride)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("✅ Crack‑detection started — press Q to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️  Webcam frame read failed")
                break

            # ── Pre‑process ────────────────────────
            img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
            img = img[:, :, ::-1].transpose((2, 0, 1))  # BGR→RGB, HWC→CHW
            img = np.ascontiguousarray(img)
            img_t = torch.from_numpy(img).float() / 255.0  # 0‑1
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            # ── Inference ───────────────────────────
            with torch.no_grad():
                pred = model(img_t)
            preds = non_max_suppression(pred, CONF_THRES, 0.45)

            # ── Draw boxes ──────────────────────────
            save_this = False
            for det in preds:
                if len(det):
                    det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        label = f"{names[int(cls)]} {conf:.2f}"
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        save_this = True

            # ── Snapshot if crack detected ──────────
            if save_this:
                fname = SAVE_DIR / f"crack_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(fname), frame)

            cv2.imshow("Crack Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")


# ────────────────────────────────────────────────────────────
# ENTRY
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mdl = load_model()
    if mdl is not None:
        run_webcam(mdl)
