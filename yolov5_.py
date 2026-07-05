#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time crack detection with YOLOv5 and a webcam.
Automatically installs any missing Python packages on first run.
"""

import sys
import subprocess
import importlib
from pathlib import Path
import os

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
MODEL_PATH = (
    r"C:\Users\kanis\runs\detect\train\weights\best.pt"     # your .pt weights
)
CONFIDENCE_THRESHOLD = 0.50                                 # 0-1
# ──────────────────────────────────────────────────────────

def install_required_packages() -> None:
    """Install all packages in `required` if they are missing."""
    required = {
        "opencv-python": "cv2",
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "seaborn": "seaborn",
        "IPython": "IPython",
        "pyyaml": "yaml",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
    }

    for pkg, import_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"⏳ Installing {pkg} …")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {pkg}: {e}")

def clear_torch_cache() -> None:
    """Remove ~/.cache/torch and ~/.cache/pip contents (if they exist)."""
    cache_dirs = [
        Path.home() / ".cache" / "torch",
        Path.home() / ".cache" / "pip",
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        try:
            # walk the directory and delete files/empty dirs
            for item in cache_dir.rglob("*"):
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        item.rmdir()
                except Exception as e:
                    print(f"⚠️ Could not remove {item}: {e}")
                    continue
            print(f"✅ Cleared cache: {cache_dir}")
        except Exception as e:
            print(f"⚠️ Error clearing {cache_dir}: {e}")

def load_yolov5_model():
    """Load a custom YOLOv5 model from MODEL_PATH."""
    try:
        install_required_packages()
        clear_torch_cache()

        import torch
        model = torch.hub.load(
            "ultralytics/yolov5:v7.0",
            "custom",
            path=MODEL_PATH,
            force_reload=True,
            skip_validation=True,
        )
        model.conf = CONFIDENCE_THRESHOLD
        model.eval()
        print("✅ Model loaded successfully")
        return model

    except Exception as exc:
        print(f"\n❌ Model loading failed: {exc}")
        print(
            """
🔧 Troubleshooting:
  1. Delete these folders manually then retry:
     • %USERPROFILE%\\.cache\\torch
     • %USERPROFILE%\\.cache\\pip
  2. Reinstall core libs:
     python -m pip install --upgrade pip
     python -m pip install torch torchvision opencv-python
  3. Verify MODEL_PATH exists:
     """
            + MODEL_PATH
        )
        return None

def main() -> None:
    if not Path(MODEL_PATH).is_file():
        print(f"❌ Error: model not found at {MODEL_PATH}")
        return

    print("\n⏳ Loading model …")
    model = load_yolov5_model()
    if model is None:
        return

    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: could not open camera")
        return

    print("\n✅ Detection started — press Q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Camera frame read failed")
                break

            results = model(frame)

            for *xyxy, conf, cls in results.xyxy[0].tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Crack Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Detection stopped")

if __name__ == "__main__":
    main()