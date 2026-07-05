import sys
import subprocess
import importlib
from pathlib import Path
import os

# ========== CONFIGURATION ==========
MODEL_PATH = r"C:\Users\kanis\runs\detect\train\weights\best.pt"
DATA_YAML = r"C:\Users\kanis\Downloads\ChatCLY.v1i.yolov5pytorch\data.yaml"
CONFIDENCE_THRESHOLD = 0.5  # Adjust detection sensitivity (0-1)
# ===================================

def install_required_packages():
    """Install all required packages including IPython"""
    required = {
        'opencv-python': 'cv2',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'seaborn': 'seaborn',
        'IPython': 'IPython',
        'pyyaml': 'yaml',
        'matplotlib': 'matplotlib'
    }
    
    for pkg, import_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"⏳ Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def clear_torch_cache():
    """Clear PyTorch hub cache more thoroughly"""
    cache_paths = [
        Path.home() / '.cache' / 'torch' / 'hub',
        Path.home() / '.cache' / 'torch' / 'checkpoints'
    ]
    
    for cache_dir in cache_paths:
        try:
            if cache_dir.exists():
                for item in cache_dir.glob('*'):
                    try:
                        if item.is_dir():
                            for subitem in item.glob('*'):
                                try:
                                    subitem.unlink()
                                except:
                                    continue
                            try:
                                item.rmdir()
                            except:
                                continue
                        else:
                            try:
                                item.unlink()
                            except:
                                continue
                print(f"✅ Cleared cache: {cache_dir}")
        except Exception as e:
            print(f"⚠️ Could not clear {cache_dir}: {e}")

def load_yolov5_model():
    """More robust model loading with dependency checks"""
    try:
        # First ensure all dependencies are installed
        install_required_packages()
        clear_torch_cache()
        
        # Now import the required modules
        import torch
        from models.experimental import attempt_load
        
        print("🔄 Loading model directly...")
        model = attempt_load(MODEL_PATH)
        model.conf = CONFIDENCE_THRESHOLD
        model.eval()
        print("✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"\n❌ Model loading failed: {str(e)}")
        print("\n🔧 Try these steps:")
        print(r"1. Delete these folders manually:")
        print(r"   - C:\Users\kanis\.cache\torch")
        print(r"   - C:\Users\kanis\.cache\pip")
        print("2. Run these commands:")
        print("   pip install --upgrade pip")
        print("   pip install torch torchvision IPython opencv-python numpy seaborn pyyaml matplotlib")
        print(f"3. Verify model exists at: {MODEL_PATH}")
        return None

def main():
    # Verify paths
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at: {MODEL_PATH}")
        return
        
    if not Path(DATA_YAML).exists():
        print(f"⚠️ Data YAML not found (may not be required): {DATA_YAML}")

    # Load model
    print("\n⏳ Loading model (may take several minutes first time)...")
    model = load_yolov5_model()
    if not model:
        return

    # Camera setup
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    print("\n✅ Detection started - Press Q to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera error")
                break
                
            # Convert and detect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame)
            
            # Draw results
            for *box, conf, cls in results.xyxy[0].tolist():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            cv2.imshow('Crack Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Detection stopped")

if __name__ == "__main__":
    # Install core requirements first
    install_required_packages()
    
    # Now import cv2 and torch after installation
    import cv2
    import torch
    
    # Run main app
    main()