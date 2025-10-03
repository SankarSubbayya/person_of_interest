#!/usr/bin/env python3
"""
Script to run the clean Celebrity Search Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path
from person_of_interest import config

def main():
    """Run the Streamlit app."""
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Resolve data directory from config.yaml (fallback to local)
    try:
        data_dir_str = config["paths"]["data"]
    except Exception:
        data_dir_str = "data/celeba"
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print("❌ CelebA dataset not found!")
        print(f"Expected at: {data_dir}")
        print("Update paths.data in config.yaml or ensure the dataset exists.")
        return
    
    # Check if images exist (auto-detect common layouts)
    candidate_image_dirs = [
        data_dir,
        data_dir / "img_align_celeba",
        data_dir / "img_align_celeba" / "img_align_celeba",
    ]
    images_dir = next((d for d in candidate_image_dirs if d.exists() and list(d.glob("*.jpg"))), None)
    if images_dir is None:
        print("❌ CelebA images not found!")
        print("Checked:")
        for d in candidate_image_dirs:
            print(f" - {d}")
        print("Please ensure the images (.jpg) are present under one of the above directories.")
        return
    
    # Run the Streamlit app
    port = "8502"
    print("🚀 Starting Clean Celebrity Search Streamlit app...")
    print(f"📱 Open your browser to http://localhost:{port}")
    print("⏹️  Press Ctrl+C to stop the app")

    env = os.environ.copy()
    env["OPENCV_OPENCL_RUNTIME"] = "disabled"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["GLOG_minloglevel"] = "3"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", port,
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()

