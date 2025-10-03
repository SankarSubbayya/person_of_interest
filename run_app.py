#!/usr/bin/env python3
"""
Script to run the clean Celebrity Search Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit app."""
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if data directory exists
    data_dir = Path("data/celeba")
    if not data_dir.exists():
        print("❌ CelebA dataset not found!")
        print("Please run the download script first:")
        print("python download_celeba.py")
        return
    
    # Check if images exist
    images_dir = data_dir / "img_align_celeba" / "img_align_celeba"
    if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
        print("❌ CelebA images not found!")
        print(f"Expected images in: {images_dir}")
        print("Please ensure the dataset is properly downloaded and extracted.")
        return
    
    # Run the Streamlit app
    print("🚀 Starting Clean Celebrity Search Streamlit app...")
    print("📱 Open your browser to http://localhost:8501")
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
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()

