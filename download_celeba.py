#!/usr/bin/env python3
"""
Simple CelebA Dataset Downloader
"""

import kagglehub
from pathlib import Path


def download_celeba(output_dir="data/celeba"):
    """
    Download CelebA dataset using kagglehub.
    
    Args:
        output_dir (str): Directory to save the dataset
        
    Returns:
        str: Path to the downloaded dataset
    """
    print("Downloading CelebA dataset...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        
        print("✓ Download completed!")
        print(f"Path to dataset files: {path}")
        
        # Copy to desired output directory
        if output_dir != str(path):
            import shutil
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Copy the dataset to the desired location
            shutil.copytree(path, output_path, dirs_exist_ok=True)
            print(f"✓ Dataset copied to: {output_path.absolute()}")
            return str(output_path)
        
        return path
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("Make sure you have kagglehub installed: uv add kagglehub")
        return None


if __name__ == "__main__":
    result = download_celeba()
    if result:
        print(f"\n🎉 CelebA dataset ready at: {result}")
        print(f"Images: {result}/img_align_celeba/img_align_celeba/")
        print(f"Annotations: {result}/")
    else:
        print("\n❌ Download failed")

