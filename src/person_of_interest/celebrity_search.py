"""
Clean Celebrity Search Engine
Extracted from the cleaned notebook code.
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import pickle
from typing import List, Tuple, Optional
class CelebritySearchEngine:
    """Clean celebrity search engine for the CelebA dataset."""
    
    
    def __init__(self, data_dir: str = "data/celeba", model_name: str = "clip-ViT-B-32", embeddings_dir: Optional[str] = None):
        """Initialize the search engine."""
        self.data_dir = Path(data_dir)
        # Resolve images directory automatically
        self.images_dir = self._resolve_images_dir(self.data_dir)
        self.model_name = model_name
        self.model = None
        self.img_names = []
        self.img_emb = None
        # Embeddings directory (defaults to data_dir if not provided)
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else self.data_dir
        # Face detection removed; use plain PIL loading
    
    
    '''
    def __init__(self, data_dir: str = "data/celeba", model_name: str = "clip-ViT-B-32",
             use_face_detection: bool = True, face_conf_threshold: float = 0.8,
             face_target_size: Tuple[int, int] = (224, 224)):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "img_align_celeba" / "img_align_celeba"
        self.model_name = model_name
        self.model = None
        self.img_names = []
        self.img_emb = None

        # RetinaFace settings (auto-disabled if lib not available)
        self.use_face_detection = bool(use_face_detection and _HAVE_RETINAFACE)
        self.face_conf_threshold = face_conf_threshold
        self.face_target_size = face_target_size
        '''

    def load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Model loaded successfully!")
        
    def load_dataset(self, batch_size: int = 1000, use_precomputed: bool = False):
        """
        Load the CelebA dataset and generate embeddings.
        
        Args:
            batch_size: Number of images to process at once
            use_precomputed: Whether to use precomputed embeddings
        """
        print("Loading CelebA dataset...")
        
        # Check for precomputed embeddings
        if use_precomputed:
            emb_filename = str(self.embeddings_dir / 'celeba-dataset-embeddings.pkl')
            if os.path.exists(emb_filename):
                print("Loading precomputed embeddings...")
                with open(emb_filename, 'rb') as fIn:
                    self.img_names, self.img_emb = pickle.load(fIn)
                # Rebase image paths if they don't exist at stored locations
                self._maybe_rebase_img_paths()
                print(f"✅ Loaded {len(self.img_names)} images with embeddings")
                return
        
        # Resolve images directory if not already
        if not self.images_dir or not self.images_dir.exists():
            self.images_dir = self._resolve_images_dir(self.data_dir)
        # Get all image paths
        self.img_names = [str(p) for p in sorted(self.images_dir.glob('*.jpg'))]
        print(f"Found {len(self.img_names)} images")
        
        if not self.img_names:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        
        # Process images in batches to avoid "too many open files" error
        print("Processing images in batches...")
        all_embeddings = []
        
        for i in range(0, len(self.img_names), batch_size):
            batch_files = self.img_names[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(self.img_names)-1)//batch_size + 1}")
            
            # Open images in this batch
            batch_images = []
            for filepath in batch_files:
                try:
                    img = Image.open(filepath)
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error opening {filepath}: {e}")
                    continue
            
            # Encode this batch
            if batch_images:
                batch_emb = self.model.encode(
                    batch_images,
                    batch_size=128,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_emb)
                
                # Close images to free file descriptors
                for img in batch_images:
                    img.close()
        
        # Concatenate all embeddings
        if all_embeddings:
            self.img_emb = torch.cat(all_embeddings, dim=0)
            print(f"✅ Generated embeddings shape: {self.img_emb.shape}")
            
            # Save embeddings for future use
            self.save_embeddings()
        else:
            raise RuntimeError("No images were successfully processed!")

    def _resolve_images_dir(self, base_dir: Path) -> Path:
        """Find a directory under base_dir that contains CelebA JPGs.
        Tries common layouts and falls back to a recursive search.
        """
        candidates = [
            base_dir,
            base_dir / "img_align_celeba",
            base_dir / "img_align_celeba" / "img_align_celeba",
        ]
        for cand in candidates:
            try:
                if cand.exists() and any(cand.glob('*.jpg')):
                    return cand
            except Exception:
                continue
        # Fallback recursive search: pick parent of first jpg found
        try:
            for p in base_dir.rglob('*.jpg'):
                return p.parent
        except Exception:
            pass
        # If nothing found, return the most likely default path
        return base_dir / "img_align_celeba" / "img_align_celeba"

    def _maybe_rebase_img_paths(self) -> None:
        """If stored image paths no longer exist (dataset moved),
        remap them to the current images directory by filename.
        """
        if not self.img_names:
            return
        num_missing = sum(1 for p in self.img_names if not os.path.exists(p))
        if num_missing == 0:
            return
        # Ensure images_dir is resolved
        if not self.images_dir.exists():
            self.images_dir = self._resolve_images_dir(self.data_dir)
        # Rebase using filenames
        rebased = [str(self.images_dir / Path(p).name) for p in self.img_names]
        self.img_names = rebased
    def save_embeddings(self, filename: str = "celeba-dataset-embeddings.pkl"):
        """Save embeddings to file."""
        if self.img_emb is None:
            print("No embeddings to save")
            return
            
        # Ensure embeddings directory exists
        try:
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        emb_path = self.embeddings_dir / filename
        with open(emb_path, 'wb') as f:
            pickle.dump((self.img_names, self.img_emb), f)
        print(f"✅ Embeddings saved to {emb_path}")
    
    def search(self, query: str, is_image: bool = False, k: int = 8) -> Tuple[List[str], List[dict]]:
        """
        Search for similar images using text or image query.
        
        Args:
            query: Text query or image path
            is_image: Whether the query is an image path
            k: Number of results to return
            
        Returns:
            Tuple of (result_paths, hits)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.img_emb is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        # Process query
        if is_image:
            if not os.path.exists(query):
                raise FileNotFoundError(f"Image not found: {query}")
            #query_image = self._crop_best_face(query) or   Image.open(query)
            query_image = Image.open(query)
        
        
            query_emb = self.model.encode([query_image], convert_to_tensor=True, show_progress_bar=False)
        else:
            query_emb = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        
        # Find similar images
        hits = util.semantic_search(query_emb, self.img_emb, top_k=k)[0]
        
        # Get result paths
        result_paths = [self.img_names[hit['corpus_id']] for hit in hits]
        
        return result_paths, hits
    
    def display_images(self, image_paths: List[str], num_images: int = 9, 
                      figsize: Tuple[int, int] = (12, 12), titles: Optional[List[str]] = None):
        """
        Display images in a grid.
        
        Args:
            image_paths: List of image file paths
            num_images: Number of images to display
            figsize: Figure size
            titles: Optional titles for each image
        """
        if not image_paths:
            print("No images to display")
            return
        
        # Limit to requested number of images
        display_paths = image_paths[:num_images]
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        # Flatten axes for easier indexing
        axes_flat = [ax for row in axes for ax in row]
        
        for i, (ax, img_path) in enumerate(zip(axes_flat, display_paths)):
            try:
                # Load and display image
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis('off')
                
                # Set title
                if titles and i < len(titles):
                    ax.set_title(titles[i], fontsize=10)
                else:
                    filename = Path(img_path).name
                    ax.set_title(filename, fontsize=8)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{Path(img_path).name}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(display_paths), len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def display_random_images(self, num_images: int = 9, random_seed: int = 42):
        """Display random images from the dataset."""
        if not self.img_names:
            print("No images loaded")
            return
            
        np.random.seed(random_seed)
        random_indices = np.random.choice(
            len(self.img_names), 
            min(num_images, len(self.img_names)), 
            replace=False
        )
        random_paths = [self.img_names[i] for i in random_indices]
        self.display_images(random_paths, num_images)
    
    def display_image_by_index(self, index: int, title: Optional[str] = None):
        """Display a single image by index."""
        if not self.img_names:
            print("No images loaded")
            return
            
        if index >= len(self.img_names):
            print(f"Index {index} is out of range. Dataset has {len(self.img_names)} images.")
            return
        
        img_path = self.img_names[index]
        img = Image.open(img_path)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Image {index}: {Path(img_path).name}")
        
        plt.show()


def main():
    """Example usage of the CelebritySearchEngine."""
    # Initialize the search engine
    search_engine = CelebritySearchEngine()
    
    # Load model and dataset
    search_engine.load_model()
    search_engine.load_dataset()
    
    # Example searches
    print("\n🔍 Example searches:")
    
    # Text search
    print("\n1. Text search: 'young smiling celebrity'")
    results, hits = search_engine.search("young smiling celebrity", k=6)
    print(f"Found {len(results)} results")
    
    # Display results
    search_engine.display_images(results, num_images=6)
    
    # Random images
    print("\n2. Random images:")
    search_engine.display_random_images(num_images=6)


if __name__ == "__main__":
    main()

