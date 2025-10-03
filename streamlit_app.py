"""
Clean Celebrity Search Streamlit App
Based on the cleaned notebook code.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from person_of_interest.celebrity_search import CelebritySearchEngine

import os, cv2, torch
os.environ.setdefault("OPENCV_OPENCL_RUNTIME","disabled")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
try: cv2.ocl.setUseOpenCL(False); cv2.setNumThreads(1)
except: pass
try: torch.set_num_threads(1); torch.set_num_interop_threads(1)
except: pass

# Page configuration
st.set_page_config(
    page_title="Celebrity Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS with larger fonts
st.markdown("""
<style>
    /* Global font size increases */
    .stApp {
        font-size: 22px;
    }
    
    /* Main header - even larger */
    .main-header {
        font-size: 6rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    /* Section headers */
    .search-section h1, .search-section h2, .search-section h3 {
        font-size: 2.2rem !important;
        margin-bottom: 1rem;
    }
    
    /* Search sections with larger text */
    .search-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2rem;
    }

     /* Sidebar text larger */
    .css-1d391kg {
        font-size: 1.1rem;
    }
    
    /* Button text larger */
    .stButton > button {
        font-size: 1.2rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600;
    }
    
    /* Input labels larger */
    .stTextInput > label, .stFileUploader > label, .stSlider > label {
        font-size: 1.2rem !important;
        font-weight: 600;
    }
    
    /* Input text larger */
    .stTextInput > div > div > input {
        font-size: 1.1rem !important;
    }
    
    /* Metric cards with larger text */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
    }
        /* Success/Warning/Error messages larger */
    .stSuccess, .stWarning, .stError, .stInfo {
        font-size: 1.2rem !important;
    }
    
    /* Image captions larger */
    .stImage > div > div > div > p {
        font-size: 1.1rem !important;
        font-weight: 500;
    }
    
    /* Sidebar headers larger */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        font-size: 1.5rem !important;
    }
    
    /* Help text larger */
    .stHelp {
        font-size: 1rem !important;
    }
    
    /* Spinner text larger */
    .stSpinner > div {
        font-size: 1.2rem !important;
    }
 /* Footer text larger */
    .css-1d391kg p {
        font-size: 1.1rem;
    }
    
    /* Result images styling */
    .result-image {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Radio button text larger */
    .stRadio > label {
        font-size: 1.2rem !important;
    }
    
    /* Slider labels larger */
    .stSlider > div > div > div > div {
        font-size: 1.1rem !important;
    }
    # Add this at the end of your CSS:
* {
    font-size: inherit !important;
}

html, body, [class*="css"] {
    font-size: 22px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_search_engine():
    """Load the search engine with caching."""
    search_engine = CelebritySearchEngine()
    
    # Check if embeddings exist
    embeddings_file = "data/celeba/celeba-dataset-embeddings.pkl"
    if os.path.exists(embeddings_file):
        try:
            search_engine.load_model()
            search_engine.load_dataset(use_precomputed=True)
            return search_engine, "Loaded from cache"
        except Exception as e:
            st.error(f"Error loading cached embeddings: {e}")
    
    # Load from scratch
    try:
        search_engine.load_model()
        search_engine.load_dataset()
        return search_engine, "Loaded and cached"
    except Exception as e:
        st.error(f"Error loading search engine: {e}")
        return None, "Error"


def display_image_grid(image_paths, titles=None, max_images=9):
    """Display images in a grid using Streamlit."""
    if not image_paths:
        st.warning("No images to display")
        return
    
    # Limit number of images
    display_paths = image_paths[:max_images]
    
    # Calculate grid dimensions
    cols = 3
    rows = (len(display_paths) + cols - 1) // cols
    
    # Create columns
    for i in range(0, len(display_paths), cols):
        cols_list = st.columns(cols)
        
        for j, col in enumerate(cols_list):
            if i + j < len(display_paths):
                img_path = display_paths[i + j]
                try:
                    img = Image.open(img_path)
                    caption = titles[i + j] if titles and i + j < len(titles) else Path(img_path).name
                    col.image(img, caption=caption, use_container_width=True)
                except Exception as e:
                    col.error(f"Error loading {Path(img_path).name}")


def filter_results_by_threshold(results, hits, threshold):
    """Filter search results based on similarity threshold."""
    filtered_results = []
    filtered_hits = []
    
    for result, hit in zip(results, hits):
        if hit['score'] >= threshold:
            filtered_results.append(result)
            filtered_hits.append(hit)
    
    return filtered_results, filtered_hits


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Celebrity Search Engine</h1>', unsafe_allow_html=True)
    st.markdown("Search through the CelebA dataset using text descriptions or images!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Search type
        search_type = st.radio(
            "Search Type",
            ["Text Search", "Image Search", "Random Images"],
            help="Choose how you want to search for celebrities"
        )
        
        # Number of results
        num_results = st.slider(
            "Number of Results",
            min_value=3,
            max_value=24,
            value=9,
            help="How many results to display"
        )

        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Minimum similarity score to show results (higher = more similar)"
        )

        
        # Model info
        st.header("ℹ️ Model Info")
        st.info("Using CLIP ViT-B-32 model for semantic search")
        
        # Dataset info
        st.header("📊 Dataset Info")
        st.info("CelebA dataset with 202,599 celebrity images")
    
    # Load search engine
    with st.spinner("Loading search engine..."):
        search_engine, load_status = load_search_engine()
    
    if search_engine is None:
        st.error("Failed to load search engine. Please check your data directory.")
        return
    
    # Show load status
    st.success(f"Search engine loaded: {load_status}")
    
    # Main content area
    if search_type == "Text Search":
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.header("🔤 Text Search")
        st.write("Describe the celebrity you're looking for using natural language.")
        
        # Text input
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'young smiling woman with glasses'",
            help="Describe the celebrity you want to find"
        )
        
        # Search button
        if st.button("🔍 Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    try:
                        results, hits = search_engine.search(query, is_image=False, k=num_results)
                        
                        # Display results
                        st.success(f"Found {len(results)} results for: '{query}'")
                        
                        # Show similarity scores
                        if hits:
                            scores = [hit['score'] for hit in hits]
                            col1, col2, col3 = st.columns(3)
                            #with col1:
                                #st.metric("Average Similarity", f"{np.mean(scores):.3f}")
                            #with col2:
                                #st.metric("Best Match Score", f"{max(scores):.3f}")
                            #with col3:
                                #st.metric("Worst Match Score", f"{min(scores):.3f}")
                        
                        # Display images
                        display_image_grid(results, max_images=num_results)
                        
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a search query")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif search_type == "Image Search":
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.header("🖼️ Image Search")
        st.write("Upload an image to find similar celebrities.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to find similar celebrities"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption="Your uploaded image", use_container_width=True)
            
            # Search button
            if st.button("🔍 Find Similar Celebrities", type="primary"):
                with st.spinner("Searching for similar celebrities..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            img.save(tmp_file.name)
                            
                            # Search
                            results, hits = search_engine.search(tmp_file.name, is_image=True, k=num_results)
                            
                            # Clean up temp file
                            os.unlink(tmp_file.name)
                        
                        # Display results
                        st.success(f"Found {len(results)} similar celebrities")
                        
                        # Show similarity scores
                        if hits:
                            scores = [hit['score'] for hit in hits]
                            col1, col2, col3 = st.columns(3)
                            
                            #with col1:
                            #    st.metric("Average Similarity", f"{np.mean(scores):.3f}")
                            #with col2:
                            #    st.metric("Best Match Score", f"{max(scores):.3f}")
                            #with col3:
                            #    st.metric("Worst Match Score", f"{min(scores):.3f}")
                        # Display images
                        display_image_grid(results, max_images=num_results)
                        
                    except Exception as e:
                        st.error(f"Search failed: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif search_type == "Random Images":
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.header("🎲 Random Images")
        st.write("Display random images from the CelebA dataset.")
        
        # Random seed
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=10000,
            value=42,
            help="Change this to get different random images"
        )
        
        # Display button
        if st.button("🎲 Show Random Images", type="primary"):
            with st.spinner("Loading random images..."):
                try:
                    # Get random images
                    np.random.seed(random_seed)
                    random_indices = np.random.choice(
                        len(search_engine.img_names), 
                        min(num_results, len(search_engine.img_names)), 
                        replace=False
                    )
                    random_paths = [search_engine.img_names[i] for i in random_indices]
                    
                    st.success(f"Showing {len(random_paths)} random images")
                    display_image_grid(random_paths, max_images=num_results)
                    
                except Exception as e:
                    st.error(f"Error loading random images: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Celebrity Search Engine | Powered by CLIP and Streamlit</p>
        <p>Dataset: CelebA | Model: CLIP ViT-B-32</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


