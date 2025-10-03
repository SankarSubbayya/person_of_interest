# Clean Celebrity Search Engine

A streamlined celebrity search application extracted from the cleaned notebook code.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure CelebA dataset is downloaded:**
   ```bash
   python download_celeba.py
   ```

3. **Run the app:**
   ```bash
   python run_clean_app.py
   ```

4. **Open browser to:** `http://localhost:8501`

## 📁 Files

- `celebrity_search_clean.py` - Core search engine class
- `streamlit_app_clean.py` - Streamlit web application
- `run_clean_app.py` - Script to launch the app
- `requirements.txt` - Minimal dependencies

## ✨ Features

- **Text Search**: Natural language queries
- **Image Search**: Upload images to find similar celebrities
- **Random Images**: Browse random images from the dataset
- **Clean Code**: Extracted from cleaned notebook
- **Fast Loading**: Cached embeddings for quick startup

## 🎯 Usage

### Text Search Examples
- "young smiling woman with glasses"
- "serious looking man with beard"
- "beautiful celebrity with blonde hair"

### Image Search
Upload any image to find similar celebrity faces.

### Random Images
Browse random images from the CelebA dataset.

## 🔧 Technical Details

- **Model**: CLIP ViT-B-32
- **Dataset**: CelebA (202,599 images)
- **Embeddings**: Cached for fast loading
- **Memory**: Efficient batch processing

## 📊 Performance

- **First run**: ~10-15 minutes (generates embeddings)
- **Subsequent runs**: ~30 seconds (loads cached embeddings)
- **Search time**: <1 second per query

---

**Clean and Simple! 🎉**

