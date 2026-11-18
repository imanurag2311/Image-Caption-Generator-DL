# Image Caption Generator Using Deep Learning

An end-to-end deep learning system that automatically generates natural language descriptions for images. The system combines InceptionV3 for image feature extraction, YOLOv8 for object detection, and a CNN-LSTM architecture for caption generation.



## Overview

This project implements an image captioning pipeline trained on the Flickr8k dataset. Given an input image, the system:

1. Extracts visual features using InceptionV3 (pretrained on ImageNet)
2. Detects objects and their spatial information using YOLOv8
3. Fuses both feature representations and feeds them into a two-layer LSTM decoder
4. Generates a descriptive caption using greedy or beam search decoding

A Streamlit web application is included for interactive use.

---

## Dataset

**Flickr8k**

- 8,091+ images with 5 human-written captions each
- Actual Local Dataset: 8,619 images (Flickr8k + extra samples)
- Local Split (60/20/20): 5,171 images for training / 1,724 images for validation / 1,724 images for testing
- Vocabulary: 7,000 words (filtered by minimum frequency)
- Maximum sequence length: 35 tokens

Download the dataset from: https://www.kaggle.com/datasets/adityajn105/flickr8k

Place images in `data/Flickr8k/Images/` and captions file at `data/Flickr8k/captions.txt`.

---

## Architecture

**Feature Extraction**

- **InceptionV3** (pretrained on ImageNet, top classification layer removed) produces a 2048-dimensional feature vector per image.
- **YOLOv8n** (Nano version, `yolov8n.pt`) is used to detect objects and encodes class IDs, confidence scores, and normalized bounding box coordinates into a 256-dimensional spatial object embedding.

**Caption Model**

- **Fusion Layer**: Concatenates InceptionV3 features and YOLOv8 embeddings into a single vector, projected to 512 dimensions.
- **Embedding Layer**: Maps partial caption tokens to 256-dimensional dense vectors.
- **LSTMs**: Two stacked Long Short-Term Memory (LSTM) layers with 512 hidden units each, incorporating a dropout rate of 0.5 to prevent overfitting.
- **Output Layer**: A Dense layer with softmax activation over the 7,000-word vocabulary.

Total parameters: 9,170,776 (approximately 9.17 million).

**Decoding Strategies**

- **Greedy Search**: Selects the word with the highest probability at each decoding step.
- **Beam Search**: Maintains a beam width of $K=3$ candidate sequences, choosing the sequence with the highest cumulative length-normalized log-probability.

---

## Project Structure

```
Image-Caption-Generator-DL/
│
├── backend/                 # Flask REST API server & config
│   ├── __init__.py
│   ├── config.py            # Environment-driven configuration
│   └── server.py            # REST endpoints, model caching, inference
│
├── data/
│   └── Flickr8k/
│       ├── Images/          # All 8,091 images
│       └── captions.txt     # Caption annotations
│
├── models/                  # Saved model weights and evaluation results
│   ├── .keras/models/       # Cached InceptionV3 weights (offline ready)
│   ├── .ultralytics/        # Cached Ultralytics settings (sandbox safe)
│   ├── final_model.keras    # Trained CNN-LSTM captioning model
│   ├── tokenizer.pkl        # Keras tokenizer (7,000 vocab)
│   ├── max_length.pkl       # Max sequence length (35)
│   └── evaluation_results.json
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Caption cleaning, tokenizer, dataset split
│   ├── feature_extraction.py # InceptionV3 feature extraction
│   ├── object_detection.py  # YOLOv8 detection and embedding
│   ├── model.py             # CNN-LSTM model definition
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # BLEU and METEOR evaluation
│   └── predict.py           # Greedy and beam search decoding
│
├── app.py                   # Streamlit entry point (launches optimized frontend)
├── frontend.py              # Modern Streamlit UI with API client, gallery & history
├── app_standalone.py        # Legacy standalone Streamlit app (direct model loading)
├── run_backend.py           # Backend server launcher
├── test_api.py              # Automated API test suite
├── main.py                  # Command-line pipeline orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.10
- TensorFlow 2.17.0
- Keras 3.x
- ultralytics (YOLOv8)
- Streamlit
- NLTK
- NumPy, Pillow, tqdm

Full list: see `requirements.txt`

---

## Setup

**1. Create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download NLTK data**

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
```

**4. Place the Flickr8k dataset**

```
data/Flickr8k/Images/    <- all .jpg images
data/Flickr8k/captions.txt
```

---

## Usage

The pipeline is controlled through `main.py`. Each stage can be run independently or all at once.

**Run the full pipeline**

```bash
python main.py --all
```

**Run individual stages**

```bash
# Stage 1: Preprocess captions
python main.py --preprocess

# Stage 2: Extract InceptionV3 image features
python main.py --extract-features

# Stage 3: Run YOLOv8 object detection
python main.py --detect-objects

# Stage 4: Train the model
python main.py --train

# Stage 5: Evaluate on test set
python main.py --evaluate
```

**Optional flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | 64 | Training batch size |
| `--method` | beam | Decoding method: `greedy` or `beam` |
| `--beam-width` | 3 | Beam width for beam search |
| `--max-images` | all | Limit evaluation to N images |

**Running the Full Application (Backend + Frontend)**

The system is organized into a dedicated Flask REST API backend and an optimized Streamlit frontend dashboard.

**Step 1: Start the Backend API Server**

In your first terminal:
```bash
python run_backend.py
```
This runs the Flask REST API at `http://localhost:5000`. It loads and caches InceptionV3, YOLOv8, and the CNN-LSTM model in memory for fast inference.

*Optional configuration flags:*
```bash
python run_backend.py --port 5000 --host 0.0.0.0
```

**Step 2: Launch the Frontend Web Dashboard**

In a second terminal:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

*Frontend features:*
- **📸 Caption Tab**: Upload any local JPG/PNG or paste an image URL. Choose between Greedy or Beam Search with adjustable beam width.
- **🖼 Gallery Tab**: Browse sample images from Flickr8k and caption them with a single click, or try a random sample with `🎲 Try a Random Sample`.
- **ℹ️ About Tab**: Live view of model metadata, parameters, vocabulary size, architecture details, and evaluation metrics.
- **⚡ Status & History**: Real-time backend connection status badge, processing time tracking, caption copy button, and persistent session history.

**Alternative: Standalone Mode (Without Backend)**
If you want to run Streamlit loading the models directly without a separate backend server:
```bash
streamlit run app_standalone.py
```

**Step 3: Verification / Testing**

To verify all backend endpoints, image captioning, and error handling without opening a browser:
```bash
python test_api.py
```
This runs a 23-test automated test suite verifying health checks, metadata, sample images, greedy search, beam search, and error handling.

---

### Backend API Endpoints

| Method | Endpoint | Description | Parameters / Payload |
|--------|----------|-------------|-----------------------|
| `GET` | `/api/health` | Health check & model load status | None |
| `GET` | `/api/model/info` | Architecture, vocab size, metrics | None |
| `POST` | `/api/caption` | Caption uploaded image file | `multipart/form-data`: `image` (file), `method` (`greedy`/`beam`), `beam_width` (1-5) |
| `POST` | `/api/caption/url` | Caption image from web URL | JSON: `{"url": "...", "method": "beam", "beam_width": 3}` |
| `GET` | `/api/sample-images` | Get 12 random sample dataset filenames | None |
| `GET` | `/api/sample-images/<filename>` | Serve a sample image file | Path parameter |

---

## Results

Evaluated on 1,619 test images using beam search (beam_width=3).

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.1653 |
| BLEU-2 | 0.0991 |
| METEOR | 0.2678 |

METEOR is the primary reliability metric here. BLEU scores are computed using per-sentence BLEU (sentence_bleu), which produces lower values than the corpus-level BLEU typically reported in published papers. The METEOR score of 0.2678 falls within the expected range of 0.24-0.28 for this class of model on Flickr8k.

Training converged at epoch 6 out of a maximum of 15, with early stopping (patience=5) halting further training when validation loss stopped improving.

---

## Limitations

- **Scene context hallucination**: Without a spatial attention mechanism, the model cannot focus on specific image regions when predicting each word. This causes it to rely on statistical co-occurrences learned during training (for example, predicting "in the snow" for outdoor dog images regardless of the actual background).
- **Vocabulary coverage**: The 7,000-word vocabulary excludes rare words, which may cause some concepts to be described imprecisely.
- **CPU training**: The model was trained on CPU, limiting the number of epochs and model size relative to GPU-trained baselines.
- **No attention mechanism**: Modern captioning systems use visual attention (or Transformer-based cross-attention) to produce more accurate and contextually grounded captions. This model uses a simple feature fusion approach.

---

## Acknowledgements

- **Flickr8k Dataset**: Hodosh et al., University of Illinois. The dataset is publicly available on Kaggle.
- **InceptionV3**: Pretrained weights from the TensorFlow / Keras model zoo (ImageNet).
- **YOLOv8**: Ultralytics open-source object detection framework.
- This project was developed as part of a summer internship.

---

## License

This project is for academic and educational purposes only. The Flickr8k dataset is subject to its own terms of use. Pretrained model weights (InceptionV3, YOLOv8) are governed by their respective licenses.
