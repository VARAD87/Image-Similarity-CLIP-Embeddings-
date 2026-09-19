# Image Similarity & Feature Search Engine

An image retrieval and similarity system built with **Python, PyTorch, Hugging Face Transformers, CLIP, NumPy, Pillow, and FastAPI**.

The project converts images into **512-dimensional normalized embeddings** using the pretrained **OpenAI CLIP ViT-B/32** model and compares those embeddings using **cosine similarity** to identify the most similar images.

---

##  Project Overview

The goal of this project is to build an image similarity search system that can:

* Accept a reference image
* Extract visual features using a pretrained CLIP model
* Convert the image into a 512-dimensional embedding
* Compare the reference embedding with candidate image embeddings
* Rank candidate images according to similarity
* Return the best match and top-K matches
* Cache previously generated embeddings to avoid unnecessary computation
* Test similarity robustness under different image transformations
* Expose the matching pipeline through a FastAPI endpoint

### Basic Workflow

```text
                 Reference Image
                        │
                        ▼
                 Image Preprocessing
                     (Pillow)
                        │
                        ▼
                  CLIP Processor
                        │
                        ▼
                  CLIP ViT-B/32
                        │
                        ▼
              512-D Image Embedding
                        │
                        ▼
                 Normalization
                        │
                        ▼
             Cosine Similarity Search
                        │
                        ▼
                 Ranking / Top-K
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        Best Match             Top Matches
```

---

##  How It Works

### 1. Image Input

The system accepts an image in formats such as:

* `.jpg`
* `.jpeg`
* `.png`

Images are loaded using **Pillow** and converted to RGB before being processed by CLIP.

---

### 2. CLIP Feature Extraction

The project uses the pretrained model:

```text
openai/clip-vit-base-patch32
```

CLIP is used as a feature extractor rather than being trained again for this project.

The image is processed and passed through CLIP to generate an image representation.

```text
Image
  ↓
CLIP Processor
  ↓
PyTorch Tensor
  ↓
CLIP Model
  ↓
Image Features
```

---

### 3. Image Embeddings

The extracted image features are represented as a **512-dimensional vector**.

Conceptually:

```text
Image
  ↓
[0.12, -0.34, 0.56, ...]
  ↓
512-dimensional embedding
```

The embedding is normalized before similarity calculation.

Normalization makes the vector have unit length, which allows the project to use cosine similarity effectively.

---

## 📐 Cosine Similarity

The project compares two image embeddings using cosine similarity.

For normalized vectors, the cosine similarity calculation can be represented as a dot product:

```text
similarity = embedding_A · embedding_B
```

The project uses NumPy for this calculation.

Higher similarity indicates that the two image embeddings point in more similar directions.

Example:

```text
Reference Image
      │
      ├── Candidate A → 0.42
      ├── Candidate B → 0.91
      ├── Candidate C → 0.67
      └── Candidate D → 0.83
```

The system ranks the candidates:

```text
1. Candidate B → 0.91
2. Candidate D → 0.83
3. Candidate C → 0.67
```

---

##  Top-K Image Retrieval

The project supports retrieving the top-K most similar images.

For example:

```python
find_top_matches(
    reference_embedding,
    candidate_embeddings,
    top_k=3
)
```

returns the three highest-scoring matches.

The results are sorted from highest similarity to lowest similarity.

---

##  Best Match & Confidence Threshold

The project also provides a best-match function.

A similarity threshold can be used to distinguish between:

```text
Confident Match
```

and:

```text
Closest Available Match
```

The current implementation uses a default threshold of:

```text
0.6
```

Conceptually:

```text
similarity >= 0.6
        ↓
confident match

similarity < 0.6
        ↓
not considered confident
```

The threshold is a project configuration and may need adjustment depending on the dataset and application requirements.

---

#  Embedding Caching

Generating embeddings for every candidate image every time would be inefficient.

The project therefore maintains a cache:

```text
embeddings/embeddings_cache.pkl
```

The workflow is:

```text
Candidate Images
      │
      ▼
Check Cache
      │
      ├── Already exists → Reuse embedding
      │
      └── New image → Generate embedding
                         │
                         ▼
                    Save to cache
```

The cache stores:

```text
filename → embedding
```

This prevents unchanged images from being processed repeatedly.

The implementation also removes cached entries for images that no longer exist in the candidate-image directory.

---

#  Robustness Testing

The project includes a small robustness evaluation framework.

A source image is transformed into different versions:

* Resize
* Rotation
* Crop
* Brightness increase
* Brightness decrease
* Gaussian blur
* JPEG compression

Example:

```text
Original Image
      │
      ├── Resize
      ├── Rotate
      ├── Crop
      ├── Brighten
      ├── Darken
      ├── Blur
      └── JPEG Compression
```

Each variation is converted into an embedding and compared with the original image embedding.

This helps evaluate how similarity scores change when the same image is modified.

---

#  FastAPI Service

The project exposes the image-matching functionality through a FastAPI backend.

### Main endpoint

```http
POST /match
```

The endpoint accepts an uploaded image and performs the complete matching pipeline.

```text
Uploaded Image
      ↓
Temporary File
      ↓
Generate Reference Embedding
      ↓
Load Candidate Embeddings
      ↓
Find Best Match
      ↓
Find Top Matches
      ↓
JSON Response
```

### Example Response

```json
{
  "best_match": {
    "filename": "example.jpg",
    "similarity": 0.9123,
    "is_confident": true
  },
  "top_matches": [
    {
      "filename": "example.jpg",
      "similarity": 0.9123
    },
    {
      "filename": "example2.jpg",
      "similarity": 0.8451
    },
    {
      "filename": "example3.jpg",
      "similarity": 0.7924
    }
  ]
}
```

The actual similarity values depend on the images being searched.

---

#  Project Structure

```text
Image-Similarity-CLIP-Embeddings/
│
├── api.py
├── build_embeddings.py
├── embedding.py
├── embedding_utils.py
├── image_utils.py
├── main.py
├── make_test_variations.py
├── similarity.py
├── test_robustness.py
├── test_threshold.py
│
├── images/
│   └── candidate images
│
├── reference/
│   └── reference images
│
├── embeddings/
│   └── embeddings_cache.pkl
│
└── test_variations/
    └── generated test images
```

---

#  File Responsibilities

| File                      | Purpose                                                                    |
| ------------------------- | -------------------------------------------------------------------------- |
| `embedding.py`            | Initial/manual CLIP embedding experiment                                   |
| `embedding_utils.py`      | Reusable function for generating normalized image embeddings               |
| `build_embeddings.py`     | Builds and caches embeddings for candidate images                          |
| `similarity.py`           | Cosine similarity, ranking, top-K matching, and threshold-based confidence |
| `image_utils.py`          | Reads basic image information such as dimensions and color mode            |
| `main.py`                 | Command-line workflow for testing image matching                           |
| `api.py`                  | FastAPI service for image matching                                         |
| `make_test_variations.py` | Generates transformed versions of an image                                 |
| `test_robustness.py`      | Tests similarity against transformed images                                |
| `test_threshold.py`       | Tests image matching and similarity scores with a reference image          |

---

#  Technologies Used

### Python

Main programming language used to implement the complete pipeline.

### PyTorch

Used to run the CLIP model and perform inference.

### Hugging Face Transformers

Provides the `CLIPModel` and `CLIPProcessor` used by the project.

### CLIP

Pretrained vision-language model used here for image feature extraction.

Model:

```text
openai/clip-vit-base-patch32
```

### Pillow

Used for image loading, RGB conversion, resizing, rotation, cropping, brightness modification, blurring, and image saving.

### NumPy

Used for numerical operations on image embeddings and cosine similarity calculations.

### Pickle

Used to save and reload the embedding cache.

### FastAPI

Used to expose the image matching functionality as a web API.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/VARAD87/Image-Similarity-CLIP-Embeddings.git
```

Move into the project:

```bash
cd Image-Similarity-CLIP-Embeddings
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install torch transformers pillow numpy fastapi uvicorn python-multipart
```

> The exact package versions should be kept consistent with the environment in which the project was developed.

---

#  Running the Project

## Option 1 — Run the command-line matcher

Run:

```bash
python main.py
```

The program:

1. Generates the reference embedding
2. Loads/builds candidate embeddings
3. Finds the best match
4. Calculates the similarity score
5. Displays the top matches

---

## Option 2 — Run the FastAPI service

Start the API:

```bash
uvicorn api:app --reload
```

The API will start locally.

The main image matching endpoint is:

```text
POST /match
```

The root endpoint can be used as a basic health check:

```text
GET /
```

Expected response:

```json
{
  "status": "Image Similarity API is running."
}
```

---

# Testing

### Generate Image Variations

Run:

```bash
python make_test_variations.py
```

This creates transformed versions of the configured source image.

---

### Test Robustness

Run:

```bash
python test_robustness.py
```

This compares the transformed images against the original image and prints similarity scores.

---

### Test Matching / Threshold

Run:

```bash
python test_threshold.py
```

This tests a reference image against the candidate image collection and displays:

```text
Testing:
Best Match:
Similarity:
```

---

#  Complete System Flow

```text
                    USER IMAGE
                        │
                        ▼
                ┌───────────────┐
                │    Pillow     │
                │ Image Loading │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ CLIPProcessor │
                └───────┬───────┘
                        │
                        ▼
                 PyTorch Tensor
                        │
                        ▼
                ┌───────────────┐
                │ CLIP ViT-B/32 │
                └───────┬───────┘
                        │
                        ▼
                  512-D Vector
                        │
                        ▼
                  Normalization
                        │
                        ▼
              Reference Embedding
                        │
                        ▼
        ┌─────────────────────────────┐
        │ Candidate Embedding Cache   │
        └──────────────┬──────────────┘
                       │
                       ▼
              Cosine Similarity
                       │
                       ▼
                  Score Ranking
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Best Match            Top-K
             │                   │
             └─────────┬─────────┘
                       ▼
                  Final Result
```

---

#  Project Goals

The project demonstrates how a pretrained deep-learning model can be used as a **feature extractor** for image retrieval without training a new image-classification model.

The main concepts demonstrated are:

* Pretrained model inference
* CLIP image feature extraction
* Image embeddings
* Vector normalization
* Cosine similarity
* Similarity ranking
* Top-K retrieval
* Embedding caching
* Image robustness testing
* Similarity thresholding
* FastAPI deployment

---

#  Possible Future Improvements

Potential extensions include:

* FAISS-based vector search for larger image collections
* Vector database integration
* GPU acceleration
* Batch embedding generation
* More efficient similarity search
* Better threshold calibration using a labeled validation dataset
* No-match detection based on evaluated thresholds
* More image preprocessing options
* Frontend integration
* Larger-scale image retrieval

---

#  Author

**Varad Nangare**

GitHub:
https://github.com/VARAD87

---

##  Project Summary

This project demonstrates an end-to-end image similarity pipeline:

```text
Image
 ↓
Pretrained CLIP
 ↓
512-D Embedding
 ↓
Normalization
 ↓
Cosine Similarity
 ↓
Ranking
 ↓
Best Match / Top-K Results
 ↓
FastAPI
```

The project combines **deep-learning-based feature extraction** with **vector similarity search** to build a practical image retrieval system.
