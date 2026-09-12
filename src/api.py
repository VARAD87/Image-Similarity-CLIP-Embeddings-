# Make sure Python can find sibling modules (embedding_utils, build_embeddings,
# similarity) regardless of how this file was launched (directly vs. via
# uvicorn importing it as part of the "src" package).
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil

from embedding_utils import embed_image
from build_embeddings import build_embeddings
from similarity import find_best_match, find_top_matches
from fastapi.staticfiles import StaticFiles
# This creates the actual FastAPI application object.
app = FastAPI()

# Enable CORS so a browser-based frontend (Angular, running on a
# different port) is allowed to call this API. In a real production
# app you'd restrict allow_origins to your actual frontend's URL
# instead of "*" (which means "allow any origin").
app.mount("/images", StaticFiles(directory="images"), name="images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_PATH = "temp_upload.jpg"


@app.post("/match")
async def match_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded reference image, compares it against all
    candidate images, and returns the best match plus top 5 matches.
    """
    # Basic validation: reject anything that isn't clearly an image file.
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File must be a .jpg, .jpeg, or .png image.")

    # Save the uploaded file to disk temporarily, since embed_image()
    # works with file paths. "wb" = write binary mode.
    with open(TEMP_UPLOAD_PATH, "wb") as buffer:
        # shutil.copyfileobj() efficiently copies the uploaded file's
        # contents into our temporary file.
        shutil.copyfileobj(file.file, buffer)

    try:
        # Reuse all our existing logic - nothing new here.
        reference_embedding = embed_image(TEMP_UPLOAD_PATH)
        candidate_embeddings = build_embeddings()

        best_filename, best_score, is_confident = find_best_match(
            reference_embedding, candidate_embeddings
        )
        top_matches = find_top_matches(reference_embedding, candidate_embeddings, top_k=3)

        # Build the JSON-friendly response. FastAPI automatically
        # converts this Python dictionary into a JSON response.
        return {
            "best_match": {
                "filename": best_filename,
                "similarity": round(best_score, 4),
                "is_confident": is_confident,
            },
            "top_matches": [
                {"filename": f, "similarity": round(s, 4)}
                for f, s in top_matches
            ],
        }
    finally:
        # Clean up the temporary file whether or not an error occurred.
        if os.path.exists(TEMP_UPLOAD_PATH):
            os.remove(TEMP_UPLOAD_PATH)


@app.get("/")
async def root():
    """A simple health-check route to confirm the API is running."""
    return {"status": "Image Similarity API is running."}