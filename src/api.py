
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

app = FastAPI()


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

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File must be a .jpg, .jpeg, or .png image.")


    with open(TEMP_UPLOAD_PATH, "wb") as buffer:
       
        shutil.copyfileobj(file.file, buffer)

    try:
        # Reuse all our existing logic - nothing new here.
        reference_embedding = embed_image(TEMP_UPLOAD_PATH)
        candidate_embeddings = build_embeddings()

        best_filename, best_score, is_confident = find_best_match(
            reference_embedding, candidate_embeddings
        )
        top_matches = find_top_matches(reference_embedding, candidate_embeddings, top_k=3)

    
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

        if os.path.exists(TEMP_UPLOAD_PATH):
            os.remove(TEMP_UPLOAD_PATH)


@app.get("/")
async def root():
    """A simple health-check route to confirm the API is running."""
    return {"status": "Image Similarity API is running."}
