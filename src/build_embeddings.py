
import os
import pickle

from embedding_utils import embed_image

IMAGES_FOLDER = "images"
CACHE_FILE = "embeddings/embeddings_cache.pkl"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_cache():
    
    if os.path.exists(CACHE_FILE):
     
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(embeddings):
 
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(embeddings, f)


def build_embeddings():
    """
    Returns a dictionary mapping filename -> embedding for every image
    in IMAGES_FOLDER. Reuses cached embeddings when available, and only
    computes embeddings for images that are new since the last run.
    """
    filenames = [
        f for f in os.listdir(IMAGES_FOLDER) 
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

   
    embeddings = load_cache()

    new_filenames = [f for f in filenames if f not in embeddings]

    if not new_filenames:
        print(f"All {len(filenames)} images already cached. Nothing to compute.")
        return embeddings

    print(f"Found {len(new_filenames)} new image(s) to process "
          f"({len(filenames) - len(new_filenames)} already cached).")

    total = len(new_filenames)
    for index, filename in enumerate(new_filenames, start=1):
        image_path = os.path.join(IMAGES_FOLDER, filename)
        print(f"Processing {index}/{total}: {filename}")
        embeddings[filename] = embed_image(image_path)

    embeddings = {f: v for f, v in embeddings.items() if f in filenames}

    save_cache(embeddings)
    print("Finished. Cache updated.")

    return embeddings


# Quick manual test - only runs if you execute this file directly.
if __name__ == "__main__":
    result = build_embeddings()
    print(f"\nTotal embeddings available: {len(result)}")
