# os.path.join() builds file paths correctly (works on Windows and Mac/Linux).
# os.path.exists() checks if a file already exists on disk.
import os

# pickle lets us save any Python object (like our embeddings dictionary)
# to a file, and load it back later exactly as it was.
import pickle

# Our reusable embedding function from Stage 5.
from embedding_utils import embed_image

IMAGES_FOLDER = "images"
CACHE_FILE = "embeddings/embeddings_cache.pkl"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_cache():
    """
    Loads previously saved embeddings from disk, if the cache file exists.
    Returns an empty dictionary if no cache exists yet.
    """
    if os.path.exists(CACHE_FILE):
        # "rb" = read binary mode, required for pickle files.
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(embeddings):
    """
    Saves the embeddings dictionary to disk so future runs can reuse it
    instead of recomputing everything.
    """
    # "wb" = write binary mode, required for pickle files.
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(embeddings, f)


def build_embeddings():
    """
    Returns a dictionary mapping filename -> embedding for every image
    in IMAGES_FOLDER. Reuses cached embeddings when available, and only
    computes embeddings for images that are new since the last run.
    """
    filenames = [
        f for f in os.listdir(IMAGES_FOLDER) # os.listdir() lets us see all filenames inside a folder.
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    # Load whatever we've already computed in previous runs.
    embeddings = load_cache()

    # Figure out which images are NOT already in our cache.
    new_filenames = [f for f in filenames if f not in embeddings]

    if not new_filenames:
        print(f"All {len(filenames)} images already cached. Nothing to compute.")
        return embeddings

    print(f"Found {len(new_filenames)} new image(s) to process "
          f"({len(filenames) - len(new_filenames)} already cached).")

    total = len(new_filenames)
    for index, filename in enumerate(new_filenames, start=1):
        image_path = os.path.join(IMAGES_FOLDER, filename)# os.path.join() builds file paths correctly (works on Windows and Mac/Linux).
        print(f"Processing {index}/{total}: {filename}")
        embeddings[filename] = embed_image(image_path)

    # Also remove entries for images that no longer exist in the folder
    # (e.g. if you deleted a file since the last run).
    embeddings = {f: v for f, v in embeddings.items() if f in filenames}

    save_cache(embeddings)
    print("Finished. Cache updated.")

    return embeddings


# Quick manual test - only runs if you execute this file directly.
if __name__ == "__main__":
    result = build_embeddings()
    print(f"\nTotal embeddings available: {len(result)}")