from embedding_utils import embed_image
from similarity import cosine_similarity
import os

SOURCE_IMAGE = "images/Andrew-Garfield-Award-Winning-Actor-PNG.png"
VARIATIONS_FOLDER = "test_variations"


def main():
    # Embed the original once.
    original_embedding = embed_image(SOURCE_IMAGE)

    # Embed and compare every variation against the original.
    variations = os.listdir(VARIATIONS_FOLDER)

    print(f"Comparing variations against: {SOURCE_IMAGE}\n")

    for filename in variations:
        path = os.path.join(VARIATIONS_FOLDER, filename)
        variation_embedding = embed_image(path)
        score = cosine_similarity(original_embedding, variation_embedding)
        print(f"{filename:20s}  similarity = {score:.4f}")


if __name__ == "__main__":
    main()