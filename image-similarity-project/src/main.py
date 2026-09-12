from embedding_utils import embed_image
from build_embeddings import build_embeddings
from similarity import find_best_match, find_top_matches

REFERENCE_IMAGE = "reference/reference.png"


def main():
    print(f"Embedding reference image: {REFERENCE_IMAGE}")
    reference_embedding = embed_image(REFERENCE_IMAGE)

    print("Loading candidate embeddings...")
    candidate_embeddings = build_embeddings()

    best_filename, best_score, is_confident = find_best_match(reference_embedding, candidate_embeddings)

    if is_confident:
        print(f"\nConfident Match: {best_filename}")
    else:
        print(f"\nNo confident match found. Closest option: {best_filename}")

    print(f"Similarity: {best_score:.2f}")
    # Get the top 3 matches.
    top_matches = find_top_matches(reference_embedding, candidate_embeddings, top_k=3)

    print("\nTop Matches:\n")
    for rank, (filename, score) in enumerate(top_matches, start=1):
        print(f"{rank}. {filename}   {score:.2f}")


if __name__ == "__main__":
    main()
