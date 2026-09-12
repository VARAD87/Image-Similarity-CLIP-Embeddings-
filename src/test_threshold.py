from embedding_utils import embed_image
from build_embeddings import build_embeddings
from similarity import find_best_match

TEST_IMAGE = "reference/thomas_shellby.png"  

def main():
    reference_embedding = embed_image(TEST_IMAGE)
    candidate_embeddings = build_embeddings()

    filename, score = find_best_match(reference_embedding, candidate_embeddings)
    print(f"Testing: {TEST_IMAGE}")
    print(f"Best Match: {filename}")
    print(f"Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
