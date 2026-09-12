# numpy gives us fast vector math - dot products, etc.
import numpy as np


def cosine_similarity(embedding_a, embedding_b):
    """
    Computes cosine similarity between two embeddings.

    Since our embeddings are already normalized to unit length
    (see embed_image() in Stage 5), cosine similarity simplifies
    to a plain dot product.
    """
    # np.dot() multiplies matching positions and sums the result -
    # exactly the dot product formula described above.
    return float(np.dot(embedding_a, embedding_b))


# Quick manual test - only runs if you execute this file directly.
if __name__ == "__main__":
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    d = np.array([-1.0, 0.0])

    print("Identical vectors:", cosine_similarity(a, b))       # expect 1.0
    print("Perpendicular vectors:", cosine_similarity(a, c))   # expect 0.0
    print("Opposite vectors:", cosine_similarity(a, d))        # expect -1.0


def find_top_matches(reference_embedding, candidate_embeddings, top_k=3):
    """
    Compares reference_embedding against every embedding in
    candidate_embeddings (a dict of filename -> embedding), and returns
    the top_k highest-scoring matches as a list of (filename, score)
    tuples, sorted highest score first.
    """
    results = []
    for filename, embedding in candidate_embeddings.items():
        score = cosine_similarity(reference_embedding, embedding)
        results.append((filename, score))

    # Sort by score descending (highest similarity first).
    results.sort(key=lambda item: item[1], reverse=True)

    # Slice the list to only keep the first top_k entries.
    return results[:top_k]


def find_best_match(reference_embedding, candidate_embeddings):
    """
    Returns just the single best match as a (filename, score) tuple.
    """
    # Reuse find_top_matches() and just take the first (highest) result.
    top_matches = find_top_matches(reference_embedding, candidate_embeddings, top_k=1)
    return top_matches[0]
def find_best_match(reference_embedding, candidate_embeddings, threshold=0.6):
    """
    Always returns the closest match. Also returns whether it's
    confident (above threshold) or just the best available guess.
    """
    top_matches = find_top_matches(reference_embedding, candidate_embeddings, top_k=1)
    filename, score = top_matches[0]

    is_confident = score >= threshold
    return filename, score, is_confident