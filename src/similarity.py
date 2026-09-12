import numpy as np
def cosine_similarity(embedding_a, embedding_b):
  
    return float(np.dot(embedding_a, embedding_b))
if __name__ == "__main__":
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    d = np.array([-1.0, 0.0])

    print("Identical vectors:", cosine_similarity(a, b))       # expect 1.0
    print("Perpendicular vectors:", cosine_similarity(a, c))   # expect 0.0
    print("Opposite vectors:", cosine_similarity(a, d))        # expect -1.0


def find_top_matches(reference_embedding, candidate_embeddings, top_k=3):
  
    results = []
    for filename, embedding in candidate_embeddings.items():
        score = cosine_similarity(reference_embedding, embedding)
        results.append((filename, score))


    results.sort(key=lambda item: item[1], reverse=True)

  
    return results[:top_k]


def find_best_match(reference_embedding, candidate_embeddings):
 
    top_matches = find_top_matches(reference_embedding, candidate_embeddings, top_k=1)
    return top_matches[0]
def find_best_match(reference_embedding, candidate_embeddings, threshold=0.6):
    
    top_matches = find_top_matches(reference_embedding, candidate_embeddings, top_k=1)
    filename, score = top_matches[0]

    is_confident = score >= threshold
    return filename, score, is_confident
