import numpy as np
def batch_cosine_similarity(query_vec: np.ndarray, doc_matrix:np.ndarray) -> np.ndarray:

    dot_product = np.dot(doc_matrix, query_vec)

    norm_v1 = np.linalg.norm(query_vec)
    norm_v2 = np.linalg.norm(doc_matrix, axis=1)

    denominator = norm_v1*norm_v2

    scores = np.divide(
        dot_product,
        denominator,
        out=np.zeros_like(dot_product, dtype=float),
        where=denominator != 0
    )
    return scores    


query = np.array([0.1, 0.5, 0.8, 0.2])    
docs = np.array([[1,2,3,4],
                [3,4,2,5],
                [9,7,9,1],
                [5,8,9,1],
                [5,0,5,1]])

similarity_scores = batch_cosine_similarity(query, docs)
print("Similarity scores:", similarity_scores)

highest_index = np.argmax(similarity_scores)
print("Index of most similar document:", highest_index)
print("Highest similarity score:", similarity_scores[highest_index])
