# Embedding analysis utilities
# analysis/embeddings.py
# Low-level embedding engine. Converts text to vectors and computes
# cosine similarity between them. Knows nothing about failure types,
# detection stages, or what the scores mean -- that's the detector's job.
# Uses sentence-transformers all-MiniLM-L6-v2: small, free, runs locally,
# no API calls needed.

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EmbeddingMatcher:
    """Handles all embedding and similarity operations.
    One instance is created and reused -- loading the model is expensive,
    so we don't reload it on every call."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load the embedding model once at initialization.
        
        Args:
            model_name: sentence-transformers model to use.
                        all-MiniLM-L6-v2 is the default -- small (80MB),
                        fast, good enough for semantic similarity tasks.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Convert a single text string into a vector.
        
        Args:
            text: any string -- task, thought, tool description, etc.
        
        Returns:
            numpy array (the vector representation of the text)
        """
        return self.model.encode(text, convert_to_numpy=True)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two texts.
        
        Returns a score between 0 and 1:
            1.0 = identical meaning
            0.0 = completely unrelated
        
        Args:
            text_a: first text
            text_b: second text
        
        Returns:
            float between 0 and 1
        """
        vec_a = self.embed(text_a).reshape(1, -1)
        vec_b = self.embed(text_b).reshape(1, -1)
        score = cosine_similarity(vec_a, vec_b)[0][0]
        return float(score)

    def rank_by_similarity(
        self, query: str, candidates: list[str]
    ) -> list[dict]:
        """Rank a list of candidate texts by similarity to a query.
        Used for tool ranking in Wrong Tool Selected (Stage 2):
        embed task against all tool descriptions, rank by score.
        
        Args:
            query: the reference text (e.g. task)
            candidates: list of texts to rank (e.g. tool descriptions)
        
        Returns:
            list of dicts, sorted by score descending:
            [
                {'text': '...', 'index': 0, 'score': 0.91},
                {'text': '...', 'index': 2, 'score': 0.43},
                ...
            ]
        """
        query_vec = self.embed(query).reshape(1, -1)
        candidate_vecs = np.array([self.embed(c) for c in candidates])

        scores = cosine_similarity(query_vec, candidate_vecs)[0]

        ranked = [
            {"text": candidates[i], "index": i, "score": float(scores[i])}
            for i in range(len(candidates))
        ]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def similarity_gap(self, ranked_results: list[dict]) -> float:
        """Compute the gap between rank-1 and rank-2 similarity scores.
        Used in Wrong Tool Selected to decide if the called tool is
        'clearly wrong' vs 'close enough to rank-1'.
        
        A large gap means rank-1 is clearly the best choice.
        A small gap means rank-1 and rank-2 are equivalent (don't flag).
        
        Args:
            ranked_results: output of rank_by_similarity()
        
        Returns:
            float gap score, or 0.0 if fewer than 2 candidates
        """
        if len(ranked_results) < 2:
            return 0.0
        return ranked_results[0]["score"] - ranked_results[1]["score"]