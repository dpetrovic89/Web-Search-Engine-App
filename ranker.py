import numpy as np
import re

class Ranker:
    """
    Simulates a Stage 3/4 Ranker using feature extraction 
    and a simple weighted scoring model (mimicking LambdaMART/Neural).
    """
    def __init__(self, weights=None):
        if weights is None:
            # Default weights for features
            self.weights = {
                "retrieval_score": 0.4,
                "title_match": 0.3,
                "exact_match": 0.2,
                "length_penalty": 0.1
            }
        else:
            self.weights = weights

    def extract_features(self, query, doc):
        features = {}
        
        # 1. Retrieval Score (already normalized RRF or BM25)
        features["retrieval_score"] = doc.get("score", 0.0)
        
        # 2. Title Match (does the query appear in the title?)
        title = doc.get("title", "").lower()
        query_words = query.lower().split()
        title_matches = sum(1 for word in query_words if word in title)
        features["title_match"] = title_matches / max(len(query_words), 1)
        
        # 3. Exact Phrase Match
        content = doc.get("content", "").lower()
        features["exact_match"] = 1.0 if query.lower() in content else 0.0
        
        # 4. Length Penalty (Prefer shorter, more concise pages for certain queries)
        content_len = len(content)
        # Normalize: 1.0 if < 1000 chars, drops to 0.0 as it approaches 50000
        features["length_penalty"] = max(0.0, 1.0 - (content_len / 50000.0))
        
        return features

    def score(self, query, doc):
        features = self.extract_features(query, doc)
        final_score = 0.0
        for feat, value in features.items():
            final_score += value * self.weights.get(feat, 0.0)
        return final_score

    def rank_results(self, query, results):
        # Add final ranker scores
        for res in results:
            res["rank_score"] = self.score(query, res)
            
        # Re-sort based on rank_score
        ranked = sorted(results, key=lambda x: x["rank_score"], reverse=True)
        return ranked

if __name__ == "__main__":
    ranker = Ranker()
    mock_query = "Python programming"
    mock_results = [
        {"title": "Intro to Python", "url": "url1", "score": 0.5, "content": "Learn python programming today."},
        {"title": "Advanced Python", "url": "url2", "score": 0.4, "content": "Complex coding in Python."},
    ]
    
    ranked = ranker.rank_results(mock_query, mock_results)
    for res in ranked:
        print(f"Title: {res['title']}, Final Score: {res['rank_score']:.4f}")
