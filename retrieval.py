import os
import pickle
import numpy as np
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class QueryProcessor:
    def __init__(self, index_dir="index"):
        self.index_dir = index_dir
        self.whoosh_dir = os.path.join(index_dir, "whoosh")
        self.tfidf_path = os.path.join(index_dir, "tfidf_model.pkl")
        self.vectors_path = os.path.join(index_dir, "vectors.npy")
        self.meta_path = os.path.join(index_dir, "metadata.pkl")
        
        self.stemmer = PorterStemmer()
        
        # Load indexes
        self.ix = open_dir(self.whoosh_dir)
        
        with open(self.tfidf_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        self.tfidf_matrix = np.load(self.vectors_path)
        
        with open(self.meta_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def process_query(self, query):
        tokens = word_tokenize(query.lower())
        stemmed = [self.stemmer.stem(t) for t in tokens]
        return " ".join(stemmed)

    def search_bm25(self, query, limit=10):
        results = []
        with self.ix.searcher() as searcher:
            parser = QueryParser("content", self.ix.schema)
            parsed_query = parser.parse(query)
            whoosh_results = searcher.search(parsed_query, limit=limit)
            for r in whoosh_results:
                results.append({
                    "url": r['url'],
                    "title": r['title'],
                    "score": r.score
                })
        return results

    def search_vector(self, query, limit=10):
        # Transform query to vector
        query_vec = self.vectorizer.transform([query]).toarray()
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "url": self.metadata['urls'][idx],
                    "title": self.metadata['titles'][idx],
                    "score": float(similarities[idx])
                })
        return results

    def reciprocal_rank_fusion(self, bm25_results, vector_results, k=60):
        scores = {}
        
        def update_scores(results):
            for rank, res in enumerate(results):
                url = res['url']
                if url not in scores:
                    scores[url] = {"score": 0.0, "title": res['title']}
                scores[url]["score"] += 1.0 / (k + rank + 1)
        
        update_scores(bm25_results)
        update_scores(vector_results)
        
        # Sort by fused score
        fused = sorted(
            [{"url": url, "title": data["title"], "score": data["score"]} for url, data in scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        return fused

    def hybrid_search(self, query, limit=10):
        processed_query = self.process_query(query)
        print(f"Searching for: {processed_query}")
        
        bm25_res = self.search_bm25(query, limit=limit*2)
        vec_res = self.search_vector(query, limit=limit*2)
        
        fused_res = self.reciprocal_rank_fusion(bm25_res, vec_res)
        return fused_res[:limit]

if __name__ == "__main__":
    qp = QueryProcessor()
    test_query = "Python programming tutorials"
    print("\nHybrid Search Results:")
    for res in qp.hybrid_search(test_query):
        print(f"- {res['title']} ({res['url']}) [Score: {res['score']:.4f}]")
