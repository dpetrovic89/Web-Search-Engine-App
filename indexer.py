import os
import json
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class Indexer:
    def __init__(self, data_dir="data", index_dir="index"):
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.whoosh_dir = os.path.join(index_dir, "whoosh")
        self.tfidf_path = os.path.join(index_dir, "tfidf_model.pkl")
        self.vectors_path = os.path.join(index_dir, "vectors.npy")
        self.meta_path = os.path.join(index_dir, "metadata.pkl")
        
        # Whoosh Schema
        self.schema = Schema(
            url=ID(stored=True, unique=True),
            title=TEXT(stored=True),
            content=TEXT(stored=True)
        )

        if not os.path.exists(self.whoosh_dir):
            os.makedirs(self.whoosh_dir)

    def build_inverted_index(self):
        print("Building inverted index...")
        ix = create_in(self.whoosh_dir, self.schema)
        writer = ix.writer()
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        for filename in files:
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                writer.add_document(
                    url=data['url'],
                    title=data['title'],
                    content=data['content']
                )
        writer.commit()
        print(f"Inverted index built with {len(files)} documents.")

    def build_vector_index(self):
        print("Building vector index (using TF-IDF)...")
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        documents = []
        urls = []
        titles = []
        
        for filename in files:
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.append(data['content'])
                urls.append(data['url'])
                titles.append(data['title'])
        
        if not documents:
            print("No documents found to index.")
            return

        # Use TfidfVectorizer as a lightweight alternative to neural embeddings
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Save vectorizer and matrix
        with open(self.tfidf_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        np.save(self.vectors_path, tfidf_matrix.toarray())
        
        # Save metadata
        metadata = {
            "urls": urls,
            "titles": titles
        }
        with open(self.meta_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Vector index built with {len(documents)} vectors.")

    def run_all(self):
        self.build_inverted_index()
        self.build_vector_index()

if __name__ == "__main__":
    indexer = Indexer()
    indexer.run_all()
