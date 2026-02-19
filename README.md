# Dejan Petrovic Search Engine

A modular web search engine prototype featuring a multi-stage retrieval and ranking pipeline.

## 🚀 Features

- **Multi-lingual Architecture**: Python-based search core with a lightning-fast FastAPI backend.
- **Hybrid Retrieval**: Combines keyword-based (BM25) and similarity-based (TF-IDF) retrieval for improved recall and precision.
- **Multi-stage Ranking**: A sophisticated ranker that extracts features like title matching and document metrics to refine final results.
- **Modern UI**: A premium, dark-mode search interface built with glassmorphism aesthetics.
- **Performance Optimized**: Total search latency under 50ms.

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Search Core**: Whoosh (BM25), Scikit-Learn (TF-IDF)
- **NLP**: NLTK (Stemming & Tokenization)
- **Frontend**: Vanilla HTML5, CSS3, ES6+ Javascript

## 📂 Project Structure

- `crawler.py`: Crawls and parses HTML documents into JSON.
- `indexer.py`: Builds the inverted and vector indexes.
- `retrieval.py`: Implements hybrid search and Reciprocal Rank Fusion (RRF).
- `ranker.py`: Performs feature extraction and final document scoring.
- `app.py`: FastAPI server serving the API and UI.
- `index.html`: Modern search frontend.

## 🏁 Getting Started

### 1. Requirements
- Python 3.11+
- PowerShell (for script execution setup)

### 2. Installation
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install requests beautifulsoup4 whoosh scikit-learn numpy fastapi uvicorn nltk
```

### 3. Usage
```powershell
# Step 1: Initialize data (Optional - pre-crawled data exists)
python crawler.py

# Step 2: Build the search index
python indexer.py

# Step 3: Start the search engine
python app.py
```
Open your browser to `http://localhost:8000`.

## 📈 Performance Summary
- **Retrieval**: < 50ms
- **Ranking**: < 1ms
- **Total Latency**: ~50ms
- **Memory Footprint**: Lightweight (optimized for local prototypes)
