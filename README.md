---
title: Modern Search Engine
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_file: app.py
---

# Dejan Petrovic Search Engine

A modular web search engine prototype featuring a multi-stage retrieval and ranking pipeline.

<img width="1909" height="919" alt="Web Search App" src="https://github.com/user-attachments/assets/f1cbffca-5d59-4a26-87fb-c72ae74becd3" />

## 🚀 Overview

This project implements a complete search engine pipeline, from web crawling to multi-stage ranking. It is designed to be lightweight, performant, and easy to deploy on platforms like Hugging Face Spaces.

### 🌟 Key Features
- **Hybrid Retrieval**: Combines keyword-based (BM25) and similarity-based (TF-IDF) retrieval.
- **Multi-stage Ranking**: Uses feature extraction (title match, phrase match, etc.) for final result refinement.
- **Modern UI**: A premium, dark-mode search interface with glassmorphism aesthetics.
- **Lightning Fast**: Total end-to-end latency is generally under 50ms.

## 🏗️ System Architecture

The engine follows a modern search pipeline architecture. For a deep dive into the technical details, components, and data flow, see our dedicated documentation:

👉 **[Read the Full Architecture Guide](architecture.md)**

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Search Core**: Whoosh (BM25), Scikit-Learn (TF-IDF)
- **Natural Language**: NLTK (Stemming & Tokenization)
- **Frontend**: Vanilla HTML5, CSS3, ES6+ Javascript
- **Deployment**: Docker, GitHub Actions (CI/CD)

## 📂 Project Structure

- `crawler.py`: Recursive web crawler and HTML parser.
- `indexer.py`: Builds inverted and vector indexes.
- `retrieval.py`: Implements hybrid search and RRF fusion.
- `ranker.py`: Feature-based re-ranking logic.
- `app.py`: FastAPI server and API endpoints.

## 🏁 Getting Started

### 1. Requirements
- Python 3.11+
- [Optional] Docker

### 2. Installation
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage
```powershell
# Step 1: Pre-build the search index (Optional)
python crawler.py && python indexer.py

# Step 2: Start the search engine
python app.py
```
Open current directory's `index.html` via the server at `http://localhost:8000`.

## 📈 Performance
- **Retrieval**: ~45ms
- **Ranking**: < 1ms
- **Total Latency**: ~50ms
