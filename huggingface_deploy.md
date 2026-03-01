# Hugging Face Spaces Deployment Guide

You can easily deploy this search engine to **Hugging Face Spaces** using the **Docker SDK**. This allows you to keep the custom HTML/CSS UI exactly as it is.

## 🚀 Steps to Deploy

### 1. Create a New Space
Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
- **Owner**: Your username
- **Space Name**: (e.g., `modern-search-engine`)
- **License**: Choose your preferred license (e.g., MIT)
- **Select the Space SDK**: Choose **Docker**

### 2. Upload Your Files
Once the Space is created, upload the following files from your project:
- `app.py`
- `crawler.py`
- `indexer.py`
- `retrieval.py`
- `ranker.py`
- `index.html`
- `requirements.txt`
- `Dockerfile` (Newly created)
- `.gitignore`
- `README.md`

> [!TIP]
> **Data and Index**: To avoid re-crawling on every startup, you should also upload the `data/` and `index/` folders. This ensures the app is ready to search immediately after deployment.

### 3. Automatic Deployment
Hugging Face will automatically detect the `Dockerfile`, build the container, and start your app. 
- The app will run on port `7860` as configured in the `Dockerfile`.
- You can access it via the public URL provided by Hugging Face (e.g., `https://huggingface.co/spaces/your-username/modern-search-engine`).

## 🛠️ Configuration Details
- **Port**: The `Dockerfile` sets `ENV PORT=7860`, which `app.py` now respects through an environment variable.
- **Dependencies**: Hugging Face will install all requirements from `requirements.txt`.
- **NLTK Data**: The build process automatically downloads the required `punkt` and `punkt_tab` resources.

## 🤖 Automated Deployment (GitHub Actions)

I have set up a GitHub Action to automatically sync your repository to Hugging Face whenever you push to the `main` branch.

### 1. Create a Hugging Face Token
1. Go to your [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens).
2. Click **"New token"**.
3. Name it (e.g., `GitHub-Sync`) and set role to **Write**.
4. **Copy the token.**

### 2. Add Secret to GitHub
1. Go to your repository on GitHub.
2. Click **Settings** > **Secrets and variables** > **Actions**.
3. Click **"New repository secret"**.
4. **Name**: `HF_TOKEN`
5. **Value**: Paste your Hugging Face token.

### 3. Push to GitHub
Now, every time you run `git push origin main`, GitHub will:
- Authenticate as `executor1389`.
- Push the latest code to `huggingface.co/spaces/executor1389/modern-search-engine`.
- Trigger a rebuild on Hugging Face Spaces.

## 📡 Live Search Tuning
If you want the Space to crawl fresh data, you can create a simple Gradio interface or an API endpoint to trigger `crawler.py` and `indexer.py` remotely, though for a free Space, it's best to upload a pre-built index.
