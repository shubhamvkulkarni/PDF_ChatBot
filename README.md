# PDF_ChatBot

A simple PDF-based chatbot using **sentence-transformers** for embeddings, **FAISS** for retrieval, and **Ollama** for local LLM generation.

---

## Requirements

- macOS or Linux  
- Conda / Miniforge  
- Python **3.11**  
- Ollama installed and running locally  

---

## Instructions
### 1. Create and activate a clean environment

We recommend using **Miniforge / conda** and Python 3.11 for maximum compatibility.

```bash
conda create -n pdfchat python=3.11 -y
conda activate pdfchat
```
Verify:

```bash
python --version
```
### 2. Install native dependencies (via conda)

FAISS should be installed via conda-forge, not pip.
```bash
conda install -c conda-forge faiss-cpu -y
```
Optional sanity check:
```bash
python -c "import faiss; print('FAISS OK')"
```
### 3. Install Python packages (via pip)

Install the remaining Python-only dependencies using pip inside the conda environment:
```bash
python -m pip install --upgrade pip

python -m pip install \
  langchain \
  langchain-community \
  langchain-ollama \
  langchain-text-splitters \
  sentence-transformers \
  pypdf
```
### 4. Install and prepare Ollama

Install Ollama from:
```bash
https://ollama.com
```

Verify installation:
```bash
ollama --version
```

Pull a model for generation (example):
```bash
ollama pull deepseek-r1:8b
```

Check available models:
```bash
ollama list
```

Make sure the Ollama daemon is running before starting the chatbot.

### 5. Embeddings cache location

Sentence-transformers models are downloaded once and cached locally.

This project uses a hardcoded cache directory:
```bash
~/data/hf_cache
```

The directory will be created automatically on the first run.
Models are not re-downloaded on subsequent runs.

### 6. Project structure

Expected file layout:
```bash
pdf_chatbot/
├── pdf_chatbot.py        # main entry point
├── core.py               # PDF loading, chunking, FAISS logic
├── embeddings.py         # sentence-transformers embeddings
├── README.md
```
### 7. Running the chatbot

From the project directory:
```bash
python pdf_chatbot.py /path/to/your/document.pdf
```

Example:
```bash
python pdf_chatbot.py papers/example.pdf
```

Once loaded, you will enter an interactive prompt:
```bash
PDF loaded. Ask questions (type 'bye' to exit).

Question:
```

Type natural-language questions about the PDF content.

### 8. Re-indexing (optional)

If you modify the PDF or want to rebuild the vector index:
```bash
python pdf_chatbot.py /path/to/document.pdf --reindex
```
