## A Minimal RAG QA Demo

This repository is a **minimal Retrieval-Augmented Generation (RAG) QA pipeline demo**.  
It is meant to:

- **Show the basic components of RAG**: embeddings, vector search, and LLM answering with retrieved context.
- **Demonstrate a simple document QA system** built on top of local PDF files.
- **Serve as a small, hackable starting point**, not a production-ready system.

---

## Files

- `rag.py`
  - A very small example using a few hard-coded text snippets.
  - Good for understanding the basic flow: **texts → embeddings → FAISS → retrieval → LLM answer**.

- `rag2_filtered.py`
  - Loads PDFs from `pdfs/`, builds a vector index over pages, and exposes:
    - `basic_retrieve`: simple similarity search.
    - `filtered_retrieve`: retrieval with page range / file filters.
    - `rag_answer_filtered`: full RAG flow (retrieve + call LLM).

- `rag3_memory.py`
  - Builds on `rag2_filtered.py`: loads PDFs from `pdfs/`, chunks (size 800, overlap 150), FAISS index.
  - Adds conversation memory (sliding window), optional history summarization (`/sum`), and one agent that switches between general chat and RAG via keyword intent.

---

## How to Run

- Python 3.9+ (recommended).
- `pip install -r requirements.txt`
- Put `OPENAI_API_KEY` in a `.env` file (see `.env.example`).
- Run `python rag.py`, `python rag2_filtered.py`, or `python rag3_memory.py`. For the latter two, put PDFs in `pdfs/`.


