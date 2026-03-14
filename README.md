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

---

## How to Run

**Requirements:**

- Python 3.9+ (recommended).


