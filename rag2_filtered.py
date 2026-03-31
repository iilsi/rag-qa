import os
from dotenv import load_dotenv
import glob
import numpy as np
import faiss
from openai import OpenAI
import PyPDF2
from dataclasses import dataclass
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pdfs")
pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))

documents = []
for pdf_path in pdf_paths:
    file_name = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i in range(len(reader.pages)):
            text = reader.pages[i].extract_text() or ""
            clean = text.replace("\n", " ").strip()
            if not clean:
                continue
            documents.append({"text": clean, "page": i + 1, "source": file_name})

    
def get_embeddings(texts, model_name="text-embedding-3-small"):
    resp = client.embeddings.create(
        input=texts,
        model=model_name
    )

    vectors = [np.array(item.embedding, dtype='float32') for item in resp.data]
    return np.vstack(vectors)

# one vector per page of PDF text
texts = [d['text'] for d in documents]

# generate embeddings
embeddings = get_embeddings(texts)

# norm
emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# FAISS index
dim = emb_norm.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(emb_norm)

@dataclass
class SimplePromptTemplate:
    template: str
    variables: list

    def format(self, **kwargs):
        for v in self.variables:
            if v not in kwargs:
                raise ValueError(f"Missing variable: {v}")
        return self.template.format(**kwargs)

rag_template = SimplePromptTemplate(
    template="""
        You are strict and only answer based on the "knowledge" below. If the answer is not in the knowledge, say "I don't know."

        [Context]:
        {context}

        [Question]:
        {question}

        Give a concise and accurate answer:
        """.strip(),
    variables=["context", "question"])

def basic_retrieve(query, top_k=3):
    q_emb = get_embeddings([query])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        doc = documents[idx]
        results.append({
            "score": float(score),
            "text": doc["text"],
            "page": doc["page"],
            "source": doc["source"],
        })
    return results


def filtered_retrieve(query, top_k=3, page_range=None, source=None):
    candidates = basic_retrieve(query, top_k=top_k * 5)

    results = []
    for r in candidates:
        # filter by page range
        if page_range is not None:
            min_page, max_page = page_range
            if r['page'] < min_page or r['page'] > max_page:
                continue
        # filter by PDF file name
        if source is not None:
            if r['source'] != source:
              continue

        results.append(r)
        if len(results) >= top_k:
           break
    return results

def rag_answer_filtered(query, top_k=3, model_name="gpt-4.1-mini", page_range=None, source=None, verbose=False):
    retrieved = filtered_retrieve(
        query,
        top_k=top_k,
        page_range=page_range,
        source=source
    )

    context_lines = []
    for r in retrieved:
        prefix = f"[{r['source']} - page {r['page']}]"
        context_lines.append(f"- {prefix} {r['text']}")
    context = "\n".join(context_lines)

    if verbose:
        print('[Retrieved document chunks]')
        print(context)
        print("-" * 60)

    prompt = rag_template.format(context=context, question=query)
    # print(prompt)

    # call LLM
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip(), retrieved

q= "how many colleges are there in UOC?"
answer, _ = rag_answer_filtered(
    query=q,
    top_k=3,
    source="uoc_2024_ugp_welcome.pdf"
)
print("question:",q, "answer:",answer)