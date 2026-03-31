import os
from dotenv import load_dotenv
import numpy as np
import faiss
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return np.array([embedding.embedding for embedding in response.data])

chunks = [
    "LangChain is a framework designed to help developers build applications with LLMs.",
    "FAISS is a vector database developed by Meta AI for efficient similarity search.",
    "Retrieval Augmented Generation combines external knowledge with LLM outputs.",
    "Python is a popular programming language for AI development.",
]


class RAGQASystem:
    def __init__(self, texts, client, embed_model='text-embedding-3-small'):
        self.texts = texts
        self.client = client
        self.embed_model = embed_model

        # 1. embedding + normalization
        embs = self._get_embeddings(texts)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        # 2. FAISS create index
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs)

    def _get_embeddings(self, texts):
        resp = self.client.embeddings.create(
            input=texts,
            model=self.embed_model
        )
        vectors = [np.array(item.embedding, dtype='float32') for item in resp.data]
        return np.vstack(vectors)
    
    def search(self, query, top_k=2):
        q_emb = self._get_embeddings([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        D, I = self.index.search(q_emb, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            results.append({
                "rank": rank,
                "score": float(score),
                "text": self.texts[idx]
            })
        return results

    def answer(self, query, top_k=2, model='gpt-4.1-mini'):
        retrieved = self.search(query, top_k = top_k)
        context = "\n".join(f"- {item['text']}" for item in retrieved)

        prompt = f"""
            You are strict and only answer based on the "Knowledge" below. 
            If the answer is not in the "Knowledge", please reply "I don't know."

            "Knowledge":
            {context}

            Question: 
            {query}

            Give a concise and accurate answer:
            """

        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content.strip(), retrieved


if __name__ == "__main__":
    rag = RAGQASystem(chunks, client)
    answer, retrieved = rag.answer("What is my FAISS?")
    print("Answer:", answer)
    print("Retrieved:")
    for r in retrieved:
        print(f"  [{r['rank']}] score={r['score']:.3f} | {r['text']}")
