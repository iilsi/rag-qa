import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
import glob
import numpy as np
import faiss
import PyPDF2

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ConversationMemory:
    def __init__(self, max_turn: int = 5):
        self.max_turn = max_turn
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_recent_message(self) -> List[Dict[str, str]]:
        max_len = self.max_turn * 2
        return self.messages[-max_len:]

    def clear(self):
        self.messages = []

class ChatAgent:
    def __init__(self, client: OpenAI, system_prompt: str=None, max_turns: int=5):
        self.client = client
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant."
            "you should answer in a concise and clear way."
            "if you're not sure, say 'I don't know'."
        )
        self.memory = ConversationMemory(max_turn=max_turns)

    def chat(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self.memory.get_recent_message()

        completion = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        answer = completion.choices[0].message.content.strip()

        self.memory.add_assistant_message(answer)
        return answer

    def reset(self):
        self.memory.clear()

agent = ChatAgent(client, max_turns=5)

# --- Uncomment to run basic chat demo (no RAG) ---
# print("Agent starts. Input content to start, and input 'exit' to end.")
# while True:
#     user_input = input("User: ")
#     if user_input.strip().lower() in ["exit", "quit"]:
#         print("Chat ends.")
#         break
#     reply = agent.chat(user_input)
#     print("Agent: ", reply)

def summarize_history(history_messages: List[Dict[str, str]]) -> str:
    history_text = ""
    for m in history_messages:
        role = "user" if m["role"]=="user" else "assistant"
        history_text += f"[{role}]: {m['content']}\n"

    prompt = f"""
        Read the conversation below and summarize the key background information in a few sentences.
        Keep only useful long-term information (e.g. user identity, preferences, task goals).
        Do not repeat the entire dialogue.

        Conversation:
        {history_text}

        Summary:
    """

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages = [{"role": "user", "content": prompt}]
    )
    summary = completion.choices[0].message.content.strip()

    return summary

class SummarizingChatAgent:
    def __init__(self, client: OpenAI, system_prompt: str=None, max_turns: int=5):
        self.client = client
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant for multi-turn conversation."
            "you have access to a summary of past conversations and recent message."
        )
        self.memory = ConversationMemory(max_turn=max_turns)
        self.summary: str = ""

    def chat(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)

        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"chat history summary: {self.summary}"})

        messages += self.memory.get_recent_message()

        completion = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        answer = completion.choices[0].message.content.strip()

        self.memory.add_assistant_message(answer)
        return answer

    def summarize_and_compress(self):
        history = self.memory.messages
        if not history:
            return

        new_summary = summarize_history(history)
        if self.summary:
            combined = f"old summary: {self.summary}\nnew summary: {new_summary}"
            self.summary = summarize_history([
                {"role": "user", "content": combined}
            ])
        else:
            self.summary = new_summary
        self.memory.clear()

    def reset(self):
        self.memory.clear()
        self.summary = ""

agent2 = SummarizingChatAgent(client, max_turns=5)

# --- Uncomment to run summarization chat demo (no RAG) ---
# while True:
#     user_input = input("User: ")
#     if user_input.strip().lower() in ["exit", "quit"]:
#         print("Chat ends.")
#         break
#     if user_input.strip() == '/sum':
#         agent2.summarize_and_compress()
#         print("Summary: ", agent2.summary)
#         continue
#     reply = agent2.chat(user_input)
#     print("Agent: ", reply)

class RAGTool:
    def __init__(self, rag_system):
        self.rag = rag_system

    def run(self, query):
        answer, ctx = self.rag.answer(query, top_k=4)
        return answer

class ToolCallingAgent:
    def __init__(self, client, rag_tool=None, max_turns=5):
        self.client = client
        self.rag_tool = rag_tool
        self.memory = ConversationMemory(max_turn=max_turns)

        self.system_prompt = """
            You are a helpful AI assistant with two modes:
              1. General chat (casual talk, explaining concepts, etc.)
              2. Document retrieval (when the user's question requires looking up the PDF knowledge base)

            If the user's question is about specific knowledge or document content, use the RAG tool.
            If the answer is not in the knowledge base, reply honestly with "I don't know."
            """

    def detect_rag_intent(self, user_input: str) -> bool:
        keywords = ["pdf", "document", "content", "introduce", "resume", "definition", "chapter", "how to use", "what is"]
        return any(k.lower() in user_input.lower() for k in keywords)

    def chat(self, user_input: str):
        self.memory.add_user_message(user_input)

        use_rag = False
        if self.rag_tool is not None:
            use_rag = self.detect_rag_intent(user_input)

        if use_rag:
            rag_answer = self.rag_tool.run(user_input)
            self.memory.add_assistant_message(rag_answer)
            return rag_answer

        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self.memory.get_recent_message()

        completion = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        answer = completion.choices[0].message.content.strip()

        self.memory.add_assistant_message(answer)
        return answer

    def reset(self):
        self.memory.clear()

pdf_dir = "./pdfs"   
pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))

documents = []

for pdf_path in pdf_paths:
    file_name = os.path.basename(pdf_path)
    print(f"\n>>> Processing: {file_name}")
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            if reader.trailer is None or "/Root" not in reader.trailer:
                print(f"  Warning: {file_name} has abnormal structure, skipped")
                continue

            num_pages = len(reader.pages)
            print("  Pages:", num_pages)

            for i in range(num_pages):
                try:
                    page = reader.pages[i]
                    text = page.extract_text()
                except Exception as e:
                    print(f"  Warning: Page {i+1} parse failed: {e}")
                    continue

                if not text:
                    continue

                clean = text.replace("\n", " ").strip()
                if clean:
                    documents.append({
                        "text": clean,
                        "page": i + 1,
                        "source": file_name
                    })

    except Exception as e:
        print(f"  Error: Cannot read {file_name}: {e}")

print(f"\nExtracted {len(documents)} page texts")

def chunk_text(text, size=800, overlap=150):
    out = []
    start = 0
    while start < len(text):
        end = start + size
        out.append(text[start:end])
        start = end - overlap
    return out

all_chunks = []
for doc in documents:
    chunks = chunk_text(doc["text"], size=800, overlap=150)
    for idx, ch in enumerate(chunks):
        all_chunks.append({
            "text": ch,
            "page": doc["page"],
            "source": doc["source"],
            "chunk_id": f"{doc['source']}-p{doc['page']}-c{idx}"
        })

print("Total chunks:", len(all_chunks))
if all_chunks:
    print("Example:", all_chunks[0]["chunk_id"], all_chunks[0]["text"][:200])

def get_embeddings(texts, model_name="text-embedding-3-small"):
    resp = client.embeddings.create(
        model=model_name,
        input=texts
    )
    vectors = [np.array(item.embedding, dtype="float32") for item in resp.data]
    return np.vstack(vectors)
texts = [c["text"] for c in all_chunks]

if not texts:
    print("No text chunks to embed. Skipping FAISS index build.")
    index = None
else:
    embeddings = get_embeddings(texts)
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dim = emb_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_norm)


class RAGQASystem:
    def __init__(self, chunks, client, embed_model="text-embedding-3-small"):
        """
        chunks: List[Dict], each element like:
            {"text": "...", "page": 1, "source": "xxx.pdf", "chunk_id": "xxx-p1-c0"}
        """
        self.chunks = chunks
        self.client = client
        self.embed_model = embed_model

        # --- Build vector index ---
        texts = [c["text"] for c in chunks]
        embs = self._embed_texts(texts)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

    # -----------------------------
    #  Embedding
    # -----------------------------
    def _embed_texts(self, texts):
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=texts
        )
        vectors = [np.array(item.embedding, dtype="float32") for item in resp.data]
        return np.vstack(vectors)

    def _embed_query(self, query):
        emb = self._embed_texts([query])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    # -----------------------------
    #  Retrieve
    # -----------------------------
    def retrieve(self, query, top_k=4):
        q_emb = self._embed_query(query)
        D, I = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            c = self.chunks[idx]
            results.append({
                "score": float(score),
                "text": c["text"],
                "page": c["page"],
                "source": c["source"],
                "chunk_id": c["chunk_id"],
            })
        return results

    # -----------------------------
    #  RAG answer
    # -----------------------------
    def answer(self, query, top_k=4, model="gpt-4.1-mini"):
        retrieved = self.retrieve(query, top_k=top_k)

        # Build context
        context = "\n".join(
            f"[{d['source']} p{d['page']}] {d['text']}" for d in retrieved
        )

        prompt = f"""
You are a precise QA assistant. Below are retrieved snippets from PDF documents. Answer strictly based on these snippets.
If the content does not provide an answer, reply: "I don't know."

[Retrieved content]:
{context}

[Question]:
{query}

Give an accurate answer in English:
"""

        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = completion.choices[0].message.content.strip()

        return answer, retrieved


if not all_chunks:
    print("No chunks from PDFs. Put PDFs in ./pdfs/ and run again.")
else:
    rag_system = RAGQASystem(all_chunks, client)

    query = "What is the main content of this resume?"
    ans, ctx = rag_system.answer(query, top_k=4)
    print("Answer:", ans)
    print("\nContext:")
    for c in ctx:
        print("-", c["source"], "page", c['page'], "score", round(c['score'], 3))

    rag_system = RAGQASystem(all_chunks, client)
    rag_tool = RAGTool(rag_system)
    agent = ToolCallingAgent(client, rag_tool=rag_tool, max_turns=5)

    print("Agent starts. Type 'exit' or 'quit' to end.")
    print("To use RAG (query your PDFs), include words like: pdf, document, content, what is, definition, chapter, resume.")
    print("Otherwise the agent replies in general chat.\n")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        reply = agent.chat(user_input)
        print("Agent: ", reply)
