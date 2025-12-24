import os
import json
import datetime
import threading
from pathlib import Path

import numpy as np
import gradio as gr
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# ================= ENV =================
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY not found")

os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

# ================= CONFIG =================
DAILY_LIMIT = 50
RATE_FILE = Path("rate_limit.json")
EMBEDDINGS_FILE = Path("embeddings.json")
MAX_HISTORY_TURNS = 3   # keep last 3 Q/A pairs

lock = threading.Lock()

# ================= MODELS =================
embedder = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
    truncate="END"
)

llm = ChatNVIDIA(
    model="mistralai/mixtral-8x22b-instruct-v0.1",
    temperature=0.2
)

# ================= LOAD DOCS =================
with open(EMBEDDINGS_FILE) as f:
    DOCS = json.load(f)

# ================= UTILS =================
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(question, k=4):
    qvec = embedder.embed_query(question)
    scored = [(cosine(qvec, d["embedding"]), d["text"]) for d in DOCS]
    scored.sort(reverse=True)
    return [t for _, t in scored[:k]]

def check_rate_limit():
    today = datetime.date.today().isoformat()
    with lock:
        data = json.loads(RATE_FILE.read_text()) if RATE_FILE.exists() else {}
        if data.get(today, 0) >= DAILY_LIMIT:
            return False
        data[today] = data.get(today, 0) + 1
        RATE_FILE.write_text(json.dumps(data))
    return True

def build_prompt(context, history, question):
    history_text = "\n".join(
        f"User: {q}\nAssistant: {a}"
        for q, a in history[-MAX_HISTORY_TURNS:]
    )

    return f"""
You are a document-grounded assistant.
Answer ONLY using the context.
If the answer is not present, say "I don't know".

Conversation so far:
{history_text}

Context:
{"\n\n---\n\n".join(context)}

User question:
{question}
""".strip()

# ================= CHAT FN (STREAMING) =================
def chat_stream(question, history):
    if not question.strip():
        yield history
        return

    if not check_rate_limit():
        history.append((question, "Daily limit reached (50 queries)."))
        yield history
        return

    context = retrieve(question)
    prompt = build_prompt(context, history, question)

    partial = ""
    for chunk in llm.stream(prompt):
        partial += chunk.content
        yield history + [(question, partial)]

# ================= UI =================
with gr.Blocks(title="Academic Regulations RAG") as demo:
    gr.Markdown("## 📘 Academic Regulations Queries")
    gr.Markdown(
        "Ask questions about the academic regulations document. "
        "Answers are generated **only** from the official document."
    )

    chatbot = gr.Chatbot(height=420)
    
    question = gr.Textbox(
        placeholder="e.g. What is the E grade?",
        label="Your question",
        scale=4
    )
    ask = gr.Button("Ask", scale=1, min_width=100)
    
    clear = gr.Button("Clear Chat")

    ask.click(chat_stream, [question, chatbot], chatbot)

    question.submit(
        chat_stream,
        inputs=[question, chatbot],
        outputs=chatbot
    )

    clear.click(lambda: [], None, chatbot)

# ================= RUN =================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
    )