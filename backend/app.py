import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import requests

from .rag_engine import initialize_index, search
from .safety import check_safety, DISCLAIMER, REFUSAL

# --- Config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"  # pick a Groq-supported free model

app = FastAPI(title="Cora - Marine Research Assistant")

# Serve frontend from /public
app.mount("/", StaticFiles(directory="public", html=True), name="static")

# Init RAG on startup
initialize_index()

SYSTEM_PROMPT = (
    "You are Cora, a marine genomics research assistant and advanced dna splice and lab expert. "
    "You ONLY provide research summaries grounded in retrieved text, "
    "cite sources (title/species/link), and include the disclaimer. "
    "You refuse lab protocols."
    "You can assist in ideas for mixing marine biology and human medicine or human advancements."
    "You only give real true information, no hallucinations or false data is ever allowed."
)

@app.post("/api/chat")
async def chat(req: Request):
    data = await req.json()
    user_text = (data.get("message") or "").strip()

    if not user_text:
        return JSONResponse({"reply": "Please enter a question."})

    if check_safety(user_text):
        return JSONResponse({"reply": f"{REFUSAL}\n\n{DISCLAIMER}"})

    # Retrieve supporting passages
    results = search(user_text, k=6)
    evidence_block = "\n".join([
        f"- {r['meta'].get('title','')} [{r['meta'].get('species','N/A')}] "
        f"— {r['meta'].get('link','')}"
        for r in results
    ])
    context = "\n\nEVIDENCE:\n" + (evidence_block if evidence_block else "No direct matches; answer with caution.")

    # Compose prompt to Groq
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_text}\n\n{context}\n\nRemember: {DISCLAIMER}"},
    ]

    if not GROQ_API_KEY:
        # Offline fallback (useful for local dev or missing key)
        fallback = (
            "Cora (offline mode): I cannot contact the LLM right now. "
            "Here are the closest items I found:\n" + evidence_block + "\n\n" + DISCLAIMER
        )
        return JSONResponse({"reply": fallback})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}

    r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    if r.status_code >= 300:
        return JSONResponse({"reply": f"LLM error: {r.text}\n\n{DISCLAIMER}"})

    data = r.json()
    reply = data["choices"][0]["message"]["content"]
    # Append disclaimer to every reply
    reply = f"{reply}\n\n—\n{DISCLAIMER}"
    return JSONResponse({"reply": reply})
