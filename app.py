
import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="Conversation Persona Bot", layout="wide")

# ---------- cached resource loaders ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_qa_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_data
def load_data():
    with open("outputs/segments_with_summaries.json") as f:
        segments = json.load(f)
    with open("outputs/chunks_with_summaries.json") as f:
        chunks = json.load(f)
    with open("outputs/messages.json") as f:
        messages = json.load(f)
    with open("outputs/persona.json") as f:
        persona = json.load(f)
    return segments, chunks, messages, persona

@st.cache_resource
def build_indexes(_embedder, segments, messages):
    topic_texts = [s["summary"] for s in segments]
    topic_embs = _embedder.encode(topic_texts, show_progress_bar=False)
    W = 5
    windows = []
    for i in range(0, len(messages), W):
        wm = messages[i:i+W]
        text = "\n".join(f"{m['speaker']}: {m['text']}" for m in wm)
        windows.append({"start_idx": i, "end_idx": i+len(wm)-1, "text": text})
    window_embs = _embedder.encode([w["text"] for w in windows], show_progress_bar=False)
    return topic_embs, windows, window_embs

# ---------- routing & generation ----------
PERSONA_KEYWORDS = {
    "personality": ["kind of person", "what is", "what's", "what are they like", "personality", "describe", "who is", "type of person"],
    "habits": ["habit", "habits", "routine", "usually", "every day", "daily", "lifestyle"],
    "communication": ["how do they talk", "talk", "communicate", "tone", "style", "writing"],
    "relationships": ["family", "friend", "relationship", "married", "partner", "kids", "children", "parents"],
    "facts": ["fact", "personal facts", "background"],
}

def route_query(q):
    q = q.lower()
    routes = set()
    for cat, kws in PERSONA_KEYWORDS.items():
        if any(kw in q for kw in kws):
            routes.add(cat)
    if not routes:
        routes.add("rag")
    return routes

def format_persona_context(persona, routes):
    parts = []
    if "personality" in routes or "facts" in routes:
        traits = persona.get("personality_traits", [])
        if traits: parts.append("PERSONALITY TRAITS: " + ", ".join(traits))
    if "habits" in routes:
        habits = persona.get("habits", [])[:8]
        if habits: parts.append("HABITS:\n" + "\n".join(f"- {h}" for h in habits))
    if "communication" in routes:
        cs = persona.get("communication_style", {})
        if cs:
            parts.append(f"COMMUNICATION STYLE: verbosity={cs.get('verbosity')}, tone={cs.get('tone')}, avg_words={cs.get('avg_message_length_words')}, emoji={cs.get('emoji_usage')}")
    if "relationships" in routes or "facts" in routes:
        pf = persona.get("personal_facts", {})
        rels = pf.get("relationships", [])
        if rels:
            parts.append("RELATIONSHIPS: " + ", ".join(f"{r['relation']} (x{r['mentions']})" for r in rels[:8]))
        if pf.get("places_mentioned"):
            parts.append("PLACES: " + ", ".join(pf["places_mentioned"][:6]))
    return "\n\n".join(parts)

def retrieve(query, embedder, segments, topic_embs, windows, window_embs, k_t=2, k_w=3):
    q_emb = embedder.encode([query])
    t_sims = cosine_similarity(q_emb, topic_embs)[0]
    top_t = t_sims.argsort()[-k_t:][::-1]
    topics = [{"topic_id": segments[i]["topic_id"], "summary": segments[i]["summary"], "score": float(t_sims[i])} for i in top_t]
    w_sims = cosine_similarity(q_emb, window_embs)[0]
    top_w = w_sims.argsort()[-k_w:][::-1]
    win = [{**windows[i], "score": float(w_sims[i])} for i in top_w]
    return topics, win

# ---------- UI ----------
st.title("Conversation Persona Bot")
st.caption("Ask about User 1 based on their full conversation history. Powered by topic-aware RAG + persona extraction.")

with st.spinner("Loading models and data (first run takes ~30s)..."):
    embedder = load_embedder()
    qa = load_qa_pipeline()
    segments, chunks, messages, persona = load_data()
    topic_embs, windows, window_embs = build_indexes(embedder, segments, messages)

# sidebar with persona snapshot
with st.sidebar:
    st.header("Persona snapshot")
    st.subheader("Personality traits")
    for t in persona.get("personality_traits", []):
        st.write(f"- {t}")
    st.subheader("Communication style")
    cs = persona.get("communication_style", {})
    for k, v in cs.items():
        st.write(f"**{k}**: {v}")
    st.subheader("Top relationships")
    for r in persona.get("personal_facts", {}).get("relationships", [])[:6]:
        st.write(f"- my {r['relation']} ({r['mentions']}x)")

# example questions
st.markdown("**Try these:**")
cols = st.columns(3)
examples = [
    "What kind of person is this user?",
    "What are their habits?",
    "How do they talk?",
]
for col, ex in zip(cols, examples):
    if col.button(ex):
        st.session_state["query"] = ex

query = st.text_input("Ask a question:", value=st.session_state.get("query", ""))

if query:
    with st.spinner("Thinking..."):
        routes = route_query(query)
        topics, wins = retrieve(query, embedder, segments, topic_embs, windows, window_embs)
        persona_ctx = format_persona_context(persona, routes)
        rag_ctx = "RELEVANT SNIPPETS:\n" + "\n".join([f"- {t['summary']}" for t in topics] + [f"- {w['text'][:300]}" for w in wins])
        ctx = (persona_ctx + "\n\n" + rag_ctx)[:2500]
        prompt = f"You are answering questions about User 1 based on their conversation history.\nUse the context below. If something is not in the context, say so.\n\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
        out = qa(prompt, max_new_tokens=180, do_sample=False)
        answer = out[0]["generated_text"]

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Routing decision"):
        st.write(f"Categories matched: {sorted(routes)}")

    with st.expander("Retrieved topic summaries"):
        for t in topics:
            st.write(f"**Topic {t['topic_id']}** (score {t['score']:.3f})")
            st.write(t["summary"])

    with st.expander("Retrieved message windows"):
        for w in wins:
            st.write(f"**Messages {w['start_idx']}-{w['end_idx']}** (score {w['score']:.3f})")
            st.code(w["text"])
