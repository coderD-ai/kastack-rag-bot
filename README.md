# Conversation Persona Bot

A topic-aware RAG system that processes day-by-day conversation history, detects topic shifts, builds checkpoint summaries, extracts a persona of User 1, and answers questions about them.

## Demo
- **Live app**: <ADD STREAMLIT URL HERE>
- **Loom walkthrough**: <ADD LOOM URL HERE>

## What it does

### Part 1 — RAG with checkpoints
- **Topic checkpoints**: detects topic shifts using sliding-window cosine similarity on sentence embeddings; produces one summary per topic segment.
- **100-message checkpoints**: independent of topics, summary every 100 messages.
- **Query handling**: retrieves both relevant topic summaries and relevant raw message windows, combines them in the prompt.

### Part 2 — Persona extraction
JSON persona with four fields:
- `habits` — extracted via regex pattern matching on "I usually / I love / I always" phrases
- `personal_facts` — spaCy NER for people/places/orgs + regex for relationship words ("my wife", "my kids")
- `personality_traits` — inferred from sentiment markers, message length, humor markers, exclamations, etc.
- `communication_style` — quantitative metrics: avg word count, emoji density, question-vs-exclamation ratio, tone classification

### Part 3 — Chatbot
A keyword-based router decides whether the query is about persona (personality/habits/communication/relationships) or factual recall (RAG). Combines both contexts into the prompt sent to a local `flan-t5-base` model.

## How topic detection works
1. Embed every message with `sentence-transformers/all-MiniLM-L6-v2` (384-d).
2. For each position `i` (with margin), compute the cosine similarity between the mean embedding of messages `[i-5, i)` and the mean embedding of messages `[i, i+5)`.
3. A point is flagged as a topic boundary if (a) similarity falls below threshold 0.45, (b) it is a local minimum within +/- 2 positions, and (c) at least 8 messages have passed since the last boundary.
4. Result on 100 sample days (~1700 messages): 137 topic segments, average 12.6 messages each. Manual inspection of random boundaries showed clean topic transitions (e.g., pets to work, hobbies to food).

## How retrieval works
Two indexes built on top of the same embedding model:
- **Topic-summary index** — one vector per topic segment summary.
- **Message-window index** — one vector per 5-message rolling window.

A query is embedded and the top-k from each index is retrieved (k=2 topics, k=3 windows by default). Both sets are formatted into the prompt under separate headers ("RELEVANT TOPIC SUMMARIES" / "RELEVANT CONVERSATION SNIPPETS"), giving the answer model both high-level themes and concrete excerpts.

## How persona is built
- Filter to User 1's messages only.
- Run regex pattern matchers on the concatenated text for habit-style sentences.
- Run spaCy `en_core_web_sm` NER on the text to pick up named entities.
- Compute communication-style metrics from message-length distribution and punctuation/emoji counts.
- Map raw signals (counts, ratios, flagged keywords) to qualitative personality traits using simple thresholds.

## Run locally
