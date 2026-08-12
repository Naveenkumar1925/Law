import faiss
import pickle
import ollama
import numpy as np
import re
import time


print("\nLoading legal knowledge base...\n")

index = faiss.read_index("knowledge/vector.index")

with open("knowledge/docs.pkl", "rb") as f:
    docs = pickle.load(f)

with open("knowledge/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("knowledge/graph.pkl", "rb") as f:
    graph = pickle.load(f)

print("Knowledge base loaded successfully\n")


# ==========================
# EMBEDDINGS
# ==========================

def embed(text):

    r = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return np.array(r["embedding"], dtype="float32")


# ==========================
# VECTOR SEARCH
# ==========================

def vector_search(query, k=5):

    q = embed(query)

    D, I = index.search(np.array([q]), k)

    return [docs[i] for i in I[0]]


# ==========================
# KEYWORD SEARCH
# ==========================

def keyword_search(query, k=5):

    scores = bm25.get_scores(query.split())

    ranked = np.argsort(scores)[::-1][:k]

    return [docs[i] for i in ranked]


# ==========================
# GRAPH SEARCH
# ==========================

def graph_search(query):

    results = []

    words = re.findall(r"\w+", query.lower())

    for w in words:

        if w in graph:

            neighbors = list(graph.neighbors(w))[:5]

            results.extend(neighbors)

    return results


# ==========================
# HYBRID RETRIEVAL
# ==========================

def retrieve_context(query):

    v = vector_search(query)

    k = keyword_search(query)

    g = graph_search(query)

    merged = list(set(v + k + g))

    return merged[:10]


# ==========================
# LAW SECTION DETECTOR
# ==========================

def detect_law_sections(text):

    patterns = [
        r"IPC\s?\d+",
        r"Section\s?\d+[A-Z]*",
        r"Article\s?\d+",
        r"MV\s?Act\s?\d+",
        r"Motor\s?Vehicle\s?Act\s?\d+",
        r"IT\s?Act\s?\d+[A-Z]*"
    ]

    found = []

    for p in patterns:

        matches = re.findall(p, text, re.IGNORECASE)

        found.extend(matches)

    return list(set(found))


# ==========================
# LEGAL DOMAIN DETECTOR
# ==========================

def detect_domain(query):

    q = query.lower()

    if any(w in q for w in ["traffic", "license", "insurance", "helmet", "challan"]):
        return "Motor Vehicle Act"

    if any(w in q for w in ["fraud", "cheat", "theft", "attack", "murder"]):
        return "Indian Penal Code"

    if any(w in q for w in ["online", "cyber", "internet", "hack"]):
        return "Information Technology Act"

    return "General Indian Law"


# ==========================
# RESPONSE GENERATOR
# ==========================

def generate_answer(query):

    context_list = retrieve_context(query)

    context = "\n".join(context_list)

    detected_domain = detect_domain(query)

    law_refs = detect_law_sections(context)

    prompt = f"""
You are an expert Indian legal counselor AI.

Domain: {detected_domain}

Detected law references:
{law_refs}

Legal knowledge:
{context}

User question:
{query}

Provide structured legal guidance.

Response format:

1. Applicable Laws
2. Explanation
3. Punishment / Penalty
4. Legal Reasoning
5. What the person should do next
6. Court procedure if dispute continues


Explain in simple language.
"""

    response = ollama.chat(
        model="mashriram/sarvam-g-lite",
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


# ==========================
# CHAT LOOP
# ==========================

print("Legal Counselor AI Ready\n")
print("Ask legal questions. Type 'exit' to quit.\n")

while True:

    query = input("User: ")

    if query.lower() in ["exit", "quit"]:
        break

    start = time.time()

    answer = generate_answer(query)

    end = time.time()

    print("\nCounselor:\n")

    print(answer)

    print(f"\nResponse time: {round(end-start,2)} sec\n")