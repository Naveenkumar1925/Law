import os
import faiss
import pickle
import ollama
import numpy as np
import re
import fitz
import time


# ==============================
# LOAD KNOWLEDGE BASE
# ==============================

print("\nLoading legal knowledge base...\n")

index = faiss.read_index("knowledge/vector.index")

with open("knowledge/docs.pkl", "rb") as f:
    docs = pickle.load(f)

with open("knowledge/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("knowledge/graph.pkl", "rb") as f:
    graph = pickle.load(f)

print("Knowledge base loaded successfully\n")


# ==============================
# LOAD CASE DOCUMENTS
# ==============================

CASE_DIR = "case"


def load_case_documents():

    text = ""

    if not os.path.exists(CASE_DIR):
        return ""

    for file in os.listdir(CASE_DIR):

        if file.endswith(".pdf"):

            path = os.path.join(CASE_DIR, file)

            doc = fitz.open(path)

            for page in doc:
                text += page.get_text() + "\n"

    return text


case_text = load_case_documents()

print("Case documents loaded\n")


# ==============================
# EMBEDDING FUNCTION
# ==============================

def embed(text):

    r = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return np.array(r["embedding"], dtype="float32")


# ==============================
# VECTOR SEARCH
# ==============================

def vector_search(query, k=5):

    q = embed(query)

    D, I = index.search(np.array([q]), k)

    return [docs[i] for i in I[0]]


# ==============================
# KEYWORD SEARCH
# ==============================

def keyword_search(query, k=5):

    scores = bm25.get_scores(query.split())

    ranked = np.argsort(scores)[::-1][:k]

    return [docs[i] for i in ranked]


# ==============================
# GRAPH SEARCH
# ==============================

def graph_search(query):

    results = []

    words = re.findall(r"\w+", query.lower())

    for w in words:

        if w in graph:

            neighbors = list(graph.neighbors(w))[:5]

            results.extend(neighbors)

    return results


# ==============================
# HYBRID RETRIEVAL
# ==============================

def retrieve_context(query):

    v = vector_search(query)

    k = keyword_search(query)

    g = graph_search(query)

    merged = list(set(v + k + g))

    return merged[:10]


# ==============================
# LAW DETECTION
# ==============================

def detect_laws(text):

    patterns = [
        r"IPC\s?\d+",
        r"Section\s?\d+[A-Z]*",
        r"Article\s?\d+",
        r"Motor\s?Vehicle\s?Act\s?\d+",
        r"MV\s?Act\s?\d+",
        r"IT\s?Act\s?\d+[A-Z]*"
    ]

    found = []

    for p in patterns:

        matches = re.findall(p, text, re.IGNORECASE)

        found.extend(matches)

    return list(set(found))


# ==============================
# LEGAL DOMAIN DETECTION
# ==============================

def detect_domain(query):

    q = query.lower()

    if any(w in q for w in ["traffic", "license", "insurance", "helmet", "challan"]):
        return "Motor Vehicle Act"

    if any(w in q for w in ["fraud", "cheat", "theft", "attack", "murder"]):
        return "Indian Penal Code"

    if any(w in q for w in ["online", "cyber", "internet", "hack"]):
        return "Information Technology Act"

    return "General Indian Law"


# ==============================
# RESPONSE GENERATOR
# ==============================

def generate_research(query):

    context_list = retrieve_context(query)

    context = "\n".join(context_list)

    laws = detect_laws(context)

    domain = detect_domain(query)

    prompt = f"""
You are an expert legal research assistant helping a lawyer prepare a case.

Case Documents:
{case_text}

Relevant Legal Knowledge:
{context}

Detected Laws:
{laws}

Legal Domain:
{domain}

Lawyer Question:
{query}

Provide detailed legal research.

Structure the response:

1. Case Summary
2. Applicable Laws
3. Section Numbers
4. Legal Reasoning
5. Evidence Analysis
6. Legal Strategy to Win the Case
7. Possible Counter Arguments
8. Court Procedure
9. Possible Outcome

Be precise and professional.
"""

    response = ollama.chat(
        model="gemma3",
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


# ==============================
# MAIN CHAT LOOP
# ==============================

print("Legal Research Assistant Ready\n")
print("Ask research questions about the case.\n")

while True:

    query = input("Lawyer: ")

    if query.lower() in ["exit", "quit"]:
        break

    start = time.time()

    answer = generate_research(query)

    end = time.time()

    print("\nResearch Analysis:\n")

    print(answer)

    print("\nResponse time:", round(end - start, 2), "sec\n")