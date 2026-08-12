import os
import faiss
import pickle
import ollama
import fitz
import numpy as np
import pandas as pd
import networkx as nx
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

DATA_DIR = "data"
SAVE_DIR = "knowledge"

os.makedirs(SAVE_DIR, exist_ok=True)

documents = []


def chunk_text(text, size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks


def load_pdfs():
    texts = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(DATA_DIR, file))
            for page in doc:
                texts.extend(chunk_text(page.get_text()))
    return texts


def load_csv():
    texts = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            rows = df.astype(str).agg(" ".join, axis=1).tolist()
            for r in rows:
                texts.extend(chunk_text(r))
    return texts


def embed(text):
    r = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return np.array(r["embedding"], dtype="float32")


def generate_embeddings(docs):

    embeddings = []

    with ThreadPoolExecutor(max_workers=8) as executor:

        results = list(
            tqdm(
                executor.map(embed, docs),
                total=len(docs),
                desc="Embedding"
            )
        )

        embeddings.extend(results)

    return np.vstack(embeddings)


def build_graph(docs):

    G = nx.Graph()

    for doc in docs:

        words = doc.split()

        for i in range(len(words)-1):

            w1 = words[i].lower()
            w2 = words[i+1].lower()

            if len(w1) > 3 and len(w2) > 3:
                G.add_edge(w1, w2)

    return G


print("\nLoading documents...\n")

documents = load_pdfs() + load_csv()

print("Total chunks:", len(documents))


print("\nGenerating embeddings...\n")

embeddings = generate_embeddings(documents)


print("\nBuilding FAISS index...\n")

dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)

index.add(embeddings)

faiss.write_index(index, os.path.join(SAVE_DIR, "vector.index"))


print("\nBuilding BM25 index...\n")

tokenized = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized)

with open(os.path.join(SAVE_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump(bm25, f)


print("\nBuilding graph...\n")

G = build_graph(documents)

with open(os.path.join(SAVE_DIR, "graph.pkl"), "wb") as f:
    pickle.dump(G, f)


with open(os.path.join(SAVE_DIR, "docs.pkl"), "wb") as f:
    pickle.dump(documents, f)


print("\nKnowledge base created successfully!\n")