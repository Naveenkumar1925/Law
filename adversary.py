import os
import faiss
import pickle
import ollama
import numpy as np
import re
import fitz
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import pyttsx3
from faster_whisper import WhisperModel


# =========================
# LOAD KNOWLEDGE BASE
# =========================

print("\nLoading legal knowledge base...\n")

index = faiss.read_index("knowledge/vector.index")

with open("knowledge/docs.pkl", "rb") as f:
    docs = pickle.load(f)

with open("knowledge/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("knowledge/graph.pkl", "rb") as f:
    graph = pickle.load(f)

print("Knowledge base loaded\n")


# =========================
# LOAD CASE DOCUMENTS
# =========================

CASE_DIR = "case"


def load_case_documents():

    text = ""

    if not os.path.exists(CASE_DIR):
        return ""

    for file in os.listdir(CASE_DIR):

        if file.endswith(".pdf"):

            doc = fitz.open(os.path.join(CASE_DIR, file))

            for page in doc:
                text += page.get_text() + "\n"

    return text


case_text = load_case_documents()

print("Case documents loaded\n")


# =========================
# VOICE ENGINE (TTS)
# =========================

engine = pyttsx3.init()


def speak(text):

    engine.say(text)
    engine.runAndWait()


# =========================
# LOAD WHISPER MODEL
# =========================

print("Loading speech recognition model...\n")

whisper_model = WhisperModel("base", compute_type="int8")

print("Speech model ready\n")


# =========================
# VOICE INPUT
# =========================

def listen():

    samplerate = 16000
    duration = 6

    print("Speak your argument...")

    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1,
                   dtype="int16")

    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    wav.write(temp_file.name, samplerate, audio)

    segments, _ = whisper_model.transcribe(temp_file.name)

    text = ""

    for segment in segments:
        text += segment.text

    text = text.strip()

    if text == "":
        print("Could not recognize speech")
        return ""

    print("User:", text)

    return text


# =========================
# EMBEDDINGS
# =========================

def embed(text):

    r = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return np.array(r["embedding"], dtype="float32")


# =========================
# VECTOR SEARCH
# =========================

def vector_search(query, k=5):

    q = embed(query)

    D, I = index.search(np.array([q]), k)

    return [docs[i] for i in I[0]]


# =========================
# KEYWORD SEARCH
# =========================

def keyword_search(query, k=5):

    scores = bm25.get_scores(query.split())

    ranked = np.argsort(scores)[::-1][:k]

    return [docs[i] for i in ranked]


# =========================
# GRAPH SEARCH
# =========================

def graph_search(query):

    results = []

    words = re.findall(r"\w+", query.lower())

    for w in words:

        if w in graph:

            neighbors = list(graph.neighbors(w))[:5]

            results.extend(neighbors)

    return results


# =========================
# HYBRID RETRIEVAL
# =========================

def retrieve_context(query):

    v = vector_search(query)

    k = keyword_search(query)

    g = graph_search(query)

    merged = list(set(v + k + g))

    return merged[:10]


# =========================
# LAW DETECTION
# =========================

def detect_laws(text):

    patterns = [
        r"IPC\s?\d+",
        r"Section\s?\d+[A-Z]*",
        r"Motor\s?Vehicle\s?Act\s?\d+",
        r"MV\s?Act\s?\d+",
        r"IT\s?Act\s?\d+[A-Z]*"
    ]

    found = []

    for p in patterns:

        matches = re.findall(p, text, re.IGNORECASE)

        found.extend(matches)

    return list(set(found))


# =========================
# ARGUMENT GENERATION
# =========================

def generate_argument(query):

    context_list = retrieve_context(query)

    context = "\n".join(context_list)

    laws = detect_laws(context)

    prompt = f"""
You are an experienced opposing lawyer in court.

You must oppose the user's argument.

Case Documents:
{case_text}

Relevant Laws:
{laws}

Legal Knowledge:
{context}

User Argument:
{query}

Respond in structured format:

1. Opposing Legal Position
2. Laws Supporting Opposition
3. Weakness in User Argument
4. Counter Legal Reasoning
5. Courtroom Strategy
6. Possible Judge Interpretation

Always argue against the user.
"""

    response = ollama.chat(
        model="gemma3",
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


# =========================
# MAIN LOOP
# =========================

print("Adversary Lawyer AI Ready\n")

print("Choose input mode each time:")
print("v = voice | t = text | exit = quit\n")


while True:

    mode = input("Mode: ")

    if mode == "exit":
        break

    if mode == "v":
        query = listen()
    else:
        query = input("Argument: ")

    if query == "":
        continue

    start = time.time()

    response = generate_argument(query)

    end = time.time()

    print("\nOpposing Lawyer:\n")

    print(response)

    speak(response)

    print("\nResponse time:", round(end - start, 2), "sec\n")