# Law — Offline Legal AI Assistant

A fully local, privacy-preserving legal AI assistant for Indian case law. It runs entirely on your own machine using [Ollama](https://ollama.com) for language models and embeddings — no cloud APIs, no data leaving your computer.

The system reads case documents (PDFs), retrieves relevant law from a knowledge base of Supreme Court judgments using a hybrid search pipeline, and generates structured legal analysis. It ships three role-based assistants: a **Researcher**, a **Counselor**, and an **Adversary** (with voice support).

---

## Features

- **Hybrid retrieval** — combines dense vector search (FAISS), keyword search (BM25), and a knowledge graph (NetworkX) for more accurate legal context.
- **Fully offline** — all inference runs locally through Ollama. Nothing is sent to external servers.
- **Case-aware** — automatically loads and reasons over PDFs placed in the `case/` folder.
- **Three specialized agents** — research, counselling, and adversarial cross-examination.
- **Voice mode** — the Adversary agent supports speech input (Whisper) and spoken replies (pyttsx3).

---

## Project Structure

```
Law/
├── data_creator.py     # Builds the knowledge base from files in data/
├── researcher.py       # Legal Research Assistant (structured case analysis)
├── counselor.py        # Legal Counselor (plain-language legal Q&A)
├── adversary.py        # Opposing-lawyer simulator with voice input/output
├── data/               # Source material (judgments.csv, source PDFs)
│   └── judgments.csv
├── case/               # Drop the PDFs for the case you're working on here
├── knowledge/          # Generated knowledge base (built by data_creator.py)
│   ├── vector.index    # FAISS vector index
│   ├── docs.pkl        # Chunked documents
│   ├── bm25.pkl        # BM25 keyword index
│   └── graph.pkl       # Knowledge graph
└── notes.txt           # Setup notes
```

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- The following Ollama models pulled locally:
  - `nomic-embed-text` (embeddings)
  - `gemma3` (used by the researcher and adversary)
  - `mashriram/sarvam-g-lite` (used by the counselor)

Pull the models:

```bash
ollama pull nomic-embed-text
ollama pull gemma3
ollama pull mashriram/sarvam-g-lite
```

### Python dependencies

```bash
pip install pymupdf pandas networkx rank-bm25 numpy tqdm ollama
```

FAISS (choose one):

```bash
# GPU
conda install -c pytorch faiss-gpu
# or CPU
conda install -c conda-forge faiss-cpu
```

For the voice-enabled Adversary agent, also install:

```bash
pip install sounddevice scipy pyttsx3 faster-whisper
```

Optional environment tuning for Ollama:

```bash
# PowerShell
$env:OLLAMA_GPU_LAYERS=100
$env:OLLAMA_NUM_PARALLEL=8
```

---

## Usage

### 1. Build the knowledge base

Place your source material (the provided `judgments.csv` and any reference PDFs) in the `data/` folder, then run:

```bash
python data_creator.py
```

This chunks the documents, generates embeddings, and writes the FAISS index, BM25 index, document store, and graph into `knowledge/`.

### 2. Add your case files

Drop the PDFs for the matter you're analyzing into the `case/` folder. All three agents automatically read from here.

### 3. Run an assistant

**Legal Research Assistant** — structured analysis of the loaded case:

```bash
python researcher.py
```

Returns a 9-part breakdown: case summary, applicable laws, section numbers, legal reasoning, evidence analysis, strategy, counter-arguments, court procedure, and likely outcome.

**Legal Counselor** — conversational legal Q&A in plain language:

```bash
python counselor.py
```

**Adversary (opposing lawyer)** — simulates cross-examination; supports voice or text each turn:

```bash
python adversary.py
```

Choose `v` for voice, `t` for text, or `exit` to quit at each prompt.

---

## How It Works

1. **Ingestion** (`data_creator.py`) — PDFs and CSV rows are split into ~500-word chunks. Each chunk is embedded with `nomic-embed-text` and indexed in FAISS, BM25, and a NetworkX graph.
2. **Retrieval** — at query time, the system runs vector + keyword search, pulls the most relevant legal passages, and detects the applicable legal domain and statute references.
3. **Generation** — the retrieved context, case text, and user question are composed into a role-specific prompt and answered by a local Ollama model.

---

## Data

The included `data/judgments.csv` contains Indian Supreme Court judgment metadata (case numbers, parties, benches, judgment dates, and document links). It seeds the legal knowledge base used for retrieval.

---

## Notes

- This project is intended for legal research and educational purposes. Its output is AI-generated and must not be treated as legal advice or a substitute for a qualified advocate.
- Because everything runs locally, first-time responses may be slower while models load into memory.

---

## License

No license file is currently included. Add one (for example, MIT) if you intend for others to reuse this code.
