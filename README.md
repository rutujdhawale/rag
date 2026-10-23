# Hybrid RAG — Retrieval-Augmented Generation with Dense + Sparse Search

A self-correcting **Hybrid Retrieval-Augmented Generation** pipeline that combines **FAISS vector search** (semantic understanding) with **BM25 keyword search** (exact matching) to answer questions about your documents — PDFs, DOCX, CSV, Excel, images (OCR), and plain text.

Built with **LangChain**, **LangGraph**, **Groq** (free LLM inference), and **HuggingFace** embeddings.

---

## Why Hybrid RAG?

Traditional RAG systems use **only** vector search (dense retrieval). This works well for semantically similar queries but fails on:

| Query Type | Vector-Only RAG | Hybrid RAG |
|---|---|---|
| `"PSK modulation"` (exact term) | ❌ May miss — relies on embedding similarity | ✅ BM25 finds exact keyword match |
| `"how does the system handle errors"` (semantic) | ✅ Embeddings capture meaning | ✅ FAISS handles this |
| `"OFDM"` (acronym) | ❌ Embedding may not map acronyms well | ✅ BM25 finds exact string |
| `"wireless communication"` vs `"radio transmission"` (synonym) | ✅ Embeddings understand synonyms | ✅ FAISS handles, BM25 may miss |

**Hybrid RAG combines both**, then fuses scores so you get the best of both worlds. This project goes further by adding a **self-correcting LangGraph flow** that automatically rewrites bad queries and retries retrieval.

### How Hybrid Fusion Works

```mermaid
graph LR
    Q["User Query"] --> F["FAISS<br/>(Semantic Search)"]
    Q --> B["BM25<br/>(Keyword Search)"]
    
    F -->|"Top-K results<br/>weight: 0.6"| E["EnsembleRetriever<br/>Score Fusion"]
    B -->|"Top-K results<br/>weight: 0.4"| E
    
    E -->|"Merged + Deduplicated<br/>Re-ranked by combined score"| R["Final Results"]

    style Q fill:#ff6b6b,color:#fff
    style E fill:#ffd43b,color:#000
    style R fill:#51cf66,color:#fff
```

**Example**: Query = `"PSK modulation technique"`

| Retriever | Finds | Score |
|---|---|---|
| FAISS (semantic) | Chunk about "digital modulation methods" | 0.82 |
| FAISS (semantic) | Chunk about "phase shift keying" | 0.79 |
| BM25 (keyword) | Chunk containing exact phrase "PSK modulation" | 0.91 |
| BM25 (keyword) | Chunk mentioning "PSK" in a table | 0.73 |

After fusion: `final_score = (0.6 × FAISS_score) + (0.4 × BM25_score)` → best chunks from both are merged and deduplicated.

---

##  Features

- **Hybrid Retrieval** — FAISS (dense/semantic) + BM25 (sparse/keyword) with configurable weights
- **Self-Correcting Pipeline** — LangGraph flow grades retrieved chunks for relevance; auto-rewrites the query and retries if results are poor
- **Multi-Format Ingestion** — PDF, DOCX, CSV, Excel (.xlsx/.xls), TXT, and images (OCR via Tesseract)
- **GPU Acceleration** — Auto-detects NVIDIA CUDA for faster embedding generation
- **Source Attribution** — Every answer cites the source file(s) it was derived from
- **Persistent Vector Index** — FAISS index is saved to disk; rebuilds only when needed
- **Configurable via `.env`** — Retrieval weights, chunk sizes, model selection, retry limits

---

##  High-Level Architecture

```mermaid
graph TD
    A["Documents<br/>(PDF, DOCX, CSV, TXT)"] --> B["Data Loader<br/>(data_loader.py)"]
    B --> C["Text Splitter<br/>(text_splitter.py)<br/>chunk_size=1000, overlap=150"]
    C --> D["Embedding Model<br/>(embedding.py)<br/>all-MiniLM-L6-v2"]
    C --> E["BM25 Index<br/>(search.py)<br/>In-memory keyword index"]
    D --> F["FAISS Vector Index<br/>(vectorstore.py)<br/>Saved to ./faiss_index/"]

    G["User Question"] --> H["Hybrid Retriever<br/>(search.py)<br/>FAISS 0.6 + BM25 0.4"]
    F --> H
    E --> H

    H --> I["Relevance Grader<br/>(graph.py)<br/>LLM grades chunks"]

    I -->|"✅ Relevant"| J["Answer Generator<br/>(llm.py)<br/>Groq LLM"]
    I -->|"❌ Not Relevant"| K["🔄 Query Rewriter<br/>(graph.py)<br/>LLM rewrites query"]
    K -->|"retry ≤ 2"| H
    K -->|"max retries hit"| L["⚠️ Fallback Answer"]

    J --> M[" Final Answer<br/>with source citations"]

    style A fill:#4a9eff,color:#fff
    style G fill:#ff6b6b,color:#fff
    style M fill:#51cf66,color:#fff
    style J fill:#ffd43b,color:#000
```

---

## Project Structure

```
hybrid-rag/
├── main.py                    # Entry point — builds pipeline, runs interactive Q&A loop
├── .env                       # API keys & configuration (not committed to Git)
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
├── data/                      # Drop your documents here (PDF, DOCX, CSV, TXT, images)
│   ├── BlackBook.pdf
│   └── ...
├── faiss_index/               # Auto-generated FAISS index (saved to disk)
│   ├── index.faiss
│   └── index.pkl
└── pipeline/                  # Core RAG pipeline modules
    ├── __init__.py
    ├── data_loader.py         # Multi-format document loader (PDF, DOCX, CSV, Excel, OCR)
    ├── text_splitter.py       # Recursive character text splitter with smart chunking
    ├── embedding.py           # HuggingFace embedding model (GPU auto-detect)
    ├── vectorstore.py         # FAISS index build/load/save with caching
    ├── search.py              # BM25 + FAISS hybrid retriever with EnsembleRetriever
    ├── llm.py                 # Groq LLM integration + RAG prompt template
    └── graph.py               # LangGraph self-correcting flow (retrieve → grade → generate)
```

---

## How It Works — Step-by-Step Pipeline Deep Dive

### Step 1: Document Ingestion (`data_loader.py`)

The loader walks the `./data` directory and routes each file to a specialized parser based on its extension.

```mermaid
graph LR
    D["📁 ./data/ directory"] --> S{"🔍 File Extension?"}

    S -->|".pdf"| P["PyPDFLoader<br/>1 Document per page"]
    S -->|".docx"| X["python-docx<br/>Extract paragraphs"]
    S -->|".csv"| C["CSVLoader<br/>1 Document per row"]
    S -->|".xlsx/.xls"| E["UnstructuredExcelLoader<br/>Element-level"]
    S -->|".txt"| T["TextLoader<br/>UTF-8 encoding"]
    S -->|".png/.jpg/.bmp"| O["Tesseract OCR<br/>Image → text"]

    P --> M["List of Documents<br/>with metadata:<br/>source, file_type"]
    X --> M
    C --> M
    E --> M
    T --> M
    O --> M

    style D fill:#4a9eff,color:#fff
    style M fill:#51cf66,color:#fff
```

**Example**: Loading `BlackBook.pdf` (98 pages)

```
Input:  BlackBook.pdf (8.5 MB, 98 pages)
Output: 98 Document objects, each containing one page of text
        metadata = {"source": "data/BlackBook.pdf", "file_type": "pdf"}
```

Every document gets metadata attached (`source` path + `file_type`) so the final answer can cite which file it came from.

---

### Step 2: Text Chunking (`text_splitter.py`)

Raw documents are too long to embed effectively. The chunker splits them into overlapping pieces while preserving semantic coherence.

```mermaid
graph TD
    D["Raw Document<br/>(e.g. 3000 chars)"] --> S["RecursiveCharacterTextSplitter"]
    
    S --> T1["Try split on: \\n\\n<br/>(paragraph breaks)"]
    T1 -->|"chunks still > 1000 chars"| T2["Try split on: \\n<br/>(line breaks)"]
    T2 -->|"chunks still > 1000 chars"| T3["Try split on: period + space<br/>(sentences)"]
    T3 -->|"chunks still > 1000 chars"| T4["Try split on: spaces<br/>(words)"]
    T4 -->|"last resort"| T5["Hard cut at<br/>1000 characters"]

    T1 -->|"✅ fits"| R["Chunk"]
    T2 -->|"✅ fits"| R
    T3 -->|"✅ fits"| R
    T4 -->|"✅ fits"| R
    T5 --> R

    style D fill:#4a9eff,color:#fff
    style R fill:#51cf66,color:#fff
    style S fill:#ffd43b,color:#000
```

**Example**: A 3000-character page becomes 3 chunks with overlap:

```
Original page (3000 chars):
┌──────────────────────────────────────────────────────────┐
│ The PSK modulation technique involves changing the phase │
│ of a carrier signal to encode data. In BPSK, two phase   │
│ states represent binary 0 and 1. QPSK uses four phase    │
│ ... (continues for 3000 characters) ...                  │
│ The BER performance depends on the SNR and channel model.│
└──────────────────────────────────────────────────────────┘

After splitting (chunk_size=1000, overlap=150):

Chunk 1 (chars 0-1000):
┌──────────────────────────┐
│ The PSK modulation ...   │
│ ... QPSK uses four ...   │
│ ... phase constellation  │◄── 150 chars overlap ──┐
└──────────────────────────┘                         │
                                                     │
Chunk 2 (chars 850-1850):                            │
┌──────────────────────────┐                         │
│ phase constellation ...  │◄────────────────────────┘
│ ... 16-QAM increases ... │
│ ... spectral efficiency  │◄── 150 chars overlap ──┐
└──────────────────────────┘                         │
                                                     │
Chunk 3 (chars 1700-2700):                           │
┌──────────────────────────┐                         │
│ spectral efficiency ...  │◄────────────────────────┘
│ ... BER performance ...  │
│ ... channel model.       │
└──────────────────────────┘
```

The **150-character overlap** ensures that information spanning a chunk boundary isn't lost.

**Smart skipping**: CSV rows (already atomic) and short documents (< 1000 chars) bypass splitting entirely.

---

### Step 3: Embedding Generation (`embedding.py`)

Each text chunk is converted into a **384-dimensional numerical vector** that captures its semantic meaning.

```mermaid
graph LR
    C["Text Chunk<br/>'PSK modulation changes<br/>the carrier phase...'"] --> M["all-MiniLM-L6-v2<br/>(22M parameters)<br/>GPU: RTX 3050"]
    M --> V["384-dim Vector<br/>[0.023, -0.156, 0.089,<br/>0.241, ..., -0.034]"]

    style C fill:#4a9eff,color:#fff
    style M fill:#ffd43b,color:#000
    style V fill:#51cf66,color:#fff
```

**How embeddings capture meaning**:
```
"PSK modulation technique"     → vector A: [0.23, -0.15, 0.08, ...]
"phase shift keying method"    → vector B: [0.21, -0.14, 0.09, ...]  ← very similar!
"chocolate cake recipe"        → vector C: [-0.45, 0.33, -0.67, ...]  ← very different!

cosine_similarity(A, B) = 0.94  ← semantically similar
cosine_similarity(A, C) = 0.12  ← semantically unrelated
```

The model runs on **GPU (CUDA)** if available, falling back to CPU. GPU provides **3-5× faster** embedding generation.

---

### Step 4: Dual Indexing — FAISS + BM25

The chunks are indexed in two completely different ways for hybrid retrieval:

```mermaid
graph TD
    C["718 Text Chunks"] --> F["FAISS Index<br/>(Dense / Semantic)"]
    C --> B["BM25 Index<br/>(Sparse / Keyword)"]

    F --> FD["How it works:<br/>• Stores 384-dim vectors<br/>• L2 distance search<br/>• Finds semantically similar text<br/>• Saved to disk (./faiss_index/)"]
    B --> BD["How it works:<br/>• Tokenizes text into words<br/>• Computes TF-IDF scores<br/>• Finds exact keyword matches<br/>• In-memory (rebuilt each run)"]

    style C fill:#4a9eff,color:#fff
    style F fill:#e599f7,color:#000
    style B fill:#74c0fc,color:#000
```

**FAISS (vectorstore.py)**:
- Embeds all 718 chunks into vectors
- Builds an `IndexFlatL2` (exact nearest-neighbor search)
- Saves to `./faiss_index/` on disk — loads from cache on subsequent runs
- Query: embed the question → find top-K nearest chunk vectors

**BM25 (search.py)**:
- Tokenizes every chunk into words
- Computes term frequency (TF) and inverse document frequency (IDF)
- Query: tokenize the question → score each chunk by keyword overlap
- Rebuilt in-memory every run (fast — takes <1 second)

---

### Step 5: Hybrid Retrieval (`search.py`)

When a question arrives, **both** retrievers search independently, and their results are fused.

```mermaid
graph TD
    Q["Query: 'What is PSK modulation?'"] --> F["FAISS Search<br/>Embed query → find nearest vectors"]
    Q --> B["BM25 Search<br/>Tokenize query → match keywords"]

    F --> FR["FAISS Results:<br/>① 'digital modulation methods...' (0.82)<br/>② 'phase shift keying...' (0.79)<br/>③ 'carrier signal encoding...' (0.71)<br/>④ 'signal processing...' (0.65)"]

    B --> BR["BM25 Results:<br/>① 'PSK modulation is a...' (0.91)<br/>② 'PSK and QAM comparison...' (0.73)<br/>③ 'modulation scheme table...' (0.68)<br/>④ 'digital modulation intro...' (0.54)"]

    FR -->|"weight: 0.6"| E["EnsembleRetriever<br/>Weighted Score Fusion"]
    BR -->|"weight: 0.4"| E

    E --> D["Deduplicate + Rank<br/>Remove overlapping chunks"]
    D --> R["Top-4 Final Chunks<br/>Best of both worlds"]

    style Q fill:#ff6b6b,color:#fff
    style E fill:#ffd43b,color:#000
    style R fill:#51cf66,color:#fff
```

**Why this works better than either alone**:
- FAISS found "phase shift keying" (synonym of PSK) — BM25 would miss this
- BM25 found the exact phrase "PSK modulation" — FAISS might rank it lower
- Together, the chunk containing both the exact term AND semantic relevance scores highest

---

### Step 6: Self-Correcting LangGraph Flow (`graph.py`)

This is the most advanced part of the pipeline. Instead of blindly generating an answer from retrieved chunks, the system **checks if the chunks are actually relevant** and retries with a rewritten query if they're not.

```mermaid
stateDiagram-v2
    [*] --> Retrieve
    
    Retrieve --> GradeRelevance: Got chunks
    
    GradeRelevance --> Generate: ✅ Chunks ARE relevant
    GradeRelevance --> RewriteQuery: ❌ Chunks NOT relevant
    
    RewriteQuery --> Retrieve: retry (≤ 2 retries)
    RewriteQuery --> Generate: ⚠️ Max retries hit
    
    Generate --> [*]: Return answer

    note right of GradeRelevance
        LLM reads chunks and answers:
        "Are these relevant to the 
        question? yes/no"
    end note

    note right of RewriteQuery
        LLM rewrites the query:
        "PSK moduation" → 
        "Phase Shift Keying modulation technique"
    end note
```

**Example of self-correction in action**:

```
Round 1:
  Query: "whats the PSK moduation methodlogy"  (typo!)
  Retrieved chunks: mostly about "methodology" in general (not PSK)
  Grade: ❌ NOT relevant
  → Rewrite: "Phase Shift Keying PSK modulation technique"

Round 2:
  Query: "Phase Shift Keying PSK modulation technique"  (cleaner!)
  Retrieved chunks: actual PSK content from BlackBook.pdf
  Grade: ✅ RELEVANT
  → Generate answer from these chunks
```

**The flow handles 4 scenarios**:

```mermaid
graph TD
    A["Query arrives"] --> B{"Retrieval<br/>found relevant<br/>chunks?"}
    
    B -->|"✅ Yes, first try"| C["Generate answer<br/>(best case — 1 LLM call)"]
    
    B -->|"❌ No"| D{"Retries<br/>remaining?"}
    
    D -->|"Yes (≤ 2)"| E["Rewrite query<br/>+ retry retrieval"]
    E --> B
    
    D -->|"No (exhausted)"| F["Return fallback:<br/>'I couldn't find enough<br/>relevant information...'"]

    style C fill:#51cf66,color:#fff
    style F fill:#ff6b6b,color:#fff
    style E fill:#ffd43b,color:#000
```

---

### Step 7: Answer Generation (`llm.py`)

Once relevant chunks are confirmed, the LLM generates a **grounded** answer.

```mermaid
graph TD
    C["Relevant Chunks<br/>(from hybrid retriever)"] --> F["Format Context<br/>[Chunk 1 | Source: BlackBook.pdf]<br/>PSK modulation changes the...<br/><br/>[Chunk 2 | Source: BlackBook.pdf]<br/>QPSK uses four phase states..."]
    
    F --> P["RAG Prompt Template:<br/>'Answer using ONLY the context.<br/>Cite source files. Don't guess.'"]
    
    P --> L["Groq LLM<br/>(openai/gpt-oss-20b)<br/>temperature=0.2"]
    
    L --> A["Grounded Answer<br/>with source citations"]

    style C fill:#4a9eff,color:#fff
    style L fill:#ffd43b,color:#000
    style A fill:#51cf66,color:#fff
```

**The RAG prompt enforces grounding**:
```
Rules:
- Answer using only the information in the context. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so clearly — don't guess.
- Cite the source file name(s) you used when possible.
- Be concise and direct.
```

**Context is formatted with source attribution**:
```
[Chunk 1 | Source: data\BlackBook.pdf]
PSK (Phase Shift Keying) is a digital modulation technique where the phase
of the carrier signal is varied to represent data...

[Chunk 2 | Source: data\BlackBook.pdf]
In QPSK, four phase states (0°, 90°, 180°, 270°) are used, allowing
2 bits per symbol...
```

This labeling enables the LLM to cite specific source files in its answer.

---

##  Complete End-to-End Example

Let's trace a real query through every step of the pipeline:

### Query
```
Question: What modulation techniques are discussed in the documents?
```

### Step-by-Step Trace

```mermaid
sequenceDiagram
    participant U as User
    participant M as Main
    participant HR as Hybrid Retriever
    participant FA as FAISS
    participant BM as BM25
    participant GR as Grader
    participant LLM as Groq LLM

    U->>M: "What modulation techniques are discussed?"
    M->>HR: invoke(question)
    
    par Parallel Retrieval
        HR->>FA: embed query → search nearest vectors
        FA-->>HR: Top-4 semantic results
        HR->>BM: tokenize query → match keywords
        BM-->>HR: Top-4 keyword results
    end
    
    HR->>HR: Fuse scores (0.6×FAISS + 0.4×BM25)
    HR->>HR: Deduplicate + rank
    HR-->>M: Top-4 merged chunks
    
    M->>GR: Grade chunks for relevance
    GR->>LLM: "Are these chunks relevant? yes/no"
    LLM-->>GR: "yes"
    GR-->>M: is_relevant = True
    
    M->>LLM: Generate answer from chunks
    LLM-->>M: Grounded answer with citations
    M-->>U: Display answer
```

### What Each Retriever Finds

| # | Retriever | Chunk Preview | Why Found |
|---|---|---|---|
| 1 | FAISS | "PSK (Phase Shift Keying) is a digital modulation technique where the phase of the carrier signal is varied..." | Semantic match: "modulation techniques" ≈ "digital modulation technique" |
| 2 | BM25 | "Table 3.2: Comparison of modulation techniques — BPSK, QPSK, 16-QAM..." | Exact keyword match: "modulation techniques" |
| 3 | FAISS | "OFDM divides the available spectrum into multiple orthogonal subcarriers for parallel data transmission..." | Semantic match: OFDM is a modulation-related concept |
| 4 | BM25 | "The modulation scheme determines the number of bits per symbol and affects the BER performance..." | Keyword match: "modulation" |

### Final Generated Answer

```
Answer:
The documents discuss several modulation techniques:

1. **PSK (Phase Shift Keying)** — a digital modulation technique where the phase of the
   carrier signal is varied to represent data bits. BPSK uses two phase states (0° and 180°),
   while QPSK uses four phase states allowing 2 bits per symbol.
   (Source: data\BlackBook.pdf)

2. **OFDM (Orthogonal Frequency Division Multiplexing)** — divides the spectrum into
   multiple orthogonal subcarriers for parallel data transmission, improving spectral
   efficiency in multipath channels.
   (Source: data\BlackBook.pdf)

3. **16-QAM** — mentioned in a comparison table alongside BPSK and QPSK, offering higher
   data rates but requiring better SNR.
   (Source: data\BlackBook.pdf)

These techniques are discussed in the context of digital communication systems and
wireless channel performance analysis.
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **Groq API Key** (free): [https://console.groq.com/keys](https://console.groq.com/keys)
- (Optional) **Tesseract OCR** for image text extraction: [Install Guide](https://github.com/tesseract-ocr/tesseract)
- (Optional) **NVIDIA GPU** with CUDA for faster embedding generation

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/hybrid-rag.git
cd hybrid-rag

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install CUDA-enabled PyTorch for GPU acceleration
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Configuration (`.env`)

```ini
# Required
GROQ_API_KEY=your_groq_api_key_here

# Document source folder
DATA_DIR=./data

# FAISS index persistence
FAISS_INDEX_DIR=./faiss_index

# Hybrid retrieval weights (must sum to 1.0)
SEMANTIC_WEIGHT=0.6
KEYWORD_WEIGHT=0.4
RETRIEVER_TOP_K=4

# Self-correction retry limit
MAX_RETRIEVAL_RETRIES=2

# LLM settings
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TEMPERATURE=0.2

# GPU: uncomment to force device ("cpu" or "cuda")
# EMBEDDING_DEVICE=cuda
```

---

## ▶ Running the Project

### Option 1: Streamlit Web Application (Recommended)

Launch the full modern product UI with grouped chat history, document management, transparent hybrid retrieval details, and streaming responses:

```bash
streamlit run app.py
```
*(or via uv: `uv run streamlit run app.py`)*

Then open **http://localhost:8501** in your browser.

#### Web Interface Highlights:
- **Chat View**: Clean typography-driven conversation interface with real-time token streaming and citation source cards.
- **Hybrid Inspection**: Expandable "Why this answer?" breakdown showing exact vector vs BM25 hits, RRF fusion scores, and retrieval latency.
- **Document Library**: Multi-format file uploader with real sequential progress tracking (Validating → Extracting → Chunking → FAISS Indexing → BM25 Indexing), document deletion, and index rebuilding.
- **Persistent History**: Conversations grouped into *Today*, *Yesterday*, *Previous 7 Days*, and *Older* with automatic query titling, manual rename, and delete actions.
- ** Config Panel**: Adjust semantic vs keyword weights, Top-K, model providers (OpenRouter / Groq), and LangGraph self-correction.

---

### Option 2: Interactive Terminal CLI

### Terminal User Interface (TUI)

<p align="center">
  <img src="assets/tui-demo.png" alt="Hybrid RAG Terminal User Interface" width="100%">
</p>

The terminal runner (`main.py`) provides an ergonomic, fast, keyboard-first **Terminal User Interface (TUI)** built with Rich:

- **Sub-Second Instant Boot**: Reuses disk-cached FAISS vectors and BM25 chunk pickles directly (< 0.2s startup).
- **Real-Time Token Streaming**: Streams generated tokens progressively with live Markdown rendering.
- **Hybrid Retrieval Telemetry**: Live breakdown table showing Dense (FAISS) rank, Sparse (BM25) rank, RRF score, and provenance.
- **Grounding Citations**: Clean source badges with document names, page numbers, and preview snippets.

#### Launch Interactive TUI
```bash
python main.py
# or with uv:
uv run python main.py
```

#### Command Palette Inside TUI
| Command | Action | Example |
|---|---|---|
| `/help` | Display command cheat-sheet and shortcuts | `/help` |
| `/telemetry` | Toggle detailed RRF candidate ranking table | `/telemetry` |
| `/sources` | Inspect full text snippets & metadata for last query | `/sources` |
| `/mode` | Toggle between Fast Streaming & LangGraph Self-Correction | `/mode` |
| `/weight <w>` | Adjust semantic vs keyword fusion weight | `/weight 0.7` |
| `/topk <k>` | Change number of retrieved context chunks | `/topk 5` |
| `/stats` | View document catalog, chunk counts & word distribution | `/stats` |
| `/rebuild` | Force re-parse documents and re-index FAISS | `/rebuild` |
| `/clear` | Clear terminal screen and redraw status header | `/clear` |
| `/exit`, `/quit` | Exit the TUI session cleanly | `/exit` |

#### Non-Interactive CLI Query (Headless Scripting)
```bash
# Direct terminal answer with citations:
python main.py --query "What is the Sherman Antitrust Act?"

# Direct terminal answer with RRF telemetry table:
python main.py --query "What is the Sherman Antitrust Act?" --telemetry
```

---

###  Running Tests & Benchmark Evaluation

#### Run Test Suite (18 Automated Tests)
```bash
pytest -v
```
*(or via uv: `uv run pytest -v`)*

#### Run Retrieval Benchmark Evaluation
Compare Dense (Vector) vs Sparse (BM25) vs Hybrid (RRF Fusion) on representative queries:
```bash
python -m evaluation.eval_retrieval
```

---

### What Happens on First Run

```mermaid
graph TD
    A["▶️ python main.py"] --> B["Step 1: Load documents<br/>from ./data/"]
    B --> C["Step 2: Split into<br/>overlapping chunks"]
    C --> D["Step 3: Generate embeddings<br/>(GPU if available)"]
    D --> E["Step 4: Build FAISS index<br/>+ save to ./faiss_index/"]
    E --> F["Step 5: Build BM25 index<br/>(in-memory)"]
    F --> G["Step 6: Compile<br/>LangGraph flow"]
    G --> H["🎉 Pipeline Ready!<br/>Interactive Q&A prompt"]

    style A fill:#4a9eff,color:#fff
    style H fill:#51cf66,color:#fff
```

### What Happens on Subsequent Runs

```mermaid
graph TD
    A["▶️ python main.py"] --> B["Step 1: Load documents"]
    B --> C["Step 2: Split into chunks"]
    C --> D["Step 3: Load FAISS index<br/>from disk cache ⚡<br/>(skips embedding!)"]
    D --> E["Step 4: Build BM25 index"]
    E --> F["Step 5: Compile flow"]
    F --> G["🎉 Pipeline Ready!<br/>(much faster!)"]

    style A fill:#4a9eff,color:#fff
    style D fill:#ffd43b,color:#000
    style G fill:#51cf66,color:#fff
```

```
Pipeline ready.

Ask a question about your documents (or type 'exit' to quit).

Question: What is this document about?

Answer:
This document covers digital communication systems, focusing on...

------------------------------------------------------------
```

---

## Technical Decisions

### Why Hybrid RAG over Pure Vector Search?

```mermaid
graph TD
    subgraph "Pure Vector RAG"
        V1["Query: 'OFDM'"] --> V2["Embed → Search"]
        V2 --> V3["❌ Might return 'orthogonal frequency'<br/>but miss exact 'OFDM' mentions"]
    end

    subgraph "Hybrid RAG (this project)"
        H1["Query: 'OFDM'"] --> H2["FAISS: semantic matches"]
        H1 --> H3["BM25: exact 'OFDM' matches"]
        H2 --> H4["✅ Gets both semantic context<br/>AND exact keyword hits"]
        H3 --> H4
    end

    style V3 fill:#ff6b6b,color:#fff
    style H4 fill:#51cf66,color:#fff
```

### Why `all-MiniLM-L6-v2`?

- **384 dimensions** — compact, fast to index and search
- **22M parameters** — lightweight enough to run on CPU, fast on GPU
- **Strong general-purpose performance** on sentence similarity benchmarks
- Free, no API key needed — runs entirely locally

### Why Recursive Character Splitting?

```mermaid
graph TD
    subgraph "Naive Splitting (fixed 1000 chars)"
        N1["'...the PSK technique. | In QPSK, four phase...'"]
        N1 --> N2["❌ Sentence cut mid-thought"]
    end

    subgraph "Recursive Splitting (this project)"
        R1["'...the PSK technique.'"]
        R2["'In QPSK, four phase states...'"]
        R1 --> R3["✅ Split on sentence boundary"]
        R2 --> R3
    end

    style N2 fill:#ff6b6b,color:#fff
    style R3 fill:#51cf66,color:#fff
```

The splitter tries `\n\n` → `\n` → `. ` → ` ` → hard cut. The 150-char overlap prevents information loss at boundaries.

### Why LangGraph Self-Correction?

```mermaid
graph TD
    subgraph "Standard RAG"
        S1["Bad retrieval"] --> S2["Generate anyway"]
        S2 --> S3["❌ Hallucinated or<br/>irrelevant answer"]
    end

    subgraph "Self-Correcting RAG (this project)"
        C1["Bad retrieval"] --> C2["Grade: NOT relevant"]
        C2 --> C3["Rewrite query"]
        C3 --> C4["Better retrieval"]
        C4 --> C5["Grade: RELEVANT"]
        C5 --> C6["✅ Accurate, grounded answer"]
    end

    style S3 fill:#ff6b6b,color:#fff
    style C6 fill:#51cf66,color:#fff
```

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain, LangGraph |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | FAISS (CPU/GPU) |
| Keyword Search | BM25 (`rank-bm25`) |
| Document Parsing | PyPDF, python-docx, Tesseract OCR |
| Language | Python 3.11+ |

---

## License

This project is for educational and personal use only. 
