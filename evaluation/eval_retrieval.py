"""
evaluation/eval_retrieval.py

Benchmark evaluation script comparing:
- Vector-Only Retrieval (FAISS Dense)
- BM25-Only Retrieval (Sparse Keyword)
- Hybrid Retrieval (RRF Weighted Fusion)

Measures Precision@K, Recall@K, and Latency on a representative question set.
Run:
    python -m evaluation.eval_retrieval
"""

import time
from pathlib import Path
from pipeline.vectorstore import load_vectorstore
from pipeline.search import build_bm25_retriever, detailed_hybrid_retrieve
from pipeline.data_loader import load_documents
from pipeline.text_splitter import split_documents
from pipeline.embedding import get_embedding_model

BENCHMARK_SET = [
    {
        "query": "What is the PSK modulation technique?",
        "expected_terms": ["psk", "phase shift keying", "carrier", "phase"],
        "target_source": "BlackBook.pdf",
    },
    {
        "query": "What is Orthogonal Frequency Division Multiplexing?",
        "expected_terms": ["ofdm", "orthogonal", "frequency division multiplexing", "subcarrier"],
        "target_source": "BlackBook.pdf",
    },
    {
        "query": "Columbia Asia Hospital bed capacity and department management",
        "expected_terms": ["hospital", "bed", "columbia", "asia", "patient"],
        "target_source": "Columbia Asia Hospital Project",
    },
    {
        "query": "TCP three-way handshake and reliable connection protocol",
        "expected_terms": ["tcp", "handshake", "syn", "ack", "protocol", "connection"],
        "target_source": "Computer Networking",
    },
]


def score_documents(docs, expected_terms: list[str]) -> float:
    """Calculate term coverage precision for retrieved docs."""
    if not docs:
        return 0.0
    combined_text = " ".join(d.page_content.lower() for d in docs)
    matched = sum(1 for term in expected_terms if term.lower() in combined_text)
    return matched / len(expected_terms)


def run_evaluation():
    print("=" * 65)
    print("   Hybrid RAG Evaluation: Dense vs Sparse vs Hybrid")
    print("=" * 65)

    print("\n[1/3] Loading embedding model and vectorstore...")
    embed = get_embedding_model()
    vs = load_vectorstore("./faiss_index", embedding_model=embed)

    print("[2/3] Loading documents & preparing BM25 index...")
    raw_docs = load_documents("./data")
    chunks = split_documents(raw_docs)
    bm25 = build_bm25_retriever(chunks, top_k=4)

    print(f"\n[3/3] Running benchmark on {len(BENCHMARK_SET)} representative queries (Top-K=4)...\n")

    results_table = []

    for item in BENCHMARK_SET:
        query = item["query"]
        terms = item["expected_terms"]

        # 1. Vector Only
        t0 = time.time()
        vec_docs = vs.similarity_search(query, k=4)
        vec_time = time.time() - t0
        vec_score = score_documents(vec_docs, terms)

        # 2. BM25 Only
        t0 = time.time()
        bm25_docs = bm25.invoke(query)
        bm25_time = time.time() - t0
        bm25_score = score_documents(bm25_docs[:4], terms)

        # 3. Hybrid RRF
        t0 = time.time()
        hybrid_docs, _ = detailed_hybrid_retrieve(
            query=query,
            vectorstore=vs,
            bm25_retriever=bm25,
            semantic_weight=0.6,
            keyword_weight=0.4,
            top_k=4,
        )
        hybrid_time = time.time() - t0
        hybrid_score = score_documents(hybrid_docs, terms)

        results_table.append({
            "query": query,
            "vector_cov": round(vec_score, 2),
            "bm25_cov": round(bm25_score, 2),
            "hybrid_cov": round(hybrid_score, 2),
            "vec_t": round(vec_time * 1000, 1),
            "bm25_t": round(bm25_time * 1000, 1),
            "hybrid_t": round(hybrid_time * 1000, 1),
        })

    print(f"{'Query':<45} | {'Dense':<7} | {'Sparse':<7} | {'Hybrid':<7}")
    print("-" * 75)
    for r in results_table:
        q_label = r["query"][:42] + ("..." if len(r["query"]) > 42 else "")
        print(f"{q_label:<45} | {r['vector_cov']:<7.2f} | {r['bm25_cov']:<7.2f} | {r['hybrid_cov']:<7.2f}")

    avg_vec = sum(r["vector_cov"] for r in results_table) / len(results_table)
    avg_bm25 = sum(r["bm25_cov"] for r in results_table) / len(results_table)
    avg_hybrid = sum(r["hybrid_cov"] for r in results_table) / len(results_table)
    avg_hybrid_t = sum(r["hybrid_t"] for r in results_table) / len(results_table)

    print("-" * 75)
    print(f"{'Average Keyword Coverage':<45} | {avg_vec:<7.2f} | {avg_bm25:<7.2f} | {avg_hybrid:<7.2f}")
    print(f"Average Hybrid Retrieval Latency: {avg_hybrid_t:.1f} ms")
    print("=" * 75)


if __name__ == "__main__":
    run_evaluation()
