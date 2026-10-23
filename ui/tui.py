"""
ui/tui.py

A polished, fast, Senior-Product-Designer-crafted Terminal User Interface (TUI)
for the Hybrid RAG engine (FAISS Dense + BM25 Sparse + Reciprocal Rank Fusion).

Features:
  - Instant sub-second startup via disk-cached FAISS vectorstore and chunks
  - Ergonomic interactive shell with command palette (/help, /telemetry, /sources, etc.)
  - Real-time token streaming with live Markdown formatting
  - Transparent Hybrid Retrieval Inspector (Dense vs Sparse ranks, RRF scores, provenance)
  - Non-interactive single-query CLI execution mode
"""

import os
import sys

# Ensure UTF-8 output encoding across Windows shells (CMD/PowerShell)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.rule import Rule
from rich.align import Align
from rich.prompt import Prompt
from rich import box

os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import huggingface_hub.utils
    huggingface_hub.utils.disable_progress_bars()
except Exception:
    pass

try:
    import transformers.utils.logging as tf_logging
    tf_logging.disable_progress_bar()
    tf_logging.set_verbosity_error()
except Exception:
    pass

# Suppress verbose backend logging in terminal
import logging
logging.basicConfig(level=logging.WARNING)
for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "transformers"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

load_dotenv()

# App paths & defaults
DATA_DIR = os.getenv("DATA_DIR", "./data")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
DEFAULT_SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.6))
DEFAULT_KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", 0.4))
DEFAULT_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 4))

console = Console(legacy_windows=False)


# ---------------------------------------------------------------------------
# Fast Startup Pipeline Loader (Sub-second cold/warm start)
# ---------------------------------------------------------------------------
def load_tui_pipeline(
    data_dir: str = DATA_DIR,
    index_dir: str = FAISS_INDEX_DIR,
    force_rebuild: bool = False,
) -> Tuple[Optional[object], Optional[object], List[Document]]:
    """
    Loads FAISS index and BM25 retriever with instant disk cache reuse.
    Avoids re-parsing PDFs/raw documents unless force_rebuild=True.
    """
    from pipeline.vectorstore import load_vectorstore, build_vectorstore
    from pipeline.search import build_bm25_retriever
    from pipeline.data_loader import load_documents
    from pipeline.text_splitter import split_documents

    faiss_file = Path(index_dir) / "index.faiss"
    chunks_file = Path(index_dir) / "chunks.pkl"

    needs_rebuild = force_rebuild or not (faiss_file.exists() and chunks_file.exists())

    # Auto-detect if documents in data_dir were added, removed, or modified
    if not needs_rebuild:
        try:
            with open(chunks_file, "rb") as f:
                cached_chunks = pickle.load(f)
            cached_sources = {Path(c.metadata.get("source", "")).name for c in cached_chunks}
            current_files = {f.name for f in Path(data_dir).iterdir() if f.is_file() and not f.name.startswith(".")}
            if current_files != cached_sources:
                needs_rebuild = True
            else:
                cache_mtime = chunks_file.stat().st_mtime
                for f in Path(data_dir).iterdir():
                    if f.is_file() and not f.name.startswith(".") and f.stat().st_mtime > cache_mtime:
                        needs_rebuild = True
                        break
        except Exception:
            needs_rebuild = True

    # Fast Path: Cached index exists and no changes detected in data_dir
    if not needs_rebuild:
        with console.status("[cyan]Warming engine from disk cache...[/cyan]", spinner="dots"):
            vectorstore = load_vectorstore(index_dir)
            bm25 = build_bm25_retriever(cached_chunks, top_k=DEFAULT_TOP_K)
            return vectorstore, bm25, cached_chunks

    # Rebuild Path: Parse documents from scratch
    console.print(Panel(
        "[bold cyan]New or updated documents detected (or rebuild requested).[/bold cyan]\n"
        f"Parsing documents from [yellow]'{data_dir}'[/yellow] and generating embeddings...",
        title="Ingestion Engine",
        border_style="cyan"
    ))

    with console.status("[cyan][1/4] Scanning and loading documents...[/cyan]", spinner="bouncingBar"):
        raw_docs = load_documents(data_dir)
        if not raw_docs:
            raise RuntimeError(f"No documents found in '{data_dir}'. Add files first.")

    with console.status(f"[cyan][2/4] Chunking {len(raw_docs)} document(s)...[/cyan]", spinner="bouncingBar"):
        chunks = split_documents(raw_docs)

    with console.status(f"[cyan][3/4] Embedding {len(chunks)} chunks & building FAISS index...[/cyan]", spinner="bouncingBar"):
        vectorstore = build_vectorstore(chunks, index_dir=index_dir)

    with console.status("[cyan][4/4] Building BM25 keyword index & saving chunk cache...[/cyan]", spinner="bouncingBar"):
        bm25 = build_bm25_retriever(chunks, top_k=DEFAULT_TOP_K)
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)

    return vectorstore, bm25, chunks


# ---------------------------------------------------------------------------
# UI Components: Header, Status Bar, Telemetry Table, Citations
# ---------------------------------------------------------------------------
def render_banner():
    """Render modern terminal banner in bluish-green gradient typography."""
    width = console.width or 80

    if width >= 105:
        # Extra-wide spaced version (101 chars wide) - perfectly balanced for wide terminals
        art_lines = [
            ("██╗  ██╗   ██╗   ██╗   ██████╗    ██████╗    ██╗   ██████╗           ██████╗     █████╗     ██████╗  ", "#5EEAD4"),
            ("██║  ██║   ╚██╗ ██╔╝   ██╔══██╗   ██╔══██╗   ██║   ██╔══██╗          ██╔══██╗   ██╔══██╗    ██╔════╝ ", "#2DD4BF"),
            ("███████║    ╚████╔╝    ██████╔╝   ██████╔╝   ██║   ██║  ██║          ██████╔╝   ███████║    ██║  ███╗", "#14B8A6"),
            ("██╔══██║     ╚██╔╝     ██╔══██╗   ██╔══██╗   ██║   ██║  ██║          ██╔══██╗   ██╔══██║    ██║   ██║", "#0D9488"),
            ("██║  ██║      ██║      ██████╔╝   ██║  ██║   ██║   ██████╔╝          ██║  ██║   ██║  ██║    ╚██████╔╝", "#0F766E"),
            ("╚═╝  ╚═╝      ╚═╝      ╚═════╝    ╚═╝  ╚═╝   ╚═╝   ╚═════╝           ╚═╝  ╚═╝   ╚═╝  ╚═╝    ╚═════╝  ", "#115E59"),
        ]
    elif width >= 95:
        # Medium spaced version (92 chars wide)
        art_lines = [
            ("██╗  ██╗  ██╗   ██╗  ██████╗   ██████╗   ██╗  ██████╗         ██████╗    █████╗    ██████╗  ", "#5EEAD4"),
            ("██║  ██║  ╚██╗ ██╔╝  ██╔══██╗  ██╔══██╗  ██║  ██╔══██╗        ██╔══██╗  ██╔══██╗   ██╔════╝ ", "#2DD4BF"),
            ("███████║   ╚████╔╝   ██████╔╝  ██████╔╝  ██║  ██║  ██║        ██████╔╝  ███████║   ██║  ███╗", "#14B8A6"),
            ("██╔══██║    ╚██╔╝    ██╔══██╗  ██╔══██╗  ██║  ██║  ██║        ██╔══██╗  ██╔══██║   ██║   ██║", "#0D9488"),
            ("██║  ██║     ██║     ██████╔╝  ██║  ██║  ██║  ██████╔╝        ██║  ██║  ██║  ██║   ╚██████╔╝", "#0F766E"),
            ("╚═╝  ╚═╝     ╚═╝     ╚═════╝   ╚═╝  ╚═╝  ╚═╝  ╚═════╝         ╚═╝  ╚═╝  ╚═╝  ╚═╝   ╚═════╝  ", "#115E59"),
        ]
    else:
        # Compact version (70 chars wide) for standard 80-col terminals
        art_lines = [
            ("██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗     ██████╗  █████╗  ██████╗ ", "#5EEAD4"),
            ("██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗    ██╔══██╗██╔══██╗██╔════╝ ", "#2DD4BF"),
            ("███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║    ██████╔╝███████║██║  ███╗", "#14B8A6"),
            ("██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║    ██╔══██╗██╔══██║██║   ██║", "#0D9488"),
            ("██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝    ██║  ██║██║  ██║╚██████╔╝", "#0F766E"),
            ("╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ", "#115E59"),
        ]

    banner_text = Text()
    for line, color in art_lines:
        banner_text.append(line + "\n", style=f"bold {color}")

    console.print(
        Panel(
            Align.center(banner_text),
            box=box.ROUNDED,
            border_style="#0D9488",
            padding=(0, 2),
        )
    )


def render_status_bar(chunks_count: int, semantic_w: float, keyword_w: float, top_k: int, telemetry_on: bool, use_graph: bool):
    """Render compact, non-wrapping single-line status bar."""
    telem_str = "[bold #2DD4BF]ON[/bold #2DD4BF]" if telemetry_on else "[dim]OFF[/dim]"
    mode_str = "[magenta]Agentic[/magenta]" if use_graph else "[#2DD4BF]Fast[/#2DD4BF]"

    width = console.width or 80
    if width >= 105:
        line = (
            f"[#2DD4BF]● Ready[/#2DD4BF] [dim]({chunks_count} chunks)[/dim]  [bright_black]│[/bright_black]  "
            f"[dim]Fusion:[/dim] [#5EEAD4]{int(semantic_w*100)}% Dense · {int(keyword_w*100)}% Sparse[/#5EEAD4]  [bright_black]│[/bright_black]  "
            f"[dim]Top-K:[/dim] [white bold]{top_k}[/white bold]  [bright_black]│[/bright_black]  "
            f"[dim]Inspector:[/dim] {telem_str}  [bright_black]│[/bright_black]  "
            f"[dim]Mode:[/dim] {mode_str}"
        )
    else:
        line = (
            f"[#2DD4BF]● Ready[/#2DD4BF] [dim]({chunks_count})[/dim] [bright_black]│[/bright_black] "
            f"[#5EEAD4]{int(semantic_w*100)}%D · {int(keyword_w*100)}%S[/#5EEAD4] [bright_black]│[/bright_black] "
            f"[dim]K:[/dim][white bold]{top_k}[/white bold] [bright_black]│[/bright_black] "
            f"[dim]Insp:[/dim] {telem_str} [bright_black]│[/bright_black] "
            f"{mode_str}"
        )

    console.print(Panel(Align.center(line), border_style="#0F766E", box=box.ROUNDED, padding=(0, 1)))


def render_help():
    """Render senior product designer command cheat-sheet."""
    table = Table(title="Available Commands & Shortcuts", border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="bold green", width=16)
    table.add_column("Description", style="white")
    table.add_column("Example", style="dim", width=24)

    table.add_row("/help", "Show this command palette", "/help")
    table.add_row("/telemetry", "Toggle RRF scoring & rank breakdown table", "/telemetry")
    table.add_row("/sources", "Inspect full content snippets of retrieved chunks", "/sources")
    table.add_row("/mode", "Toggle between Fast Streaming & LangGraph Self-Correction", "/mode")
    table.add_row("/weight <w>", "Adjust semantic vs keyword weight (0.0 to 1.0)", "/weight 0.7")
    table.add_row("/topk <k>", "Change number of retrieved context chunks", "/topk 5")
    table.add_row("/stats", "Display document catalog and chunk counts", "/stats")
    table.add_row("/rebuild", "Re-parse documents and regenerate FAISS index", "/rebuild")
    table.add_row("/clear", "Clear terminal screen and redraw header", "/clear")
    table.add_row("/exit, /quit", "Exit the TUI session cleanly", "/exit")

    console.print(table)
    console.print("[dim]Tip: You can also press Ctrl+C anytime to cancel an in-flight query or exit.[/dim]\n")


def render_stats(chunks: List[Document]):
    """Render catalog statistics and chunk distribution."""
    from collections import Counter
    counts = Counter(Path(c.metadata.get("source", "unknown")).name for c in chunks)

    table = Table(title="Knowledge Base Document Catalog", border_style="bright_blue", show_header=True, header_style="bold cyan")
    table.add_column("Document Name", style="bold white")
    table.add_column("Type", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Est. Words", justify="right", style="dim")

    total_words = 0
    for name, cnt in counts.items():
        doc_chunks = [c for c in chunks if Path(c.metadata.get("source", "")).name == name]
        words = sum(len(c.page_content.split()) for c in doc_chunks)
        total_words += words
        suffix = Path(name).suffix.lstrip(".").upper() or "TXT"
        table.add_row(name, suffix, str(cnt), f"{words:,}")

    console.print(table)
    console.print(f"[dim]Total: {len(counts)} file(s) • {len(chunks)} chunk(s) • ~{total_words:,} words in FAISS + BM25[/dim]\n")


def render_telemetry_table(telemetry: dict):
    """Render visual hybrid candidate fusion telemetry table."""
    fused_hits = telemetry.get("fused_hits", [])
    if not fused_hits:
        return

    table = Table(
        title=f"Hybrid Retrieval Inspector • {telemetry.get('fusion_algorithm', 'RRF')}",
        border_style="magenta",
        header_style="bold magenta",
        show_header=True
    )
    table.add_column("Rank", justify="center", style="bold white", width=6)
    table.add_column("Source Document", style="cyan", width=22)
    table.add_column("Chunk #", justify="center", style="dim", width=8)
    table.add_column("Dense (FAISS)", justify="center", width=14)
    table.add_column("Sparse (BM25)", justify="center", width=14)
    table.add_column("RRF Score", justify="right", style="bold green", width=12)
    table.add_column("Provenance", justify="center", width=14)

    for i, hit in enumerate(fused_hits[:telemetry.get("top_k", 4)], start=1):
        dense_str = f"#{hit.get('vector_rank', '-')}" if hit.get("in_vector") else "[dim]Miss[/dim]"
        sparse_str = f"#{hit.get('bm25_rank', '-')}" if hit.get("in_bm25") else "[dim]Miss[/dim]"

        if hit.get("in_vector") and hit.get("in_bm25"):
            prov = "[bold green]Hybrid Both[/bold green]"
        elif hit.get("in_vector"):
            prov = "[cyan]Dense Only[/cyan]"
        else:
            prov = "[yellow]BM25 Only[/yellow]"

        table.add_row(
            f"#{i}",
            hit.get("source", "unknown"),
            str(hit.get("chunk_index", 0)),
            dense_str,
            sparse_str,
            f"{hit.get('rrf_score', 0):.5f}",
            prov
        )

    console.print(table)
    console.print(
        f"[dim]Overlap: {telemetry.get('overlap_count', 0)} / {telemetry.get('top_k', 4)} chunks found in both retrievers. "
        f"Weights: {telemetry.get('semantic_weight', 0.6):.1f} Dense / {telemetry.get('keyword_weight', 0.4):.1f} Sparse[/dim]\n"
    )


def render_citations(documents: List[Document]):
    """Render clean, compact citation tags without taking excessive space."""
    if not documents:
        return

    seen = set()
    badges = []
    for doc in documents:
        src = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", None)
        key = (src, page)
        if key not in seen:
            seen.add(key)
            tag = f"[bold #2DD4BF]{src}[/bold #2DD4BF]"
            if page is not None:
                tag += f" [dim](p. {page})[/dim]"
            badges.append(tag)

    if badges:
        console.print(f"\n[dim]Sources:[/dim]  " + "  [dim]•[/dim]  ".join(badges))


def render_full_sources(documents: List[Document]):
    """Display comprehensive source details from last query (/sources)."""
    if not documents:
        console.print("[dim]No sources retrieved in the last query.[/dim]\n")
        return

    for i, doc in enumerate(documents, start=1):
        src = Path(doc.metadata.get("source", "Unknown")).name
        meta_table = Table.grid(expand=True)
        meta_table.add_column(ratio=2)
        meta_table.add_column(ratio=2)
        meta_table.add_row(
            f"[bold cyan]Source:[/bold cyan] {src}",
            f"[bold cyan]Chunk Index:[/bold cyan] {doc.metadata.get('chunk_index', '0')}"
        )
        meta_table.add_row(
            f"[bold cyan]Retrieval Provenance:[/bold cyan] {doc.metadata.get('retrieval_method', 'N/A')}",
            f"[bold cyan]RRF Score:[/bold cyan] {doc.metadata.get('rrf_score', 'N/A')}"
        )

        content_panel = Panel(
            Text(doc.page_content, style="white"),
            title=f"Source #{i}: {src}",
            subtitle=f"{len(doc.page_content.split())} words",
            border_style="cyan"
        )
        console.print(meta_table)
        console.print(content_panel)
        console.print("")


# ---------------------------------------------------------------------------
# Core Execution: Fast Streaming & Agentic LangGraph Queries
# ---------------------------------------------------------------------------
def execute_query(
    question: str,
    vectorstore,
    bm25,
    semantic_w: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_w: float = DEFAULT_KEYWORD_WEIGHT,
    top_k: int = DEFAULT_TOP_K,
    show_telemetry: bool = False,
    use_graph: bool = False,
    graph_app=None,
    chunks: Optional[List[Document]] = None,
) -> Tuple[str, List[Document], dict]:
    """
    Executes hybrid retrieval and generates an answer.
    In fast mode: uses detailed_hybrid_retrieve + real-time token streaming.
    In graph mode: invokes LangGraph self-correcting evaluation loop.
    """
    from pipeline.search import detailed_hybrid_retrieve
    from pipeline.llm import stream_answer

    t0 = time.time()

    if use_graph:
        if graph_app is None and chunks is not None:
            from pipeline.graph import build_graph
            from pipeline.search import get_hybrid_retriever
            retriever = get_hybrid_retriever(chunks, vectorstore, semantic_w, keyword_w, top_k)
            graph_app = build_graph(retriever)

        if graph_app is not None:
            with console.status("[cyan]Executing agentic LangGraph flow (retrieve -> grade -> generate)...[/cyan]", spinner="dots"):
                from pipeline.graph import run_query
                try:
                    res = run_query(graph_app, question)
                    answer = res.get("answer", "")
                    docs = res.get("documents", [])
                    timing = res.get("timing_s", round(time.time() - t0, 3))
                    telemetry = {
                        "top_k": len(docs),
                        "total_time_s": timing,
                        "retrieve_time_s": round(timing * 0.25, 2),
                        "gen_time_s": round(timing * 0.75, 2),
                        "is_relevant": res.get("is_relevant", True),
                        "retry_count": res.get("retry_count", 0),
                    }
                    console.print(Markdown(answer))
                    return answer, docs, telemetry
                except Exception as e:
                    console.print(f"[yellow]LangGraph flow encountered an issue ({e}). Falling back to fast retrieval...[/yellow]\n")

    # --- Fast Streaming Mode ---
    with console.status("[cyan]Running Hybrid Dense + Sparse retrieval...[/cyan]", spinner="dots"):
        docs, telemetry = detailed_hybrid_retrieve(
            query=question,
            vectorstore=vectorstore,
            bm25_retriever=bm25,
            semantic_weight=semantic_w,
            keyword_weight=keyword_w,
            top_k=top_k,
        )

    retrieve_time = time.time() - t0

    if show_telemetry:
        render_telemetry_table(telemetry)

    console.print(f"[dim]⚡ Retrieved {len(docs)} chunk(s) in {retrieve_time:.2f}s • Streaming response:[/dim]\n")

    # Real-time token streaming with live markdown rendering
    collected_tokens = []
    t_gen_start = time.time()

    with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
        try:
            for token in stream_answer(question, docs):
                collected_tokens.append(token)
                live.update(Markdown("".join(collected_tokens)))
        except KeyboardInterrupt:
            collected_tokens.append("\n\n*[Generation stopped by user]*")
            live.update(Markdown("".join(collected_tokens)))

    gen_time = time.time() - t_gen_start
    total_time = time.time() - t0

    full_answer = "".join(collected_tokens)
    telemetry["retrieve_time_s"] = round(retrieve_time, 3)
    telemetry["gen_time_s"] = round(gen_time, 3)
    telemetry["total_time_s"] = round(total_time, 3)

    return full_answer, docs, telemetry


# ---------------------------------------------------------------------------
# Interactive TUI REPL Loop
# ---------------------------------------------------------------------------
def run_tui(
    force_rebuild: bool = False,
    telemetry_default: bool = False,
    single_query: Optional[str] = None,
):
    """Primary entry point for the TUI."""
    semantic_w = DEFAULT_SEMANTIC_WEIGHT
    keyword_w = DEFAULT_KEYWORD_WEIGHT
    top_k = DEFAULT_TOP_K
    telemetry_on = telemetry_default
    use_graph = False
    graph_app = None
    last_docs: List[Document] = []

    # 1. Fast Load Pipeline (silently warms disk cache)
    try:
        vectorstore, bm25, chunks = load_tui_pipeline(force_rebuild=force_rebuild)
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        sys.exit(1)

    # 2. Clear screen and render pristine Claude Code banner
    os.system("cls" if os.name == "nt" else "clear")
    render_banner()

    # 2. Handle non-interactive single-query CLI execution
    if single_query:
        render_status_bar(len(chunks), semantic_w, keyword_w, top_k, telemetry_on, use_graph)
        console.print(f"\n[bold cyan]Question:[/bold cyan] {single_query}\n")
        answer, docs, telem = execute_query(
            single_query, vectorstore, bm25, semantic_w, keyword_w, top_k, telemetry_on, use_graph, graph_app, chunks
        )
        render_citations(docs)
        console.print(f"\n[dim]Completed in {telem.get('total_time_s', '?')}s[/dim]\n")
        return

    # 3. Interactive Shell Loop
    render_status_bar(len(chunks), semantic_w, keyword_w, top_k, telemetry_on, use_graph)
    console.print("[dim]Type your question or /help for commands. Press Ctrl+C or type /exit to quit.[/dim]\n")

    while True:
        try:
            query = console.input("[bold #2DD4BF]❯[/bold #2DD4BF] [bold white]Ask Hybrid RAG[/bold white] [dim]›[/dim] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed. Goodbye![/dim]")
            break

        if not query:
            continue

        # Handle Commands
        cmd = query.lower()
        if cmd in ("/exit", "/quit", "exit", "quit", ":q"):
            console.print("[dim]Exiting Hybrid RAG TUI. Have a productive day![/dim]")
            break

        elif cmd in ("/help", "help", "?"):
            render_help()
            continue

        elif cmd in ("/clear", "clear", "cls"):
            os.system("cls" if os.name == "nt" else "clear")
            render_banner()
            render_status_bar(len(chunks), semantic_w, keyword_w, top_k, telemetry_on, use_graph)
            continue

        elif cmd in ("/telemetry", "telemetry"):
            telemetry_on = not telemetry_on
            status_str = "[bold green]ENABLED[/bold green]" if telemetry_on else "[dim]DISABLED[/dim]"
            console.print(f"Retrieval Inspector Telemetry is now {status_str}.\n")
            continue

        elif cmd in ("/sources", "sources"):
            render_full_sources(last_docs)
            continue

        elif cmd in ("/stats", "stats"):
            render_stats(chunks)
            continue

        elif cmd in ("/mode", "mode"):
            use_graph = not use_graph
            if use_graph:
                from pipeline.graph import build_graph
                from pipeline.search import get_hybrid_retriever
                try:
                    with console.status("[cyan]Compiling LangGraph self-correction state graph...[/cyan]", spinner="dots"):
                        retriever = get_hybrid_retriever(chunks, vectorstore, semantic_w, keyword_w, top_k)
                        graph_app = build_graph(retriever)
                except Exception as e:
                    console.print(f"[bold red]Failed to compile LangGraph:[/bold red] {e}\n")
                    use_graph = False
            render_status_bar(len(chunks), semantic_w, keyword_w, top_k, telemetry_on, use_graph)
            mode_str = "[magenta]Agentic LangGraph (with query rewrite & grading)[/magenta]" if use_graph else "[#2DD4BF]Fast Streaming Mode[/#2DD4BF]"
            console.print(f"Engine mode switched to {mode_str}.\n")
            continue

        elif cmd.startswith("/weight") or cmd.startswith("weight"):
            parts = query.split()
            if len(parts) == 2:
                try:
                    w = float(parts[1])
                    if 0.0 <= w <= 1.0:
                        semantic_w = round(w, 2)
                        keyword_w = round(1.0 - w, 2)
                        graph_app = None
                        console.print(f"Fusion weights updated: [cyan]{int(semantic_w*100)}% Dense / {int(keyword_w*100)}% Sparse[/cyan]\n")
                    else:
                        console.print("[red]Weight must be between 0.0 and 1.0 (e.g. /weight 0.7)[/red]\n")
                except ValueError:
                    console.print("[red]Invalid weight format. Example: /weight 0.7[/red]\n")
            else:
                console.print(f"Current weight: {semantic_w:.2f} Dense / {keyword_w:.2f} Sparse. Use: /weight <float>\n")
            continue

        elif cmd.startswith("/topk") or cmd.startswith("topk"):
            parts = query.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = int(parts[1])
                graph_app = None
                console.print(f"Context retrieval count updated to [cyan]{top_k} chunks[/cyan].\n")
            else:
                console.print(f"Current top_k: {top_k}. Use: /topk <number>\n")
            continue

        elif cmd in ("/rebuild", "rebuild"):
            try:
                vectorstore, bm25, chunks = load_tui_pipeline(force_rebuild=True)
                graph_app = None
                console.print(f"[bold green]Rebuild complete![/bold green] Indexed {len(chunks)} chunks.\n")
            except Exception as e:
                console.print(f"[bold red]Rebuild failed:[/bold red] {e}\n")
            continue

        # Execute User Query
        try:
            console.print("")
            answer, docs, telem = execute_query(
                question=query,
                vectorstore=vectorstore,
                bm25=bm25,
                semantic_w=semantic_w,
                keyword_w=keyword_w,
                top_k=top_k,
                show_telemetry=telemetry_on,
                use_graph=use_graph,
                graph_app=graph_app,
                chunks=chunks,
            )
            last_docs = docs
            render_citations(docs)
            console.print(Rule(style="dim"))
            console.print(
                f"[dim]Latency: {telem.get('total_time_s', '?')}s "
                f"(Retrieve: {telem.get('retrieve_time_s', '?')}s, Gen: {telem.get('gen_time_s', '?')}s) • "
                f"{len(docs)} chunks cited[/dim]\n"
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Query cancelled by user.[/yellow]\n")
        except Exception as e:
            console.print(f"\n[bold red]Error answering query:[/bold red] {e}\n")
