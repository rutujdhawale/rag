"""
main.py

Primary terminal entrypoint for the Hybrid RAG system.
Features:
  - Senior-Product-Designer quality TUI with real-time streaming, command palette,
    hybrid retrieval telemetry, and source citations.
  - Sub-second instant startup via pre-built FAISS index & chunk cache.
  - CLI arguments support:
      python main.py                  # Launches interactive TUI
      python main.py --query "..."    # Single-shot fast query in terminal
      python main.py --telemetry      # Launch TUI with retrieval telemetry enabled
      python main.py --rebuild        # Force rebuild of FAISS & chunk cache
"""

import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ui.tui import run_tui


def main():
    parser = argparse.ArgumentParser(
        description="⚡ Hybrid RAG — Dense + Sparse AI Retrieval Engine (TUI & CLI)"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Run a single question non-interactively and print the answer."
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable detailed Reciprocal Rank Fusion (RRF) candidate telemetry."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-parsing documents in ./data and rebuild FAISS index."
    )

    args = parser.parse_args()

    try:
        run_tui(
            force_rebuild=args.rebuild,
            telemetry_default=args.telemetry,
            single_query=args.query,
        )
    except KeyboardInterrupt:
        print("\nSession interrupted. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()