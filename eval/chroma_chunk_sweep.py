"""Chroma chunk config sweep: build vault_* collections and run retrieval + generation eval for each."""

import argparse
import re
import subprocess
import sys

from eval.common import ROOT  # side effect: adds project root to sys.path

CONFIGS = [
    {"name": "vault_small", "size": 300, "overlap": 50},
    {"name": "vault_medium", "size": 500, "overlap": 100},
    {"name": "vault_large", "size": 1000, "overlap": 200},
]


def build_collection(config: dict) -> None:
    from chroma.store import build_index_for_config

    name, size, overlap = config["name"], config["size"], config["overlap"]
    print(f"\nBuilding collection '{name}' (chunk_size={size}, chunk_overlap={overlap})...")
    build_index_for_config(name, size, overlap)
    print(f"  Done.\n")


def run_retrieval_eval(collection_name: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "eval.evaluate_retrieval", "chroma", "--collection", collection_name],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def run_generation_eval(collection_name: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "eval.evaluate_generation", "chroma", "--collection", collection_name],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def parse_recall(text: str) -> str | None:
    """Extract last 'Recall...: XX.XX%' line if present."""
    m = re.search(r"Recall[^:]*:\s*([\d.]+)%", text)
    return f"{m.group(1)}%" if m else None


def parse_time_avg_median(text: str) -> tuple[str, str] | None:
    """Extract 'Time (avg): X.XXXs  |  Time (median): X.XXXs'. Returns (avg_str, median_str) or None."""
    m = re.search(r"Time \(avg\):\s*([\d.]+)s\s*\|\s*Time \(median\):\s*([\d.]+)s", text)
    return (m.group(1), m.group(2)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Chroma collections for multiple chunk configs and run retrieval + generation eval."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building collections (use existing vault_* collections).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval eval, not generation.",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Only run generation eval, not retrieval.",
    )
    args = parser.parse_args()

    print("Chroma chunk config sweep")
    print("Configs:", [c["name"] for c in CONFIGS])

    if not args.skip_build:
        for config in CONFIGS:
            build_collection(config)

    retrieval_recalls = {}
    retrieval_times = {}  # name -> (avg_str, median_str)
    generation_recalls = {}
    generation_times = {}  # name -> (avg_str, median_str)

    for config in CONFIGS:
        name = config["name"]
        if not args.generation_only:
            out = run_retrieval_eval(name)
            retrieval_recalls[name] = parse_recall(out)
            t = parse_time_avg_median(out)
            retrieval_times[name] = (f"{t[0]}s", f"{t[1]}s") if t else ("—", "—")
            print(out)
        if not args.retrieval_only:
            out = run_generation_eval(name)
            generation_recalls[name] = parse_recall(out)
            t = parse_time_avg_median(out)
            generation_times[name] = (f"{t[0]}s", f"{t[1]}s") if t else ("—", "—")
            print(out)

    # Summary table: Collection | Retrieval Recall | Retr Avg | Retr Med | Gen Recall | Gen Avg | Gen Med
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(
        f"{'Collection':<16} {'Ret Recall':<12} {'Ret Avg':<10} {'Ret Med':<10} "
        f"{'Gen Recall':<12} {'Gen Avg':<10} {'Gen Med':<10}"
    )
    print("-" * 95)
    for config in CONFIGS:
        name = config["name"]
        r = retrieval_recalls.get(name) or "—"
        ra, rm = retrieval_times.get(name) or ("—", "—")
        g = generation_recalls.get(name) or "—"
        ga, gm = generation_times.get(name) or ("—", "—")
        print(f"{name:<16} {r:<12} {ra:<10} {rm:<10} {g:<12} {ga:<10} {gm:<10}")
    print("=" * 95)


if __name__ == "__main__":
    main()
