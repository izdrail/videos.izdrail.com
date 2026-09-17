#!/usr/bin/env python3
"""Repeatable, dependency-light selector quality/latency benchmark."""
import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media_scoring import rerank_pooled_candidates
from core.media.identity import candidate_identity


def evaluate(payload, top_k=5):
    started = time.perf_counter()
    candidates = payload.get("candidates_by_source", {})
    before = sum(len(v) for v in candidates.values())
    selected = rerank_pooled_candidates(
        narration_text=payload.get("narration_text", ""),
        keyword_text=payload.get("keyword_text"),
        candidates_by_source=candidates,
        top_k=top_k,
    )
    identities = [candidate_identity(item, item.get("_source")) for item in selected]
    return {
        "candidate_count": before,
        "unique_pool_count": len({candidate_identity(c, s) for s, values in candidates.items() for c in values}),
        "final_selection_size": len(selected),
        "unique_selected": len(set(identities)),
        "duplicate_rate": 0 if not selected else 1 - len(set(identities)) / len(selected),
        "mean_relevance": 0 if not selected else sum(x.get("_relevance_score", 0) for x in selected) / len(selected),
        "selection_time_ms": round((time.perf_counter() - started) * 1000, 3),
        "selected": [{"identity": str(i), "score": item.get("_score"), "source": item.get("_source")} for i, item in zip(identities, selected)],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    with open(args.dataset) as handle:
        print(json.dumps(evaluate(json.load(handle), args.top_k), indent=2))
