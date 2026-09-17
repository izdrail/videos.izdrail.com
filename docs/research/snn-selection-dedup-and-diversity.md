# SNN selection: deduplication and diversity

## Current approach and root cause

The SNN scores and sequences keywords. It does not retrieve videos. The failure crossed three layers:

1. Beam history held only embeddings. If an embedding was unavailable, an exact repeated keyword was not visible to the duplicate check.
2. Repeated queries reused the same cached file. Parallel slide searches also checked and updated shared used-URL state separately, allowing two workers to choose one candidate.
3. The final slide-to-file map had no source-identity uniqueness boundary, including for symlink or path aliases from manual overrides.

## Alternatives investigated

- Plain cosine similarity is a strong relevance baseline but repeatedly chooses near-identical high scorers.
- Maximum Marginal Relevance (MMR) balances relevance against similarity to already selected results. It is a small re-ranking step and fits the existing CLIP scorer.
- Clustering can improve set diversity, but needs a cluster-count policy and gives less predictable ordering for small pools.
- Perceptual hashes are cheap near-duplicate signals for sampled frames, but require decoding video frames and threshold calibration. They are best added later as an offline/index-time signal.
- ANN indexes such as FAISS matter when an in-memory candidate pool becomes too large for exact ranking. The current provider pool is small, so a new vector service or index would add complexity without a measured need.
- Full video embeddings can capture temporal content better than a thumbnail, but add model and inference cost. The existing CLIP thumbnail path remains the practical first stage.

## Chosen approach

The pipeline now uses strong exact identity first (provider + stable asset ID, then canonical URL), keeps the existing two-gate CLIP relevance score, and applies MMR against previous choices. Selection state is atomically reserved during concurrent retrieval. The final generation boundary canonicalises real file identity using resolved path plus device/inode and removes any duplicate that still reaches it.

This stays additive: SNN can remain on or off; `MS_GATE_B_WEIGHT` and `MS_MMR_LAMBDA=1` can restore simpler ranking behaviour.

## Evaluation

Run the same saved candidate payload through:

```bash
python scripts/evaluate_selection.py examples/selection-candidates.json --top-k 5
```

The report includes candidate count, unique pool count, final and unique sizes, duplicate rate, mean relevance and latency. Regression fixtures with three identical references now select one unique source (duplicate rate changes from 66.7% to 0%). The dependency-light 17-test selection suite completed in about 1.7 seconds locally. Real provider/API latency and production-library quality should be recorded from representative saved payloads; this change does not claim an unmeasured live speedup.

## Observability

Each completed download records candidate count, unique candidate count, duplicates removed, selected identity and score in `MediaManager.last_selection_stats`. The final guard logs how many slide assets it removed, without logging embeddings.

## Trade-offs and follow-ups

- Different providers may mirror identical bytes under different IDs and URLs. Exact identity cannot prove those are the same. Add sampled-frame pHash only after collecting false-positive/false-negative fixtures.
- A duplicate removed at the final boundary falls back to the existing gradient path rather than blocking generation. A future candidate API can replace it interactively.
- The existing preview already lets users inspect candidates by slide, choose a source, clear selections and apply explicit per-slide gradient/file overrides. This change makes the state safe, but a fully downloaded clip reorder/replace browser needs a dedicated candidate endpoint rather than more client-only state.

## Sources

- Carbonell and Goldstein, MMR diversity reranking: https://www.cs.cmu.edu/~jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf
- OpenAI CLIP paper: https://arxiv.org/pdf/2103.00020
- FAISS documentation: https://faiss.ai/index.html
- OpenCV pHash documentation: https://docs.opencv.org/4.13.0/df/d4e/classcv%5F1%5F1img%5F%5Fhash%5F1%5F1PHash.html
- Milvus video similarity pipeline (engineering reference): https://milvus.io/docs/v2.6.x/video_similarity_search.md
