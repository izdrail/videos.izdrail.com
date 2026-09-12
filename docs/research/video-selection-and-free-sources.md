# Video Selection & Free Media Sources — Deep Research

**Date:** 2026-09-12 · **Scope:** improve the "brain simulator" selection pipeline for better clip choice, catalogue additional free/no-API-key video sources, and fix the failed-render restart path.
**Method:** full read of the repo pipeline, external literature review, and mining of the G06 patent library in `izdrail/maicoach` (`research/patents/library`, 2,902 full-text patents).

---

## 1. How the current selection pipeline works (as found in code)

| Stage | Where | What it does |
|---|---|---|
| Script → sentences | `main.py` | Splits narration into slides/sentences. |
| Keyword candidates | `core/nlp/keyword_extractor.py`, `core/nlp/neuron_extractor.py` | spaCy + local LLM (Ollama) + "neuron signal" heuristic scoring produce ranked keywords per sentence. |
| SNN engagement ("brain simulator") | `core/nlp/brain_simulator.py` | Brian2 Izhikevich spiking simulation over 12 named "regions" (reward, emotion, trust, fear, …). LLM/heuristic signals become input currents; spike rates are combined by **hand-set weights** into a `biological_score` in [-1, 3]. Optional via `use_snn`. |
| Sequence optimisation | `keyword_extractor.optimize_keyword_sequence` | Beam search (width 4) blending engagement + pairwise keyword embedding coherence − duplicate penalty. |
| Source search | `core/media/manager.py` | Searches ALL eligible providers in parallel (Openverse, Wikimedia, SearXNG, Internet Archive, YouTube/yt-dlp first; keyed Pexels/Pixabay/Unsplash/Giphy after), per query, with local-folder reuse and query-simplification fallback. |
| Pooled rerank | `media_scoring.py` | `rerank_pooled_candidates`: 55% CLIP relevance (narration vs **thumbnail**), 25% resolution/aspect quality, 10% source freshness, 10% source diversity, 5% DB `clip_performance`, +0.15 preferred-source boost. Returns top_k=1. |
| Temporal coherence | `core/visual/temporal_coherence.py` | Brightness/colour-histogram continuity between consecutive chosen clips. |
| Feedback | `core/database.py` `clip_performance` | Persisted per-clip performance reused as the 5% score term. |

## 2. Weaknesses found (code-grounded)

1. **SNN weights are fixed and unvalidated.** `_compute_biological_score` uses constants (reward 1.5, pain −2.5, …) with no learning from the feedback the system already logs. The SNN is a fixed nonlinear transform of its inputs — nothing today proves it ranks better than the heuristic it replaces, and `use_snn` defaults to off in the UI path.
2. **SNN memory priming is unvalidated.** `clear_memory()` IS called at job start (via `keyword_extractor.clear_used()`), so cross-job leakage is handled — but within one job the spike-rate priming across sentences is an untested assumption (a scary early sentence biases all later scores upward/downward).
3. **Relevance is thumbnail-only.** A stock clip's thumbnail is often a title card or first frame, not representative of the clip. One 5-second-timeout image decides 55% of the score.
4. **No semantic near-duplicate control.** `used_urls` blocks exact URL repeats only; the same scene from two providers (or two crops of one clip) can land on consecutive slides. `manager.search()` dedupes by canonical URL, but the bandit path does not dedupe pooled candidates at all.
5. **No diversity objective across slides.** Reranking is per-sentence top-1 with only a *source-level* diversity term (10%). Two slides can pick visually identical clips from different sources.
6. **Keyword↔theme mismatch is unchecked.** A clip can match the keyword but drift from the sentence's actual meaning (classic stock-footage failure: "apple" the fruit in a tech narration).
7. **No aesthetic/technical quality signal.** Resolution is the only quality proxy; a 4K but ugly/watermarked clip beats a beautiful 1080p one.
8. **Source learning is in-memory only.** `source_success_counts` resets every restart, so bad sources are re-tried forever and good ones re-earned.
9. **Failed jobs can't be restarted from the UI.** The Jobs dropdown only lists `status == 'completed'` jobs (`main.py` `_refresh_jobs`), so the "🔁 Retry Failed" button can never receive a failed job id — the exact "can't restart a failed render" complaint. (`JobManager.retry_job` itself works.)

## 3. Improvement directions (with sources)

### 3.1 Two-stage cross-modal consistency gate — from patent CN121661575B
*Operator low-quality video material determining method based on element object matching degree* ([abstract in library](https://github.com/izdrail/maicoach/blob/main/research/patents/library/CN121661575B/CN121661575B.md)) scores a clip twice: (a) **intra-clip consistency** — do the clip's own text/audio elements match its visual elements? and (b) **theme match** — do the theme keywords match those elements? A clip failing stage (a) is rejected before stage (b) is even computed.

**Applied here:** keep the cheap CLIP(text, thumbnail) score but split the text into two gates:
- Gate A (keyword↔thumbnail) — the current behaviour.
- Gate B (full narration sentence↔thumbnail) — catches the "apple fruit vs Apple Inc" drift the keyword alone misses.
Final relevance = `0.5·A + 0.5·B` (weights env-tunable). Zero new model cost: same model, second text embedding.

### 3.2 Diversity as a first-class objective: MMR / DPP — from patent CN121660751B + classic IR
*Advertisement video key frame identification* ([library](https://github.com/izdrail/maicoach/blob/main/research/patents/library/CN121660751B/CN121660751B.md)) builds single-frame scores, then runs a **Determinantal Point Process (DPP)** over the score×similarity kernel to pick a *diverse* key-frame set, and updates the model online with policy-gradient rewards from click data. DPP for diverse video summarization is established: Gong et al., *Diverse Sequential Subset Selection for Supervised Video Summarization*, NeurIPS 2014 ([paper](https://papers.neurips.cc/paper_files/paper/2014/file/5d3b9e06117de70a7e5076cc3ed89e18-Paper.pdf); seqDPP follow-up [arXiv:1807.10957](https://doi.org/10.48550/arxiv.1807.10957)).

**Applied here:** full DPP needs the whole clip set at once; our selection is per-sentence online. The streaming-equivalent is **Maximal Marginal Relevance** (Carbonell & Goldstein 1998, [paper](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)): `MMR = λ·relevance − (1−λ)·max similarity-to-already-selected`. Implementation: when reranking sentence N's pool, penalise each candidate by its CLIP-embedding similarity to the clips already chosen for slides 1..N−1 (embeddings we already compute for relevance, so the extra cost is one cosine per candidate). This kills near-duplicate consecutive slides without new models.

### 3.3 Learn the weights from logged feedback — from patents CN121660023B / CN121660023B-adjacent + LinUCB
*Reinforced-learning reward function construction* ([library](https://github.com/izdrail/maicoach/blob/main/research/patents/library/CN121660023B/CN121660023B.md)) uses a **two-stage decoupled reward**: train a reward model offline on rich signals, deploy it where those signals are unavailable. That is exactly our situation with the SNN: expensive, hand-weighted, unvalidated. The `clip_performance` + `keyword_selection_audit` tables already log outcomes; a tiny logistic/linear model (or per-region learned weights) fit on that history replaces the constants, and the SNN becomes the offline "rich signal" generator rather than a runtime dependency.

For **source selection**, the canonical method is the **contextual bandit LinUCB** (Li et al., WWW 2010, [arXiv:1003.0146](https://doi.org/10.48550/arxiv.1003.0146)): per-source ridge regression over context features (query embedding, orientation, media type) with an upper-confidence exploration bonus. It subsumes today's freshness/diversity heuristics, persists to SQLite, and explores dead sources just enough to notice when they recover. Related cold-start treatment: [arXiv:1405.7544](https://ar5iv.labs.arxiv.org/html/1405.7544).

### 3.4 Short/long-term memory for sources and styles — from patent CN121934722A
*Federal dynamic multi-modal memory recommendation* ([library](https://github.com/izdrail/maicoach/blob/main/research/patents/library/CN121934722A/CN121934722A.md)) keeps a **short-term buffer** (recent interactions, high sensitivity) and a **long-term buffer** (only updated when novelty ≥ threshold), sampling candidates from both. Applied: persist `source_success_counts` (long-term, per source per media-type) and a short-term per-job buffer; source ordering = long-term prior + short-term recency. Fixes weakness #8 with a one-table change.

### 3.5 Dynamic weight rebalancing — from patent CN121883133A
*Intelligent product recommendation based on a large model* ([library](https://github.com/izdrail/maicoach/blob/main/research/patents/library/CN121883133A/CN121883133A.md)) rebalances feature weights from implicit feedback (clicks vs cancels) via a weight-rebalancing tree. Applied: make the rerank weights (0.55/0.25/0.10/0.10/0.05) **config-driven and periodically re-fit** from `clip_performance` instead of frozen literals.

### 3.6 Aesthetic & technical quality signal
Resolution ≠ quality. The LAION **improved aesthetic predictor** ([github.com/christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)) is a tiny MLP on CLIP embeddings trained on 176k human aesthetic ratings — it rides on the embeddings we already compute, adding ~1ms per candidate. Optional env flag `AESTHETIC_SCORING=1`, weight folded into the quality term.

### 3.7 Better relevance models (when GPU budget allows)
The pipeline pins ViT-B-32. SigLIP-family checkpoints (e.g. via OpenCLIP `ViT-B-16-SigLIP`) measurably improve text↔image retrieval zero-shot, drop-in via the existing `model_name` parameter; multi-frame CLIP (sample 3–4 frames after download, re-verify before final render) addresses thumbnail-only scoring — long-video retrieval practice summarised in *A CLIP-Hitchhiker's Guide to Long Video Retrieval* ([arXiv:2205.08508](https://ar5iv.labs.arxiv.org/html/2205.08508)) and CLIP4Video-Sampling ([paper](https://www.scirp.org/pdf/jcc20241211_21732954.pdf)).

### 3.8 SNN-specific recommendations
- Keep the SNN **optional and offline** (weight-learning signal generator), not a runtime gate, until its ranking uplift vs the heuristic is measured on `keyword_selection_audit`.
- Call `clear_memory()` at job start; priming across unrelated jobs is leakage, not "memory".
- Fit the region weights from feedback (3.3) instead of the current constants.

## 4. Free, no-API-key video sources

Already integrated: **Openverse** (images/audio), **Wikimedia Commons**, **Internet Archive**, **YouTube via yt-dlp**, **SearXNG** (images). New candidates, verified 2026-09-12:

| Source | Access | Key? | License / attribution | Fit |
|---|---|---|---|---|
| **NASA Image & Video Library** | Real JSON API: `GET https://images-api.nasa.gov/search?media_type=video&q=…`, then follow the per-item asset manifest JSON | **No** (optional key only raises rate limits) | NASA-created media is free to use; occasional partner content needs a check ([API docs](https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf)) | Space/science/Earth footage; excellent for tech & documentary scripts |
| **Library of Congress** | Real JSON API: `GET https://www.loc.gov/search/?fo=json&q=…&fa=format:film,video` | **No** (rate-limited; docs: [loc.gov/apis/json-and-yaml](https://www.loc.gov/apis/json-and-yaml/), [endpoints](https://www.loc.gov/apis/json-and-yaml/requests/endpoints/), [limits](https://www.loc.gov/apis/json-and-yaml/working-within-limits/)) | Much is public domain / no known restrictions; per-item rights field | Historical/archival clips; unique texture no stock site has |
| **Mixkit** (Envato) | No API — scrape search pages `https://mixkit.co/free-stock-video/<slug>/` (working reference: [OpenMontage mixkit.py](https://github.com/calesthio/OpenMontage/blob/main/tools/video/stock_sources/mixkit.py)) | **No** | [Mixkit License](https://mixkit.co/license/): free for commercial use, no attribution | Curated modern B-roll; the highest average visual quality of the free sources |
| **Pond5 Public Domain** | Public-domain collection browsable/scrapable at `https://www.pond5.com/free` (reference: [OpenMontage pond5_pd.py](https://github.com/calesthio/OpenMontage/blob/main/tools/video/stock_sources/pond5_pd.py)) | **No** for the PD collection | Public domain (CC0-equivalent) | Historical/newsreel/early-cinema footage |
| **Videvo** | No public API — scrape only | **No** | Mixed: many clips under licenses **requiring attribution**; check per-clip ([licensing](https://www.videvo.net/licensing/)) | Use only with per-clip license capture; lower priority |
| **Dailymotion & 1,800+ sites** | URL-level via the existing yt-dlp path (`YouTubeAPI` generalises to a `YtDlpProvider`) | **No** | Per-video; must respect each site's terms | Opportunistic; last resort |
| ~~Coverr~~ | JSON API | **Yes (free registration)** | Free license | Excluded from "no key" list — needs an account key ([docs](https://api.coverr.co/docs/start/)) |
| ~~Europeana~~ | JSON API | **Yes (free registration)** | Open metadata | Excluded — key required |

**Recommendation:** add NASA + Library of Congress (real APIs, stable, no scraping fragility) and Mixkit (best quality-per-effort, scraping pattern already proven in the wild). Insert them into `preferred_order` after the existing open providers: NASA and LoC are niche, so they should not outrank Openverse/Wikimedia for generic queries — the pooled rerank decides per candidate anyway.

## 5. What was implemented (companion PRs)

1. **This document** (`docs/research/video-selection-and-free-sources.md`).
2. **Failed-render restart fix:** the Jobs dropdown now lists failed and canceled jobs (with status labels), so "🔁 Retry Failed" can actually target them; retry is rejected for completed/processing jobs with a clear message.
3. **Selection improvements (3.1 + 3.2):** two-gate relevance (keyword + full sentence), MMR diversity penalty against already-selected clips, and canonical-URL dedupe in the bandit pool. All weights env-tunable.
4. **New providers (4):** NASA, Library of Congress, Mixkit providers + manager registration + offline tests.

Larger items deliberately left as follow-ups (need logged data or GPU budget): LinUCB source bandit (3.3), learned rerank weights (3.5), aesthetic predictor (3.6), SigLIP upgrade + multi-frame verification (3.7), SNN weight fitting (3.8).
