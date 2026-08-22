# Voice RAG — MSMARCO-XI

Multilingual voice-enabled RAG pipeline over `ai4bharat/MSMARCO-XI`. A user speaks a question in Hindi, Marathi, Tamil, or Telugu; the system transcribes it, retrieves grounded evidence, and returns a cited answer — or explicitly abstains if the evidence doesn't support one.

```
audio → Sarvam STT → input guardrail → BGE-M3 embed
      → hybrid dense+sparse retrieval (Qdrant) → RRF fusion
      → BGE-reranker-v2-m3 → confidence gate
      → Groq (gpt-oss-20b) → citation + NLI grounding → answer
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally on port 6333 (`docker run -p 6333:6333 qdrant/qdrant`)
- API keys: `GROQ_API_KEY` (required for generation), `SARVAM_API_KEY` (required for voice input)

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Fill in GROQ_API_KEY and SARVAM_API_KEY
```

### Build the index

```bash
# Demo profile: 70 rows × 4 languages (~8,500 chunks, takes ~50 min on CPU)
python scripts/build_index.py --mode demo

# Dev profile: 200 rows × hi+mr only, faster
python scripts/build_index.py --mode dev
```

### Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser. Type a question or hold the mic button to speak.

### Without API keys (text-only, mock generation)

```bash
GENERATION_BACKEND=mock uvicorn app.main:app --port 8000
```

The pipeline runs end-to-end; generation returns an extractive answer from the top retrieved passage. Set `SARVAM_API_KEY` to also enable voice input.

---

## Latency numbers

Measured on CPU (Intel i5-1240P, 16 threads, int8 reranker), demo corpus, 40 test queries, language-filtered retrieval. No STT — RAG-only latency from transcript received to validated answer.

### Retrieval only (no reranker)

| arm         | P50 ms | P95 ms | mean ms | recall@5 | nDCG@10 |
|-------------|-------:|-------:|--------:|--------:|--------:|
| dense       |  38.2  | 132.3  |   84.9  |  0.78   |  0.65   |
| sparse      |  14.9  |  78.2  |   24.0  |  0.60   |  0.58   |
| hybrid RRF  |  28.6  | 131.1  |   45.2  |  0.79   |  0.64   |

Hybrid RRF is the default retrieval mode.

### Full pipeline (retrieval + rerank, top-10)

| arm            | P50 ms  | P95 ms   | mean ms  | recall@5 | nDCG@10 |
|----------------|--------:|---------:|---------:|--------:|--------:|
| hybrid+rerank  | 2,170.7 | 12,727.9 |  5,085.8 |  0.79   |  0.60   |

The reranker (bge-reranker-v2-m3, ~560M params) dominates on CPU. The retrieval-only path (hybrid RRF, P50 29ms) comfortably meets the 200ms target. Full rerank does not on CPU; a GPU reduces this to ~50–150ms.

### Confidence gate (calibrated thresholds)

| metric    | value  |
|-----------|--------|
| threshold | -0.978 |
| precision | 0.72   |
| recall    | 0.89   |
| F1        | 0.79   |

Thresholds were fitted on the calibration split (40% of queries, disjoint from test). The chosen operating point maximises F1 — earlier versions used a precision≥0.85 floor which caused ~77% of answerable queries to abstain.

---

## Architecture

### Chunking (4 strategies, offline)

Every passage gets its native (verbatim) representation plus at most one child strategy chosen by passage shape:

| passage shape                  | child strategy       |
|-------------------------------|----------------------|
| < 3 sentences                 | none                 |
| ≥ 3 sentences, normal length  | `sentence_window`    |
| ≥ 320 tokens                  | `semantic_split`     |
| ≥ 1024 tokens (pathological)  | `fixed_fallback`     |

- **native** — whole passage verbatim; always indexed; used as generation context
- **sentence_window** — overlapping windows of 2 sentences, stride 1; ensures facts straddling a sentence boundary appear whole in at least one chunk
- **semantic_split** — BGE-M3 sentence embeddings → cosine-distance breakpoints at the top-25% most-dissimilar consecutive pairs; merges undersized segments
- **fixed_fallback** — XLM-R tokenizer-aligned 256-token windows with ~17.5% overlap for pages and tables without sentence punctuation

Child chunks are indexed for retrieval; `expand_to_parents()` collapses back to full parent passages before generation so the LLM always sees the complete context.

Demo corpus breakdown (70 rows × 4 languages):

| language | native | sentence_window | semantic_split | fixed_fallback | total chunks |
|----------|-------:|----------------:|---------------:|---------------:|-------------:|
| hi       |    699 |           1,496 |              0 |             12 |        2,207 |
| mr       |    699 |           1,474 |              0 |             25 |        2,198 |
| ta       |    699 |           1,316 |              4 |              7 |        2,026 |
| te       |    698 |           1,368 |              0 |             12 |        2,078 |

### Retrieval

- **Embedder**: `BAAI/bge-m3` — dense (1024-d CLS, L2-normalised) + sparse (learned lexical weights via `sparse_linear`)
- **Store**: Qdrant with named `dense` + `sparse` vector fields; payload indexes on `language`, `strategy`, `parent_id`
- **Fusion**: Reciprocal Rank Fusion (`k=60`) over concurrent dense and sparse branch queries
- **Reranker**: `BAAI/bge-reranker-v2-m3`, int8-quantised on CPU (5,230ms → 4,063ms with margin preserved)
- **Language routing**: Sarvam returns a language tag + confidence. Confident single-language → language-filtered search. Uncertain or code-mixed → cross-lingual search across all indexed languages

### Guardrails (5 layers)

1. **Input safety** — compiled-regex tier-1 (self-harm, weapons, injection, hate, illicit); tier-2 informational-framing adjudicator optional
2. **Confidence gate** — pre-LLM; abstains if reranker top score < threshold or margin too narrow
3. **Citation validation** — every cited ID must exist in the retrieved set; set membership, microseconds
4. **NLI entailment** — `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`; each factual answer sentence checked against cited evidence; deterministic string-containment shortcut for extractive answers
5. **Output escalation** — citations invalid → ABSTAIN; unsupported sentences → REGENERATE once (narrowed context) → ABSTAIN if still fails

Never fails open: model error, unavailable service, or timeout → abstain.

### Pipeline harness

- Formal state machine (`ALLOWED_TRANSITIONS`) — illegal stage hops raise `StateMachineError`
- Typed Pydantic schemas at every stage boundary: `STTResult → GuardrailResult → QueryEmbeddingResult → RetrievalResult → RerankResult → GroundingDecision → GenerationResult → ValidationResult → FinalResponse`
- CPU-bound work (embed, rerank, NLI) dispatched via `asyncio.to_thread`
- Bounded retries with exponential backoff on every external call (Qdrant, Sarvam, Groq)
- Per-request latency breakdown across 15 named stages, P50/P70/P95/P100 via in-process ring buffer

### STT

Sarvam Saaras v3 (`saaras:v3`), WebSocket streaming with REST fallback. Primary endpoint: `wss://api.sarvam.ai/speech-to-text/ws` (not the translate endpoint, which forces English output). Language set to `unknown` for auto-detection; confidence feeds the language-routing decision in retrieval. Code-mixing detection (Latin vs Indic codepoint ratio) forces cross-lingual search when triggered.

---

## Project layout

```
app/
  pipeline/       orchestrator + state machine
  stt/            Sarvam client (WS streaming + REST fallback)
  chunking/       4 strategies + routing engine
  retrieval/      embedder, Qdrant store, hybrid retriever, reranker, confidence gate
  generation/     Groq client, prompt builder, mock test double
  guardrails/     input safety, citation validator, NLI grounder, output escalation
  observability/  metrics registry (ring buffer, percentiles), structured tracing
  schemas/        Pydantic models for every stage boundary
  evaluation/     Recall@k, MRR, nDCG
scripts/
  build_index.py           offline ingestion
  benchmark_latency.py     P50/P70/P95/P100 report
  calibrate_thresholds.py  fits confidence gate on calibration split
  evaluate_retrieval.py    Recall/MRR/nDCG on test split
  evaluate_chunking.py     per-strategy retrieval comparison
configs/
  dev.yaml / demo.yaml / full.yaml   corpus profiles
  thresholds.json                    calibrated gate thresholds
reports/
  retrieval_eval_topk10.md   retrieval metrics (40 queries, 4 arms)
  calibration.md             threshold sweep table
frontend/
  index.html / app.js / styles.css
```

---

## Running evaluations

```bash
# Retrieval eval on test split
python scripts/evaluate_retrieval.py

# Latency benchmark (200 queries, balanced across languages)
python scripts/benchmark_latency.py

# Re-calibrate thresholds after changing the reranker or device
python scripts/calibrate_thresholds.py
```

Reports are written to `reports/`.

---

## Known limitations

- **Reranker latency on CPU**: P50 ~2s, P95 ~13s on the demo corpus. The retrieval-only path (hybrid RRF) meets 200ms at P50 (29ms). GPU inference or a lighter cross-encoder reduces full-pipeline latency to the 200ms range.
- **Demo corpus is small**: 70 rows × 4 languages (~8,500 chunks). The `full.yaml` profile covers 13 languages without a row cap but requires GPU for practical indexing time.
- **Telugu**: No train split upstream; demo uses the validation split, which may differ in distribution.
- **STT latency not in benchmarks**: `benchmark_latency.py` measures RAG-only latency (from transcript). End-to-end voice latency (including Sarvam round trip) requires pre-recorded audio clips and a valid `SARVAM_API_KEY`.
