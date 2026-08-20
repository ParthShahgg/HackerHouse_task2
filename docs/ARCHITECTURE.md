# Architecture

Design decisions and the reasoning behind them. Measured results live in
[`reports/`](../reports); this document explains *why* the system is shaped the
way it is, including the places where the honest answer is "the hardware could
not meet the target".

---

## 1. Shape of the system

```
                        ┌──────────────── OFFLINE (never on the query path) ────────────────┐
                        │                                                                   │
   HF: ai4bharat/MSMARCO-XI                                                                 │
        │  stream parquet row groups (pre_buffer=False, column projection)                   │
        ▼                                                                                    │
   normalise (NFC, whitespace, control chars)                                                │
        │                                                                                    │
        ▼                                                                                    │
   deduplicate   sha256(language + normalized_text)  ──►  eval label store                   │
        │                                                  (query, Answer, is_selected)      │
        ▼                                                        │                           │
   chunk: native + at most one child strategy                     │  data/eval/ only         │
        │                                                        │  NEVER indexed            │
        ▼                                                        ▼                           │
   BGE-M3 encode (dense 1024-d + learned sparse)          retrieval / chunking evaluation    │
        │                                                  threshold calibration             │
        ▼                                                                                    │
   Qdrant collection: named vectors {dense, sparse}                                          │
                        └───────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────── ONLINE (latency-critical) ────────────────────────────┐
   │                                                                                    │
   │  microphone ─► Sarvam Saaras v3 (streaming WS, REST fallback)                       │
   │                     │ transcript + detected language + confidence                   │
   │                     ▼                                                               │
   │              input guardrail (normalise → tiered safety)                            │
   │                     ▼                                                               │
   │              BGE-M3 query encode (dense + sparse)                                   │
   │                     ▼                                                               │
   │              hybrid retrieval: dense ∥ sparse ──► RRF                               │
   │                     ▼                                                               │
   │              bge-reranker-v2-m3 (cross-encoder, top-30)                             │
   │                     ▼                                                               │
   │              confidence gate (CALIBRATED)  ──► ABSTAIN (LLM never called)           │
   │                     ▼ GENERATE                                                      │
   │              parent expansion (children → parent passages, deduped)                 │
   │                     ▼                                                               │
   │              Groq openai/gpt-oss-20b, structured output {answer, citations}          │
   │                     ▼                                                               │
   │              output guardrail: citation membership → NLI entailment                  │
   │                     ├─ PASS       → answer                                          │
   │                     ├─ REGENERATE → once, supported context only                     │
   │                     └─ ABSTAIN                                                      │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

The hard boundary between the two halves is the central latency decision. **The
live path never chunks, never embeds a passage, never touches the dataset.** It
does one query encode, two vector searches, one cross-encoder pass, one LLM call
and one entailment pass.

---

## 2. Dataset reality, and what it forced

Verified against the live repo (not the README) on 2026-08:

| fact | consequence for this repo |
|---|---|
| Every shard is a **single Parquet row group** (`train/hintrain.parquet`: 778,638 rows, 3.72 GB compressed / 9.73 GB uncompressed) | Row-group iteration is useless. Bounded-memory streaming relies on `pre_buffer=False` + page-level reads. |
| `passages.English_passages` is ~38% of the bytes | Nested column projection (`passages.Translated_passages`, `passages.is_selected`) skips it. |
| **`train/teltrain.parquet` does not exist** although the README lists `teltrain.jsonl` | `app/languages.py` records `has_train=False` for Telugu; the corpus builder transparently falls back to `validation` and logs it. |
| File stems are **not** ISO-639-1 and not derivable by truncation (`or`→`ori`, `pa`→`pan`, `te`→`tel`) | Explicit mapping table, unit-tested. |
| Repo ships a legacy loading script; `datasets>=4` refuses to execute it | The documented `load_dataset("ai4bharat/MSMARCO-XI", "hi")` API is dead. We read parquet directly with pyarrow, which is also what makes streaming possible. |

Measured on `validation/tamval.parquet`: first 64-row batch in ~18.5 s (connection
+ first pages), the next 576 rows in ~2.0 s. So `--max-rows-per-language`
genuinely bounds work rather than pretending to.

**55.6 GB is never downloaded.** Only the selected language shards are touched,
and only the projected columns, and only up to the row cap.

---

## 3. What is indexed, and what must never be

Indexed: **translated passage text only.**

Deliberately excluded from the index:

- the dataset `query`
- the dataset `Answer`
- `is_selected`
- translation metadata

`Answer` is ground truth for evaluation. `is_selected` is ground truth for
retrieval evaluation. Indexing either — or building a `query + answer + passage`
document — would leak the answer into the corpus and make every retrieval metric
meaningless.

This is enforced **structurally, not by convention**:

- `Chunk` (the only thing uploaded) has no field for a label. `Chunk.to_payload()`
  emits a fixed key set.
- `EvalExample` (the only thing holding labels) is written to a different
  directory and is never read by the serving path.
- `RetrievalCandidate` has no label field, so a label cannot re-enter ranking.
- `verify_no_leakage()` runs at the end of every build and asserts no chunk text
  equals any query or answer. `tests/test_dedup_and_leakage.py` covers the same
  invariants.

Optional: English source passages can be indexed as a separate
`language="en"` namespace (`--include-english`) for cross-lingual fallback. It is
a *separate* retrieval unit, never mixed into a target-language namespace.

---

## 4. Deduplication

MS MARCO is query-centric: ~10 candidate passages per row, and popular passages
recur across many queries. Ingesting rows naively produces one document per
(row, passage) pair, which inflates the index, lets one passage occupy several
candidate slots, and makes Recall@k ill-defined because "the relevant passage"
exists under several IDs.

Key: `sha256(language + normalized_text)`, with a `\x1f` separator so
`("hi","xy")` cannot collide with `("hix","y")`. Language is part of the key
because the same passage in Hindi and Marathi is genuinely two retrieval units.

Normalisation must be **idempotent** — the hash is the dedup key *and* the stable
`doc_id`, so drift between runs would silently duplicate the corpus. NFC is
applied because Devanagari can encode the same grapheme precomposed or as
base+nukta, which would otherwise defeat dedup. ZWNJ/ZWJ are folded for the hash
but preserved in stored text, since they are orthographically meaningful.

Provenance is kept (`source_query_ids`), capped so a passage referenced by 50k
queries cannot blow up memory.

Identifiers: `doc_id`, `parent_id`, `chunk_id`, `language`, `strategy`,
`source_split`, `content_hash`.

---

## 5. Chunking

MSMARCO passages are already human-curated retrieval units (~200-240 chars
median, 1-6 sentences in the demo corpus). So:

- **The default representation is the passage itself.** Cutting every passage
  into 500-token windows would be strictly destructive.
- **No cross-passage merging, anywhere.** Every chunk's text is a contiguous
  span of exactly one source passage. "Semantic merging" of independent MSMARCO
  passages is not implemented — it would fabricate documents that never existed
  and manufacture false co-occurrence evidence.

| strategy | when | metadata |
|---|---|---|
| **A `native`** | always | `strategy=native`, `chunk_id == parent_id == doc_id` |
| **B `sentence_window`** | ≥3 sentences | `sentence_start/end`, `parent_id` |
| **C `semantic_split`** | ≥`semantic_split_min_tokens` | breakpoints from neighbouring-sentence cosine distance, within one passage |
| **D `fixed_fallback`** | ≥`fixed_fallback_min_tokens` | token-based, ~17.5% overlap |

Each passage gets **native + at most one child strategy**, chosen by shape.
Emitting all four would store three near-duplicate representations of the same
text, inflating the index and letting one passage crowd out genuine diversity.
`chunk_forced()` exists so the evaluation can build a single-strategy index per
arm.

Two bugs the tests caught here, both worth recording:

1. `find_breakpoints` originally used nearest-rank percentile, which returns an
   *observed* distance. Since the cut test is strictly `>`, the single largest gap
   could never become a breakpoint — with few sentences that silently disabled
   the strategy. Now uses linear interpolation (the standard formulation).
2. The character-window fallback resumed the overlap mid-word, so following
   chunks began with a fragment. Now snaps forward to a word boundary.

### Parent-child retrieval

Children are tighter *retrieval* targets; a 2-sentence window is poor
*generation* context (dangling pronouns). So retrieval runs against children and
generation reads reconstructed parents:

```
child_17 → parent_A      final context:
child_22 → parent_A          parent_A
child_51 → parent_B    ⇒     parent_B
child_73 → parent_C          parent_C
```

A parent's score is the **best** child score, not a sum — otherwise a heavily
fragmented passage climbs the ranking purely by producing more windows.

---

## 6. Embedding: BGE-M3

One model emitting both retrieval representations, in 100+ languages, explicitly
trained for hybrid retrieval + reranking. Dense and sparse therefore share
tokenisation and vocabulary, so a query cannot match lexically in one space and
miss in the other for tokenisation reasons. It is also one 2.3 GB model load
instead of two.

Implemented directly on `transformers` rather than via `FlagEmbedding`:

- `FlagEmbedding` pulls `datasets`, `accelerate`, `peft`, `sentence-transformers`,
  hides batching/threading behind its own scheduler, and instantiates its own
  model copy.
- The query path needs explicit control of thread count, padding and batch
  composition.

The maths matches the reference implementation exactly:

- **dense** = L2-normalised CLS token of the last hidden state
- **sparse** = `relu(sparse_linear(hidden_states))`, max-pooled per token id,
  special tokens dropped

`sparse_linear.pt` is the published head. If it fails to load we **raise** rather
than fall back — an untrained sparse head looks like working hybrid retrieval
while contributing pure noise to fusion.

Dense dimension is read from `config.hidden_size` (1024), never hardcoded at call
sites. Max-pooling rather than summing means a term repeated five times does not
get five times the weight.

Verified behaviour (`tests/test_models_integration.py`, and reproduced during the
build): unit norms; `q_en·relevant = 0.784` vs `q_en·unrelated = 0.321`;
cross-lingual `q_en·q_hi = 0.784`; top sparse term for "What is a corporation?" is
`▁corporation` (0.359) and for the Hindi passage `▁निगम` (0.400).

### Pinned revisions

All three models are pinned to commit SHAs. This is not pedantry — it fixed a
real bug. `snapshot_download` resolved `main` to `5617a9f6…` while a later
`AutoModel.from_pretrained` resolved to `9a0624b8…`, so the backbone and the
separately-downloaded `sparse_linear.pt` could come from **different commits**,
silently corrupting the sparse branch. It also cost an extra 2.3 GB download.
Pinning additionally makes the reported metrics reproducible.

---

## 7. Vector database: Qdrant

Dense and sparse live in the **same point** under named vectors, and RRF is
available server-side. So hybrid retrieval is one round trip against one
consistent snapshot instead of two stores that can drift. Chroma has no
first-class sparse-vector support, so BGE-M3's lexical branch would have to be
bolted on separately — which is why it is not used.

```
dense  : 1024-d, cosine
sparse : BGE-M3 learned lexical weights
```

Cosine rather than dot: vectors are already L2-normalised so they rank
identically, but declaring cosine keeps the metric correct if an un-normalised
vector is ever inserted.

**No IDF modifier on the sparse vector.** BGE-M3's weights are already learned
term importances; Qdrant's IDF would re-weight by corpus frequency a second time.
IDF is right for raw BM25-style counts, which these are not.

Payload indexes on `language`, `strategy`, `parent_id`, `source_split` — without
them language filtering degrades to a full scan.

Point IDs are UUIDv5 of the readable `chunk_id`, so rebuilds upsert rather than
duplicate.

---

## 8. Hybrid retrieval and RRF

Cosine similarity (~0.3-0.95, tightly clustered) and BGE-M3 lexical scores
(unbounded, magnitude depends on query length and match count) are not
comparable. `alpha·dense + (1-alpha)·sparse` therefore lets whichever branch has
the larger numeric range dominate, and the "optimal" alpha shifts per query and
per language. RRF uses only ranks:

```
RRF(d) = Σ_branches 1 / (k + rank_b(d))      k = 60
```

No calibration, no per-language tuning, robust to one branch producing
pathological magnitudes, and a document must do well in *both* branches to win.

Two fusion implementations, both benchmarked:

- `server` — one round trip, Qdrant's native `FusionQuery(RRF)` over two
  prefetches.
- `client` (default) — two branch queries issued **concurrently**, fused locally.
  Costs one extra round trip but yields per-branch latency and per-branch ranks,
  which the required latency table and the debug drawer need. Because the
  branches are concurrent, wall clock is `max(dense, sparse)` + fusion, not their
  sum.

Ties break deterministically on `chunk_id` so benchmark percentiles do not jitter
for reasons unrelated to latency.

### Language-aware retrieval

| condition | behaviour |
|---|---|
| code-mixed speech | **never** filter — the answer may be in either namespace |
| confident detection of an indexed language | filter to it (quality + smaller HNSW candidate set) |
| uncertain / unknown | cross-lingual over all configured languages |

There is **no query translation**. BGE-M3 already embeds Hindi and English into a
shared space (measured above), so a translation hop would have to *prove* a
retrieval gain to earn its latency. It has not, so it is not implemented.

---

## 9. Query preprocessing

Deterministic and cheap: NFC + whitespace normalisation, then conservative ASR
artifact removal (fillers, bracketed annotations, immediate stutter, repeated
punctuation). Named entities and script are preserved — they are exactly what
retrieval keys on. Any removal that would empty the query is reverted.

**No LLM query rewriting.** A second model call in the latency-critical path is
not justified without a measured retrieval gain large enough to pay for it.

---

## 10. Reranking

`bge-reranker-v2-m3` shares the XLM-RoBERTa backbone and tokenizer with BGE-M3,
so it is multilingual over the same vocabulary with no per-language config.

Bi-encoders score a query against a *precomputed* vector — the document never
sees the query. A cross-encoder attends over the pair, fixing the cases
bi-encoders systematically get wrong (negation, quantities, near-duplicates
differing in one decisive detail). It also supplies **the only calibrated signal
the abstention gate can use**: RRF scores are rank-derived and carry no
information about whether the top document is actually relevant.

Runs on the fused candidate set only (default 30), never the corpus. Scores are
raw logits, deliberately **not** squashed before thresholding — sigmoid saturates
at both tails and destroys the margin resolution the ambiguity check needs.

---

## 11. Abstention gate

Runs **before** generation, so a bad-evidence query costs no generation latency
and gets no opportunity to hallucinate.

Two independent signals:

- `top_score` — is anything here actually relevant?
- `margin = top − second` — can the reranker *tell them apart*? A high top score
  with a negligible margin is retrieval ambiguity: several passages look equally
  plausible, and committing to one invites a confidently-wrong answer. A
  sufficiently decisive `top_score` overrides this, so two passages that both
  genuinely answer are not punished.

**Nothing is a guessed constant.** `scripts/calibrate_thresholds.py` fits the
thresholds and records the exact configuration used. Labelling targets the
question that matters at serving time: *is the top reranked passage a gold
(`is_selected`) passage?*

Genuine out-of-corpus negatives are essential and are constructed honestly — we
stream *additional* dataset rows beyond those ingested, so their gold passages
were never indexed, and discard any whose gold passage is present anyway. Without
real negatives every query is answerable and the fitted threshold is meaningless.

The operating point is **the lowest threshold meeting a precision floor**, not
max-F1: a false GENERATE is worse than a false ABSTAIN, because abstaining is
honest and answering from irrelevant evidence is not.

If the calibration artefact is missing, the gate runs in an explicitly
**uncalibrated** state and `thresholds_calibrated=false` propagates into the API
response, the debug drawer, `/health` and the logs. An uncalibrated guess is never
presented as an empirical threshold.

Thresholds depend on reranker precision, so the artefact records
`int8_quantized`/`device` and must be re-fitted if those change.

---

## 12. Generation

Groq, `openai/gpt-oss-20b`, configurable via `GROQ_MODEL`.
`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are deprecated on Groq and
are referenced nowhere in this codebase.

Streaming is used **even though the response is a JSON object**, for one reason:
time-to-first-token can only be *measured* by streaming. A non-streaming call
yields one number conflating queueing, prefill and full decode.

`reasoning_effort=low` — gpt-oss are reasoning models and extended reasoning is
pure added latency for a two-sentence extractive answer.

Structured output, cheapest first: `json_schema` where supported, else
`json_object` with the schema in the prompt; on a parse failure **exactly one**
retry with strict JSON instructions; still unparseable ⇒ **fail closed**. Output
is never "best-effort repaired" into an answer.

Answers are constrained to two short sentences so completion latency is
predictable and the response is speakable.

Every failure mode — no key, auth rejection, timeout, rate limit, malformed
output, model refusal — becomes an **abstention**, never a fabricated answer.

---

## 13. Prompt injection

MS MARCO is real scraped web text, so the corpus must be assumed to contain
strings that look like instructions. If retrieved text is treated as
instructions, anyone who can get a page into the corpus controls the assistant.

Invariant: **retrieved passages are DATA, never instructions.** Enforced
structurally, not by asking nicely:

1. The system prompt states the rule before any evidence is seen.
2. Every passage is wrapped in a labelled `<<<EVIDENCE … EVIDENCE>>>` envelope.
3. Delimiter sequences *inside* passage text are neutralised, so a passage cannot
   forge the end of its own envelope and escape into the instruction context.
   Chat-template role markers (`<|im_start|>` etc.) are defused too.
4. The "ignore embedded instructions" reminder is repeated **after** the
   evidence. Instructions nearest the end of the context carry the most weight,
   so the final word is ours.

---

## 14. Output guardrails

Ordering is a cost decision: citation validation is set membership
(microseconds) and catches the most deceptive failure, so it runs first and a
response with invented citations never reaches the NLI model.

**Citation validation** — every cited id must be in the retrieved set. Pure set
membership; no fuzzy matching that could let a near-miss through. Any invalid
citation **rejects the whole output** (not "drop the bad citation and keep the
answer": if the model referenced evidence that does not exist, the reasoning is
already untrustworthy). A non-empty answer with zero valid citations is also
rejected.

**NLI entailment** — citations prove the model *pointed at* real evidence; they
cannot prove the answer follows from it. Each factual answer sentence becomes a
hypothesis against the selected retrieved context. Model:
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (configurable) —
genuinely multilingual, and ~560 MB fits alongside two 2.3 GB models on one CPU
box. Label indices are read from `config.id2label`; hardcoding "entailment = 0"
silently inverts the guardrail on checkpoints that order labels differently.

Cost control: non-factual sentences skipped; a sentence that is a near-verbatim
span of the context is grounded **deterministically** by containment with no model
call (the common case for extractive answers, and provably safe); remaining pairs
batched. NLI never runs over the corpus.

Escalation:

```
citations invalid            → ABSTAIN   (never repaired, never retried)
factual sentence unsupported → REGENERATE once with supported context only
                               → still unsupported → ABSTAIN
```

**Never fails open.** `UNKNOWN` on a factual sentence counts as not-grounded —
"the evidence does not say" is not permission to assert. A guardrail that cannot
run is treated as a failed guardrail.

---

## 15. Input guardrails

Two tiers, for latency. Tier 1 runs on every query: compiled regexes over the
normalised string, tens of microseconds. Tier 2 (a deeper check) runs **only when
tier 1 fires**, so the normal path never pays for it, and it may *overturn* a
tier-1 block — the regexes are intentionally blunt and a web-QA system that
refuses "history of nuclear weapons" is not useful. Actionable
weapons/self-harm categories can never be overturned.

Normalisation runs **before** screening, otherwise zero-width joiners would evade
the patterns.

This layer answers "is this unsafe?" — deliberately **not** "can we answer it?".
MSMARCO-XI is broad web text with no topical boundary, so no static rule can
predict answerability; that is settled empirically by retrieval and the
calibrated gate. The accurate framing is **in-corpus / answerable vs
out-of-corpus**, not "in-domain vs out-of-domain".

---

## 16. Orchestration

An explicit state machine whose transition table is **data**
(`app/pipeline/states.py`), not control flow buried in `if` statements. That
means illegal transitions raise instead of silently producing a half-built
response, and the executed path is recorded per request and surfaced in the debug
drawer — so "why did this abstain?" is answerable from the trace alone.

Typed Pydantic models at every stage boundary: `STTResult → GuardrailResult →
QueryEmbeddingResult → RetrievalResult → RerankResult → GroundingDecision →
GenerationResult → ValidationResult → FinalResponse`.

Every external call has a timeout, bounded retries with exponential backoff +
jitter, a structured error, and a fallback where one is technically meaningful —
and the fallback is always *recorded*:

| failure | behaviour |
|---|---|
| Sarvam transient close (1006/1011/1001) | retry with backoff, replaying buffered audio |
| Sarvam 4xxx / 401 / 429 | **no retry** (retrying a bad key just burns the budget), REST fallback, clean error |
| Qdrant error | retry once, then abstain |
| Reranker failure | fall back to RRF order, `fallback_used=true` propagated — RRF cannot feed the calibrated gate, so such a response is strictly less trustworthy and must not look identical to a normal one |
| Groq failure | bounded retry, then abstain |
| Grounding failure | regenerate once, then abstain |

CPU-bound model work is dispatched via `asyncio.to_thread` so one slow inference
cannot stall the event loop for every other in-flight request. Inference is
serialised behind a lock: torch CPU inference with a fixed intra-op thread pool
degrades badly under concurrency (threads oversubscribe, p99 explodes), and
predictable latency matters more than raw throughput here.

---

## 17. Latency engineering, honestly

Instrumentation uses `time.perf_counter_ns()` — monotonic. `time.time` is
wall-clock and can step backwards (NTP, DST), silently corrupting percentiles.

`LatencyBreakdown` keeps `None` (**not measured**) distinct from `0.0`
(**measured as sub-microsecond**), and every report renders `None` as `n/a`.
Coercing an unmeasured stage to zero is how fake benchmark numbers get published.

Three different aggregates, never conflated:

| metric | definition |
|---|---|
| `total_rag_latency` | transcript received → final validated answer (**excludes STT**) |
| `total_voice_latency` | audio submitted → **first** answer token (includes STT) |
| `total_completion_latency` | audio/transcript in → final answer token |

### The CPU finding

The reference machine has **no CUDA**. Measured on an i5-1240P (4P+8E, 16
logical), 30 candidates:

| config | time |
|---|---|
| fp32, 8 threads | 8091 ms |
| fp32, 12 threads | 5568 ms |
| fp32, 16 threads | 5230 ms |
| **int8 dynamic, 16 threads** | **4063 ms** |

`bge-reranker-v2-m3` is XLM-R-large (568M params, 24 layers, hidden 1024).
Reranking 30 query-passage pairs is ~30 full forward passes and it dominates
everything else.

**The aggressive targets (<100 ms retrieval+rerank, <200 ms TTFT) are not
attainable in this configuration, and this repo does not claim they are.**
`reports/latency.md` reports what was measured, with the device and settings
recorded in the artefact.

What was done about it:

- int8 dynamic quantization of the reranker (CPU only): 5230 → 4063 ms, and
  ranking is preserved — the relevant/irrelevant logit margin moved 11.41 → 11.06.
- Thread count set from measurement (16), not defaults.
- Length-sorted batching so short passages stop paying the padding cost of the
  longest item in the batch.
- Dense and sparse branches run concurrently.
- `RERANK_TOP_K` is configurable, with the measured cost/quality trade-off in the
  reports.
- The **embedder is deliberately not quantized**: index and query vectors must
  come from the identical model, so quantizing it would require rebuilding the
  index or silently mixing precisions.

On a GPU box, `DEVICE=cuda` enables fp16 and the same code path; nothing else
changes.

---

## 18. Evaluation methodology

Queries are partitioned deterministically by hashing `query_id`:

- `calibration` (~40%) — fits abstention thresholds
- `test` (~60%) — every reported metric

So no threshold is ever tuned on the queries it is scored against.

Metrics are computed over **unique passage content hashes**, not chunk ids. A
passage indexed as native + several windows shares one hash; ranking over chunk
ids would let one passage occupy ranks 1-4 and Recall@5 would measure
fragmentation rather than retrieval. This is also what makes the chunking arms
comparable: an arm emitting 6 vectors/passage is scored on the same footing as
native.

Recall@k uses `|relevant|` as denominator (not `min(|R|, k)`), which would
otherwise inflate Recall@1 whenever multiple passages are relevant. Macro
average over queries, so a query with many relevant passages does not dominate.

The evaluator uses the **same `RetrievalService` as the serving path** — if it had
its own retrieval code, the reported metrics would describe code that never
serves a request.

---

## 19. Deliberate omissions

Per the "remove what does not measurably help" principle:

- **No LangChain / LlamaIndex.** Direct FastAPI + Qdrant + model SDK calls. The
  orchestration here is a typed state machine; a framework would add indirection
  and latency without adding capability.
- **No query rewriting**, **no query translation**, **no HyDE** — each is a model
  hop on the critical path, none has demonstrated a retrieval gain here.
- **No second embedding model.** BGE-M3 covers dense + sparse + sentence
  embedding for semantic chunking.
- **No semantic merging of independent passages.** It would fabricate documents.
- **No multiple sequential LLM calls** in the critical path. Exactly one, plus at
  most one regeneration when grounding fails.
- **No Prometheus client.** Deployment target is a single container; the
  requirement is observability of stage latencies and abstention behaviour, met
  by an in-process registry with bounded ring buffers.

---

## 20. Security

- Secrets only via env/`.env`; `.env` is gitignored, `.env.example` documents
  every variable.
- Startup validates required secrets and reports them on `/health`. Missing
  secrets cause affected stages to **fail closed**, never to degrade silently.
- API keys are never logged and never attached to trace context; `redact()`
  scrubs sensitive keys from structured log payloads.
- Raw microphone audio is never logged — only byte counts and duration.
- Transcripts are redacted unless `LOG_REQUEST_BODIES=true`.
- The frontend contains no credentials; the browser only talks to this app's own
  origin.
- The global exception handler returns a generic error, never a stack trace or
  internal path.
- `GENERATION_BACKEND=mock` exists for exercising the output guardrails without a
  Groq key. It is **never** auto-selected when the key is missing — that would
  turn "generation unavailable" into a plausible-looking answer, the exact
  failure this system is built to prevent. Mock responses are labelled
  `model="mock-extractive"` everywhere.
