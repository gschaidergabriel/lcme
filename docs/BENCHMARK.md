# Titan Memory: Empirical Benchmark

## Abstract

We benchmark Titan Memory v1.0.0 on a corpus of 30 realistic conversational memory items with 20 labeled retrieval queries, averaged over 3 runs. We also compare deployment characteristics against Mem0, Graphiti, and Letta.

**Results (averaged over 3 runs):**

| Metric | Titan |
|--------|-------|
| **Precision@1** | **0.900** |
| **Precision@5** | **1.000** |
| **MRR** | **0.942** |
| Ingestion latency | 27.6 ms/item |
| Retrieval P50 | 14.1 ms |
| Retrieval P95 | 16.9 ms |
| RAM delta | 16.9 MB |
| Disk footprint | 0.64 MB |
| Neural parameters | 77,580 |
| Languages supported | 8 |
| Cross-session persistence | Yes |
| Contradiction detection | Yes |
| External dependencies | None |

## 1. Systems Under Test

| System | Version | Can Run Fully Local | External Dependencies |
|--------|---------|--------------------|-----------------------|
| **Titan** | v1.0.0 | Yes | None (SQLite + PyTorch CPU) |
| **Mem0** | v1.0.7 | Partial | ChromaDB + LLM endpoint (required) |
| **Graphiti** | v0.28.2 | No | Neo4j server + LLM API key (required) |
| **Letta** | v0.16.6 | No | Letta server + PostgreSQL (required) |

Graphiti and Letta could not be benchmarked — both require external infrastructure that prevents fully local operation. This is a valid finding: **Titan is the only system that runs with zero external dependencies.**

## 2. Methodology

### 2.1 Corpus

30 realistic, diverse conversational memory items covering:
- User facts ("Alice is a senior backend engineer at Google")
- Technical details ("The authentication service listens on port 8080")
- Decisions ("We decided to postpone the microservices migration")
- Observations ("Elena raised security concerns about the API flow")
- Costs and metrics ("The observability stack costs about 2000 USD per month")

Items are intentionally diverse in vocabulary and structure to reflect real agent conversations, not template-generated.

### 2.2 Queries

20 labeled queries with keyword ground truth, covering:
- Direct fact recall ("Where does Alice work?")
- Technical queries ("What port does auth use?")
- Semantic queries ("What monitoring tools?")
- Person-specific queries ("Who built spam detection?")
- Decision recall ("Microservices migration decision?")

### 2.3 Configuration

- **CPU-only**: AMD Ryzen 9 7940HS (no GPU acceleration)
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim)
- **3 runs averaged** for all latency and quality metrics
- **Cold start**: Each run starts from scratch

## 3. Results

### 3.1 Retrieval Quality

```
Precision@1:     0.900   (18/20 queries, correct result at position 1)
Precision@5:     1.000   (20/20 queries, correct result in top 5)
MRR:             0.942   (average reciprocal rank of first hit)
```

Consistent across all 3 runs (zero variance). The two queries where the correct result was not at position 1 returned it at positions 2-3 (related items ranked slightly higher due to shared vocabulary).

### 3.2 Ingestion Performance

```
Per-item latency:   27.6 ms   (30 items in 0.83s)
Breakdown:
  - Embedding computation:  ~20 ms (all-MiniLM-L6-v2 on CPU)
  - SQLite writes:          ~3 ms  (nodes, claims, FTS index, events)
  - Neural cortex scoring:  ~1 ms  (MIS importance + ET emotion)
  - Entity/claim extraction: <1 ms (regex, no LLM)
```

### 3.3 Retrieval Latency

```
P50:    14.1 ms
P95:    16.9 ms
Mean:   14.0 ms
```

Sub-20ms retrieval across all queries. This includes FTS5 search, vector cosine similarity, RRF fusion, entity filtering, and score computation.

### 3.4 Resource Usage

```
RAM delta:       16.9 MB   (after ingesting 30 items)
Disk footprint:  0.64 MB   (SQLite + vectors + FTS index)
Cold start:      7.83 s    (includes embedding model load)
Neural params:   77,580    (6 cortex micro-networks)
```

### 3.5 Persistence and Contradiction

- **Cross-session persistence**: Verified. Shutdown and restart preserves all nodes, edges, vectors, and claims.
- **Contradiction detection**: Verified. After ingesting "Alice works at Google" then "Alice now works at Microsoft", querying "Where does Alice work now?" returns the Microsoft entry.

## 4. Comparison with Other Systems

### 4.1 Why Only Titan Has Numbers

| System | Benchmarked? | Reason |
|--------|-------------|--------|
| Titan | Yes | Runs fully local, zero dependencies |
| Mem0 | No (v1 only) | Requires LLM endpoint for extraction. Previous benchmark (pre-fix) showed 11.8s/item ingestion, P@5=0.867 |
| Graphiti | No | Requires running Neo4j server. Confirmed: `OpenAIError` without API key |
| Letta | No | Requires running Letta server. Confirmed: `APIConnectionError` without server |

### 4.2 Feature Comparison Matrix

```
Feature                              Titan    Mem0     Graphiti   Letta
──────────────────────────────────────────────────────────────────────────
STORAGE
  Tri-hybrid (SQL + Vector + Graph)  Yes      No       No         No
  Full-text search (FTS5)            Yes      No       No         No
  Vector similarity search           Yes      Yes      No         Yes
  Knowledge graph                    Yes      No       Yes        No
  Temporal validity tracking         No       No       Yes        No

EXTRACTION
  No LLM required for ingestion      Yes      No       No         No
  Claim-based extraction             Yes      No       No         No
  Entity recognition                 Yes      Yes      Yes        Yes
  Counter-hypotheses                 Yes      No       No         No

NEURAL COMPONENTS
  Importance scoring (MIS)           Yes      No       No         No
  Emotion tagging (ET)               Yes      No       No         No
  Learned retrieval weights (RWL)    Yes      No       No         No
  Hopfield associative memory        Yes      No       No         No
  Self-training consolidation        Yes      No       No         No
  Trainable parameters               77K+     0        0          0

LIFECYCLE
  Time-weighted confidence decay     Yes      No       No         No
  Controlled forgetting              Yes      Manual   No         Yes
  Memory graduation                  Yes      No       No         No

MULTILINGUAL
  Stop word filtering                8 langs  N/A      N/A        N/A
  Temporal word detection            8 langs  N/A      N/A        N/A
  Emotion word detection             8 langs  N/A      N/A        N/A

DEPLOYMENT
  Fully local, zero dependencies     Yes      No       No         No
  No API keys required               Yes      No       No         No
  No servers required                Yes      Yes*     No         No
  Single pip install                 Yes      Yes*     No         No

  * Mem0 requires LLM endpoint for extraction
```

## 5. Supported Languages

Titan's FTS stop-word filtering, temporal detection, emotion detection, and negation detection support 8 languages:

| Language | Stop Words | Temporal | Emotion | Negation |
|----------|-----------|----------|---------|----------|
| English | ~80 | 25 | 27 | 7 |
| German | ~70 | 22 | 23 | 8 |
| Spanish | ~55 | 19 | 22 | 8 |
| French | ~50 | 16 | 22 | 7 |
| Portuguese | ~45 | 16 | 22 | 7 |
| Mandarin Chinese | ~45 | 26 | 23 | 7 |
| Hindi | ~35 | 26 | 21 | 7 |
| Arabic | ~35 | 24 | 23 | 8 |

The embedding model (`all-MiniLM-L6-v2`) supports 100+ languages for semantic similarity. The stop-word and feature detection lists above enhance FTS precision and query understanding for these specific languages.

## 6. Limitations

1. **30-item corpus.** Small by benchmarking standards. Scaling behavior at 10K+ items may differ as vector search, FTS, and graph traversal costs grow.

2. **CPU-only.** All measurements on AMD Ryzen 9 7940HS. GPU-accelerated embedding computation would reduce ingestion latency.

3. **Cold-start neural components.** Titan's Hippocampus (226K params) was not loaded in standalone mode. The cortex networks (77K params) were in cold-start phase. A production instance with trained networks and consolidation history would show different characteristics.

4. **No head-to-head with competitors.** Graphiti and Letta require external infrastructure. Mem0 requires an LLM endpoint. A fully controlled comparison would need identical infrastructure for all systems.

5. **Keyword ground truth.** Quality measured by keyword matching. Semantically correct results using different wording may be undercounted.

## Hardware

```
CPU:      AMD Ryzen 9 7940HS w/ Radeon 780M Graphics
RAM:      24.4 GB
Python:   3.12.3
PyTorch:  2.5.0+cu124 (CPU mode)
OS:       Linux 6.17.0-19-generic (Ubuntu 24.04)
```

## Reproducing

```bash
cd titan-memory/
pip install -e .
python benchmarks/benchmark.py
```

Results are saved to `benchmarks/results.json`.

---

*Benchmark conducted March 25, 2026. All numbers averaged over 3 runs on identical hardware.*
