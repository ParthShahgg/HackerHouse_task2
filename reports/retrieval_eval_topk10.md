# Retrieval Evaluation

- dataset: `ai4bharat/MSMARCO-XI`
- collection: `msmarco_xi_demo`
- eval split: `test` (disjoint from threshold calibration)
- queries: **40** {'hi': 23, 'mr': 17}
- retrieval: language-filtered
- device: `cpu`, int8 reranker: `True`, threads: `16`
- embedding: `BAAI/bge-m3`
- reranker: `BAAI/bge-reranker-v2-m3` (top-10)

Metrics are macro-averaged over queries, computed on unique passage
content hashes so multi-chunk strategies are not rewarded for
fragmentation. Ground truth is the dataset's `is_selected` label.

## Overall

| arm           |   queries |   recall@1 |   recall@3 |   recall@5 |   recall@10 |   mrr |   ndcg@10 |   latency_mean_ms |   latency_p50_ms |   latency_p95_ms |
|---------------|-----------|------------|------------|------------|-------------|-------|-----------|-------------------|------------------|------------------|
| dense         |        40 |       0.45 |       0.64 |       0.78 |        0.81 |  0.62 |      0.65 |             84.87 |            38.22 |           132.33 |
| sparse        |        40 |       0.4  |       0.56 |       0.6  |        0.74 |  0.56 |      0.58 |             23.99 |            14.86 |            78.17 |
| hybrid_rrf    |        40 |       0.45 |       0.61 |       0.79 |        0.79 |  0.6  |      0.64 |             45.18 |            28.61 |           131.07 |
| hybrid_rerank |        40 |       0.38 |       0.7  |       0.79 |        0.79 |  0.56 |      0.6  |           5085.82 |          2170.72 |         12727.9  |


## Per language

| language   | arm           |   queries |   recall@1 |   recall@3 |   recall@5 |   recall@10 |   mrr |   ndcg@10 |   latency_mean_ms |   latency_p50_ms |   latency_p95_ms |
|------------|---------------|-----------|------------|------------|------------|-------------|-------|-----------|-------------------|------------------|------------------|
| hi         | dense         |        23 |       0.54 |       0.72 |       0.83 |        0.83 |  0.68 |      0.7  |            123.22 |            46.96 |           443.79 |
| hi         | sparse        |        23 |       0.43 |       0.61 |       0.63 |        0.74 |  0.59 |      0.6  |             30.91 |            21.64 |           113.82 |
| hi         | hybrid_rrf    |        23 |       0.5  |       0.63 |       0.83 |        0.83 |  0.63 |      0.67 |             62.64 |            43.47 |           138.8  |
| hi         | hybrid_rerank |        23 |       0.46 |       0.76 |       0.83 |        0.83 |  0.61 |      0.65 |           7454.28 |          8661.08 |         14563.4  |
| mr         | dense         |        17 |       0.32 |       0.53 |       0.71 |        0.79 |  0.54 |      0.58 |             32.98 |            29.08 |            53.28 |
| mr         | sparse        |        17 |       0.35 |       0.5  |       0.56 |        0.74 |  0.52 |      0.55 |             14.64 |            11.24 |            31.88 |
| mr         | hybrid_rrf    |        17 |       0.38 |       0.59 |       0.74 |        0.74 |  0.57 |      0.59 |             21.56 |            19.7  |            38.33 |
| mr         | hybrid_rerank |        17 |       0.26 |       0.62 |       0.74 |        0.74 |  0.49 |      0.54 |           1881.44 |          1809.88 |          2983.86 |
