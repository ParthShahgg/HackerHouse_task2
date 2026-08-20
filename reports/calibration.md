# Abstention Threshold Calibration

- collection: `msmarco_xi_demo`
- fitted on the **calibration** split only (test split is held out for reporting)
- in-corpus queries: 48
- out-of-corpus queries: 60 (gold passages verified absent from the index)
- positives: 35, negatives: 73
- reranker: `BAAI/bge-reranker-v2-m3` int8=`True` device=`cpu`

**Label**: a gold (`is_selected`) passage appears within the top 5 reranked candidates - i.e. inside the context the generator
actually receives. Labelling on rank 1 alone would be pessimistic by
construction, because MS MARCO marks only ~1 of ~10 passages as selected,
so a genuinely relevant top hit is frequently unlabelled.

**Chosen**: `rerank_abstain_below = 6.38`, `rerank_margin_min = 0.018`

**Objective actually used**: lowest threshold with precision >= 0.85

Requested precision floor `0.85` was MET.

Rationale for preferring a precision floor: a false GENERATE is worse than
a false ABSTAIN - abstaining is honest, answering from irrelevant evidence
is not.

## Threshold sweep

|   threshold |   tp |   fp |   fn |   tn |   precision |   recall |   f1 |   youden_j |
|-------------|------|------|------|------|-------------|----------|------|------------|
|       -7.64 |   35 |   73 |    0 |    0 |        0.32 |     1    | 0.49 |       0    |
|       -6.43 |   35 |   66 |    0 |    7 |        0.35 |     1    | 0.51 |       0.1  |
|       -5.66 |   35 |   58 |    0 |   15 |        0.38 |     1    | 0.55 |       0.21 |
|       -4.89 |   35 |   51 |    0 |   22 |        0.41 |     1    | 0.58 |       0.3  |
|       -4.23 |   35 |   45 |    0 |   28 |        0.44 |     1    | 0.61 |       0.38 |
|       -3.95 |   34 |   38 |    1 |   35 |        0.47 |     0.97 | 0.64 |       0.45 |
|       -3.52 |   33 |   32 |    2 |   41 |        0.51 |     0.94 | 0.66 |       0.5  |
|       -2.92 |   32 |   27 |    3 |   46 |        0.54 |     0.91 | 0.68 |       0.54 |
|       -2.12 |   32 |   19 |    3 |   54 |        0.63 |     0.91 | 0.74 |       0.65 |
|       -1.03 |   31 |   14 |    4 |   59 |        0.69 |     0.89 | 0.78 |       0.69 |
|        1.12 |   27 |   11 |    8 |   62 |        0.71 |     0.77 | 0.74 |       0.62 |
|        2.07 |   22 |    8 |   13 |   65 |        0.73 |     0.63 | 0.68 |       0.52 |
|        2.93 |   16 |    7 |   19 |   66 |        0.7  |     0.46 | 0.55 |       0.36 |
|        4.13 |   11 |    6 |   24 |   67 |        0.65 |     0.31 | 0.42 |       0.23 |
|        6.33 |    8 |    2 |   27 |   71 |        0.8  |     0.23 | 0.36 |       0.2  |
|        8.43 |    3 |    0 |   32 |   73 |        1    |     0.09 | 0.16 |       0.09 |
