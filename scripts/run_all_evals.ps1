# Runs the full evaluation suite in dependency order and logs everything.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run_all_evals.ps1
#
# Ordering is deliberate:
#   * thresholds are calibrated FIRST, otherwise every later run reports
#     `thresholds_calibrated=false`;
#   * the latency benchmark runs LAST and alone, because anything else competing
#     for CPU would inflate its percentiles and the numbers would be worthless.
#
# Sizes are fitted to the MEASURED cost on the CPU-only reference box, where a
# single query with rerank_top_k=30 costs ~10-14 s. On a GPU raise these freely.

$ErrorActionPreference = 'Continue'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$py = '.\.venv\Scripts\python.exe'

function Step($name, $cmd) {
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "  STEP: $name   ($(Get-Date -Format HH:mm:ss))"
    Write-Host ("=" * 78)
    $start = Get-Date
    Invoke-Expression $cmd
    $mins = ((Get-Date) - $start).TotalMinutes
    Write-Host ("  -> $name finished in {0:N1} min (exit={1})" -f $mins, $LASTEXITCODE)
}

# 1. Calibrate abstention thresholds.
#    Label = a gold passage lands inside the top FINAL_TOP_K context the
#    generator actually receives. Negatives are real out-of-corpus queries.
Step 'calibrate_thresholds' "$py scripts\calibrate_thresholds.py --limit 90 --negatives 60 --target-precision 0.85"

# 2. Retrieval evaluation on the held-out test split: dense / sparse / RRF / +rerank.
Step 'evaluate_retrieval' "$py scripts\evaluate_retrieval.py --limit 100"

# 3. Demo scenarios under the REAL configured backend, so the
#    generation-unavailable -> abstain path is demonstrated as it ships.
Step 'demo_scenarios' "$py scripts\run_demo_scenarios.py"

# 4. Chunking strategy comparison (builds and embeds one index per arm).
Step 'evaluate_chunking' "$py scripts\evaluate_chunking.py --max-rows-per-language 8 --limit 20 --rerank-top-k 10"

# 5. Latency benchmark - LAST, so nothing else is competing for the CPU.
#    GENERATION_BACKEND=mock so the generation, NLI and output-guardrail stages
#    actually execute and get measured; the artefact records the backend so these
#    can never be mistaken for Groq numbers.
Step 'benchmark_latency' "$py scripts\benchmark_latency.py --queries 100 --generation-backend mock"

# 6. Consolidate everything into reports/SUMMARY.md
Step 'make_summary' "$py scripts\make_summary.py"

Write-Host ""
Write-Host "ALL EVALUATIONS COMPLETE"
Get-ChildItem reports | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
