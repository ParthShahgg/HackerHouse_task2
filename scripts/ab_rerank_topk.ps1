# A/B: does reducing the candidate pool cost retrieval quality?
#
#   powershell -ExecutionPolicy Bypass -File scripts/ab_rerank_topk.ps1
#
# Arm A (old): dense/sparse/rrf = 30, rerank_top_k = 30
# Arm B (new): dense/sparse/rrf = 15, rerank_top_k = 10
#
# Both arms run the SAME queries in the same deterministic order, so the only
# variable is the candidate pool size. Writes reports/rerank_topk_ab.md.
#
# Why this matters: the reduction was made to stop the cross-encoder starving the
# co-located Qdrant container of CPU. That is only a good trade if quality holds.

$ErrorActionPreference = 'Continue'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$env:PYTHONIOENCODING = 'utf-8'
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$py = '.\.venv\Scripts\python.exe'
$limit = 40

Write-Host "=== Arm A: pool 30 / rerank 30 ===" -ForegroundColor Cyan
$env:DENSE_TOP_K = '30'; $env:SPARSE_TOP_K = '30'; $env:RRF_TOP_K = '30'
& $py scripts\evaluate_retrieval.py --limit $limit --rerank-top-k 30 --tag topk30
Write-Host "arm A exit=$LASTEXITCODE"

Write-Host "=== Arm B: pool 15 / rerank 10 ===" -ForegroundColor Cyan
$env:DENSE_TOP_K = '15'; $env:SPARSE_TOP_K = '15'; $env:RRF_TOP_K = '15'
& $py scripts\evaluate_retrieval.py --limit $limit --rerank-top-k 10 --tag topk10
Write-Host "arm B exit=$LASTEXITCODE"

Remove-Item Env:\DENSE_TOP_K, Env:\SPARSE_TOP_K, Env:\RRF_TOP_K -ErrorAction SilentlyContinue

& $py scripts\compare_topk_ab.py
Write-Host "compare exit=$LASTEXITCODE"
