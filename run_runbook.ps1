<#
.SYNOPSIS
    run_runbook.ps1 — implementa fedelmente RUNBOOK.md (rerun completo 4-way BIC).
    NON è run_all.ps1: qui l'ordine, i reset e i backup JSON seguono il runbook.

.USAGE
    .\run_runbook.ps1 -Setup    # prima volta: crea venv 3.12 + dipendenze, poi esegue tutto
    .\run_runbook.ps1           # venv già pronto: esegue tutto (a→h)

    Log completo in runbook_run.log (gitignorato).
#>

param([switch]$Setup)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Start-Transcript -Path (Join-Path $Root "runbook_run.log") -Append

function Step($desc, [scriptblock]$block, [switch]$Fatal, [switch]$Tolerated) {
    Write-Host ("`n[{0}] ==== {1} ====" -f (Get-Date -Format "HH:mm:ss"), $desc) -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne 0) {
        if ($Fatal) { Stop-Transcript; throw "[runbook] $desc FAILED (exit $LASTEXITCODE) — ABORT" }
        elseif ($Tolerated) { Write-Warning "[runbook] $desc FAILED — tollerato da runbook, si prosegue" }
        else { Write-Warning "[runbook] $desc FAILED (exit $LASTEXITCODE)" }
    }
}

# ── Setup una-tantum (RUNBOOK §1) ───────────────────────────────────────────
if ($Setup) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Step "Install uv" { powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" } -Fatal
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
    Step "uv venv (Python 3.12, in-repo)" { uv venv --python 3.12 } -Fatal
    . .venv\Scripts\Activate.ps1
    Step "torch cu128 (Blackwell)" { uv pip install torch --index-url https://download.pytorch.org/whl/cu128 } -Fatal
    Step "Dipendenze + pacchetto editable" { uv pip install -r requirements-lock.txt; uv pip install -e . } -Fatal
} else {
    . .venv\Scripts\Activate.ps1
}

# ── Ambiente (RUNBOOK §0/§1/§3) ─────────────────────────────────────────────
$env:HF_HOME = Join-Path $Root "hf_cache"          # cache modelli DENTRO il repo
$secrets = Join-Path $Root "secrets.local.ps1"
if (Test-Path $secrets) { . $secrets }             # imposta HF_TOKEN (gitignorato)
$env:ELA_SUBSAMPLE = "500000"
$env:ELA_SUBSAMPLE_SEED = "12345"

Write-Host "`n[runbook] Controlla che la GPU sia libera (nvidia-smi) prima di proseguire."
nvidia-smi

# ── §2 Sanity check: i test DEVONO passare ──────────────────────────────────
Step "pytest (47 test)" { python -m pytest tests/ -v } -Fatal

# ── (a) Layerwise 4-way, 15 modelli — core result ───────────────────────────
Step "(a) reset checkpoint (vecchio protocollo gpt2)" { python scripts\run_layerwise_incremental.py --reset } -Fatal
Step "(a) layerwise 15 modelli (~30-90 min)" { python scripts\run_layerwise.py } -Fatal
Copy-Item results\layerwise_model_comparison.json results\layerwise_main.json -Force
Write-Host "[runbook] (a) salvato in results\layerwise_main.json"

# ── (b) Pipeline 8-layer + random-init ──────────────────────────────────────
Step "(b) run_pipeline" { python scripts\run_pipeline.py }

# ── (c) Risultati citati nel paper mai salvati ──────────────────────────────
Step "(c) L1 regularization" { python scripts\l1_regularization_test.py }
Step "(c) bootstrap" { python scripts\bootstrap_analysis.py }

# ── (d) Controlli randomized-label (GPU) ────────────────────────────────────
Step "(d) control_short"    { python scripts\control_short.py }
Step "(d) control_long"     { python scripts\control_long.py }
Step "(d) control_shuffled" { python scripts\control_shuffled.py }

# ── (e) Analisi secondarie ──────────────────────────────────────────────────
Step "(e) init_analysis"       { python scripts\init_analysis.py }
Step "(e) head_level (BART può fallire: logga, non blocca)" { python scripts\head_level_analysis.py } -Tolerated
Step "(e) mlp_analysis"        { python scripts\mlp_analysis.py }
Step "(e) checkpoint_analysis" { python scripts\checkpoint_analysis.py }

# ── (f) Modelli estesi (HF_TOKEN per LLaMA; Phi-2 senza) ────────────────────
$env:INCLUDE_OPTIONAL_MODELS = "1"
Step "(f) modern_llms_ext" { python scripts\modern_llms_ext.py }

# ── (g) Stabilità subsampling (appendice) — (a) GIÀ salvato come layerwise_main.json
Step "(g) reset per full-tensor" { python scripts\run_layerwise_incremental.py --reset }
$env:ELA_SUBSAMPLE = "0"
Step "(g) full-tensor distilbert" { python scripts\run_layerwise_incremental.py distilbert-base-uncased }
Step "(g) full-tensor t5-small"   { python scripts\run_layerwise_incremental.py t5-small }
Step "(g) full-tensor electra"    { python scripts\run_layerwise_incremental.py google/electra-small-discriminator }
Copy-Item results\layerwise_model_comparison.json results\layerwise_fulltensor.json -Force

Step "(g) reset per seed alternativo" { python scripts\run_layerwise_incremental.py --reset }
$env:ELA_SUBSAMPLE = "500000"; $env:ELA_SUBSAMPLE_SEED = "99"
Step "(g) seed 99 distilbert" { python scripts\run_layerwise_incremental.py distilbert-base-uncased }
Copy-Item results\layerwise_model_comparison.json results\layerwise_seed99.json -Force

# Ripristina il risultato principale (a) come file canonico per figure e commit
Copy-Item results\layerwise_main.json results\layerwise_model_comparison.json -Force
$env:ELA_SUBSAMPLE_SEED = "12345"

# ── (h) Figure ──────────────────────────────────────────────────────────────
Step "(h) generate_figures" { python scripts\generate_figures.py }

Write-Host ("`n[{0}] RUN COMPLETO. Verifica results\ e results\figures\, poi commit (RUNBOOK §6)." -f (Get-Date -Format "HH:mm:ss")) -ForegroundColor Green
Stop-Transcript
