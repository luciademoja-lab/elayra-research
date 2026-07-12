<#
run_extend.ps1 - one-command incremental run for THIS round.

Confirms the two code fixes (head-level BART, control_short) and runs the new
MLP + embedding 4-way distribution analysis. Assumes the in-repo venv from the
previous run already exists (.venv\Scripts\python.exe) - it does NOT create or
modify it, and does NOT run the full pipeline.

Usage (from the repo root, PowerShell):
    .\run_extend.ps1            # run all three steps
    .\run_extend.ps1 -Quick     # skip control_short (the GPU training step)

Outputs land in results\ and a full transcript is saved to
results\extend_run.log so nothing is lost to the console.
#>
param([switch]$Quick, [switch]$WithModern)

# Run from the repo root regardless of where the command was invoked.
Set-Location -Path $PSScriptRoot

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: venv not found at $py" -ForegroundColor Red
    Write-Host "The venv from the previous run is missing. See RUNBOOK section 1." -ForegroundColor Red
    exit 1
}

# Reuse the previous run's in-repo model cache (no re-download).
$env:HF_HOME = "$PWD\hf_cache"

New-Item -ItemType Directory -Force -Path "results" | Out-Null
$log = "results\extend_run.log"
"=== extend run $(Get-Date -Format o) ===" | Out-File -FilePath $log -Encoding utf8

# Cheap/CPU steps first; the GPU training step (control_short) last.
$steps = @(
    @{ name = "mlp_embedding (new)";      script = "scripts\mlp_embedding_analysis.py"; out = "results\mlp_embedding_4way.json" },
    @{ name = "head_level (BART fix)";     script = "scripts\head_level_analysis.py";     out = "results\head_level_results.json" }
)
if (-not $Quick) {
    $steps += @{ name = "control_short (fix)"; script = "scripts\control_short.py"; out = "results\extended_control_results.json" }
}
if ($WithModern) {
    # LLaMA-3.2-1B is gated: set $env:HF_TOKEN and accept the HF access form first
    # (see RUNBOOK). The script reports a clear message and continues if it is blocked.
    $steps += @{ name = "layerwise modern (LLaMA+Phi)"; script = "scripts\run_layerwise_modern.py"; out = "results\layerwise_model_comparison.json" }
}

$summary = @()
foreach ($s in $steps) {
    Write-Host "`n>>> $($s.name) ..." -ForegroundColor Cyan
    "`n>>> $($s.name)" | Out-File -FilePath $log -Append -Encoding utf8
    & $py $s.script 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $summary += [pscustomobject]@{
        step   = $s.name
        exit   = $code
        status = if ($code -eq 0) { "OK" } else { "FAILED" }
        output = $s.out
    }
}

Write-Host "`n================= SUMMARY ================="
$summary | Format-Table step, status, exit, output -AutoSize
Write-Host "Full transcript: $log"

$failed = @($summary | Where-Object { $_.exit -ne 0 }).Count
if ($failed -gt 0) {
    Write-Host "$failed step(s) failed - read $log, do not assume success." -ForegroundColor Yellow
    exit 1
}
Write-Host "All steps completed. Outputs saved in results\." -ForegroundColor Green
