<#
.SYNOPSIS
    run_all.ps1 — Windows-native runner for elayra-research.
    Equivalent to `make all` on Unix/WSL but works in plain PowerShell.

.USAGE
    .\run_all.ps1          # everything
    .\run_all.ps1 -Target pipeline   # single target
#>

param(
    [ValidateSet("pipeline","layerwise","controls","init-analysis","bootstrap",
                 "l1","checkpoint","heads","modern-llms","mlp","all","test")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py  = "python"
$Scripts = "scripts"

function Invoke-Step($desc, $cmd) {
    Write-Host "`n[run_all] $desc" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Warning "[run_all] $desc FAILED (exit $LASTEXITCODE)" }
}

switch ($Target) {
    "pipeline"      { Invoke-Step "Pipeline (15-model)"         "$Py $Scripts\run_pipeline.py" }
    "layerwise"     { Invoke-Step "Layerwise comparison"        "$Py $Scripts\run_layerwise.py" }
    "controls"      { Invoke-Step "Control: short"              "$Py $Scripts\control_short.py"
                      Invoke-Step "Control: long"               "$Py $Scripts\control_long.py"
                      Invoke-Step "Control: shuffled"           "$Py $Scripts\control_shuffled.py" }
    "init-analysis" { Invoke-Step "Initialisation analysis"     "$Py $Scripts\init_analysis.py" }
    "bootstrap"     { Invoke-Step "Bootstrap CI"                "$Py $Scripts\bootstrap_analysis.py" }
    "l1"            { Invoke-Step "L1 regularisation test"      "$Py $Scripts\l1_regularization_test.py" }
    "checkpoint"    { Invoke-Step "Checkpoint trajectory"       "$Py $Scripts\checkpoint_analysis.py" }
    "heads"         { Invoke-Step "Head-level analysis"         "$Py $Scripts\head_level_analysis.py" }
    "modern-llms"   { Invoke-Step "Modern LLM extension"        "$Py $Scripts\modern_llms_ext.py" }
    "mlp"           { Invoke-Step "MLP layer analysis"          "$Py $Scripts\mlp_analysis.py" }
    "test"          { Invoke-Step "Tests"                       "$Py -m pytest tests/ -v" }
    "all"           {
        Invoke-Step "Pipeline"        "$Py $Scripts\run_pipeline.py"
        Invoke-Step "Layerwise"       "$Py $Scripts\run_layerwise.py"
        Invoke-Step "Init analysis"   "$Py $Scripts\init_analysis.py"
        Invoke-Step "Control: short"  "$Py $Scripts\control_short.py"
        Invoke-Step "Control: long"   "$Py $Scripts\control_long.py"
        Invoke-Step "Control: shuffle" "$Py $Scripts\control_shuffled.py"
        Invoke-Step "Bootstrap CI"    "$Py $Scripts\bootstrap_analysis.py"
        Invoke-Step "L1 test"         "$Py $Scripts\l1_regularization_test.py"
        Invoke-Step "Checkpoint"      "$Py $Scripts\checkpoint_analysis.py"
        Invoke-Step "Head-level"      "$Py $Scripts\head_level_analysis.py"
        Invoke-Step "MLP analysis"    "$Py $Scripts\mlp_analysis.py"
        Invoke-Step "Modern LLMs"     "$Py $Scripts\modern_llms_ext.py"
        Invoke-Step "Tests"           "$Py -m pytest tests/ -v"
    }
}
