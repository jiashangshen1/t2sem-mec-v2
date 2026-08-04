Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);'
# ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x1) -> keep system awake while this shell lives
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null
$env:PYTHONIOENCODING = "utf-8"

function Log($m) { "$m $(Get-Date -Format o)" | Out-File results/paper_suite.log -Append -Encoding utf8 }

Log "SUITE_START"

# Step 1: Pareto sweep (skip if already produced its CSV)
if (-not (Test-Path results/pareto/pareto_points.csv)) {
    python -u scripts/pareto_sweep.py --episodes 150 --seeds 42 123 --eval-episodes 15 *> results/pareto_full.log
    Log "PARETO_EXIT=$LASTEXITCODE"
} else { Log "SKIP pareto (csv exists)" }

# Step 2: mask sensitivity (zero-retrain; uses official checkpoints)
if (-not (Test-Path results/improved_pertier/mask_sensitivity.csv)) {
    python -u scripts/mask_sensitivity.py --eval-episodes 10 *> results/mask_sens.log
    Log "MASKSENS_EXIT=$LASTEXITCODE"
} else { Log "SKIP masksens (csv exists)" }

# Step 3: full ablation (T1 single-component + T2 cumulative anchor)
if (-not (Test-Path results/ablation/ablation_summary.csv)) {
    python -u scripts/ablation_run.py --episodes 150 --seeds 42 123 --eval-episodes 15 *> results/ablation_full.log
    Log "ABLATION_EXIT=$LASTEXITCODE"
} else { Log "SKIP ablation (csv exists)" }

Log "SUITE_DONE"
