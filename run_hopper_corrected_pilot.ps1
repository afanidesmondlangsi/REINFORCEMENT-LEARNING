$ErrorActionPreference = "Stop"

$variants = @(
    @{ Name = "NORMALIZED"; Reduction = "normalized" },
    @{ Name = "UNNORMALIZED"; Reduction = "mean" }
)

$outputRoot = "logs_hopper_corrected_pilot"
$shift = "0.1"
$seed = 0

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

foreach ($variant in $variants) {

    $runName = "$($variant.Name)_hopper_shift_${shift}_seed_${seed}"
    $runDir = Join-Path $outputRoot $runName

    $successFile = Join-Path $runDir "_SUCCESS.txt"

    if (Test-Path $successFile) {
        Write-Host "Skipping completed run: $runName"
        continue
    }

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $params = @{
        eval_freq = 1000
        eval_episode = 10

        ea_gate_bc = $true
        ea_force_unit_weights = $false

        ea_gate_uncertainty = $true
        ea_uncertainty_temperature = 4.0

        ea_adaptive_gate = $true
        ea_adaptive_max_strength = 0.50
        ea_adaptive_error_low = 0.75
        ea_adaptive_error_high = 0.95

        ea_min_dynamics_updates = 20
        ea_reliability_mode = "relative"
        ea_max_relative_error = 0.90

        ea_dynamics_seed = 104729
        ea_bc_reduction = $variant.Reduction
    }

    $json = ConvertTo-Json -InputObject $params -Compress -Depth 10
    $escapedJson = $json.Replace('"', '\"')

    $json | Set-Content (Join-Path $runDir "run_config.json")

    Write-Host ""
    Write-Host "==========================================="
    Write-Host "RUN: $runName"
    Write-Host "Reduction: $($variant.Reduction)"
    Write-Host "Corrected Hopper: ENABLED"
    Write-Host "==========================================="

    python .\train.py `
        --policy ea_ntc_bc_sac `
        --env hopper-friction `
        --shift_level $shift `
        --corrected_hopper_friction `
        --mode 1 `
        --srctype medium `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $escapedJson `
        --dir $runDir

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $runName"
    }

    $eventFiles = @(
        Get-ChildItem -Path $runDir -Recurse -File `
            -Filter "events.out.tfevents.*" `
            -ErrorAction SilentlyContinue
    )

    if ($eventFiles.Count -eq 0) {
        throw "No TensorBoard events found: $runName"
    }

    "Training process completed; TensorBoard events found." |
        Set-Content $successFile

    Write-Host "Completed: $runName"
}

Write-Host ""
Write-Host "CORRECTED HOPPER PILOT FINISHED"
