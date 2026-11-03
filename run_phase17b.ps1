$ErrorActionPreference = "Stop"

Set-Location "C:\Users\LANGS\Documents\off-dynamics-rl"

$root = "logs_ea_ntc_phase17b_normalized"
$shifts = @("0.1", "0.5")
$seeds = @(0, 1, 2)

if (Test-Path $root) {
    throw "Output directory already exists: $root"
}

New-Item -ItemType Directory -Path $root | Out-Null

$completed = 0
$total = 6

foreach ($shift in $shifts) {
    foreach ($seed in $seeds) {

        $runName = "normalized_shift_${shift}_seed_${seed}"
        $runDir = Join-Path $root $runName

        $params = @{
            eval_freq = 1000
            ea_gate_bc = $true
            ea_force_unit_weights = $false
            ea_gate_uncertainty = $true
            ea_uncertainty_temperature = 4.0
            ea_adaptive_gate = $true
            ea_adaptive_max_strength = 0.50
            ea_adaptive_error_low = 0.35
            ea_adaptive_error_high = 0.75
            ea_min_dynamics_updates = 20
            ea_reliability_mode = "relative"
            ea_max_relative_error = 0.75
            ea_dynamics_seed = 104729
            ea_bc_reduction = "normalized"
        }

        $json = ConvertTo-Json -InputObject $params -Compress
        $escapedJson = $json.Replace('"', '\"')

        Write-Host ""
        Write-Host "===================================="
        Write-Host "PHASE 17B: $($completed + 1) / $total"
        Write-Host "Shift: $shift | Seed: $seed"
        Write-Host "Output: $runDir"
        Write-Host "===================================="

        python .\train.py `
            --policy ea_ntc_bc_sac `
            --env halfcheetah-friction `
            --shift_level $shift `
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

        $tb = @(
            Get-ChildItem `
                -Path $runDir `
                -Directory `
                -Recurse `
                -Filter tb
        )

        if ($tb.Count -ne 1) {
            throw "TensorBoard validation failed: $runName"
        }

        $completed++
        Write-Host "PASS: $runName"
    }
}

Write-Host "PHASE 17B COMPLETE: $completed / $total"
