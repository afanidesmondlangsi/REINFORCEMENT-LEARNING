$ErrorActionPreference = "Stop"

Set-Location "C:\Users\LANGS\Documents\off-dynamics-rl"

$root = "logs_ea_ntc_phase17c_confirmation"

if (Test-Path $root) {
    throw "Output directory already exists: $root"
}

$source = ".\algo\offline_online\ea_ntc_bc_sac.py"

if (-not (Test-Path $source)) {
    throw "EA-NTC source file not found."
}

if (-not (Select-String -Path $source -Pattern 'ea_bc_reduction' -Quiet)) {
    throw "Phase 17B BC-loss reduction option not found."
}

python -m py_compile $source

if ($LASTEXITCODE -ne 0) {
    throw "Python syntax validation failed."
}

$shifts = @("0.1", "0.5")
$seeds = @(3, 4, 5)
$reductions = @("mean", "normalized")

New-Item -ItemType Directory -Path $root | Out-Null

$completed = 0
$total = 12

foreach ($shift in $shifts) {
    foreach ($seed in $seeds) {
        foreach ($reduction in $reductions) {

            $name = "${reduction}_shift_${shift}_seed_${seed}"
            $runDir = Join-Path $root $name

            if (Test-Path $runDir) {
                throw "Existing run directory: $runDir"
            }

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
                ea_bc_reduction = $reduction
            }

            $json = ConvertTo-Json -InputObject $params -Compress
            $escapedJson = $json.Replace('"', '\"')

            Write-Host ""
            Write-Host "========================================"
            Write-Host "PHASE 17C: $($completed + 1) / $total"
            Write-Host "Friction: $shift"
            Write-Host "Seed: $seed"
            Write-Host "Reduction: $reduction"
            Write-Host "Output: $runDir"
            Write-Host "========================================"

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
                throw "Training failed: $name"
            }

            $tbDirs = @(
                Get-ChildItem `
                    -Path $runDir `
                    -Directory `
                    -Recurse `
                    -Filter tb
            )

            if ($tbDirs.Count -ne 1) {
                throw "TensorBoard directory validation failed: $name"
            }

            $eventFiles = @(
                Get-ChildItem `
                    -Path $tbDirs[0].FullName `
                    -File `
                    -Recurse `
                    -Filter "events.out.tfevents.*"
            )

            if ($eventFiles.Count -eq 0) {
                throw "No TensorBoard event files found: $name"
            }

            $completed++
            Write-Host "PASS: $name"
        }
    }
}

Write-Host ""
Write-Host "PHASE 17C COMPLETE: $completed / $total"
