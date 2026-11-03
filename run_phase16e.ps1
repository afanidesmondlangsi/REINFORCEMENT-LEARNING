$ErrorActionPreference = "Stop"

Set-Location "C:\Users\LANGS\Documents\off-dynamics-rl"

$root = "logs_ea_ntc_phase16e"
$shifts = @("0.1", "0.5")
$seeds = @(0, 1, 2)
$strengths = @(0.25, 0.50)

if (Test-Path $root) {
    throw "Experiment root already exists: $root"
}

New-Item -ItemType Directory -Path $root | Out-Null

$completed = 0
$total = $shifts.Count * $seeds.Count * $strengths.Count

foreach ($strength in $strengths) {

    $strengthName = ("{0:F2}" -f $strength).Replace(".", "p")

    foreach ($shift in $shifts) {

        foreach ($seed in $seeds) {

            $runName = "lambda_${strengthName}_shift_${shift}_seed_${seed}"
            $runDir = Join-Path $root $runName

            if (Test-Path $runDir) {
                throw "Run directory already exists: $runDir"
            }

            $params = @{
                eval_freq = 1000
                ea_gate_bc = $true
                ea_force_unit_weights = $false
                ea_gate_uncertainty = $true
                ea_uncertainty_temperature = 4.0
                ea_gate_strength = $strength
                ea_min_dynamics_updates = 20
                ea_reliability_mode = "relative"
                ea_max_relative_error = 0.75
                ea_dynamics_seed = 104729
            }

            $json = ConvertTo-Json -InputObject $params -Compress
            $escapedJson = $json.Replace('"', '\"')

            Write-Host ""
            Write-Host "================================================"
            Write-Host "PHASE 16E"
            Write-Host "Gate strength: $strength"
            Write-Host "Friction shift: $shift"
            Write-Host "Seed: $seed"
            Write-Host "Run: $($completed + 1) / $total"
            Write-Host "================================================"

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

            $tbDirs = @(
                Get-ChildItem -Path $runDir `
                    -Directory -Recurse -Filter tb
            )

            if ($tbDirs.Count -ne 1) {
                throw "Expected one TensorBoard directory for $runName; found $($tbDirs.Count)"
            }

            $completed++

            Write-Host "PASS: $runName"
        }
    }
}

Write-Host ""
Write-Host "PHASE 16E COMPLETE: $completed / $total"
