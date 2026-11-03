$ErrorActionPreference = "Stop"

$shifts = @("0.1", "0.5", "2.0", "5.0")
$seeds = @(0, 1, 2, 3, 4, 5)

$variants = @(
    @{ Name = "NORMALIZED"; Reduction = "normalized" },
    @{ Name = "UNNORMALIZED"; Reduction = "mean" }
)

$maxStep = 5000
$evalFreq = 1000
$evalEpisodes = 10
$targetInterval = 10

$outputRoot = "logs_hopper_corrected_gate_48runs"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$manifest = Join-Path $outputRoot "experiment_manifest.csv"

$total = 48
$completed = 0
$skipped = 0

foreach ($variant in $variants) {
    foreach ($shift in $shifts) {
        foreach ($seed in $seeds) {

            $runName = "$($variant.Name)_hopper_shift_${shift}_seed_${seed}"
            $runDir = Join-Path $outputRoot $runName
            $successFile = Join-Path $runDir "_SUCCESS.txt"

            if (Test-Path $successFile) {
                Write-Host "SKIPPING completed run: $runName"
                $skipped++
                continue
            }

            $params = @{
                eval_freq = $evalFreq
                eval_episode = $evalEpisodes

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

            Write-Host ""
            Write-Host "===================================================="
            Write-Host "RUN: $runName"
            Write-Host "Reduction: $($variant.Reduction)"
            Write-Host "Shift: $shift"
            Write-Host "Seed: $seed"
            Write-Host "===================================================="

            New-Item -ItemType Directory -Path $runDir -Force | Out-Null

            $json | Set-Content (Join-Path $runDir "run_config.json")

            $startTime = Get-Date

            python .\train.py `
                --policy ea_ntc_bc_sac `
                --env hopper-friction `
                --corrected_hopper_friction `
                --shift_level $shift `
                --mode 1 `
                --srctype medium `
                --seed $seed `
                --max_step $maxStep `
                --tar_env_interact_interval $targetInterval `
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
                throw "No TensorBoard event file found: $runName"
            }

            $endTime = Get-Date

            [PSCustomObject]@{
                treatment = $variant.Name
                reduction = $variant.Reduction
                environment = "hopper-friction"
                shift = $shift
                seed = $seed
                max_step = $maxStep
                reliability_threshold = 0.90
                adaptive_error_low = 0.75
                adaptive_error_high = 0.95
                start_time = $startTime.ToString("o")
                end_time = $endTime.ToString("o")
                status = "PROCESS_COMPLETED"
                run_directory = $runDir
            } | Export-Csv -Path $manifest -NoTypeInformation -Append

            "Training process completed; TensorBoard event file found." |
                Set-Content $successFile

            $completed++

            Write-Host "Completed this session: $completed"
            Write-Host "Skipped: $skipped"
            Write-Host "Total expected: $total"
        }
    }
}

Write-Host ""
Write-Host "===================================================="
Write-Host "CORRECTED HOPPER LAUNCHER FINISHED"
Write-Host "Newly completed: $completed"
Write-Host "Previously completed/skipped: $skipped"
Write-Host "Expected total: $total"
Write-Host "Output: $outputRoot"
Write-Host "===================================================="

