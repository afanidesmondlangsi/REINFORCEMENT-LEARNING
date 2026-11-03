
$ErrorActionPreference = "Stop"

$root = "C:\Users\LANGS\Documents\off-dynamics-rl"
Set-Location $root

# ============================================================
# EA-NTC NORMALIZED VS UNNORMALIZED
# FOUR ENVIRONMENTS / FOUR SUPPORTED SHIFTS / SIX SEEDS
# RESUME-SAFE VERSION
# ============================================================

$environments = @(
    "hopper",
    "halfcheetah",
    "walker2d",
    "ant"
)

# IMPORTANT:
# These are the four shifts currently supported by the
# repository's MuJoCo XML/environment implementation.

$shifts = @("0.1", "0.5", "2.0", "5.0")

$seeds = @(0, 1, 2)

$variants = @(
    @{
        Name = "NORMALIZED"
        Reduction = "normalized"
    },
    @{
        Name = "UNNORMALIZED"
        Reduction = "mean"
    }
)

# Training protocol
$maxStep = 10000
$evalFreq = 1000
$evalEpisodes = 10
$targetInterval = 10

# Reuse the existing experiment directory.
$outputRoot = "experiments_ea_ntc_10k_96"

$manifestPath = Join-Path $outputRoot "experiment_manifest.csv"

if (!(Test-Path $outputRoot)) {
    New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
}

# Load previous manifest records.
$manifest = [System.Collections.Generic.List[object]]::new()

if (Test-Path $manifestPath) {
    Import-Csv $manifestPath | ForEach-Object {
        $manifest.Add($_)
    }
}

$total = $environments.Count * $shifts.Count * $seeds.Count * $variants.Count
$attempted = 0
$skipped = 0
$failed = 0

Write-Host ""
Write-Host "============================================"
Write-Host "EA-NTC RESUMABLE EXPERIMENT"
Write-Host "============================================"
Write-Host "Environments: $($environments -join ', ')"
Write-Host "Shifts: $($shifts -join ', ')"
Write-Host "Seeds: $($seeds -join ', ')"
Write-Host "Treatments: NORMALIZED / UNNORMALIZED"
Write-Host "Expected runs: $total"
Write-Host "============================================"

foreach ($variant in $variants) {

    foreach ($envName in $environments) {

        foreach ($shift in $shifts) {

            foreach ($seed in $seeds) {

                $runName = "$($variant.Name)_${envName}_shift_${shift}_seed_${seed}"

                $runDir = Join-Path $outputRoot $runName

                # Skip previously verified completed runs.
                $successFile = Join-Path $runDir "_SUCCESS.txt"

                if (Test-Path $successFile) {

                    Write-Host "SKIPPING COMPLETED: $runName" `
                        -ForegroundColor Cyan

                    $skipped++
                    continue
                }

                # Preserve incomplete attempts before retrying.
                if (Test-Path $runDir) {

                    $archiveRoot = Join-Path $outputRoot "_incomplete_attempts"

                    New-Item -ItemType Directory `
                        -Path $archiveRoot -Force | Out-Null

                    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

                    $archivePath = Join-Path $archiveRoot `
                        "${runName}_${timestamp}"

                    Move-Item -Path $runDir -Destination $archivePath

                    Write-Host "Archived incomplete attempt: $archivePath"
                }

                $attempted++

                Write-Host ""
                Write-Host "============================================"
                Write-Host "RUN: $runName"
                Write-Host "Treatment: $($variant.Name)"
                Write-Host "Reduction: $($variant.Reduction)"
                Write-Host "Environment: $envName"
                Write-Host "Shift: $shift"
                Write-Host "Seed: $seed"
                Write-Host "============================================"

                # Common settings for both EA-NTC treatments.
                $params = @{
                    eval_freq = $evalFreq
                    eval_episode = $evalEpisodes

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

                    # The only treatment difference:
                    ea_bc_reduction = $variant.Reduction
                }

                $json = ConvertTo-Json -InputObject $params -Compress -Depth 10

                $escapedJson = $json.Replace('"', '\"')

                $status = "FAILED"
                $errorMessage = ""

                try {

                    $environmentArg = "${envName}-friction"

                    python .\train.py `
                        --policy ea_ntc_bc_sac `
                        --env $environmentArg `
                        --shift_level $shift `
                        --mode 1 `
                        --srctype medium `
                        --seed $seed `
                        --max_step $maxStep `
                        --tar_env_interact_interval $targetInterval `
                        --params $escapedJson `
                        --dir $runDir

                    if ($LASTEXITCODE -ne 0) {
                        throw "Training returned exit code $LASTEXITCODE"
                    }

                    # Verify TensorBoard event files.

                    $tbDirs = @(
                        Get-ChildItem -Path $runDir `
                            -Directory -Recurse -Filter "tb"
                    )

                    if ($tbDirs.Count -ne 1) {
                        throw "Expected exactly one TensorBoard directory."
                    }

                    $events = @(
                        Get-ChildItem -Path $tbDirs[0].FullName `
                            -File -Filter "events.out.tfevents.*"
                    )

                    if ($events.Count -eq 0) {
                        throw "TensorBoard event files missing."
                    }

                    $status = "TRAINING_COMPLETE_EVENTS_PRESENT"

                    Set-Content `
                        -Path (Join-Path $runDir "_SUCCESS.txt") `
                        -Value "Training process completed; event files verified."

                    Write-Host "SUCCESS: $runName" -ForegroundColor Green

                }
                catch {

                    $status = "FAILED"
                    $errorMessage = $_.Exception.Message
                    $failed++

                    Write-Host "FAILED: $runName" -ForegroundColor Red
                    Write-Host $errorMessage
                }

                # Record every attempt.
                $record = [PSCustomObject]@{
                    treatment = $variant.Name
                    reduction = $variant.Reduction
                    environment = $envName
                    shift = $shift
                    seed = $seed
                    max_step = $maxStep
                    eval_freq = $evalFreq
                    eval_episode = $evalEpisodes
                    target_interval = $targetInterval
                    run_path = $runDir
                    status = $status
                    error = $errorMessage
                }

                $manifest.Add($record)

                $manifest |
                    Export-Csv -Path $manifestPath `
                        -NoTypeInformation

                if ($status -eq "FAILED") {
                    throw "Stopped after failure. Manifest preserved. Correct the problem and rerun."
                }
            }
        }
    }
}

Write-Host ""
Write-Host "============================================"
Write-Host "EXPERIMENT LOOP FINISHED"
Write-Host "============================================"

Write-Host "Total planned: $total"
Write-Host "New runs attempted: $attempted"
Write-Host "Previously completed/skipped: $skipped"
Write-Host "Failures: $failed"

Write-Host ""
Write-Host "Manifest: $manifestPath"

$manifest |
    Group-Object treatment |
    ForEach-Object {
        Write-Host "$($_.Name): $($_.Count) manifest records"
    }

