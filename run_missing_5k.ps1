param(
    [switch]$Run,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path ".").Path
$outputRoot = Join-Path $root "experiments_10k_576"
$resultsRoot = Join-Path $outputRoot "results"

$environments = @(
    "hopper-friction",
    "halfcheetah-friction",
    "walker2d-friction",
    "ant-friction"
)

$shifts = @("0.1", "5.0")
$seeds = @(0, 1, 2)

$algorithms = @(
    @{
        Name = "SAC_TARGET_ONLY"
        Policy = "sac_target_only"
        Mode = 0
    },
    @{
        Name = "RLPD"
        Policy = "rlpd"
        Mode = 1
    }
)

$params = @{
    eval_freq = 1000
    eval_episode = 10
    save_freq = 5000
}

$checker = Join-Path $outputRoot "verify_5k_run.py"

if (-not (Test-Path $checker)) {
    throw "Missing 5k completion checker: $checker"
}

function Test-Complete {
    param([string]$RunDir)

    $result = & python $checker $RunDir 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        return $true
    }

    if ($exitCode -eq 1) {
        return $false
    }

    throw "Verification failed for $RunDir : $result"
}

$manifest = @()
$executed = 0

foreach ($shift in $shifts) {

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STARTING SHIFT: $shift" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    foreach ($environment in $environments) {

        # Hopper 0.1 is already available for both algorithms.
        if ($shift -eq "0.1" -and $environment -eq "hopper-friction") {
            Write-Host "SKIP EXISTING: $environment shift=$shift"
            continue
        }

        foreach ($algorithm in $algorithms) {

            foreach ($seed in $seeds) {

                $variantRoot = Join-Path $resultsRoot $algorithm.Name

                if ($algorithm.Mode -eq 0) {
                    $runDir = Join-Path `
                        (Join-Path $variantRoot $algorithm.Policy) `
                        "$environment-$shift"
                }
                else {
                    $runDir = Join-Path `
                        (Join-Path $variantRoot $algorithm.Policy) `
                        "$environment-srcdatatype-medium-$shift"
                }

                $runDir = Join-Path $runDir "r$seed"

                $manifest += [pscustomobject]@{
                    Algorithm = $algorithm.Name
                    Environment = $environment
                    Shift = $shift
                    Seed = $seed
                    RunDirectory = $runDir
                    Status = "PENDING"
                }

                if (-not $Run) {
                    continue
                }

                if ($Limit -gt 0 -and $executed -ge $Limit) {
                    continue
                }

                if (Test-Path $runDir) {

                    if (Test-Complete -RunDir $runDir) {
                        Write-Host "SKIP COMPLETE: $($algorithm.Name) $environment shift=$shift seed=$seed" -ForegroundColor Green
                        continue
                    }

                    throw "Existing incomplete directory. Refusing to overwrite: $runDir"
                }

                $arguments = @(
                    ".\train.py",
                    "--policy", $algorithm.Policy,
                    "--env", $environment,
                    "--shift_level", $shift,
                    "--mode", $algorithm.Mode,
                    "--seed", $seed,
                    "--max_step", 5000,
                    "--tar_env_interact_interval", 10,
                    "--params", ($params | ConvertTo-Json -Compress -Depth 10),
                    "--dir", $variantRoot
                )

                if ($algorithm.Mode -eq 1) {
                    $arguments += @("--srctype", "medium")
                }

                if ($environment -eq "hopper-friction") {
                    $arguments += "--corrected_hopper_friction"
                }

                $logDir = Join-Path $outputRoot "execution_logs_5k"
                New-Item -ItemType Directory -Path $logDir -Force | Out-Null

                $logName = "$($algorithm.Name)_${environment}_${shift}_seed$seed"

                $stdout = Join-Path $logDir "$logName.stdout.log"
                $stderr = Join-Path $logDir "$logName.stderr.log"

                Write-Host "`nSTART: $logName" -ForegroundColor Yellow

                $jsonIndex = [Array]::IndexOf($arguments, "--params") + 1
                $arguments[$jsonIndex] = $arguments[$jsonIndex].Replace('"', '\"')

                $previousPreference = $ErrorActionPreference
                $ErrorActionPreference = "Continue"

                try {
                    & python @arguments 1> $stdout 2> $stderr
                    $trainingExitCode = $LASTEXITCODE
                }
                finally {
                    $ErrorActionPreference = $previousPreference
                }

                if ($trainingExitCode -ne 0) {
                    throw "Training failed: $logName. Check $stderr"
                }

                if (-not (Test-Complete -RunDir $runDir)) {
                    throw "5,000-step verification failed: $logName"
                }

                $executed++

                Write-Host "COMPLETE: $logName" -ForegroundColor Green
            }
        }
    }

    Write-Host "`nFINISHED SHIFT: $shift" -ForegroundColor Green
}

$manifestPath = Join-Path $outputRoot "experiment_manifest_missing_5k.csv"

$manifest | Export-Csv $manifestPath -NoTypeInformation

Write-Host "`n========== 5K EXPERIMENT SUMMARY ==========" -ForegroundColor Cyan
Write-Host "Configurations prepared: $($manifest.Count)"
Write-Host "Training runs executed: $executed"
Write-Host "Expected missing runs: 42"
Write-Host "Manifest: $manifestPath"
