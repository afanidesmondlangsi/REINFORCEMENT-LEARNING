param(
    [switch]$Run,
    [ValidateSet("ALL","BC_PAR","BC_SAC","RLPD","SAC_TARGET_ONLY","EA_NTC_MEAN","EA_NTC_NORMALIZED")]
    [string]$SelectedAlgorithm = "ALL",
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

$shifts = @("0.1", "0.5", "2.0", "5.0")
$seeds = 0..5

$algorithms = @(
    @{ Name = "BC_PAR"; Policy = "bc_par"; Mode = 1 },
    @{ Name = "BC_SAC"; Policy = "bc_sac"; Mode = 1 },
    @{ Name = "RLPD"; Policy = "rlpd"; Mode = 1 },
    @{ Name = "SAC_TARGET_ONLY"; Policy = "sac_target_only"; Mode = 0 },
    @{ Name = "EA_NTC_MEAN"; Policy = "ea_ntc_bc_sac"; Mode = 1; Reduction = "mean" },
    @{ Name = "EA_NTC_NORMALIZED"; Policy = "ea_ntc_bc_sac"; Mode = 1; Reduction = "normalized" }
)

$eaParams = @{
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
}

function Test-Complete {
    param([string]$RunDir)

    $checker = Join-Path $outputRoot "verify_10k_run.py"

    if (-not (Test-Path $checker)) {
        throw "TensorBoard completion checker missing: $checker"
    }

    $result = & python $checker $RunDir 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        return $true
    }

    if ($exitCode -eq 1) {
        return $false
    }

    throw "Completion checker failed for ${RunDir}: $result"
}

$executed = 0
$manifest = @()

foreach ($algorithm in $algorithms) {
    if ($SelectedAlgorithm -ne "ALL" -and $algorithm.Name -ne $SelectedAlgorithm) { continue }
    foreach ($environment in $environments) {
        foreach ($shift in $shifts) {
            foreach ($seed in $seeds) {

                $variantRoot = Join-Path $resultsRoot $algorithm.Name
                $runDir = Join-Path $variantRoot "$environment-srcdatatype-medium-$shift"
                if ($algorithm.Mode -eq 0) { $runDir = Join-Path (Join-Path $variantRoot $algorithm.Policy) "$environment-$shift" } else { $runDir = Join-Path (Join-Path $variantRoot $algorithm.Policy) "$environment-srcdatatype-medium-$shift" }; $runDir = Join-Path $runDir "r$seed"

                $params = @{
                    eval_freq = 1000
                    eval_episode = 10
                }

                if ($algorithm.Name -like "EA_NTC_*") {
                    foreach ($key in $eaParams.Keys) {
                        $params[$key] = $eaParams[$key]
                    }
                    $params["ea_bc_reduction"] = $algorithm.Reduction
                }

                $arguments = @(
                    ".\train.py",
                    "--policy", $algorithm.Policy,
                    "--env", $environment,
                    "--shift_level", $shift,
                    "--mode", $algorithm.Mode,
                    "--seed", $seed,
                    "--max_step", 10000,
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

                $manifest += [pscustomobject]@{
                    Algorithm = $algorithm.Name
                    Environment = $environment
                    Shift = $shift
                    Seed = $seed
                    Status = "PENDING"
                    RunDirectory = $runDir
                }

                if ($Run -and ($Limit -le 0 -or $executed -lt $Limit)) {

    if (Test-Path $runDir) {
        if (Test-Complete -RunDir $runDir) {
            Write-Host "SKIP COMPLETE: $($algorithm.Name) $environment $shift seed=$seed"
            continue
        }

        throw "Existing incomplete run directory: $runDir. Refusing to overwrite."
    }

    $logDir = Join-Path $outputRoot "execution_logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null

    $logName = "$($algorithm.Name)_${environment}_${shift}_seed$seed"
    $stdout = Join-Path $logDir "$logName.stdout.log"
    $stderr = Join-Path $logDir "$logName.stderr.log"

    Write-Host "START: $logName"

    $jsonIndex = [Array]::IndexOf($arguments, "--params") + 1
    $arguments[$jsonIndex] = $arguments[$jsonIndex].Replace('"', '\"')

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @arguments 1> $stdout 2> $stderr
        $trainingExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($trainingExitCode -ne 0) {
        throw "Training failed: $logName. Check $stderr"
    }

    if (-not (Test-Complete -RunDir $runDir)) {
        throw "Training finished without ten verified evaluations: $logName"
    }

    $executed++
    Write-Host "COMPLETE: $logName"
}
            }
        }
    }
}

$manifestPath = Join-Path $outputRoot "experiment_manifest_$($SelectedAlgorithm)_10k.csv"
$manifest | Export-Csv $manifestPath -NoTypeInformation

Write-Host "Configurations prepared: $($manifest.Count)"
Write-Host "Manifest: $manifestPath"
Write-Host "Training runs executed: $executed"












