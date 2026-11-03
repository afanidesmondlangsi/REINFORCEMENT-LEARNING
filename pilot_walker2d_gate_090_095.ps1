$ErrorActionPreference = "Stop"

$envName = "walker2d"
$shift = "0.5"
$seed = 0

$outputRoot = "logs_walker2d_gate_sensitivity_090_095"

$variants = @(
    @{ Name = "NORMALIZED"; Reduction = "normalized" },
    @{ Name = "UNNORMALIZED"; Reduction = "mean" }
)

foreach ($variant in $variants) {

    $runName = "$($variant.Name)_walker2d_shift_${shift}_seed_${seed}"
    $runDir = Join-Path $outputRoot $runName

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

        # Diagnostic sensitivity setting.
        ea_max_relative_error = 0.90

        ea_dynamics_seed = 104729
        ea_bc_reduction = $variant.Reduction
    }

    $json = ConvertTo-Json -InputObject $params -Compress -Depth 10
    $escapedJson = $json.Replace('"', '\"')

    Write-Host ""
    Write-Host "Running $runName"
    Write-Host "Reduction: $($variant.Reduction)"
    Write-Host "Reliability threshold: 0.90"

    python .\train.py `
        --policy ea_ntc_bc_sac `
        --env "walker2d-friction" `
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
}
