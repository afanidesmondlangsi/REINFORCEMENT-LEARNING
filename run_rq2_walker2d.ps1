# ============================================================
# RQ2 GENERALIZATION EXPERIMENTS
# Environment: Walker2d friction
#
# Algorithms:
#   1. SAC_TARGET_ONLY
#   2. BC_SAC
#   3. RLPD
#
# Shift levels:
#   0.5
#   2.0
#
# Seeds:
#   0, 1, 2
#
# Total:
#   3 algorithms x 2 shifts x 3 seeds = 18 runs
# ============================================================

$params = '{\"eval_freq\":1000}'

$shifts = @("0.5", "2.0")
$seeds  = @(0, 1, 2)


# ============================================================
# 1. SAC_TARGET_ONLY
# ============================================================

foreach ($shift in $shifts) {

    foreach ($seed in $seeds) {

        Write-Host ""
        Write-Host "================================================"
        Write-Host "SAC_TARGET_ONLY"
        Write-Host "Environment: walker2d-friction"
        Write-Host "Shift: $shift"
        Write-Host "Seed: $seed"
        Write-Host "================================================"

        python train.py `
            --policy SAC_TARGET_ONLY `
            --env walker2d-friction `
            --shift_level $shift `
            --mode 0 `
            --seed $seed `
            --max_step 5000 `
            --tar_env_interact_interval 10 `
            --params $params `
            --dir rq2_generalization

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: SAC_TARGET_ONLY shift=$shift seed=$seed"
            exit 1
        }
    }
}


# ============================================================
# 2. BC_SAC
# ============================================================

foreach ($shift in $shifts) {

    foreach ($seed in $seeds) {

        Write-Host ""
        Write-Host "================================================"
        Write-Host "BC_SAC"
        Write-Host "Environment: walker2d-friction"
        Write-Host "Shift: $shift"
        Write-Host "Seed: $seed"
        Write-Host "================================================"

        python train.py `
            --policy BC_SAC `
            --env walker2d-friction `
            --shift_level $shift `
            --srctype medium `
            --mode 1 `
            --seed $seed `
            --max_step 5000 `
            --tar_env_interact_interval 10 `
            --params $params `
            --dir rq2_generalization

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: BC_SAC shift=$shift seed=$seed"
            exit 1
        }
    }
}


# ============================================================
# 3. RLPD
# ============================================================

foreach ($shift in $shifts) {

    foreach ($seed in $seeds) {

        Write-Host ""
        Write-Host "================================================"
        Write-Host "RLPD"
        Write-Host "Environment: walker2d-friction"
        Write-Host "Shift: $shift"
        Write-Host "Seed: $seed"
        Write-Host "================================================"

        python train.py `
            --policy RLPD `
            --env walker2d-friction `
            --shift_level $shift `
            --srctype medium `
            --mode 1 `
            --seed $seed `
            --max_step 5000 `
            --tar_env_interact_interval 10 `
            --params $params `
            --dir rq2_generalization

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: RLPD shift=$shift seed=$seed"
            exit 1
        }
    }
}


Write-Host ""
Write-Host "================================================"
Write-Host "WALKER2D RQ2 EXPERIMENTS FINISHED"
Write-Host "18 / 18 runs attempted successfully"
Write-Host "================================================"