# ============================================================
# RQ2 CROSS-TASK GENERALIZATION
# TASK: HALFCHEETAH
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
        Write-Host "=============================================="
        Write-Host "SAC_TARGET_ONLY | HalfCheetah"
        Write-Host "Shift = $shift | Seed = $seed"
        Write-Host "=============================================="

        python train.py `
            --policy SAC_TARGET_ONLY `
            --env halfcheetah-friction `
            --shift_level $shift `
            --mode 0 `
            --seed $seed `
            --max_step 5000 `
            --tar_env_interact_interval 10 `
            --params $params `
            --dir rq2_generalization
    }
}


# ============================================================
# 2. BC_SAC
# ============================================================

foreach ($shift in $shifts) {

    foreach ($seed in $seeds) {

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "BC_SAC | HalfCheetah"
        Write-Host "Shift = $shift | Seed = $seed"
        Write-Host "=============================================="

        python train.py `
            --policy BC_SAC `
            --env halfcheetah-friction `
            --shift_level $shift `
            --srctype medium `
            --mode 1 `
            --seed $seed `
            --max_step 5000 `
            --tar_env_interact_interval 10 `
            --params $params `
            --dir rq2_generalization
    }
}


# ============================================================
# 3. RLPD
# ============================================================

foreach ($shift in $shifts) {

    foreach ($seed in $seeds) {

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "RLPD | HalfCheetah"
        Write-Host "Shift = $shift | Seed = $seed"
        Write-Host "=============================================="

        python train.py `
            --policy RLPD `
            --env halfcheetah-friction `
            --shift_level $shift `
            --srctype medium `
            --mode 1 `
            --seed $seed `
            --max_step 5000 `
            --tar_env_interact_interval 10 `
            --params $params `
            --dir rq2_generalization
    }
}


Write-Host ""
Write-Host "=============================================="
Write-Host "HALFCHEETAH RQ2 RUNS FINISHED"
Write-Host "=============================================="