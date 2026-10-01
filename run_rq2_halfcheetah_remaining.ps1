# ============================================================
# RQ2 GENERALIZATION EXPERIMENTS
# TASK: HalfCheetah friction
#
# Already completed:
#   SAC_TARGET_ONLY | shift 0.5 | seed 0
#   SAC_TARGET_ONLY | shift 2.0 | seed 0
#   BC_SAC          | shift 0.5 | seed 0
#   RLPD            | shift 0.5 | seed 0
#
# This script runs ONLY the remaining 14 experiments.
# ============================================================

$params = '{\"eval_freq\":1000}'


# ============================================================
# 1. SAC_TARGET_ONLY
# Remaining seeds: 1 and 2 for both shifts
# ============================================================

foreach ($shift in @("0.5", "2.0")) {

    foreach ($seed in @(1,2)) {

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
#
# shift 0.5: seed 0 already complete -> run seeds 1 and 2
# shift 2.0: run seeds 0, 1 and 2
# ============================================================

foreach ($seed in @(1,2)) {

    Write-Host ""
    Write-Host "=============================================="
    Write-Host "BC_SAC | HalfCheetah"
    Write-Host "Shift = 0.5 | Seed = $seed"
    Write-Host "=============================================="

    python train.py `
        --policy BC_SAC `
        --env halfcheetah-friction `
        --shift_level "0.5" `
        --srctype medium `
        --mode 1 `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $params `
        --dir rq2_generalization
}


foreach ($seed in @(0,1,2)) {

    Write-Host ""
    Write-Host "=============================================="
    Write-Host "BC_SAC | HalfCheetah"
    Write-Host "Shift = 2.0 | Seed = $seed"
    Write-Host "=============================================="

    python train.py `
        --policy BC_SAC `
        --env halfcheetah-friction `
        --shift_level "2.0" `
        --srctype medium `
        --mode 1 `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $params `
        --dir rq2_generalization
}


# ============================================================
# 3. RLPD
#
# shift 0.5: seed 0 already complete -> run seeds 1 and 2
# shift 2.0: run seeds 0, 1 and 2
# ============================================================

foreach ($seed in @(1,2)) {

    Write-Host ""
    Write-Host "=============================================="
    Write-Host "RLPD | HalfCheetah"
    Write-Host "Shift = 0.5 | Seed = $seed"
    Write-Host "=============================================="

    python train.py `
        --policy RLPD `
        --env halfcheetah-friction `
        --shift_level "0.5" `
        --srctype medium `
        --mode 1 `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $params `
        --dir rq2_generalization
}


foreach ($seed in @(0,1,2)) {

    Write-Host ""
    Write-Host "=============================================="
    Write-Host "RLPD | HalfCheetah"
    Write-Host "Shift = 2.0 | Seed = $seed"
    Write-Host "=============================================="

    python train.py `
        --policy RLPD `
        --env halfcheetah-friction `
        --shift_level "2.0" `
        --srctype medium `
        --mode 1 `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $params `
        --dir rq2_generalization
}


Write-Host ""
Write-Host "=============================================="
Write-Host "HALFCHEETAH REMAINING RQ2 RUNS FINISHED"
Write-Host "=============================================="