# ============================================================
# RQ1 - HALFCHEETAH FRICTION
#
# Research Question 1:
# How can offline source-task data reduce exploration
# needed in the target task?
#
# Algorithms:
#   SAC_TARGET_ONLY
#   BC_SAC
#   RLPD
#
# Shift: 0.1
# Seeds: 0, 1, 2
# Source dataset: medium
#
# Total runs: 9
# ============================================================

$params = '{\"eval_freq\":1000}'

$seeds = @(0, 1, 2)

$completed = 0
$total = 9


# ============================================================
# 1. SAC_TARGET_ONLY
# ============================================================

foreach ($seed in $seeds) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RQ1 HALFCHEETAH"
    Write-Host "SAC_TARGET_ONLY | shift=0.1 | seed=$seed"
    Write-Host "============================================================"

    python train.py `
        --policy SAC_TARGET_ONLY `
        --env halfcheetah-friction `
        --shift_level 0.1 `
        --mode 0 `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $params `
        --dir rq1_generalization

    if ($LASTEXITCODE -ne 0) {

        Write-Host ""
        Write-Host "ERROR:"
        Write-Host "SAC_TARGET_ONLY failed at seed=$seed"
        exit 1
    }

    $completed++

    Write-Host "Completed $completed / $total"
}


# ============================================================
# 2. BC_SAC
# ============================================================

foreach ($seed in $seeds) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RQ1 HALFCHEETAH"
    Write-Host "BC_SAC | shift=0.1 | seed=$seed"
    Write-Host "============================================================"

    python train.py `
        --policy BC_SAC `
        --env halfcheetah-friction `
        --shift_level 0.1 `
        --srctype medium `
        --mode 1 `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $params `
        --dir rq1_generalization

    if ($LASTEXITCODE -ne 0) {

        Write-Host ""
        Write-Host "ERROR:"
        Write-Host "BC_SAC failed at seed=$seed"
        exit 1
    }

    $completed++

    Write-Host "Completed $completed / $total"
}


# ============================================================
# 3. RLPD
# ============================================================

foreach ($seed in $seeds) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RQ1 HALFCHEETAH"
    Write-Host "RLPD | shift=0.1 | seed=$seed"
    Write-Host "============================================================"

    python train.py `
        --policy RLPD `
        --env halfcheetah-friction `
        --shift_level 0.1 `
        --srctype medium `
        --mode 1 `
        --seed $seed `
        --max_step 5000 `
        --tar_env_interact_interval 10 `
        --params $params `
        --dir rq1_generalization

    if ($LASTEXITCODE -ne 0) {

        Write-Host ""
        Write-Host "ERROR:"
        Write-Host "RLPD failed at seed=$seed"
        exit 1
    }

    $completed++

    Write-Host "Completed $completed / $total"
}


# ============================================================
# FINISHED
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "HALFCHEETAH RQ1 EXPERIMENTS FINISHED"
Write-Host "$completed / $total runs attempted successfully"
Write-Host "============================================================"