# ============================================================
# RQ1 ANT SMOKE TESTS
# ============================================================
#
# Environment : ant-friction
# Shift       : 0.1
# Seed        : 0
# Steps       : 100
# Source data : medium
#
# Algorithms:
#   1. SAC_TARGET_ONLY
#   2. BC_SAC
#   3. RLPD
# ============================================================

$params = '{\"eval_freq\":1000}'

Write-Host ""
Write-Host "============================================================"
Write-Host "RQ1 ANT SMOKE TESTS"
Write-Host "============================================================"


# ============================================================
# 1. SAC_TARGET_ONLY
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "1/3 - SAC_TARGET_ONLY"
Write-Host "Environment: ant-friction"
Write-Host "Shift: 0.1"
Write-Host "Seed: 0"
Write-Host "============================================================"

python train.py `
    --policy SAC_TARGET_ONLY `
    --env ant-friction `
    --shift_level 0.1 `
    --mode 0 `
    --seed 0 `
    --max_step 100 `
    --tar_env_interact_interval 10 `
    --params $params `
    --dir rq1_smoke

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: SAC_TARGET_ONLY"
    Write-Host "Stopping smoke tests."
    exit 1
}

Write-Host ""
Write-Host "SAC_TARGET_ONLY: PASSED"


# ============================================================
# 2. BC_SAC
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "2/3 - BC_SAC"
Write-Host "Environment: ant-friction"
Write-Host "Shift: 0.1"
Write-Host "Source data: medium"
Write-Host "Seed: 0"
Write-Host "============================================================"

python train.py `
    --policy BC_SAC `
    --env ant-friction `
    --shift_level 0.1 `
    --srctype medium `
    --mode 1 `
    --seed 0 `
    --max_step 100 `
    --tar_env_interact_interval 10 `
    --params $params `
    --dir rq1_smoke

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: BC_SAC"
    Write-Host "Stopping smoke tests."
    exit 1
}

Write-Host ""
Write-Host "BC_SAC: PASSED"


# ============================================================
# 3. RLPD
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "3/3 - RLPD"
Write-Host "Environment: ant-friction"
Write-Host "Shift: 0.1"
Write-Host "Source data: medium"
Write-Host "Seed: 0"
Write-Host "============================================================"

python train.py `
    --policy RLPD `
    --env ant-friction `
    --shift_level 0.1 `
    --srctype medium `
    --mode 1 `
    --seed 0 `
    --max_step 100 `
    --tar_env_interact_interval 10 `
    --params $params `
    --dir rq1_smoke

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED: RLPD"
    Write-Host "Stopping smoke tests."
    exit 1
}

Write-Host ""
Write-Host "RLPD: PASSED"


# ============================================================
# FINISHED
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "RQ1 ANT SMOKE TESTS COMPLETE"
Write-Host ""
Write-Host "SAC_TARGET_ONLY : PASSED"
Write-Host "BC_SAC          : PASSED"
Write-Host "RLPD            : PASSED"
Write-Host ""
Write-Host "3 / 3 smoke tests passed."
Write-Host "============================================================"