# Final 5,000-Step Reinforcement Learning Analysis

## Experimental design

- Four environments.
- Four friction shifts.
- Three matched seeds: 0, 1 and 2.
- Six algorithms.
- Five evaluation checkpoints.
- 288 experiment records.

Primary outcome: test/target_return.

AUC is calculated using trapezoidal integration from 1,000 to 5,000 steps.

## RQ1: Negative transfer

### BC-SAC
- Mean paired AUC advantage: 827549.976
- Positive transfer: 87.5%
- Negative transfer: 12.5%
### RLPD
- Mean paired AUC advantage: -262189.229
- Positive transfer: 45.8%
- Negative transfer: 54.2%
### BC-PAR
- Mean paired AUC advantage: 661429.213
- Positive transfer: 54.2%
- Negative transfer: 45.8%
### EA-NTC Mean
- Mean paired AUC advantage: 837069.300
- Positive transfer: 91.7%
- Negative transfer: 8.3%
### EA-NTC Normalized
- Mean paired AUC advantage: 856733.021
- Positive transfer: 87.5%
- Negative transfer: 12.5%

## RQ2: Task relatedness

Inspect RQ2_task_relatedness_summary.csv and RQ2_EA_NTC_pairwise_comparisons.csv.

Friction shift is treated as the experimental mismatch parameter, not as a direct measurement of source-target dynamics distance.

## RQ3: Adaptation

Inspect RQ3_adaptation_summary.csv, RQ3_EA_NTC_normalized_vs_mean_paired.csv and the EA-NTC diagnostic outputs.

## Important limitations

- The six-method primary comparison uses raw target returns.
- Normalized-score observations are not available for all six methods in the matched main dataset.
- EA adaptive controller and transfer-quality controller were disabled in the recorded configurations.
- Three seeds provide limited precision for statistical generalization.
- Positive AUC advantage indicates better cumulative target return over the measured interval, not proof of causal transfer benefit.