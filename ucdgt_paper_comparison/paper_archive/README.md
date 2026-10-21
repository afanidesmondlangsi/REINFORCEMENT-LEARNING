# UCDGT Research Archive

## Research scope

This archive preserves results supporting the
offline-to-online reinforcement learning paper.

The original RQ1-RQ3 experiments and the UCDGT
adaptive-temperature extension are separate analyses.

## Adaptive-temperature experiment

- Environments: Ant, HalfCheetah, Hopper, Walker2d
- Friction shifts: 0.5 and 2.0
- Seeds: 0, 1, 2
- Completed adaptive runs: 24
- Initial temperature: 10
- ESS target: 0.85
- Temperature bounds: [5, 20]
- Controller updates per run: 8
- Target training interactions: 10,000

## Main adaptive findings

1. The ESS controller responds differently across environments.
2. Ant, HalfCheetah and Walker2d generally decrease temperature.
3. Hopper exhibits mixed or increasing temperature adaptation.
4. ESS tracking does not guarantee stable target-policy performance.
5. Adaptive temperature outperforms fixed tau=10 in all four
   currently matched HalfCheetah/Hopper conditions.
6. Adaptive temperature does not consistently outperform the
   best fixed temperature.

## Experimental limitations

- Three seeds per condition.
- Fixed-temperature baselines for Ant and Walker2d are missing
  from the matched ablation analysis.
- Controller and evaluation logging use different time axes.
- Observed associations do not establish causality.
- The archive contains derived outputs rather than raw
  TensorBoard event files or checkpoints.

## Organization

adaptive_results/
    Run metrics, learning curves, AUC and comparisons.

fixed_temperature/
    Fixed-temperature ablation results.

controller_diagnostics/
    ESS tracking, temperature updates, learning stability.

figures/
    Controller and learning-curve plots.

original_results/
    Selected earlier RQ1-RQ3 and UCDGT results.

methodology/
    Selected experiment configuration files.

## Reproducibility

See the experiment configurations, source implementations,
and analysis scripts in the parent repository.

A complete raw-data backup should additionally preserve
TensorBoard event files, checkpoints when required,
dependencies, and source-dataset provenance.
