import pandas as pd
from pathlib import Path

p = Path("matched_final_5000/ea_ntc_diagnostics")
d = pd.read_csv(p / "ea_ntc_diagnostics_timeseries.csv")

tags = [
    "ea_ntc/adaptive_gate_strength",
    "ea_ntc/mean_source_weight",
    "ea_ntc/source_weight_ess",
    "ea_ntc/heldout_relative_mse",
    "ea_ntc/gate_ready"
]

d = d[d["tag"].isin(tags)].copy()

summary = (
    d.groupby(["environment", "shift", "method", "tag"])
    ["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

for tag in tags:
    print("\n" + "=" * 75)
    print(tag.upper())

    x = summary[summary["tag"] == tag]

    print(
        x.pivot_table(
            index=["environment", "shift"],
            columns="method",
            values="mean"
        ).round(4).to_string()
    )

out = Path("matched_final_5000/final_three_RQ_analysis/paper_tables")
summary.to_csv(out / "RQ3_gate_response_by_shift.csv", index=False)

print("\nSaved RQ3_gate_response_by_shift.csv")
