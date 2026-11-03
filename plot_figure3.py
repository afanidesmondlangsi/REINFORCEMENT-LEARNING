import numpy as np
import matplotlib.pyplot as plt

# Approximate values read from Figure 3
environments = ["Hopper", "HalfCheetah", "Walker2d", "Ant"]

normalized_1k_5k = [4.25, 5.10, 5.70, 12.70]
mean_1k_5k       = [4.15, 5.10, 5.70, 12.70]
normalized_5k_10k = [5.55, 5.70, 5.65, 11.70]
mean_5k_10k       = [5.90, 6.45, 5.50, 11.80]

x = np.arange(len(environments))
width = 0.21

fig, ax = plt.subplots(figsize=(12, 6.5))

ax.bar(x - 1.5*width, normalized_1k_5k,
       width, color="#5B91ED", label="Normalized 1K–5K")

ax.bar(x - 0.5*width, mean_1k_5k,
       width, color="#63C481", label="Mean 1K–5K")

ax.bar(x + 0.5*width, normalized_5k_10k,
       width, color="#F08C61", label="Normalized 5K–10K")

ax.bar(x + 1.5*width, mean_5k_10k,
       width, color="#F5D463", label="Mean 5K–10K")

ax.set_title(
    "Figure 3. Average adaptation across friction shifts",
    fontsize=14, fontweight="bold", loc="left", pad=22
)

ax.set_xticks(x)
ax.set_xticklabels(environments, fontsize=11)

ax.set_ylim(0, 16.5)
ax.set_yticks([0, 4, 8, 12, 16])
ax.grid(axis="y", linestyle="--", alpha=0.2)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(axis="both", length=0)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False,
    fontsize=10
)

plt.tight_layout()

output = "figure3_average_adaptation.png"
plt.savefig(output, dpi=300, bbox_inches="tight")

print("Figure saved:", output)

plt.show()
