# %% Importing packages
import numpyro.distributions as dist
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from jax import random
import matplotlib.patches as patches
from src.non_hierarchical.plotting_functions import plotting_quantile_mapping

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1

legend_fontsize = 6
cm = 1 / 2.54
text_width = 17.68 * cm
page_width = 21.6 * cm

out_path = "/home/jez/Bias_Correction/results/Paper_Images/"

# %% Figure: PDFs and Quantile Mapping Example
fig, ax = plt.subplots(figsize=(13 * cm, 5.0 * cm), dpi=300)

xs = np.linspace(-1, 5, 100)
ys = norm.pdf(xs, 3, 0.5)
zs = norm.pdf(xs, 2, 1.0)
plot = ax.plot(xs, ys, lw=2, linestyle="dashed")
plt.hist(
    dist.Normal(3.0, 0.5).sample(rng_key, (10000,)),
    density=True,
    bins=100,
    color=plot[0].get_color(),
    alpha=0.7,
    label="$Y\sim \mathcal{N}(3,0.5)$",
)
plot2 = ax.plot(xs, zs, lw=2, linestyle="dashed")
plt.hist(
    dist.Normal(2.0, 1.0).sample(rng_key, (10000,)),
    density=True,
    bins=100,
    color=plot2[0].get_color(),
    alpha=0.7,
    label="$Z\sim \mathcal{N}(2,1)$",
)

y_percentile = np.percentile(dist.Normal(3.0, 0.5).sample(rng_key, (10000,)), 20)
z_percentile = np.percentile(dist.Normal(2.0, 1.0).sample(rng_key, (10000,)), 20)
plt.vlines(
    y_percentile, 0, 0.8, color="k", linestyle="dotted", label="20th Percentiles"
)
plt.vlines(z_percentile, 0, 0.8, color="k", linestyle="dotted")
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")
arrow = patches.FancyArrowPatch(
    (z_percentile, 0.8), (y_percentile, 0.8), **kw, connectionstyle="arc3,rad=-.2"
)
plt.gca().add_patch(arrow)

ax.set_ylabel("Probability Density")
ax.set_xlabel("Value Measured")
plt.legend(fontsize=legend_fontsize)
plt.show()
fig.savefig(f"{out_path}fig01.pdf", dpi=300, bbox_inches="tight")

# %% Figure: Bias Corrected Output Example
inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"
qm_scenario = np.load(
    f"{inpath}quantile_mapping_scenario.npy", allow_pickle="TRUE"
).item()
fig, ax = plt.subplots(1, 1, figsize=(12 * cm, 7.0 * cm), dpi=300)

plotting_quantile_mapping(ax, qm_scenario, ms=15)
ax.set_xticklabels([])
ax.set_xlabel("Time")
ax.set_ylabel("Value")
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    fontsize=legend_fontsize,
    bbox_to_anchor=(0.5, -0.05),
    ncols=3,
    loc=10,
)
plt.tight_layout()
plt.show()
fig.savefig(f"{out_path}fig02.pdf", dpi=300, bbox_inches="tight")


# %%
