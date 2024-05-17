
# %% Importing packages
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from scipy.stats import gamma
import pandas as pd
import arviz as az
from src.non_hierarchical.plotting_functions import plot_underlying_data_1d
from src.non_hierarchical.plotting_functions import plot_underlying_data_1d_singleprocess
from src.non_hierarchical.plotting_functions import plot_predictions_1d

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

jax.config.update("jax_enable_x64", True)

# %% Loading data
inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"

scenario_base_joint = np.load(
    f"{inpath}scenario_base_joint.npy", allow_pickle="TRUE"
).item()
scenario_base_joint2 = np.load(
    f"{inpath}scenario_base_joint2.npy", allow_pickle="TRUE"
).item()
scenario_base_product = np.load(
    f"{inpath}scenario_base_product.npy", allow_pickle="TRUE"
).item()
scenario_base_correlated = np.load(
    f"{inpath}scenario_base_correlated.npy", allow_pickle="TRUE"
).item()

# %%
fig, axs = plt.subplots(4, 1, figsize=(17 * cm, 16.5 * cm), dpi=300)

scenarios = [scenario_base_joint,scenario_base_joint2,scenario_base_product,scenario_base_correlated]

X = jnp.arange(-20, 120, 0.05)

for scenario,ax in zip(scenarios,axs):

    plot_predictions_1d(
        X,
        scenario,
        "truth_posterior_predictive_realisations",
        ax,
        ms=20,
        color="tab:purple",
    )
    plot_predictions_1d(
        X,
        scenario,
        "bias_posterior_predictive_realisations",
        ax,
        ms=20,
        color="tab:red",
    )
    ax.plot(np.NaN, np.NaN, '-', color='none', label=' ')
    plot_underlying_data_1d(scenario, ax, ms=20)

handles, labels = axs[-1].get_legend_handles_labels()
labels = [
    "Unbiased Parameter Post.Pred. Exp. $E[\phi_Y(\mathbf{s}_z)]$",
    "Unbiased Parameter Post.Pred. Std.Dev. $\sigma[\phi_Y(\mathbf{s}_z)]$",
    "Bias Parameter Post.Pred. Exp. $E[\phi_B(\mathbf{s}_z)]$",
    "Bias Parameter Post.Pred. Std.Dev. $\sigma[\phi_B(\mathbf{s}_z)]$",
    "",
    "Unbiased Parameter Field $\phi_Y(\mathbf{s}^{\star})$",
    "Bias Parameter Field $\phi_B(\mathbf{s}^{\star})$",
    "Climate Model Parameter Field $\phi_Z(\mathbf{s}^{\star})$",
    "In Situ Parameter Observations $\phi_Y(\mathbf{s}_y)$",
    "Climate Model Parameter Observations $\phi_Z(\mathbf{s}_z)$",
]

for ax in axs:
    ax.set_xlabel("Location (s)",labelpad=0)
    ax.set_ylabel("Parameter Value")
    ax.get_legend().remove()

axs[2].set_ylabel("Parameter Value",labelpad=10)

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_xlabel("")

axs[0].annotate("(a) Scenario A",
                xy=(0.01, 0.92),
                xycoords="axes fraction",
                fontsize=6)
axs[1].annotate("(b) Scenario B",
                xy=(0.01, 0.92),
                xycoords="axes fraction",
                fontsize=6)
axs[2].annotate("(c) Scenario C",
                xy=(0.01, 0.92),
                xycoords="axes fraction",
                fontsize=6)
axs[3].annotate("(d) Scenario D",
                xy=(0.01, 0.92),
                xycoords="axes fraction",
                fontsize=6)

axs[-1].legend(
    handles,
    labels,
    fontsize=legend_fontsize,
    bbox_to_anchor=(0.5, -0.65),
    ncols=2,
    loc=10,
)

plt.tight_layout()
plt.show()
fig.savefig(f"{out_path}fig13.pdf", dpi=300, bbox_inches="tight")

# %%
# %% Table: Posterior distribution statistics
parameters = [
    "In-Situ Kernel Variance $v_{\phi_Y}$",
    "In-Situ Kernel Lengthscale $l_{\phi_Y}$",
    "In-Situ Mean Constant $m_{\phi_Y}$",
    "In-Situ Observation Noise $\sigma_{\phi_Y}$",
    "Bias Kernel Variance $v_{\phi_B}$ ",
    "Bias Kernel Lengthscale $l_{\phi_B}$",
    "Bias Mean Constant $m_{\phi_B}$",
]

desired_index_order = [
    "kern_var",
    "lengthscale",
    "mean",
    "onoise",
    "bkern_var",
    "blengthscale",
    "bmean",
]

columns = ["Exp.", "Std. Dev.", "95\% C.I. L.B.", "95\% C.I. U.B."]

desired_columns = ["mean", "sd", "hdi_2.5%", "hdi_97.5%"]

scenarios = [scenario_base_joint, scenario_base_joint2, scenario_base_product,scenario_base_correlated]

for scenario in scenarios:
    df = az.summary(scenario["mcmc"].posterior, hdi_prob=0.95)
    df = df.reindex(desired_index_order)
    df = df.set_index(np.array(parameters))
    df = df[desired_columns]
    df.columns = columns

    df["Distribution"] = "Po."
    df_conc = df.copy()
    # df_conc = pd.concat([df, df_singleprocess, df_prior])
    df_conc = df_conc.set_index([df_conc.index, "Distribution"])
    df_conc = df_conc.unstack()
    df_conc = df_conc.reindex(parameters)
    df_conc = df_conc.reindex(columns=["Po."], level="Distribution")

    df_conc = df_conc.swaplevel(axis="columns")
    df_conc = df_conc.reindex(columns=["Po."], level="Distribution")

    # df_conc.insert(0, "Specified Value", np.array(values)[[0, 1, 2, 3, 4, 5, 6]])
    df_conc = df_conc.astype(float)
    df_conc = df_conc.round(2)
    print(df_conc.to_latex(escape=False))