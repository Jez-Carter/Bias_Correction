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
from src.non_hierarchical.plotting_functions import plot_underlying_data_1d_lima
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
scenario_ampledata = np.load(
    f"{inpath}scenario_ampledata.npy", allow_pickle="TRUE"
).item()
scenario_sparse_smooth = np.load(
    f"{inpath}scenario_sparse_smooth.npy", allow_pickle="TRUE"
).item()
scenario_sparse_complex = np.load(
    f"{inpath}scenario_sparse_complex.npy", allow_pickle="TRUE"
).item()
scenario_ampledata_lima = np.load(
    f"{inpath}scenario_ampledata_lima.npy", allow_pickle="TRUE"
).item()
scenario_sparse_smooth_lima = np.load(
    f"{inpath}scenario_sparse_smooth_lima.npy", allow_pickle="TRUE"
).item()
scenario_sparse_complex_lima = np.load(
    f"{inpath}scenario_sparse_complex_lima.npy", allow_pickle="TRUE"
).item()

# %% Table: Parameters used to generate the data
parameters = [
    "In-Situ Kernel Variance ($v_{\phi_Y}$)",
    "In-Situ Kernel Lengthscale ($l_{\phi_Y}$)",
    "In-Situ Mean Constant ($m_{\phi_Y}$)",
    "In-Situ Observation Noise ($\sigma_{\phi_Y}$)",
    "Bias Kernel Variance ($v_{\phi_B}$)",
    "Bias Kernel Lengthscale ($l_{\phi_B}$)",
    "Bias Mean Constant ($m_{\phi_B}$)",
    "\# In-Situ Observations",
    "\# Climate Model Predictions",
]
parameters_shorthand = [
    "t_variance",
    "t_lengthscale",
    "t_mean",
    "onoise",
    "b_variance",
    "b_lengthscale",
    "b_mean",
    "ox",
    "cx",
]

values_list = []
for scenario in [scenario_ampledata, scenario_sparse_smooth, scenario_sparse_complex]:
    values = []
    for i in parameters_shorthand:
        if i == "ox" or i == "cx":
            value = len(scenario[i])
        else:
            value = scenario[i]
        values.append(value)
    values_list.append(values)
values = np.array(values_list)

scenarios = ["Scenario 1", "Scenario 2", "Scenario 3"]
df = pd.DataFrame(data=values.T, index=parameters, columns=scenarios)
caption = "A table showing the parameters used to generate the realisations and data for 3 scenarios."
print(df.to_latex(escape=False, caption=caption))

# %% Figure: Visualising the data
fig, axs = plt.subplots(3, 1, figsize=(12 * cm, 12.0 * cm), dpi=300)
plot_underlying_data_1d(scenario_ampledata, axs[0], ms=20)
plot_underlying_data_1d(scenario_sparse_smooth, axs[1], ms=20)
plot_underlying_data_1d(scenario_sparse_complex, axs[2], ms=20)

for ax in axs:
    ax.set_xlabel("s")
    ax.set_ylabel("Value")
    ax.get_legend().remove()

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_xlabel("")

axs[0].annotate("(a) Scenario 1", xy=(0.01, 0.92), xycoords="axes fraction")
axs[1].annotate("(b) Scenario 2", xy=(0.01, 0.92), xycoords="axes fraction")
axs[2].annotate("(c) Scenario 3", xy=(0.01, 0.92), xycoords="axes fraction")

handles, labels = axs[0].get_legend_handles_labels()
labels = [
    "$\phi_Y$",
    "$\phi_B$",
    "$\phi_Z$",
    "In-Situ Observations",
    "Climate Model Output",
]
fig.legend(
    handles, labels, fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0), ncols=5, loc=10
)
plt.tight_layout()
plt.show()
fig.savefig(f"{out_path}fig05.pdf", dpi=300, bbox_inches="tight")


# %% Table: Prior and posterior distribution statistics
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

prior_keys = [
    "t_variance_prior",
    "t_lengthscale_prior",
    "t_mean_prior",
    "b_variance_prior",
    "b_lengthscale_prior",
    "b_mean_prior",
    "onoise_prior",
]

scenarios = [scenario_ampledata, scenario_sparse_smooth, scenario_sparse_complex]
scenarios_lima = [
    scenario_ampledata_lima,
    scenario_sparse_smooth_lima,
    scenario_sparse_complex_lima,
]

for scenario, scenario_lima, values in zip(scenarios, scenarios_lima, values_list):
    df = az.summary(scenario["mcmc"].posterior, hdi_prob=0.95)
    df = df.reindex(desired_index_order)
    df = df.set_index(np.array(parameters))
    df = df[desired_columns]
    df.columns = columns

    df_lima = az.summary(scenario_lima["mcmc_lima"].posterior, hdi_prob=0.95)
    df_lima = df_lima.reindex(desired_index_order)
    df_lima = df_lima.set_index(np.array(parameters))
    df_lima = df_lima[desired_columns]
    df_lima.columns = columns

    expectations = []
    standard_deviations = []
    LB_CIs = []
    UB_CIs = []
    for key in prior_keys:
        distribution = scenario[key]
        expectation = distribution.mean
        variance = distribution.variance
        standard_deviation = jnp.sqrt(variance)
        if distribution.reparametrized_params == ["concentration", "rate"]:
            LB_CI = gamma.ppf(
                0.025, distribution.concentration, loc=0, scale=1 / distribution.rate
            )
            UB_CI = gamma.ppf(
                0.975, distribution.concentration, loc=0, scale=1 / distribution.rate
            )
        else:
            LB_CI = distribution.icdf(0.025)
            UB_CI = distribution.icdf(0.975)
        expectations.append(expectation)
        standard_deviations.append(standard_deviation)
        LB_CIs.append(LB_CI)
        UB_CIs.append(UB_CI)
    d = {
        columns[0]: expectations,
        columns[1]: standard_deviations,
        columns[2]: LB_CIs,
        columns[3]: UB_CIs,
    }
    df_prior = pd.DataFrame(data=d, index=parameters)

    df["Distribution"] = "Po.2"
    df_lima["Distribution"] = "Po.1"
    df_prior["Distribution"] = "Pr."

    df_conc = pd.concat([df, df_lima, df_prior])
    df_conc = df_conc.set_index([df_conc.index, "Distribution"])
    df_conc = df_conc.unstack()
    df_conc = df_conc.reindex(parameters)
    df_conc = df_conc.reindex(columns=["Pr.", "Po.2", "Po.1"], level="Distribution")

    df_conc = df_conc.swaplevel(axis="columns")
    df_conc = df_conc.reindex(columns=["Pr.", "Po.2", "Po.1"], level="Distribution")

    df_conc.insert(0, "Specified Value", np.array(values)[[0, 1, 2, 3, 4, 5, 6]])
    df_conc = df_conc.astype(float)
    df_conc = df_conc.round(2)
    print(df_conc.to_latex(escape=False))

# %% Figure: Visualising posterior predictives
fig, axs = plt.subplots(3, 2, figsize=(16 * cm, 15.0 * cm), dpi=300)
scenarios = [
    scenario_ampledata_lima,
    scenario_sparse_smooth_lima,
    scenario_sparse_complex_lima,
]
for scenario, ax in zip(scenarios, axs[:, 0]):
    plot_predictions_1d(
        scenario["cx"],
        scenario,
        "truth_posterior_predictive_realisations_lima",
        ax,
        ms=20,
        color="tab:purple",
    )
    plot_underlying_data_1d_lima(scenario, ax, ms=20)

scenarios = [scenario_ampledata, scenario_sparse_smooth, scenario_sparse_complex]
for scenario, ax in zip(scenarios, axs[:, 1]):
    plot_predictions_1d(
        scenario["cx"],
        scenario,
        "truth_posterior_predictive_realisations",
        ax,
        ms=20,
        color="tab:purple",
    )
    plot_predictions_1d(
        scenario["cx"],
        scenario,
        "bias_posterior_predictive_realisations",
        ax,
        ms=20,
        color="tab:red",
    )
    plot_underlying_data_1d(scenario, ax, ms=20)

for ax in axs.ravel():
    ax.set_xlabel("s")
    ax.set_ylabel("Value")
    ax.get_legend().remove()

for ax in axs[:2, :].ravel():
    ax.set_xticklabels([])
    ax.set_xlabel("")

for ax in axs[:, 1].ravel():
    ax.set_yticklabels([])
    ax.set_ylabel("")

axs[0, 0].set_ylim(axs[0, 1].get_ylim())
axs[1, 0].set_ylim(axs[1, 1].get_ylim())
axs[2, 0].set_ylim(axs[2, 1].get_ylim())

axs[0, 0].annotate(
    "(a) Scenario 1 (1 Process)", xy=(0.01, 1.01), xycoords="axes fraction"
)
axs[0, 1].annotate(
    "(b) Scenario 1 (2 Process)", xy=(0.01, 1.01), xycoords="axes fraction"
)
axs[1, 0].annotate(
    "(c) Scenario 2 (1 Process)", xy=(0.01, 1.01), xycoords="axes fraction"
)
axs[1, 1].annotate(
    "(d) Scenario 2 (2 Process)", xy=(0.01, 1.01), xycoords="axes fraction"
)
axs[2, 0].annotate(
    "(e) Scenario 3 (1 Process)", xy=(0.01, 1.01), xycoords="axes fraction"
)
axs[2, 1].annotate(
    "(f) Scenario 3 (2 Process)", xy=(0.01, 1.01), xycoords="axes fraction"
)

handles, labels = axs[0, 1].get_legend_handles_labels()
labels = [
    "$\phi_Y$ Expectation",
    "$\phi_Y$ 1$\sigma$ Uncertainty",
    "$\phi_B$ Expectation",
    "$\phi_B$ 1$\sigma$ Uncertainty",
    "$\phi_Y$ Underlying Field",
    "$\phi_B$ Underlying Field",
    "$\phi_Z$ Underlying Field",
    "In-Situ Observations",
    "Climate Model Output",
]
fig.legend(
    handles,
    labels,
    fontsize=legend_fontsize,
    bbox_to_anchor=(0.5, -0.01),
    ncols=5,
    loc=10,
)
# plt.subplots_adjust(hspace=0.01)
plt.tight_layout()
plt.show()
fig.savefig(f"{out_path}fig06.pdf", dpi=300, bbox_inches="tight")

# %% Table of R2 Scores
scenario_names = ["Scenario 1", "Scenario 2", "Scenario 3"]
scenario_pairs = [
    [scenario_ampledata, scenario_ampledata_lima],
    [scenario_sparse_smooth, scenario_sparse_smooth_lima],
    [scenario_sparse_complex, scenario_sparse_complex_lima],
]

r2_scores_2process_exp = []
r2_scores_1process_exp = []
r2_scores_2process_std = []
r2_scores_1process_std = []

for scenario_pair, scenario_name in zip(scenario_pairs, scenario_names):
    y_true = scenario_pair[0]["cdata_o"]
    y_pred = scenario_pair[0]["truth_posterior_predictive_realisations"]
    y_pred_lima = scenario_pair[1]["truth_posterior_predictive_realisations_lima"]

    r2_2process = az.r2_score(y_true, y_pred)
    r2_scores_2process_exp.append(r2_2process[0])
    r2_scores_2process_std.append(r2_2process[1])

    r2_1process = az.r2_score(y_true, y_pred_lima)
    r2_scores_1process_exp.append(r2_1process[0])
    r2_scores_1process_std.append(r2_1process[1])

r2_df = pd.DataFrame(
    {
        "2process_exp": r2_scores_2process_exp,
        "2process_std": r2_scores_2process_std,
        "1process_exp": r2_scores_1process_exp,
        "1process_std": r2_scores_1process_std,
    }
)
r2_df.index = scenario_names
r2_df = r2_df.round(2)
print(r2_df.to_latex(escape=False))

# %%
