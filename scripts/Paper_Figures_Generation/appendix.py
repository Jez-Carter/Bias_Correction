# %% Importing packages
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
import matplotlib.gridspec as gridspec
from src.non_hierarchical.plotting_functions import plot_priors
from src.non_hierarchical.plotting_functions import plot_posteriors

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1

legend_fontsize = 6
cm = 1 / 2.54  # centimeters in inches
text_width = 17.68 * cm
page_width = 21.6 * cm

out_path = "/home/jez/Bias_Correction/results/Paper_Images/"

jax.config.update("jax_enable_x64", True)

# %% Figure: 1D Non-hierarchical case, visualising prior and posterior distributions
prior_keys = [
    "t_variance_prior",
    "t_lengthscale_prior",
    "t_mean_prior",
    "b_variance_prior",
    "b_lengthscale_prior",
    "b_mean_prior",
    "onoise_prior",
]
posterior_keys = [
    "kern_var",
    "lengthscale",
    "mean",
    "bkern_var",
    "blengthscale",
    "bmean",
    "onoise",
]
titles = [
    "(a) Kernel Variance $v_{\phi_Y}$",
    "(b) Kernel Lengthscale $l_{\phi_Y}$",
    "(c) Mean Constant $m_{\phi_Y}$",
    "(d) Kernel Variance $v_{\phi_B}$",
    "(e) Kernel Lengthscale $l_{\phi_B}$",
    "(f) Mean Constant $m_{\phi_B}$",
    "(g) Noise $\sigma_{\phi_Y}$",
]

inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"
scenario_ampledata = np.load(
    f"{inpath}scenario_ampledata.npy", allow_pickle="TRUE"
).item()
scenario = scenario_ampledata

fig = plt.figure(figsize=(17 * cm, 16 * cm), dpi=300)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.8)
gs.update(hspace=0.3)

axs = [
    plt.subplot(gs[0, :2]),
    plt.subplot(gs[0, 2:4]),
    plt.subplot(gs[0, 4:6]),
    plt.subplot(gs[1, :2]),
    plt.subplot(gs[1, 2:4]),
    plt.subplot(gs[1, 4:6]),
    plt.subplot(gs[2, 2:4]),
]

rng_key = random.PRNGKey(5)
plot_priors(scenario, prior_keys, axs, rng_key, 0.75)
plot_posteriors(scenario["mcmc"].posterior, posterior_keys, axs)
singleprocess_posterior_keys = [posterior_keys[i] for i in [0, 1, 2]]
singleprocess_axs = [axs[i] for i in [0, 1, 2]]
# plot_posteriors(scenario_singleprocess['mcmc_singleprocess'].posterior,singleprocess_posterior_keys,singleprocess_axs)

for ax, title in zip(axs, titles):
    ax.set_title(title, pad=3, loc="left", fontsize=8)

axs[-1].legend(
    fontsize=legend_fontsize, labels=["Specified", "Prior", "Posterior"], loc=[1.1, 0.7]
)
axs[1].set_ylim(0, 0.5)
axs[1].set_xlim(0, 12)

plt.tight_layout()
plt.show()
fig.savefig(f"{out_path}fig11.pdf", dpi=300, bbox_inches="tight")

# %% Figure: 1D hierarchical case, visualising prior and posterior distributions

prior_keys = [
    "MEAN_T_variance_prior",
    "MEAN_T_lengthscale_prior",
    "MEAN_T_mean_prior",
    "LOGVAR_T_variance_prior",
    "LOGVAR_T_lengthscale_prior",
    "LOGVAR_T_mean_prior",
    "MEAN_B_variance_prior",
    "MEAN_B_lengthscale_prior",
    "MEAN_B_mean_prior",
    "LOGVAR_B_variance_prior",
    "LOGVAR_B_lengthscale_prior",
    "LOGVAR_B_mean_prior",
]
posterior_keys = [
    "mt_kern_var",
    "mt_lengthscale",
    "mt_mean",
    "lvt_kern_var",
    "lvt_lengthscale",
    "lvt_mean",
    "mb_kern_var",
    "mb_lengthscale",
    "mb_mean",
    "lvb_kern_var",
    "lvb_lengthscale",
    "lvb_mean",
]
titles = [
    "(a) $v_{\mu_Y}$",
    "(b) $l_{\mu_Y}$",
    "(c) $m_{\mu_Y}$",
    r"(d) $v_{\tilde{\sigma}^2_Y}$",
    r"(e) $l_{\tilde{\sigma}^2_Y}$",
    r"(f) $m_{\tilde{\sigma}^2_Y}$",
    "(g) $v_{\mu_B}$",
    "(h) $l_{\mu_B}$",
    "(i) $m_{\mu_B}$",
    r"(j) $v_{\tilde{\sigma}^2_B}$",
    r"(k) $l_{\tilde{\sigma}^2_B}$",
    r"(l) $m_{\tilde{\sigma}^2_B}$",
]

inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"
scenario_base_hierarchical = np.load(
    f"{inpath}scenario_base_hierarchical.npy", allow_pickle="TRUE"
).item()
scenario = scenario_base_hierarchical

fig = plt.figure(figsize=(17 * cm, 16 * cm), dpi=300)
gs = gridspec.GridSpec(4, 3)
gs.update(wspace=0.2)
gs.update(hspace=0.2)

axs = [
    plt.subplot(gs[0, 0]),
    plt.subplot(gs[0, 1]),
    plt.subplot(gs[0, 2]),
    plt.subplot(gs[1, 0]),
    plt.subplot(gs[1, 1]),
    plt.subplot(gs[1, 2]),
    plt.subplot(gs[2, 0]),
    plt.subplot(gs[2, 1]),
    plt.subplot(gs[2, 2]),
    plt.subplot(gs[3, 0]),
    plt.subplot(gs[3, 1]),
    plt.subplot(gs[3, 2]),
]

rng_key = random.PRNGKey(5)
plot_priors(scenario, prior_keys, axs, rng_key, 0.5)
plot_posteriors(scenario["mcmc"].posterior, posterior_keys, axs)

for ax, title in zip(axs, titles):
    ax.set_title(title, pad=3, loc="left", fontsize=8)

for ax in axs[::3]:
    ax.set_ylabel("Prob. Density")

y_axis_limits = [2, 0.5, 1.5]
y_axis_ticks = [[0, 2], [0, 0.5], [0, 1.5]]
y_axis_ticks_minor = [[0.5, 1, 1.5], [0.1, 0.2, 0.3, 0.4], [0.5, 1.0]]

for i in [0, 1, 2]:
    for ax in axs[i::3]:
        ax.set_yticks(y_axis_ticks[i])
        ax.set_yticks(y_axis_ticks_minor[i], minor=True)
        ax.set_ylim([0, y_axis_limits[i]])

for ax in axs[:-3]:
    ax.set_xticklabels([])
    ax.set_xlabel("")

for ax in axs[-3:]:
    ax.set_xlabel("Value")

labels = ["Specified", "Prior", "Posterior"]
fig.legend(
    labels, fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.025), ncols=7, loc=10
)

plt.tight_layout()
plt.show()
fig.savefig(f"{out_path}fig12.pdf", dpi=300, bbox_inches="tight")
# %%
