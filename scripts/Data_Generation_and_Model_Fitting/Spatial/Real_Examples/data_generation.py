# %% Importing Packages
import numpy as np
import numpyro.distributions as dist
from numpy.random import RandomState
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tinygp import kernels, GaussianProcess


plt.rcParams["lines.markersize"] = 3
plt.rcParams["lines.linewidth"] = 0.4
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 1

legend_fontsize = 6
cm = 1 / 2.54
text_width = 17.68 * cm
page_width = 21.6 * cm

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.non_hierarchical.data_generation_functions import generate_underlying_data
from src.non_hierarchical.plotting_functions import plot_underlying_data_1d
from src.non_hierarchical.plotting_functions import plot_underlying_data_2d

outpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"

# %% Specifying parameters for different scenarios

min_x, max_x = 0, 100
X = jnp.arange(min_x, max_x, 0.1)
X_left = np.split(X,2)[0]
X_right = np.split(X,2)[1]

min_x_left, max_x_left = 0, 50
min_x_right, max_x_right = 50, 100


# Scenario Base: Relatively Smooth Bias and Medium Number of Observations

scenario_base = {
    "onoise": 1e-1,
    "bnoise": 1e-1,
    "cnoise": 1e-3,
    "jitter": 1e-10,
    "t_variance": 1.0,
    "t_lengthscale": 3.0,
    "t_mean": 1.0,
    "b_variance": 1.0,
    "b_lengthscale": 10.0,
    "b_mean": -1.0,
    "ox": RandomState(0).uniform(low=min_x, high=max_x, size=(40,)),
    "cx": np.linspace(min_x, max_x, 80),
    "X": X,
    "t_variance_prior": dist.Gamma(1.0, 1.5),
    "t_lengthscale_prior": dist.Gamma(3.0, 0.2),
    "t_mean_prior": dist.Normal(0.0, 2.0),
    "b_variance_prior": dist.Gamma(1.0, 0.5),
    "b_lengthscale_prior": dist.Gamma(3.0, 0.2),
    "b_mean_prior": dist.Normal(0.0, 2.0),
    "onoise_prior": dist.Uniform(0.0, 0.5),
    "cnoise_prior": dist.Uniform(0.0, 0.5),
    "nx": X[::5],
}

scenario_base_product = scenario_base.copy()

scenario_base_correlated = scenario_base.copy()

scenario_base_left = scenario_base.copy()
scenario_base_left.update(
    {
        "t_lengthscale": 5.0,
        "X":X_left,
        "ox": RandomState(0).uniform(low=min_x_left, high=max_x_left, size=(20,)),
        "cx": np.linspace(min_x_left, max_x_left, 40),
    }
)

scenario_base_right = scenario_base.copy()
scenario_base_right.update(
    {
        "t_lengthscale": 1.0,
        "X":X_right,
        "ox": RandomState(0).uniform(low=min_x_right, high=max_x_right, size=(20,)),
        "cx": np.linspace(min_x_right, max_x_right, 40),
    }
)

scenario_base_left2 = scenario_base.copy()
scenario_base_left2.update(
    {
        "t_lengthscale": 5.0,
        "X":X_left,
        "ox": RandomState(0).uniform(low=min_x_left, high=max_x_left, size=(20,)),
        "cx": np.linspace(min_x_left, max_x_left, 40),
    }
)

scenario_base_right2 = scenario_base.copy()
scenario_base_right2.update(
    {
        "t_lengthscale": 1.0,
        "b_lengthscale": 2.0,
        "X":X_right,
        "ox": RandomState(0).uniform(low=min_x_right, high=max_x_right, size=(20,)),
        "cx": np.linspace(min_x_right, max_x_right, 40),
    }
)


# %% Data Generation Functions

def generate_underlying_data_parts(scenario, rng_key):
    rng_key, rng_key_ = random.split(rng_key)

    # middle = jnp.array(scenario["X"][-1]/2)
    middle = X[np.array([round(len(X)/2)-1,round(len(X)/2)+1])]

    GP_T = GaussianProcess(
        scenario["t_variance"] * kernels.ExpSquared(scenario["t_lengthscale"]),
        middle,
        diag=scenario["jitter"],
        mean=scenario["t_mean"],
    )
    GP_B = GaussianProcess(
        scenario["b_variance"] * kernels.ExpSquared(scenario["b_lengthscale"]),
        middle,
        diag=scenario["jitter"],
        mean=scenario["b_mean"],
    )

    GP_T = GP_T.condition(scenario["t_mean"], scenario["X"]).gp
    GP_B = GP_B.condition(scenario["b_mean"], scenario["X"]).gp

    scenario["T"] = GP_T.sample(rng_key)
    scenario["B"] = GP_B.sample(rng_key_)

    # scenario["T"] = GP_T.condition(scenario["t_mean"], scenario["X"]).gp.sample(rng_key)
    # scenario["B"] = GP_B.condition(scenario["b_mean"], scenario["X"]).gp.sample(rng_key)
    scenario["C"] = scenario["T"] + scenario["B"]

    rng_key, rng_key_ = random.split(rng_key)
    scenario["odata"] = GP_T.condition(scenario["T"], scenario["ox"]).gp.sample(rng_key)
    odata_noise = dist.Normal(0.0, scenario["onoise"]).sample(
        rng_key_, scenario["odata"].shape
    )
    scenario["odata"] = scenario["odata"] + odata_noise

    rng_key, rng_key_ = random.split(rng_key)
    scenario["cdata_o"] = GP_T.condition(scenario["T"], scenario["cx"]).gp.sample(
        rng_key
    )
    scenario["cdata_b"] = GP_B.condition(scenario["B"], scenario["cx"]).gp.sample(
        rng_key_
    )
    scenario["cdata"] = scenario["cdata_o"] + scenario["cdata_b"]
    rng_key, rng_key_ = random.split(rng_key)
    cdata_noise = dist.Normal(0.0, scenario["cnoise"]).sample(
        rng_key, scenario["cdata"].shape
    )
    scenario["cdata"] = scenario["cdata"] + cdata_noise


def generate_underlying_data_product(scenario, rng_key):
    rng_key, rng_key_ = random.split(rng_key)

    tkernel = (scenario["t_variance"] * kernels.ExpSquared(scenario["t_lengthscale"]) + 
               0.2 * scenario["t_variance"] * kernels.ExpSquared(scenario["t_lengthscale"]*0.2))
    # bkernel = kernels.ExpSquared(scenario["b_lengthscale"]) * kernels.ExpSquared(scenario["b_lengthscale"]*0.2)

    GP_T = GaussianProcess(
        tkernel,
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["t_mean"],
    )
    GP_B = GaussianProcess(
        scenario["b_variance"] * kernels.ExpSquared(scenario["b_lengthscale"]),
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["b_mean"],
    )

    scenario["T"] = GP_T.sample(rng_key)
    scenario["B"] = GP_B.sample(rng_key_)
    scenario["C"] = scenario["T"] + scenario["B"]

    rng_key, rng_key_ = random.split(rng_key)
    scenario["odata"] = GP_T.condition(scenario["T"], scenario["ox"]).gp.sample(rng_key)
    odata_noise = dist.Normal(0.0, scenario["onoise"]).sample(
        rng_key_, scenario["odata"].shape
    )
    scenario["odata"] = scenario["odata"] + odata_noise

    rng_key, rng_key_ = random.split(rng_key)
    scenario["cdata_o"] = GP_T.condition(scenario["T"], scenario["cx"]).gp.sample(
        rng_key
    )
    scenario["cdata_b"] = GP_B.condition(scenario["B"], scenario["cx"]).gp.sample(
        rng_key_
    )
    scenario["cdata"] = scenario["cdata_o"] + scenario["cdata_b"]
    rng_key, rng_key_ = random.split(rng_key)
    cdata_noise = dist.Normal(0.0, scenario["cnoise"]).sample(
        rng_key, scenario["cdata"].shape
    )
    scenario["cdata"] = scenario["cdata"] + cdata_noise

def generate_underlying_data_correlated(scenario, rng_key):
    rng_key, rng_key_ = random.split(rng_key)

    # tkernel = (scenario["t_variance"] * kernels.ExpSquared(scenario["t_lengthscale"]) + 
    #            0.2 * scenario["t_variance"] * kernels.ExpSquared(scenario["t_lengthscale"]*0.2))
    # bkernel = kernels.ExpSquared(scenario["b_lengthscale"]) * kernels.ExpSquared(scenario["b_lengthscale"]*0.2)

    GP_T = GaussianProcess(
        scenario["t_variance"] * kernels.ExpSquared(scenario["t_lengthscale"]),
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["t_mean"],
    )
    GP_B_Ind = GaussianProcess(
        scenario["b_variance"] * kernels.ExpSquared(scenario["b_lengthscale"]),
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["b_mean"],
    )

    scenario["T"] = GP_T.sample(rng_key)
    scenario["B_Ind"] = GP_B_Ind.sample(rng_key_)

    scenario["B"] = 0.3*scenario["T"] + scenario["B_Ind"]
    scenario["C"] = scenario["T"] + scenario["B"]

    rng_key, rng_key_ = random.split(rng_key)
    scenario["odata"] = GP_T.condition(scenario["T"], scenario["ox"]).gp.sample(rng_key)
    odata_noise = dist.Normal(0.0, scenario["onoise"]).sample(
        rng_key_, scenario["odata"].shape
    )
    scenario["odata"] = scenario["odata"] + odata_noise

    rng_key, rng_key_ = random.split(rng_key)
    scenario["cdata_o"] = GP_T.condition(scenario["T"], scenario["cx"]).gp.sample(
        rng_key
    )
    scenario["cdata_b"] = GP_B_Ind.condition(scenario["B_Ind"], scenario["cx"]).gp.sample(
        rng_key_
    )
    scenario["cdata"] = scenario["cdata_o"] + 0.3*scenario["cdata_o"] + scenario["cdata_b"]
    rng_key, rng_key_ = random.split(rng_key)
    cdata_noise = dist.Normal(0.0, scenario["cnoise"]).sample(
        rng_key, scenario["cdata"].shape
    )
    scenario["cdata"] = scenario["cdata"] + cdata_noise

# %% Generating the underlying simulated data

rng_key = random.PRNGKey(0)
generate_underlying_data_parts(scenario_base_left, rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data_parts(scenario_base_right, rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data_parts(scenario_base_left2, rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data_parts(scenario_base_right2, rng_key)
rng_key, rng_key_ = random.split(rng_key)

generate_underlying_data_product(scenario_base_product, rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data_correlated(scenario_base_correlated, rng_key)
rng_key, rng_key_ = random.split(rng_key)

# %% Joining Sides

scenario_base_joint = scenario_base_left.copy()
scenario_base_joint.update(
    {   
        "X":jnp.append(X_left,X_right),
        "T":jnp.append(scenario_base_left['T'],scenario_base_right['T']),
        "B":jnp.append(scenario_base_left['B'],scenario_base_right['B']),
        "C":jnp.append(scenario_base_left['C'],scenario_base_right['C']),
        "ox":jnp.append(scenario_base_left['ox'],scenario_base_right['ox']),
        "odata":jnp.append(scenario_base_left['odata'],scenario_base_right['odata']),
        "cx":jnp.append(scenario_base_left['cx'],scenario_base_right['cx']),
        "cdata":jnp.append(scenario_base_left['cdata'],scenario_base_right['cdata']),
    }
)

scenario_base_joint2 = scenario_base_left2.copy()
scenario_base_joint2.update(
    {   
        "X":jnp.append(X_left,X_right),
        "T":jnp.append(scenario_base_left2['T'],scenario_base_right2['T']),
        "B":jnp.append(scenario_base_left2['B'],scenario_base_right2['B']),
        "C":jnp.append(scenario_base_left2['C'],scenario_base_right2['C']),
        "ox":jnp.append(scenario_base_left2['ox'],scenario_base_right2['ox']),
        "odata":jnp.append(scenario_base_left2['odata'],scenario_base_right2['odata']),
        "cx":jnp.append(scenario_base_left2['cx'],scenario_base_right2['cx']),
        "cdata":jnp.append(scenario_base_left2['cdata'],scenario_base_right2['cdata']),
    }
)

# %% Plot showing generated 1D data

fig, axs = plt.subplots(4, 1, figsize=(17 * cm, 16.5 * cm), dpi=300)

plot_underlying_data_1d(scenario_base_joint, axs[0], ms=20)
plot_underlying_data_1d(scenario_base_joint2, axs[1], ms=20)
plot_underlying_data_1d(scenario_base_product, axs[2], ms=20)
plot_underlying_data_1d(scenario_base_correlated, axs[3], ms=20)


handles, labels = axs[0].get_legend_handles_labels()
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
    ax.set_xlabel("Location (s)",labelpad=-5)
    ax.set_ylabel("Parameter Value")
    ax.get_legend().remove()

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_xlabel("")

axs[-1].legend(
    handles,
    labels,
    fontsize=legend_fontsize,
    bbox_to_anchor=(0.5, -0.45),
    ncols=2,
    loc=10,
)

plt.tight_layout()
plt.show()


# %% Saving Dictionaries

np.save(f'{outpath}scenario_base_joint.npy', scenario_base_joint)
np.save(f'{outpath}scenario_base_joint2.npy', scenario_base_joint2)
np.save(f'{outpath}scenario_base_product.npy', scenario_base_product)
np.save(f'{outpath}scenario_base_correlated.npy', scenario_base_correlated)

# np.save(f'{outpath}scenario_ampledata.npy', scenario_ampledata)
# np.save(f'{outpath}scenario_sparse_smooth.npy', scenario_sparse_smooth)
# np.save(f'{outpath}scenario_sparse_complex.npy', scenario_sparse_complex)
# np.save(f'{outpath}scenario_2d.npy', scenario_2d)

# %%
