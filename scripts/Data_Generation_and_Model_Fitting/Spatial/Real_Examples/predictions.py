# %% Importing Packages
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
import matplotlib.pyplot as plt

plt.rcParams["lines.markersize"] = 3
plt.rcParams["lines.linewidth"] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.non_hierarchical.prediction_functions import (
    generate_posterior_predictive_realisations,
)

inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"

# %% Loading data
scenario_base_joint = np.load(f"{inpath}scenario_base_joint.npy", allow_pickle="TRUE").item()
scenario_base_joint2 = np.load(f"{inpath}scenario_base_joint2.npy", allow_pickle="TRUE").item()
scenario_base_product = np.load(f"{inpath}scenario_base_product.npy", allow_pickle="TRUE").item()
scenario_base_correlated = np.load(f"{inpath}scenario_base_correlated.npy", allow_pickle="TRUE").item()

# %% Generating posterior predictive realisations

X = jnp.arange(-20, 120, 0.05)

generate_posterior_predictive_realisations(
    X, scenario_base_joint, 200, 1
)
rng_key, rng_key_ = random.split(rng_key)
generate_posterior_predictive_realisations(
    X, scenario_base_joint2, 200, 1
)
rng_key, rng_key_ = random.split(rng_key)
generate_posterior_predictive_realisations(
    X, scenario_base_product, 200, 1
)

rng_key, rng_key_ = random.split(rng_key)
generate_posterior_predictive_realisations(
    X, scenario_base_correlated, 200, 1
)

# %% Saving Dictionaries
np.save(f'{inpath}scenario_base_joint.npy', scenario_base_joint)
np.save(f'{inpath}scenario_base_joint2.npy', scenario_base_joint2)
np.save(f'{inpath}scenario_base_product.npy', scenario_base_product)
np.save(f'{inpath}scenario_base_correlated.npy', scenario_base_correlated)

# %%
scenario_base_product['truth_posterior_predictive_realisations']