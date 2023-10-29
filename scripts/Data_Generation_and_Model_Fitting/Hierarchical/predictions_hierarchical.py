# %% Importing Packages
import numpy as np
import jax
from jax import random

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.hierarchical.prediction_functions import (
    generate_posterior_predictive_realisations_hierarchical_mean,
)
from src.hierarchical.prediction_functions import (
    generate_posterior_predictive_realisations_hierarchical_std,
)

inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"

# %% Loading data
scenario_base = np.load(
    f"{inpath}scenario_base_hierarchical.npy", allow_pickle="TRUE"
).item()
scenario_2d = np.load(
    f"{inpath}scenario_2d_hierarchical.npy", allow_pickle="TRUE"
).item()

# %% Generating posterior predictive realisations 1D
generate_posterior_predictive_realisations_hierarchical_mean(
    scenario_base["cx"], scenario_base, 400, 1
)
generate_posterior_predictive_realisations_hierarchical_std(
    scenario_base["cx"], scenario_base, 400, 1
)

# %% Generating posterior predictive realisations 2D
generate_posterior_predictive_realisations_hierarchical_mean(scenario_2d, 20, 20)
generate_posterior_predictive_realisations_hierarchical_std(scenario_2d, 20, 20)

# %% Saving the output
outpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"
np.save(f"{outpath}scenario_base_hierarchical.npy", scenario_base)
np.save(f"{outpath}scenario_2d_hierarchical.npy", scenario_2d)

# %%
