# %% Importing Packages
import numpy as np
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
scenario_base = np.load(f"{inpath}scenario_base.npy", allow_pickle="TRUE").item()
scenario_ampledata = np.load(
    f"{inpath}scenario_ampledata.npy", allow_pickle="TRUE"
).item()
scenario_sparse_smooth = np.load(
    f"{inpath}scenario_sparse_smooth.npy", allow_pickle="TRUE"
).item()
scenario_sparse_complex = np.load(
    f"{inpath}scenario_sparse_complex.npy", allow_pickle="TRUE"
).item()
scenario_2d = np.load(f"{inpath}scenario_2d.npy", allow_pickle="TRUE").item()

# %% Generating posterior predictive realisations
# generate_posterior_predictive_realisations(scenario_base['cx'],scenario_base,20,20)
generate_posterior_predictive_realisations(
    scenario_ampledata["cx"], scenario_ampledata, 20, 20
)
generate_posterior_predictive_realisations(
    scenario_sparse_smooth["cx"], scenario_sparse_smooth, 20, 20
)
generate_posterior_predictive_realisations(
    scenario_sparse_complex["cx"], scenario_sparse_complex, 20, 20
)
# generate_posterior_predictive_realisations(scenario_2d,20,20)

# %% Saving Dictionaries
# np.save(f'{inpath}scenario_base.npy', scenario_base)
# np.save(f'{inpath}scenario_ampledata.npy', scenario_ampledata)
# np.save(f'{inpath}scenario_sparse_smooth.npy', scenario_sparse_smooth)
# np.save(f'{inpath}scenario_sparse_complex.npy', scenario_sparse_complex)
# np.save(f'{inpath}scenario_2d.npy', scenario_2d)
