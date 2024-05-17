# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import arviz as az

plt.rcParams["lines.markersize"] = 3
plt.rcParams["lines.linewidth"] = 0.4
plt.rcParams.update({"font.size": 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.non_hierarchical.model_fitting_functions import generate_posterior

inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"

# %% Loading data
scenario_base_joint = np.load(f"{inpath}scenario_base_joint.npy", allow_pickle="TRUE").item()
scenario_base_joint2 = np.load(f"{inpath}scenario_base_joint2.npy", allow_pickle="TRUE").item()
scenario_base_product = np.load(f"{inpath}scenario_base_product.npy", allow_pickle="TRUE").item()
scenario_base_correlated = np.load(f"{inpath}scenario_base_correlated.npy", allow_pickle="TRUE").item()

# %% Fitting the model: Ample Data Scenario (~6mins)
generate_posterior(scenario_base_joint, rng_key, 1000, 2000, 2)
rng_key, rng_key_ = random.split(rng_key)
generate_posterior(scenario_base_joint2, rng_key, 1000, 2000, 2)
rng_key, rng_key_ = random.split(rng_key)
generate_posterior(scenario_base_product, rng_key, 1000, 2000, 2)
rng_key, rng_key_ = random.split(rng_key)
generate_posterior(scenario_base_correlated, rng_key, 1000, 2000, 2)
rng_key, rng_key_ = random.split(rng_key)

# %% Summary statistics from MCMC
az.summary(scenario_base_joint["mcmc"].posterior, hdi_prob=0.95)
az.summary(scenario_base_joint2["mcmc"].posterior, hdi_prob=0.95)
az.summary(scenario_base_product["mcmc"].posterior, hdi_prob=0.95)
az.summary(scenario_base_correlated["mcmc"].posterior, hdi_prob=0.95)


# az.summary(scenario_sparse_smooth['mcmc'].posterior,hdi_prob=0.95)
# az.summary(scenario_sparse_complex['mcmc'].posterior,hdi_prob=0.95)
# az.summary(scenario_2d['mcmc'].posterior,hdi_prob=0.95)

# %% Saving Dictionaries
np.save(f"{inpath}scenario_base_joint.npy", scenario_base_joint)
np.save(f"{inpath}scenario_base_joint2.npy", scenario_base_joint2)
np.save(f"{inpath}scenario_base_product.npy", scenario_base_product)
np.save(f"{inpath}scenario_base_correlated.npy", scenario_base_correlated)

# %%
