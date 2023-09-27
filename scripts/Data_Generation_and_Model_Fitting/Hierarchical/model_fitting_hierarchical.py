# %% Importing Packages
import numpy as np
import jax
from jax import random
import arviz as az

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.hierarchical.model_fitting_functions import generate_posterior_hierarchical

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Loading data
scenario_base = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d_hierarchical.npy',allow_pickle='TRUE').item()

# %% Fitting the model 1D
generate_posterior_hierarchical(scenario_base,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)

# %% Fitting the model 2D
generate_posterior_hierarchical(scenario_2d,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)

# %% Summary statistics from MCMC
az.summary(scenario_base['mcmc'].posterior,hdi_prob=0.95)
az.summary(scenario_2d['mcmc'].posterior,hdi_prob=0.95)

# %% Saving the output
# outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
# np.save(f'{outpath}scenario_base_hierarchical.npy', scenario_base) 
# np.save(f'{outpath}scenario_2d_hierarchical.npy', scenario_2d) 
