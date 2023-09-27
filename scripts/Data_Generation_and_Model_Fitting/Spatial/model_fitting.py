# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import arviz as az

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams.update({'font.size': 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.non_hierarchical.model_fitting_functions import generate_posterior

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Loading data
scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d.npy',allow_pickle='TRUE').item()

# %% Fitting the model: Ample Data Scenario (~6mins)
generate_posterior(scenario_ampledata,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)

# %% Fitting the model: Sparse Data Smooth Bias Scenario
generate_posterior(scenario_sparse_smooth,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)

# %% Fitting the model: Sparse Data Complex Bias Scenario
generate_posterior(scenario_sparse_complex,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)

# %% Fitting the model: 2D Scenario
generate_posterior(scenario_2d,rng_key,1000,2000,2)

# %% Summary statistics from MCMC
az.summary(scenario_ampledata['mcmc'].posterior,hdi_prob=0.95)
# az.summary(scenario_sparse_smooth['mcmc'].posterior,hdi_prob=0.95)
# az.summary(scenario_sparse_complex['mcmc'].posterior,hdi_prob=0.95)
# az.summary(scenario_2d['mcmc'].posterior,hdi_prob=0.95)

# %% Saving Dictionaries
# np.save(f'{inpath}scenario_ampledata.npy', scenario_ampledata) 
# np.save(f'{inpath}scenario_sparse_smooth.npy', scenario_sparse_smooth) 
# np.save(f'{inpath}scenario_sparse_complex.npy', scenario_sparse_complex) 
# np.save(f'{inpath}scenario_2d.npy', scenario_2d)
