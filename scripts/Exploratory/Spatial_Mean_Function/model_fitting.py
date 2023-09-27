# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import arviz as az

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams.update({'font.size': 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions_meanfunction import generate_posterior
from src.simulated_data_functions_meanfunction import plot_underlying_data_1d
from src.simulated_data_functions import plot_underlying_data_2d
from src.simulated_data_functions import plot_priors
from src.simulated_data_functions import plot_posteriors

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
scenario_ampledata = np.load(f'{inpath}scenario_ampledata_meanfunction.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth_meanfunction.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex_meanfunction.npy',allow_pickle='TRUE').item()

# %%
from functools import partial

def mean_function(params,x):
    mean = params["beta0"]+params["beta1"]*(x-50)+params["beta2"]*(x-50)**2
    return(mean)

mean_params = {'beta0':scenario_ampledata['t_mean_beta0'],
                'beta1':scenario_ampledata['t_mean_beta1'],
                'beta2':scenario_ampledata['t_mean_beta2']}
t_mean = partial(mean_function, mean_params)

# %%
c_mean = lambda x : t_mean(x) + 5
c_mean(15)
# %%
# generate_posterior(scenario_ampledata,rng_key,1000,2000,1)
# rng_key, rng_key_ = random.split(rng_key)
# generate_posterior(scenario_sparse_smooth,rng_key,1000,2000,1)
rng_key, rng_key_ = random.split(rng_key)
generate_posterior(scenario_sparse_complex,rng_key,1000,2000,1)

# %%
# np.save(f'{inpath}scenario_ampledata_meanfunction.npy', scenario_ampledata) 
# np.save(f'{inpath}scenario_sparse_smooth_meanfunction.npy', scenario_sparse_smooth) 
np.save(f'{inpath}scenario_sparse_complex_meanfunction.npy', scenario_sparse_complex) 

# %%
scenario_sparse_complex['mcmc']

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plot_underlying_data_1d(scenario_ampledata,ax,ms=20)
