# %%

#Importing Packages
import numpy as np
import numpyro.distributions as dist
from numpy.random import RandomState
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.simulated_data_functions_hierarchical import generate_underlying_data_hierarchical
from src.simulated_data_functions_hierarchical import plot_underlying_data_mean_1d
from src.simulated_data_functions_hierarchical import plot_underlying_data_std_1d
from src.simulated_data_functions_hierarchical import plot_pdfs_1d
from src.simulated_data_functions_hierarchical import plot_underlying_data_2d

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

# Scenario: Similar Lengthscales, Sparse Observations
scenario_base = {
    'jitter': 1e-5,
    'MEAN_T_variance': 1.0,
    'MEAN_T_lengthscale': 3.0,
    'MEAN_T_mean': 1.0,
    'LOGVAR_T_variance': 1.0,
    'LOGVAR_T_lengthscale': 3.0,
    'LOGVAR_T_mean': 1.0,
    'MEAN_B_variance': 1.0,
    'MEAN_B_lengthscale': 10.0,
    'MEAN_B_mean': -1.0,
    'LOGVAR_B_variance': 1.0,
    'LOGVAR_B_lengthscale': 10.0,
    'LOGVAR_B_mean': -1.0,
    'osamples':20,
    'csamples':100,
    'ox': RandomState(0).uniform(low=min_x, high=max_x, size=(40,)),
    'cx': np.linspace(min_x,max_x,80) ,
    'X': X,
    'MEAN_T_variance_prior': dist.Gamma(1.0,1.5),
    'MEAN_T_lengthscale_prior': dist.Gamma(3.0,0.2),
    'MEAN_T_mean_prior': dist.Normal(0.0, 2.0),
    'LOGVAR_T_variance_prior': dist.Gamma(1.0,1.5),
    'LOGVAR_T_lengthscale_prior': dist.Gamma(3.0,0.2),
    'LOGVAR_T_mean_prior': dist.Normal(0.0, 2.0),
    'MEAN_B_variance_prior': dist.Gamma(1.0,1.5),
    'MEAN_B_lengthscale_prior': dist.Gamma(3.0,0.2),
    'MEAN_B_mean_prior': dist.Normal(0.0, 2.0),
    'LOGVAR_B_variance_prior': dist.Gamma(1.0,1.5),
    'LOGVAR_B_lengthscale_prior': dist.Gamma(3.0,0.2),
    'LOGVAR_B_mean_prior': dist.Normal(0.0, 2.0),
    'nx':X[::5]
}

# Scenario: 2D
scenario_2d = scenario_base.copy()
X1,X2 = np.meshgrid(jnp.linspace(min_x,max_x,21),jnp.linspace(min_x,max_x,21))
X = np.dstack([X1.ravel(),X2.ravel()])[0]
CX1,CX2 = np.meshgrid(jnp.linspace(min_x,max_x,11),jnp.linspace(min_x,max_x,11))
CX = np.dstack([CX1.ravel(),CX2.ravel()])[0]
scenario_2d.update( 
    {'X1':X1,
     'X2':X2,
     'X':X,
     'CX1':CX1,
     'CX2':CX2,
     'cx': CX,
     'ox': RandomState(0).uniform(low=min_x, high=max_x, size=(80,2)),
     'MEAN_T_lengthscale': 10.0,
     'LOGVAR_T_lengthscale': 10.0,
     'MEAN_B_lengthscale': 50.0,
     'LOGVAR_B_lengthscale': 50.0,
     'nx':X}
)

# %%
rng_key = random.PRNGKey(0)
generate_underlying_data_hierarchical(scenario_base,rng_key)
generate_underlying_data_hierarchical(scenario_2d,rng_key)

# %%
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
plot_underlying_data_mean_1d(scenario_base,axs[0])
plot_underlying_data_std_1d(scenario_base,axs[1])

# %%
scenario_2d['STDEV_T'] = jnp.sqrt(jnp.exp(scenario_2d['LOGVAR_T']))
scenario_2d['STDEV_B'] = jnp.sqrt(jnp.exp(scenario_2d['LOGVAR_B']))
scenario_2d['STDEV_C'] = jnp.sqrt(jnp.exp(scenario_2d['LOGVAR_C']))

variables_mean = ['MEAN_T','MEAN_B','MEAN_C']
variables_std = ['STDEV_T','STDEV_B','STDEV_C']

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
plot_underlying_data_2d(scenario_2d,variables_mean,axs[0],1,True,'RdBu_r')
plot_underlying_data_2d(scenario_2d,variables_std,axs[1],1,None,'viridis')

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plot_pdfs_1d(scenario_base,ax,100)

# %% Saving Dictionaries
# np.save(f'{outpath}scenario_base_hierarchical.npy', scenario_base) 
# np.save(f'{outpath}scenario_2d_hierarchical.npy', scenario_2d) 
