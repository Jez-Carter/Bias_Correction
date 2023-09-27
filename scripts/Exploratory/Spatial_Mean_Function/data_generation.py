# %%

#Importing Packages
import numpy as np
import numpyro.distributions as dist
from numpy.random import RandomState
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions_meanfunction import generate_underlying_data
from src.simulated_data_functions_meanfunction import plot_underlying_data_1d
from src.simulated_data_functions_meanfunction import plot_underlying_data_2d
outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

# Scenario: Similar Lengthscales, Sparse Observations
scenario_base = {
    'onoise': 1e-1,
    'bnoise': 1e-1,
    'cnoise': 1e-3,
    'jitter': 1e-10,
    't_variance': 1.0,
    't_lengthscale': 3.0,
    't_mean_beta0': 1.0,
    't_mean_beta1':-0.01,
    't_mean_beta2':0.01,
    'b_variance': 1.0,
    'b_lengthscale': 10.0,
    'b_mean': -1.0,
    'ox': RandomState(0).uniform(low=min_x, high=max_x, size=(40,)),
    'cx': np.linspace(min_x,max_x,80) ,
    'X': X,
    't_variance_prior': dist.Gamma(1.0,1.5),
    't_lengthscale_prior': dist.Gamma(3.0,0.2),
    't_mean_prior': dist.Normal(0.0,2.0),
    't_mean_beta0_prior': dist.Normal(0.0, 2.0),
    't_mean_beta1_prior': dist.Normal(0.0, 2.0),
    't_mean_beta2_prior': dist.Normal(0.0, 2.0),
    'b_variance_prior': dist.Gamma(1.0,0.5),
    'b_lengthscale_prior': dist.Gamma(3.0,0.2),
    'b_mean_prior': dist.Normal(0.0, 2.0),
    'onoise_prior':dist.Uniform(0.0,0.5),
    'cnoise_prior':dist.Uniform(0.0,0.5),
    'nx':X[::5]
}

# Scenario: Lots of Observations and Climate Model Output
scenario_ampledata = scenario_base.copy()
scenario_ampledata.update( 
    {'ox': np.sort(RandomState(5).uniform(low=min_x, high=max_x, size=(80,))),
     'cx': np.linspace(min_x,max_x,100)}
)

# Scenario: Sparse Observations, Simple Bias
scenario_sparse_smooth = scenario_base.copy()
scenario_sparse_smooth.update( 
    {'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(20,))),
     'b_lengthscale': 20.0,}
)

# Scenario: Sparse Observations, Complex Bias
scenario_sparse_complex = scenario_base.copy()
scenario_sparse_complex.update( 
    {'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(20,))),
     'b_lengthscale': 5.0,}
)

# %%
rng_key = random.PRNGKey(0)
# generate_underlying_data(scenario_ampledata,rng_key)
# generate_underlying_data(scenario_sparse_smooth,rng_key)
generate_underlying_data(scenario_sparse_complex,rng_key)

# np.save(f'{outpath}scenario_ampledata_meanfunction.npy', scenario_ampledata) 
# np.save(f'{outpath}scenario_sparse_smooth_meanfunction.npy', scenario_sparse_smooth) 
# np.save(f'{outpath}scenario_sparse_complex_meanfunction.npy', scenario_sparse_complex) 

# %%

scenario_sparse_complex['ox'].shape

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
# plot_underlying_data_1d(scenario_ampledata,ax,ms=20)
# plot_underlying_data_1d(scenario_sparse_smooth,ax,ms=20)
plot_underlying_data_1d(scenario_sparse_complex,ax,ms=20)

# %%
