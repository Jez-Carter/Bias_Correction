# %% Importing Packages
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

from src.non_hierarchical.data_generation_functions import generate_underlying_data
from src.non_hierarchical.plotting_functions import plot_underlying_data_1d
from src.non_hierarchical.plotting_functions import plot_underlying_data_2d
outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Specifying parameters for different scenarios

min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

# Scenario Base: Relatively Smooth Bias and Medium Number of Observations
scenario_base = {
    'onoise': 1e-1,
    'bnoise': 1e-1,
    'cnoise': 1e-3,
    'jitter': 1e-10,
    't_variance': 1.0,
    't_lengthscale': 3.0,
    't_mean': 1.0,
    'b_variance': 1.0,
    'b_lengthscale': 10.0,
    'b_mean': -1.0,
    'ox': RandomState(0).uniform(low=min_x, high=max_x, size=(40,)),
    'cx': np.linspace(min_x,max_x,80) ,
    'X': X,
    't_variance_prior': dist.Gamma(1.0,1.5),
    't_lengthscale_prior': dist.Gamma(3.0,0.2),
    't_mean_prior': dist.Normal(0.0,2.0),
    'b_variance_prior': dist.Gamma(1.0,0.5),
    'b_lengthscale_prior': dist.Gamma(3.0,0.2),
    'b_mean_prior': dist.Normal(0.0, 2.0),
    'onoise_prior':dist.Uniform(0.0,0.5),
    'cnoise_prior':dist.Uniform(0.0,0.5),
    'nx':X[::5]
}

# Scenario: Increased number of Observations and Climate Model Output
scenario_ampledata = scenario_base.copy()
scenario_ampledata.update( 
    {'ox': np.sort(RandomState(5).uniform(low=min_x, high=max_x, size=(80,))),
     'cx': np.linspace(min_x,max_x,100)}
)

# Scenario: Decreased Number of Observations, Smoother Bias
scenario_sparse_smooth = scenario_base.copy()
scenario_sparse_smooth.update( 
    {'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(20,))),
     'b_lengthscale': 20.0,}
)

# Scenario: Decreased Number of Observations, Less Smooth Bias
scenario_sparse_complex = scenario_base.copy()
scenario_sparse_complex.update( 
    {'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(20,))),
     'b_lengthscale': 5.0,}
)

# Scenario: 2D Relatively Smooth Bias and Medium Number of Observations
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
     't_lengthscale': 10.0,
     'b_lengthscale': 50.0,
     'nx':X}
)

# %% Generating the underlying simulated data

rng_key = random.PRNGKey(0)
generate_underlying_data(scenario_base,rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data(scenario_ampledata,rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data(scenario_sparse_smooth,rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data(scenario_sparse_complex,rng_key)
rng_key, rng_key_ = random.split(rng_key)
generate_underlying_data(scenario_2d,rng_key)

# %% Plot showing generated 1D data

fig, axs = plt.subplots(3,1,figsize=(12, 12),dpi= 300)

plot_underlying_data_1d(scenario_ampledata,axs[0],ms=20)
plot_underlying_data_1d(scenario_sparse_smooth,axs[1],ms=20)
plot_underlying_data_1d(scenario_sparse_complex,axs[2],ms=20)

plt.tight_layout()
plt.show()

# %% Plot showing generated 2D data
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plot_underlying_data_2d(scenario_2d,axs,1)

# %% Saving Dictionaries
# np.save(f'{outpath}scenario_base.npy', scenario_base) 
# np.save(f'{outpath}scenario_ampledata.npy', scenario_ampledata) 
# np.save(f'{outpath}scenario_sparse_smooth.npy', scenario_sparse_smooth) 
# np.save(f'{outpath}scenario_sparse_complex.npy', scenario_sparse_complex) 
# np.save(f'{outpath}scenario_2d.npy', scenario_2d) 

# %%
