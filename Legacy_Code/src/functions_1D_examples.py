# %%

#Importing Packages
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.random import RandomState
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import multivariate_normal
from src.model_fitting_functions import run_inference
from src.examples_functions import tinygp_2process_model
from src.examples_functions import realisations_2process
from src.examples_functions import plot_underlying_data,plotting_output_2process
from src.examples_functions import tinygp_model
from src.examples_functions import singleprocess_posterior_predictive_realisations

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
def generate_underlying_data(scenario):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    GP_T = GaussianProcess(
        scenario['t_variance'] * kernels.ExpSquared(scenario['t_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['t_mean'])
    GP_B = GaussianProcess(
        scenario['b_variance'] * kernels.ExpSquared(scenario['b_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['b_mean'])
    
    scenario['T'] = GP_T.sample(rng_key)
    scenario['B'] = GP_B.sample(rng_key_)
    scenario['C'] = scenario['T']+scenario['B']

    scenario['odata'] = GP_T.condition(scenario['T'],scenario['ox']).gp.sample(rng_key)
    odata_noise = dist.Normal(0.0,scenario['onoise']).sample(rng_key,scenario['odata'].shape)
    scenario['odata'] = scenario['odata'] + odata_noise

    scenario['cdata_o'] = GP_T.condition(scenario['T'],scenario['cx']).gp.sample(rng_key)
    scenario['cdata_b'] = GP_B.condition(scenario['B'],scenario['cx']).gp.sample(rng_key_)
    scenario['cdata'] = scenario['cdata_o']+scenario['cdata_b']
    cdata_noise = dist.Normal(0.0,scenario['cnoise']).sample(rng_key,scenario['cdata'].shape)
    scenario['cdata'] = scenario['cdata'] + cdata_noise

def tinygp_2process_model(scenario):
    """
   Example model where the climate data is generated from 2 GPs,
   one of which also generates the observations and one of
   which generates bias in the climate model.
    """
    kern_var = numpyro.sample("kern_var", scenario['t_variance_prior'])
    lengthscale = numpyro.sample("lengthscale", scenario['t_lengthscale_prior'])
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", scenario['t_mean_prior'])
    onoise = numpyro.sample("onoise", scenario['onoise_prior'])
    gp = GaussianProcess(kernel, scenario['ox'], diag=onoise, mean=mean)
    numpyro.sample("obs_temperature", gp.numpyro_dist(),obs=scenario['odata'])

    bkern_var = numpyro.sample("bkern_var", scenario['b_variance_prior'])
    blengthscale = numpyro.sample("blengthscale", scenario['b_lengthscale_prior'])
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", scenario['b_mean_prior'])

    ckernel = kernel+bkernel
    cnoise = numpyro.sample("cnoise", scenario['cnoise_prior'])
    cgp = GaussianProcess(ckernel, scenario['cx'], diag=cnoise, mean=mean+bmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=scenario['cdata'])