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

    bkern_var = numpyro.sample("bkern_var", scenario['b_variance_prior'])
    blengthscale = numpyro.sample("blengthscale", scenario['b_lengthscale_prior'])
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", scenario['b_mean_prior'])

    ckernel = kernel+bkernel
    cgp = GaussianProcess(ckernel, scenario['cx'], diag=scenario['cnoise'], mean=mean+bmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=scenario['cdata'])

def generate_posterior(scenario):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    mcmc_2process = run_inference(
        tinygp_2process_model, rng_key, 1000, 2000,scenario)
    
    idata_2process = az.from_numpyro(mcmc_2process)
    scenario['mcmc'] = idata_2process

def posterior_predictive(scenario,posterior_param_realisation,num_posterior_pred_realisations):
        
    kern_var = posterior_param_realisation['t_variance_realisation']
    lengthscale = posterior_param_realisation['t_lengthscale_realisation']
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = posterior_param_realisation['t_mean_realisation']

    bkern_var = posterior_param_realisation['b_variance_realisation']
    blengthscale = posterior_param_realisation['b_lengthscale_realisation']
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = posterior_param_realisation['b_mean_realisation']

    ckernel = kernel+bkernel
    cgp = GaussianProcess(ckernel, scenario['cx'], diag=scenario['cnoise'], mean=mean+bmean)
    cgp_cond = cgp.condition(scenario['cdata'], scenario['nx']).gp

    realisation = cgp_cond.sample(rng_key,(num_posterior_pred_realisations,))

    return(realisation)

def generate_posterior_predictive_realisations(
        scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior = scenario['mcmc'].posterior

    realisations_list = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            't_variance_realisation': posterior['kern_var'].data[0,:][i],
            't_lengthscale_realisation': posterior['lengthscale'].data[0,:][i],
            't_mean_realisation': posterior['mean'].data[0,:][i],
            'b_variance_realisation': posterior['bkern_var'].data[0,:][i],
            'b_lengthscale_realisation': posterior['blengthscale'].data[0,:][i],
            'b_mean_realisation': posterior['bmean'].data[0,:][i]
        }
        
        realisations = posterior_predictive(scenario,posterior_param_realisation,num_posterior_pred_realisations)
        realisations_list.append(realisations)

    realisations = np.array(realisations_list)
    scenario['realisations']=realisations.reshape(-1,realisations.shape[-1])

def tinygp_model(x,data=None,noise=None,priors=None):
    """
    Example model where the data is generated from a GP.
    Args:x (jax device array): array of coordinates for data, shape [#points,dimcoords]
    data (jax device array): array of data values, shape [#points,]
    """
    kern_var = numpyro.sample("kern_var", priors['kern_var'])
    lengthscale = numpyro.sample("lengthscale", priors['lengthscale'])
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", priors['mean'])
    gp = GaussianProcess(kernel, x, diag=noise, mean=mean)
    numpyro.sample("data", gp.numpyro_dist(),obs=data)

def generate_posterior_single(scenario):
    mcmc = run_inference(
        tinygp_model, rng_key_, 1000, 2000,
        scenario['cx'],
        data=scenario['cdata'],
        noise=scenario['cnoise'],
        priors={'kern_var':scenario['t_variance_prior'],
                'lengthscale':scenario['t_lengthscale_prior'],
                'mean':scenario['t_mean_prior']})
    idata = az.from_numpyro(mcmc)
    scenario['mcmc_single'] = idata

def generate_conditional_gp_single(scenario,
                                   posterior_param_realisation):
    kern_var = posterior_param_realisation['variance_realisation']
    lengthscale = posterior_param_realisation['lengthscale_realisation']
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = posterior_param_realisation['mean_realisation']

    cgp = GaussianProcess(kernel, scenario['cx'], diag=scenario['cnoise'], mean=mean)
    cgp_cond = cgp.condition(scenario['cdata'], scenario['nx']).gp

    return(cgp_cond)

def generate_posterior_predictive_realisations_single(
        scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior = scenario['mcmc_single'].posterior

    realisations_list = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            'variance_realisation': posterior['kern_var'].data[0,:][i],
            'lengthscale_realisation': posterior['lengthscale'].data[0,:][i],
            'mean_realisation': posterior['mean'].data[0,:][i]
        }

        cgp_cond = generate_conditional_gp_single(scenario,
                                   posterior_param_realisation)
        
        realisations = cgp_cond.sample(rng_key,(num_posterior_pred_realisations,))
        realisations_list.append(realisations)

    realisations = np.array(realisations_list)
    scenario['realisations_single']=realisations.reshape(-1,realisations.shape[-1])

# %%
def plotting_underlying_data(scenario,ax,ms=None,ylims=None):
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=1.0,linewidth=2)
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.7,s=ms)
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plotting_output(scenario,key,ax,ms=None,ylims=None):
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=1.0,linewidth=2)
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.7,s=ms)

    pred_mean = scenario[key].mean(axis=0)
    pred_std = scenario[key].std(axis=0)
    ax.plot(scenario['nx'],pred_mean,label='Mean',color='m',alpha=1.0,linewidth=2)
    ax.fill_between(scenario['nx'],pred_mean+pred_std,pred_mean-pred_std,label='StdDev',color='m',alpha=0.3)
    
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

# %%
min_x,max_x = 0,200
X = jnp.arange(min_x,max_x,0.1)

# Scenario: Similar Lengthscales, Sparse Observations
scenario_base = {
    'onoise': 1e-1,
    'bnoise': 1e-1,
    'cnoise': 1e-1,
    'jitter': 1e-5,
    't_variance': 1.0,
    't_lengthscale': 1.0,
    't_mean': 1.0,
    'b_variance': 1.0,
    'b_lengthscale': 10.0,
    'b_mean': -1.0,
    'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(20,))),
    'cx': np.linspace(min_x,max_x,40) ,
    'X': X,
    't_variance_prior': dist.Gamma(1.0,1.5),
    't_lengthscale_prior': dist.Gamma(3.0,0.2),
    't_mean_prior': dist.Normal(0.0, 2.0),
    'b_variance_prior': dist.Gamma(1.0,0.5),
    'b_lengthscale_prior': dist.Gamma(3.0,0.2),
    'b_mean_prior': dist.Normal(0.0, 2.0),
    'nx':X[::5]
}

# %%
generate_underlying_data(scenario_base)
generate_posterior(scenario_base)
generate_posterior_predictive_realisations(scenario_base,20,20)

# %%
generate_posterior_single(scenario_base)
generate_posterior_predictive_realisations_single(scenario_base,20,20)

# %%

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

plotting_underlying_data(scenario_base,ax,ms=20)

# %%

fig, axs = plt.subplots(2, 1, figsize=(15, 15))

plotting_output(scenario_base,'realisations',axs[0],ms=20)
plotting_output(scenario_base,'realisations_single',axs[1],ms=20)
