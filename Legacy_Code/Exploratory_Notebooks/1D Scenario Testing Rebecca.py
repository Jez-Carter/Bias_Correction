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
        scenario['X'],diag=scenario['t_noise'],mean=scenario['t_mean'])
    GP_B = GaussianProcess(
        scenario['b_variance'] * kernels.ExpSquared(scenario['b_lengthscale']),
        scenario['X'],diag=scenario['b_noise'],mean=scenario['b_mean'])
    
    scenario['T'] = GP_T.sample(rng_key)
    scenario['odata'] = GP_T.condition(scenario['T'],scenario['ox']).gp.sample(rng_key)
    scenario['cdata_o'] = GP_T.condition(scenario['T'],scenario['cx']).gp.sample(rng_key)

    scenario['B'] = GP_B.sample(rng_key_)
    scenario['cdata_b'] = GP_B.condition(scenario['B'],scenario['cx']).gp.sample(rng_key_)

    scenario['C'] = scenario['T']+scenario['B']
    scenario['cdata'] = scenario['cdata_o']+scenario['cdata_b']

def generate_predictions(scenario):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    mcmc_2process = run_inference(
        tinygp_2process_model, rng_key, 1000, 2000,scenario)
    
    idata_2process = az.from_numpyro(mcmc_2process)
    scenario['mcmc'] = idata_2process
    scenario['nx'] = scenario['X'][::5] # locations where predictions will be made

    truth_realisations,bias_realisations = realisations_2process(scenario,20,20)
    
    scenario['truth_realisations'] = truth_realisations
    scenario['bias_realisations'] = bias_realisations

def generate_predictions_lima_method(scenario):
    mcmc = run_inference(
        tinygp_model, rng_key_, 1000, 2000,
        scenario['ox'],
        data=scenario['odata'],
        noise=scenario['onoise'],
        priors={'kern_var':scenario['t_variance_prior'],
                'lengthscale':scenario['t_lengthscale_prior'],
                'mean':scenario['t_mean_prior']})
    idata = az.from_numpyro(mcmc)
    scenario['mcmc_lima'] = idata
    realisations = singleprocess_posterior_predictive_realisations(
        scenario['nx'],scenario['ox'],idata,scenario['onoise'],20,20)
    
    scenario['truth_realisations_lima'] = realisations.reshape(-1,realisations.shape[-1])

def generate_predictions_step_by_step_method(scenario):
    mcmc_climate = run_inference(
        tinygp_model, rng_key_, 1000, 2000,
        scenario['cx'],
        data=scenario['cdata'],
        noise=scenario['cnoise'],
        priors={'kern_var':scenario['t_variance_prior'],
                'lengthscale':scenario['t_lengthscale_prior'],
                'mean':scenario['t_mean_prior']})
    
    idata_climate = az.from_numpyro(mcmc_climate)
    scenario['mcmc_stb_climate'] = idata_climate

    climate_realisations = singleprocess_posterior_predictive_realisations(
        scenario['ox'],
        scenario['cx'],
        idata_climate,
        scenario['cnoise'],
        20,20)
    bdata_realisations = climate_realisations - scenario['odata']
    bdata_expecation = bdata_realisations.mean(axis=(0,1))
    bnoise = bdata_realisations.std(axis=(0,1)).mean()

    mcmc_bias = run_inference(
        tinygp_model, rng_key_, 1000, 2000,
        scenario['ox'],
        data=bdata_expecation,
        noise=bnoise,
        priors={'kern_var':scenario['b_variance_prior'],
                'lengthscale':scenario['b_lengthscale_prior'],
                'mean':scenario['b_mean_prior']})

    idata_bias = az.from_numpyro(mcmc_bias)
    scenario['mcmc_stb_bias'] = idata_bias
    bias_realisations = singleprocess_posterior_predictive_realisations(
        scenario['nx'],
        scenario['ox'],
        idata_bias,
        bnoise,
        20,20)
    
    climate_realisations = singleprocess_posterior_predictive_realisations(
        scenario['nx'],
        scenario['cx'],
        idata_climate,
        scenario['cnoise'],
        20,20)
    truth_realisations = climate_realisations - bias_realisations

    truth_realisations = truth_realisations.reshape(-1,truth_realisations.shape[-1])
    bias_realisations = bias_realisations.reshape(-1,bias_realisations.shape[-1])

    scenario['truth_realisations_step_by_step'] = truth_realisations
    scenario['bias_realisations_step_by_step'] = bias_realisations

def plot_underlying_data(scenario,ms,ax):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=0.6)
    ax.plot(scenario['X'],scenario['B'],label='Bias',alpha=0.6)
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=0.6)

    ax.scatter(scenario['ox'],scenario['odata'],label='Observations',alpha=0.8,s=ms)
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.8,s=ms)

    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plotting_output_2process(scenario,ax,ms=None,ylims=None):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=1.0,linewidth=2)
    ax.plot(scenario['X'],scenario['B'],label='Bias',alpha=1.0,linewidth=2)
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=1.0,linewidth=2)
    ax.scatter(scenario['ox'],scenario['odata'],color='b',label='Observations',alpha=0.7,s=ms)
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.7,s=ms)

    truth_pred_mean = scenario['truth_realisations'].mean(axis=0)
    truth_pred_std = scenario['truth_realisations'].std(axis=0)
    ax.plot(scenario['nx'],truth_pred_mean,label='Truth Mean',color='m',alpha=1.0,linewidth=2)
    ax.fill_between(scenario['nx'],truth_pred_mean+truth_pred_std,truth_pred_mean-truth_pred_std,label='Truth StdDev',color='m',alpha=0.3)
    bias_pred_mean = scenario['bias_realisations'].mean(axis=0)
    bias_pred_std = scenario['bias_realisations'].std(axis=0)
    ax.plot(scenario['nx'],bias_pred_mean,label='Bias Mean',color='r',alpha=1.0,linewidth=2)
    ax.fill_between(scenario['nx'],bias_pred_mean+bias_pred_std,bias_pred_mean-bias_pred_std,label='Bias StdDev',color='r',alpha=0.3)

    # ax.set_ylim(ylims[0],ylims[1])
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plotting_output_lima(scenario,ax,ms=None,ylims=None):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=1.0,linewidth=2)
    ax.plot(scenario['X'],scenario['B'],label='Bias',alpha=1.0,linewidth=2)
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=1.0,linewidth=2)
    ax.scatter(scenario['ox'],scenario['odata'],color='b',label='Observations',alpha=0.7,s=ms)
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.7,s=ms)

    truth_pred_mean = scenario['truth_realisations_lima'].mean(axis=0)
    truth_pred_std = scenario['truth_realisations_lima'].std(axis=0)
    ax.plot(scenario['nx'],truth_pred_mean,label='Truth Mean',color='m',alpha=1.0,linewidth=2)
    ax.fill_between(scenario['nx'],truth_pred_mean+truth_pred_std,truth_pred_mean-truth_pred_std,label='Truth StdDev',color='m',alpha=0.3)
    
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plotting_output_step_by_step(scenario,ax,ms=None,ylims=None):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=1.0,linewidth=2)
    ax.plot(scenario['X'],scenario['B'],label='Bias',alpha=1.0,linewidth=2)
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=1.0,linewidth=2)
    ax.scatter(scenario['ox'],scenario['odata'],color='b',label='Observations',alpha=0.7,s=ms)
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.7,s=ms)

    truth_pred_mean = scenario['truth_realisations_step_by_step'].mean(axis=0)
    truth_pred_std = scenario['truth_realisations_step_by_step'].std(axis=0)
    ax.plot(scenario['nx'],truth_pred_mean,label='Truth Mean',color='m',alpha=1.0,linewidth=2)
    ax.fill_between(scenario['nx'],truth_pred_mean+truth_pred_std,truth_pred_mean-truth_pred_std,label='Truth StdDev',color='m',alpha=0.3)
    bias_pred_mean = scenario['bias_realisations_step_by_step'].mean(axis=0)
    bias_pred_std = scenario['bias_realisations_step_by_step'].std(axis=0)
    ax.plot(scenario['nx'],bias_pred_mean,label='Bias Mean',color='r',alpha=1.0,linewidth=2)
    ax.fill_between(scenario['nx'],bias_pred_mean+bias_pred_std,bias_pred_mean-bias_pred_std,label='Bias StdDev',color='r',alpha=0.3)

    # ax.set_ylim(ylims[0],ylims[1])
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

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
    gp = GaussianProcess(kernel, scenario['ox'], diag=scenario['onoise'], mean=mean)
    numpyro.sample("obs_temperature", gp.numpyro_dist(),obs=scenario['odata'])

    bkern_var = numpyro.sample("bkern_var", scenario['b_variance_prior'])
    blengthscale = numpyro.sample("blengthscale", scenario['b_lengthscale_prior'])
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", scenario['b_mean_prior'])

    ckernel = kernel+bkernel
    cgp = GaussianProcess(ckernel, scenario['cx'], diag=scenario['cnoise'], mean=mean+bmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=scenario['cdata'])

def diagonal_noise(coord,noise):
    return(np.diag(np.full(coord.shape[0],noise)))

def truth_posterior_predictive(scenario,posterior_param_realisation):

    t_variance_realisation = posterior_param_realisation['t_variance_realisation']
    t_lengthscale_realisation = posterior_param_realisation['t_lengthscale_realisation']
    b_variance_realisation = posterior_param_realisation['b_variance_realisation']
    b_lengthscale_realisation = posterior_param_realisation['b_lengthscale_realisation']
    omean = posterior_param_realisation['t_mean_realisation']
    bmean = posterior_param_realisation['t_mean_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    onoise = scenario['onoise']
    cnoise = scenario['cnoise']
    jitter = scenario['jitter']
    odata = scenario['odata']
    cdata = scenario['cdata']

    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)

    y2 = np.hstack([odata,cdata]) 
    u1 = np.full(nx.shape[0], omean)
    u2 = np.hstack([np.full(ox.shape[0], omean),np.full(cx.shape[0], omean+bmean)])
    k11 = kernelo(nx,nx) + diagonal_noise(nx,jitter)
    k12 = np.hstack([kernelo(nx,ox),kernelo(nx,cx)])
    k21 = np.vstack([kernelo(ox,nx),kernelo(cx,nx)])
    k22_upper = np.hstack([kernelo(ox,ox)+diagonal_noise(ox,onoise),kernelo(ox,cx)])
    k22_lower = np.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,cnoise)])
    k22 = np.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = np.linalg.inv(k22)

    u1g2 = u1 + np.matmul(np.matmul(k12,k22i),y2-u2)

    l22 = np.linalg.cholesky(k22)
    l22i = np.linalg.inv(l22)
    p21 = np.matmul(l22i,k21)
    k1g2 = k11 - np.matmul(p21.T,p21)

    mvn = multivariate_normal(u1g2,k1g2)
    return(mvn)

def bias_posterior_predictive(scenario,posterior_param_realisation):

    t_variance_realisation = posterior_param_realisation['t_variance_realisation']
    t_lengthscale_realisation = posterior_param_realisation['t_lengthscale_realisation']
    b_variance_realisation = posterior_param_realisation['b_variance_realisation']
    b_lengthscale_realisation = posterior_param_realisation['b_lengthscale_realisation']
    omean = posterior_param_realisation['t_mean_realisation']
    bmean = posterior_param_realisation['t_mean_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    onoise = scenario['onoise']
    cnoise = scenario['cnoise']
    jitter = scenario['jitter']
    odata = scenario['odata']
    cdata = scenario['cdata']

    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)

    y2 = np.hstack([odata,cdata]) 
    u1 = np.full(nx.shape[0], bmean)
    u2 = np.hstack([np.full(ox.shape[0], omean),np.full(cx.shape[0], omean+bmean)])
    k11 = kernelb(nx,nx) + diagonal_noise(nx,jitter)
    k12 = np.hstack([np.full((len(nx),len(ox)),0),kernelb(nx,cx)])
    k21 = np.vstack([np.full((len(ox),len(nx)),0),kernelb(cx,nx)])
    k22_upper = np.hstack([kernelo(ox,ox)+diagonal_noise(ox,onoise),kernelo(ox,cx)])
    k22_lower = np.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,cnoise)])
    k22 = np.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = np.linalg.inv(k22)

    u1g2 = u1 + np.matmul(np.matmul(k12,k22i),y2-u2)

    l22 = np.linalg.cholesky(k22)
    l22i = np.linalg.inv(l22)
    p21 = np.matmul(l22i,k21)
    k1g2 = k11 - np.matmul(p21.T,p21)

    mvn = multivariate_normal(u1g2,k1g2)
    return(mvn)

def posterior_predictive_realisations(
        posterior_pred_func,scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior = scenario['mcmc'].posterior

    realisations_list = []
    for i in range(num_parameter_realisations):
        posterior_param_realisation = {
            't_variance_realisation': posterior['kern_var'].data[0,:][i],
            't_lengthscale_realisation': posterior['lengthscale'].data[0,:][i],
            't_mean_realisation': posterior['mean'].data[0,:][i],
            'b_variance_realisation': posterior['bkern_var'].data[0,:][i],
            'b_lengthscale_realisation': posterior['blengthscale'].data[0,:][i],
            'b_mean_realisation': posterior['bmean'].data[0,:][i]
        }
        
        postpred = posterior_pred_func(scenario,posterior_param_realisation)
        realisations = postpred.rvs(num_posterior_pred_realisations)
        realisations_list.append(realisations)

    return(np.array(realisations_list))

def realisations_2process(scenario,npr,nppr):

    truth_realisations = posterior_predictive_realisations(
        truth_posterior_predictive,scenario,
        npr,nppr)

    bias_realisations = posterior_predictive_realisations(
        bias_posterior_predictive,scenario,
        npr,nppr)
      
    truth_realisations = truth_realisations.reshape(-1,truth_realisations.shape[-1])
    bias_realisations = bias_realisations.reshape(-1,bias_realisations.shape[-1])

    return(truth_realisations,bias_realisations)

def remove_outliers(array, perc=[0.001,0.99]):
  lower_threshold = np.quantile(array,perc[0])
  upper_threshold = np.quantile(array,perc[1])
  outlier_condition = (array > upper_threshold) | (array < lower_threshold)

  return(array[outlier_condition==False])

def plot_priors(scenario,prior_keys,axs):

    for key,ax in zip(prior_keys,axs):
        variable = key.split('_prior')[0]
        value = scenario[variable]
        prior_sample = scenario[key].sample(rng_key,(10000,))
        prior_sample = remove_outliers(prior_sample)

        ax.hist(prior_sample,density=True,bins=100,label=f'{key}',alpha=0.8)
        ax.axvline(x=value, ymin=0, ymax=1,linestyle='--')

        ax.legend()

def plot_posteriors(scenario,posterior_keys,axs):

    for key,ax in zip(posterior_keys,axs):
        posterior_sample = scenario['mcmc'].posterior[key].data.reshape(-1)
        posterior_sample = remove_outliers(posterior_sample)

        ax.hist(posterior_sample,density=True,bins=100,label=f'posterior',alpha=0.8)
        ax.legend()

# %%
min_x,max_x = 0,200
X = jnp.arange(min_x,max_x,0.1)

# %%
# Scenario: Similar Lengthscales, Sparse Observations
scenario_base = {
    'onoise': 1e-2,
    'bnoise': 1e-2,
    'cnoise': 1e-2,
    'jitter': 1e-5,
    't_variance': 1.0,
    't_lengthscale': 3.0,
    't_mean': 1.0,
    'b_variance': 1.0,
    'b_lengthscale': 10.0,
    'b_mean': -1.0,
    'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(20,))),
    'cx': np.linspace(min_x,max_x,80) ,
    'X': X,
    't_variance_prior': dist.Gamma(1.0,1.5),
    't_lengthscale_prior': dist.Gamma(3.0,0.2),
    't_mean_prior': dist.Normal(0.0, 2.0),
    'b_variance_prior': dist.Gamma(1.0,0.5),
    'b_lengthscale_prior': dist.Gamma(3.0,0.2),
    'b_mean_prior': dist.Normal(0.0, 2.0),
}

# Scenario: Flat Bias, Sparse Observations
scenario_1 = scenario_base.copy()
scenario_1.update( 
    {'b_lengthscale': 100.0,
     'b_variance': 0.0}
)

# Scenario: Linearly Increasing Bias, Sparse Observations
scenario_2 = scenario_base.copy()
scenario_2.update( 
    {'b_lengthscale': 100.0}
)

# Scenario: Sparse Climate Model Output, Sparse Observations, Smooth Bias
scenario_3 = scenario_base.copy()
scenario_3.update( 
    {'cx': np.linspace(min_x,max_x,20)}
)

# Scenario: Sparse Climate Model Output, Lots of Observations, Complex Bias
scenario_4 = scenario_base.copy()
scenario_4.update( 
    {'cx': np.linspace(min_x,max_x,20),
     'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(80,))),
     'b_lengthscale': 1.0,}
)

# Scenario: Lots of Climate Model Output, Sparse Observations, Simple Bias
scenario_5 = scenario_base.copy()
scenario_5.update( 
    {'b_lengthscale': 100.0}
)

# %%
scenarios = [scenario_base,scenario_3,scenario_4,scenario_5]

for scenario in scenarios:
    generate_underlying_data(scenario)

# %%

# Underlying Data
scenarios = [scenario_base,scenario_3,scenario_4,scenario_5]

fig, axs = plt.subplots(2, 2, figsize=(15, 6*2))

for scenario,i in zip(scenarios,range(len(scenarios))):
    plot_underlying_data(scenario,ms=20,ax=axs.ravel()[i])

# %%
scenarios = [scenario_base,scenario_3,scenario_4,scenario_5]

for scenario in scenarios:

    generate_underlying_data(scenario)
    generate_predictions(scenario)
    # generate_predictions_lima_method(scenario)
    # generate_predictions_step_by_step_method(scenario)

# %%

# generate_underlying_data(scenario_base)
generate_predictions(scenario_base)
# generate_predictions_lima_method(scenario_base)
# generate_predictions_step_by_step_method(scenario_base)

# %%

def plot_posteriors(posterior,keys,axs):

    for key,ax in zip(keys,axs):
        posterior_sample = posterior[key].data.reshape(-1)
        posterior_sample = remove_outliers(posterior_sample)

        ax.hist(posterior_sample,density=True,bins=100,label=f'posterior',alpha=0.8)
        ax.legend()

# %%
list(scenario_base.keys())

# %%
scenario_base['mcmc']

# %%
scenario_base['mcmc_stb_bias']

# %%

prior_keys = []
for key in list(scenario.keys()):
    if 'prior' in key:
        prior_keys.append(key)

posterior_keys = ['kern_var','lengthscale','mean',
                  'bkern_var','blengthscale','bmean']

posterior_keys_sbs = ['kern_var','lengthscale','mean']

fig, axs = plt.subplots(2, 3, figsize=(10, 5))
plot_priors(scenario_base,prior_keys,axs.ravel())
plot_posteriors(scenario_base['mcmc'].posterior,posterior_keys,axs.ravel())
plot_posteriors(scenario_base['mcmc_stb_bias'].posterior,posterior_keys_sbs,axs.ravel()[3:])
plt.tight_layout()

# %%

# Lima Methodology
scenarios = [scenario_4,scenario_5]

fig, axs = plt.subplots(2, 2, figsize=(15, 6*2))

for scenario,i in zip(scenarios,range(len(scenarios))):
    plot_underlying_data(scenario,ms=20,ax=axs[i,0])
    plotting_output_lima(scenario,axs[i,1],ms=20)

# %%

# Step-by-step Methodology
scenarios = [scenario_4,scenario_5]

fig, axs = plt.subplots(2, 2, figsize=(15, 6*2))

for scenario,i in zip(scenarios,range(len(scenarios))):
    plot_underlying_data(scenario,ms=20,ax=axs[i,0])
    plotting_output_step_by_step(scenario,axs[i,1],ms=20)

# %%

# 2 Process Methodology
scenarios = [scenario_base,scenario_5]

fig, axs = plt.subplots(2, 2, figsize=(15, 6*2))

for scenario,i in zip(scenarios,range(len(scenarios))):
    plot_underlying_data(scenario,ms=20,ax=axs[i,0])
    plotting_output_2process(scenario,axs[i,1],ms=20)

# %%

# Lima v's Step-by-step Methodology
scenario = scenario_3

fig, axs = plt.subplots(2, 2, figsize=(15, 6*2))

plot_underlying_data(scenario,ms=20,ax=axs[0,0])
plotting_output_lima(scenario,axs[0,1],ms=20)
plot_underlying_data(scenario,ms=20,ax=axs[1,0])
plotting_output_step_by_step(scenario,axs[1,1],ms=20)

# %%

# Methodology
scenario = scenario_3

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

plot_underlying_data(scenario,ms=20,ax=axs[0])
plotting_output_2process(scenario,axs[1],ms=20)


# %%
fig, axs = plt.subplots(4, len(scenarios), figsize=(15, 6*4))
axs[0,0]
# %%
scenarios = [scenario_3]

fig, axs = plt.subplots(4, 1, figsize=(15, 6*4))

plot_underlying_data(scenarios[0],ms=20,ax=axs[0])
plotting_output_lima(scenarios[0],axs[1],ms=20)
plotting_output_step_by_step(scenarios[0],axs[2],ms=20)
plotting_output_2process(scenarios[0],axs[3],ms=20)

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

plot_underlying_data(scenario_1,ms=20,ax=axs[0])
plot_underlying_data(scenario_2,ms=20,ax=axs[1])

plt.tight_layout()
    
# %%
generate_predictions(scenario_1)
generate_predictions(scenario_2)

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plotting_output_2process(scenario_1,axs[0],ms=20)
plotting_output_2process(scenario_2,axs[1],ms=20)

# %%
generate_predictions_lima_method(scenario_1)
generate_predictions_lima_method(scenario_2)

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plotting_output_lima(scenario_1,axs[0],ms=20)
plotting_output_lima(scenario_2,axs[1],ms=20)

# %%
generate_predictions_step_by_step_method(scenario_1)
generate_predictions_step_by_step_method(scenario_2)
# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plotting_output_step_by_step(scenario_1,axs[0],ms=20)
plotting_output_step_by_step(scenario_2,axs[1],ms=20)

# %%
axs[0,1].shape

# %%
fig, axs = plt.subplots(4, 2, figsize=(15, 6*4))

plot_underlying_data(scenario_1,ms=20,ax=axs[0,0])
plot_underlying_data(scenario_2,ms=20,ax=axs[0,1])
plotting_output_lima(scenario_1,axs[1,0],ms=20)
plotting_output_lima(scenario_2,axs[1,1],ms=20)
plotting_output_step_by_step(scenario_1,axs[2,0],ms=20)
plotting_output_step_by_step(scenario_2,axs[2,1],ms=20)
plotting_output_2process(scenario_1,axs[3,0],ms=20)
plotting_output_2process(scenario_2,axs[3,1],ms=20)

# %%
generate_underlying_data(scenario_base)
generate_predictions(scenario_base)
generate_predictions_lima_method(scenario_base)
generate_predictions_step_by_step_method(scenario_base)

# %%
fig, axs = plt.subplots(4, 3, figsize=(15, 6*4))

plot_underlying_data(scenario_1,ms=20,ax=axs[0,0])
plot_underlying_data(scenario_2,ms=20,ax=axs[0,1])
plot_underlying_data(scenario_base,ms=20,ax=axs[0,2])
plotting_output_lima(scenario_1,axs[1,0],ms=20)
plotting_output_lima(scenario_2,axs[1,1],ms=20)
plotting_output_lima(scenario_base,axs[1,2],ms=20)
plotting_output_step_by_step(scenario_1,axs[2,0],ms=20)
plotting_output_step_by_step(scenario_2,axs[2,1],ms=20)
plotting_output_step_by_step(scenario_base,axs[2,2],ms=20)
plotting_output_2process(scenario_1,axs[3,0],ms=20)
plotting_output_2process(scenario_2,axs[3,1],ms=20)
plotting_output_2process(scenario_base,axs[3,2],ms=20)

# %%
fig, axs = plt.subplots(1, 1, figsize=(15, 2*4))

plotting_output_2process(scenario_base,axs,ms=20)

# %%
fig, axs = plt.subplots(1, 1, figsize=(15, 2*4))

plotting_output_step_by_step(scenario_base,axs,ms=20)