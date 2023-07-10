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
    cmean = mean+bmean
    cnoise = numpyro.sample("cnoise", scenario['cnoise_prior'])
    cgp = GaussianProcess(ckernel, scenario['cx'], diag=cnoise, mean=cmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=scenario['cdata'])

    onoise = numpyro.sample("onoise", scenario['onoise_prior'])
    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(scenario,ckernel,cmean,cnoise,kernel,mean,onoise)
    numpyro.sample("obs_temperature", obs_conditional_climate_dist,obs=scenario['odata'])

    # gp = cgp.condition(scenario['cdata'],scenario['ox'], kernel=cgp.kernel.kernel1).gp
    # onoise = numpyro.sample("onoise", scenario['onoise_prior'])
    # gp.noise = tinygp.noise.Diagonal(jnp.full(scenario['ox'].shape,onoise))
    # gp.mean = jnp.full(scenario['ox'].shape,mean)
    # numpyro.sample("obs_temperature", gp.numpyro_dist(),obs=scenario['odata'])

def generate_posterior(scenario):
    rng_key = random.PRNGKey(0)
    
    mcmc_2process = run_inference(
        tinygp_2process_model, rng_key, 1000, 2000,scenario)
    
    idata_2process = az.from_numpyro(mcmc_2process)
    scenario['mcmc'] = idata_2process

def diagonal_noise(coord,noise):
    return(jnp.diag(jnp.full(coord.shape[0],noise)))

def generate_obs_conditional_climate_dist(scenario,ckernel,cmean,cnoise,okernel,omean,onoise):

    ox = scenario['ox']
    cx = scenario['cx']
    cdata = scenario['cdata']

    y2 = cdata
    u1 = jnp.full(ox.shape[0], omean)
    u2 = jnp.full(cx.shape[0], cmean)
    k11 = okernel(ox,ox) + diagonal_noise(ox,onoise)
    k12 = okernel(ox,cx)
    k21 = okernel(cx,ox) 
    k22 = ckernel(cx,cx) + diagonal_noise(cx,cnoise)
    k22i = jnp.linalg.inv(k22)

    # print(u1.shape,k12.shape,k22i.shape,)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)

    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)

    mvn_dist = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn_dist)

def generate_truth_predictive_dist(scenario,
                                   posterior_param_realisation):
    t_variance_realisation = posterior_param_realisation['t_variance_realisation']
    t_lengthscale_realisation = posterior_param_realisation['t_lengthscale_realisation']
    t_mean_realisation = posterior_param_realisation['t_mean_realisation']

    b_variance_realisation = posterior_param_realisation['b_variance_realisation']
    b_lengthscale_realisation = posterior_param_realisation['b_lengthscale_realisation']
    b_mean_realisation = posterior_param_realisation['b_mean_realisation']

    onoise_realisation = posterior_param_realisation['onoise_realisation']
    cnoise_realisation = posterior_param_realisation['cnoise_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    odata = scenario['odata']
    cdata = scenario['cdata']

    omean = t_mean_realisation
    bmean = b_mean_realisation
    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)
    onoise = onoise_realisation
    cnoise = cnoise_realisation

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

def generate_bias_predictive_dist(scenario,
                                   posterior_param_realisation):
    t_variance_realisation = posterior_param_realisation['t_variance_realisation']
    t_lengthscale_realisation = posterior_param_realisation['t_lengthscale_realisation']
    t_mean_realisation = posterior_param_realisation['t_mean_realisation']

    b_variance_realisation = posterior_param_realisation['b_variance_realisation']
    b_lengthscale_realisation = posterior_param_realisation['b_lengthscale_realisation']
    b_mean_realisation = posterior_param_realisation['b_mean_realisation']

    onoise_realisation = posterior_param_realisation['onoise_realisation']
    cnoise_realisation = posterior_param_realisation['cnoise_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    odata = scenario['odata']
    cdata = scenario['cdata']

    omean = t_mean_realisation
    bmean = b_mean_realisation
    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)
    onoise = onoise_realisation
    cnoise = cnoise_realisation

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

def generate_posterior_predictive_realisations(
        scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior = scenario['mcmc'].posterior

    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            't_variance_realisation': posterior['kern_var'].data[0,:][i],
            't_lengthscale_realisation': posterior['lengthscale'].data[0,:][i],
            't_mean_realisation': posterior['mean'].data[0,:][i],
            'b_variance_realisation': posterior['bkern_var'].data[0,:][i],
            'b_lengthscale_realisation': posterior['blengthscale'].data[0,:][i],
            'b_mean_realisation': posterior['bmean'].data[0,:][i],
            'onoise_realisation': posterior['onoise'].data[0,:][i],
            'cnoise_realisation': posterior['cnoise'].data[0,:][i]
        }
        
        truth_predictive_dist = generate_truth_predictive_dist(scenario,
                                   posterior_param_realisation)
        bias_predictive_dist = generate_bias_predictive_dist(scenario,
                                   posterior_param_realisation)

        truth_predictive_realisations = truth_predictive_dist.rvs(num_posterior_pred_realisations)
        bias_predictive_realisations = bias_predictive_dist.rvs(num_posterior_pred_realisations)

        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = np.array(truth_posterior_predictive_realisations)
    bias_posterior_predictive_realisations = np.array(bias_posterior_predictive_realisations)
    truth_posterior_predictive_realisations = truth_posterior_predictive_realisations.reshape(-1,truth_posterior_predictive_realisations.shape[-1])
    bias_posterior_predictive_realisations = bias_posterior_predictive_realisations.reshape(-1,bias_posterior_predictive_realisations.shape[-1])

    scenario['truth_posterior_predictive_realisations'] = truth_posterior_predictive_realisations
    scenario['bias_posterior_predictive_realisations'] = bias_posterior_predictive_realisations

# %%

def plot_underlying_data(scenario,ax,ms):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=0.6)
    ax.plot(scenario['X'],scenario['B'],label='Bias',alpha=0.6)
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=0.6)

    ax.scatter(scenario['ox'],scenario['odata'],label='Observations',alpha=0.8,s=ms)
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.8,s=ms)

    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plotting_output(scenario,key,ax,ms=None,ylims=None,color=None):
    pred_mean = scenario[key].mean(axis=0)
    pred_std = scenario[key].std(axis=0)
    ax.plot(scenario['nx'],pred_mean,label='Mean',color=color,alpha=0.5,linewidth=2)
    ax.fill_between(scenario['nx'],pred_mean+pred_std,pred_mean-pred_std,label='StdDev',color=color,alpha=0.3)
    
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

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

def plot_posteriors(posterior,posterior_keys,axs):

    for key,ax in zip(posterior_keys,axs):
        posterior_sample = posterior[key].data.reshape(-1)
        posterior_sample = remove_outliers(posterior_sample)

        ax.hist(posterior_sample,density=True,bins=100,label=f'posterior',alpha=0.8)
        ax.legend()

# %%
min_x,max_x = 0,400
X = jnp.arange(min_x,max_x,0.1)

# Scenario: Similar Lengthscales, Sparse Observations
scenario_base = {
    'onoise': 1e-1,
    'bnoise': 1e-1,
    'cnoise': 1e-1,
    'jitter': 1e-5,
    't_variance': 1.0,
    't_lengthscale': 3.0,
    't_mean': 1.0,
    'b_variance': 1.0,
    'b_lengthscale': 10.0,
    'b_mean': -1.0,
    'ox': np.sort(RandomState(0).uniform(low=min_x, high=max_x, size=(40,))),
    'cx': np.linspace(min_x,max_x,80) ,
    'X': X,
    't_variance_prior': dist.Gamma(1.0,1.5),
    't_lengthscale_prior': dist.Gamma(3.0,0.2),
    't_mean_prior': dist.Normal(0.0, 2.0),
    'b_variance_prior': dist.Gamma(1.0,0.5),
    'b_lengthscale_prior': dist.Gamma(3.0,0.2),
    'b_mean_prior': dist.Normal(0.0, 2.0),
    'onoise_prior':dist.Uniform(0.0,0.5),
    'cnoise_prior':dist.Uniform(0.0,0.5),
    'nx':X[::5]
}

# %%
generate_underlying_data(scenario_base)
generate_posterior(scenario_base)
# generate_posterior_predictive_realisations(scenario_base,20,20)