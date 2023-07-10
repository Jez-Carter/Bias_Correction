from tqdm import tqdm
import numpy as np
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from jax import random
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

# def singleGP_model(x=None,data=None,noise=None):
#     """
#     Example model where the data is generated from a GP.
#     Args:x (jax device array): array of coordinates for data, shape [#points,dimcoords]
#     data (jax device array): array of data values, shape [#points,]
#     """
#     kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
#     lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
#     kernel = kern_var * kernels.ExpSquared(lengthscale)
#     mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
#     gp = GaussianProcess(kernel, x, diag=noise, mean=mean)
#     numpyro.sample("data", gp.numpyro_dist(),obs=data)

def singleGP_model(x=None,data=None,noise=None,kern=None):
    """
    Example model where the data is generated from a GP.
    Args:x (jax device array): array of coordinates for data, shape [#points,dimcoords]
    data (jax device array): array of data values, shape [#points,]
    """
    kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
    lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
    kernel = kern_var * kern(lengthscale)
    mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
    gp = GaussianProcess(kernel, x, diag=noise, mean=mean)
    numpyro.sample("data", gp.numpyro_dist(),obs=data)

def singleprocess_posterior_predictive_realisations(nx,x,idata,noise,
                                                    num_parameter_realisations,num_posterior_pred_realisations):
    data = idata.observed_data.data.data
    posterior = idata.posterior
    #parameter realisations 
    kern_vars = posterior['kern_var'].data[0,:]
    lengthscales = posterior['lengthscale'].data[0,:]
    means = posterior['mean'].data[0,:]

    realisations_list = []
    for kern_var,lengthscale,mean in tqdm(list(zip(kern_vars,lengthscales,means))[:num_parameter_realisations]):
        kernel = kern_var * kernels.ExpSquared(lengthscale)
        # kernel = kern_var * kernels.Matern32(lengthscale)
        gp = GaussianProcess(kernel, x, diag=noise, mean=mean)
        gp_cond = gp.condition(data, nx).gp
        realisations = gp_cond.sample(rng_key,(num_posterior_pred_realisations,))
        realisations_list.append(realisations)
    return(np.array(realisations_list))