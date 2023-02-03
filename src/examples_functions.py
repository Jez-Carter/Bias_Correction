from tqdm import tqdm
import numpy as np
import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
from scipy.stats import multivariate_normal

from jax import random
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def tinygp_model(x,data=None):
	"""
	Example model where the data is generated from a GP.
	Args:x (jax device array): array of coordinates for data, shape [#points,dimcoords]
	data (jax device array): array of data values, shape [#points,]
	"""
	kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
	lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
	kernel = kern_var * kernels.ExpSquared(lengthscale)
	mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
	gp = GaussianProcess(kernel, x, diag=1e-5, mean=mean)
	numpyro.sample("data", gp.numpyro_dist(),obs=data)

def tinygp_2process_model(cx,ox=None,cdata=None,odata=None):
    """
   Example model where the climate data is generated from 2 GPs, one of which also generates the observations and one of which generates bias in the climate model.
    Args:
        cx (jax device array): array of coordinates for climate model, shape [#gridcells,dimcoords]
        ox (jax device array): array of coordinates for observations, shape [#sites,dimcoords]
        cdata (jax device array): array of data values for climate model, shape [#gridcells,]
        odata (jax device array): array of data values for climateervations, shape [#sites,]
    """

    #GP that generates the truth (& so the climateervations directly)
    kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
    lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
    gp = GaussianProcess(kernel, ox, diag=1e-5, mean=mean)
    numpyro.sample("obs_temperature", gp.numpyro_dist(),obs=odata)

    bkern_var = numpyro.sample("bkern_var", dist.Gamma(3.0,0.5))
    blengthscale = numpyro.sample("blengthscale", dist.Gamma(3.0,0.5))
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", dist.Normal(0.0, 2.0))

    ckernel = kernel+bkernel
    cgp = GaussianProcess(ckernel, cx, diag=1e-5, mean=mean+bmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=cdata)

def diagonal_noise(square_matrix,noise):
    return(np.diag(np.full(square_matrix.shape[0],noise)))

def truth_posterior_predictive(nx,ox,cx,odata,cdata,omean,bmean,kernelo,kernelb):
    y2 = np.hstack([odata,cdata]) 
    u1 = np.full(nx.shape, omean)
    u2 = np.hstack([np.full(ox.shape, omean),np.full(cx.shape, omean+bmean)])
    k11 = kernelo(nx,nx)
    k11 = k11+diagonal_noise(k11,1e-5)
    k12 = np.hstack([kernelo(nx,ox),kernelo(nx,cx)])
    k21 = np.vstack([kernelo(ox,nx),kernelo(cx,nx)])
    k22_upper = np.hstack([kernelo(ox,ox),kernelo(ox,cx)])
    k22_lower = np.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)])
    k22 = np.vstack([k22_upper,k22_lower])
    k22 = k22+diagonal_noise(k22,1e-5)
    k22i = np.linalg.inv(k22)

    u1g2 = u1 + np.matmul(np.matmul(k12,k22i),y2-u2)

    l22 = np.linalg.cholesky(k22)
    l22i = np.linalg.inv(l22)
    p21 = np.matmul(l22i,k21)
    k1g2 = k11 - np.matmul(p21.T,p21)

    mvn = multivariate_normal(u1g2,k1g2)
    return(mvn)

def bias_posterior_predictive(nx,ox,cx,odata,cdata,omean,bmean,kernelo,kernelb):
    y2 = np.hstack([odata,cdata]) 
    u1 = np.full(nx.shape, bmean)
    u2 = np.hstack([np.full(ox.shape, omean),np.full(cx.shape, omean+bmean)])
    k11 = kernelb(nx,nx)
    k11 = k11+diagonal_noise(k11,1e-5)
    k12 = np.hstack([np.full((len(nx),len(ox)),0),kernelb(nx,cx)])
    k21 = np.vstack([np.full((len(ox),len(nx)),0),kernelb(cx,nx)])
    k22_upper = np.hstack([kernelo(ox,ox),kernelo(ox,cx)])
    k22_lower = np.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)])
    k22 = np.vstack([k22_upper,k22_lower])
    k22 = k22+diagonal_noise(k22,1e-5)
    k22i = np.linalg.inv(k22)

    u1g2 = u1 + np.matmul(np.matmul(k12,k22i),y2-u2)

    l22 = np.linalg.cholesky(k22)
    l22i = np.linalg.inv(l22)
    p21 = np.matmul(l22i,k21)
    k1g2 = k11 - np.matmul(p21.T,p21)

    mvn = multivariate_normal(u1g2,k1g2)
    return(mvn)

def posterior_predictive_realisations(posterior_pred_func,nx,ox,cx,odata,cdata,idata,num_parameter_realisations,num_posterior_pred_realisations):
    posterior = idata.posterior
    #parameter realisations for process W
    kern_vars = posterior['kern_var'].data[0,:]
    lengthscales = posterior['lengthscale'].data[0,:]
    means = posterior['mean'].data[0,:]
    #parameter realisations for process D
    bkern_vars = posterior['bkern_var'].data[0,:]
    blengthscales = posterior['blengthscale'].data[0,:]
    bmeans = posterior['bmean'].data[0,:]

    realisations_list = []
    for kern_var,lengthscale,mean,bkern_var,blengthscale,bmean in tqdm(list(zip(kern_vars,lengthscales,means,bkern_vars,blengthscales,bmeans))[:num_parameter_realisations]):
        okernel = kern_var * kernels.ExpSquared(lengthscale)
        bkernel = bkern_var * kernels.ExpSquared(blengthscale)
        postpred = posterior_pred_func(nx,ox,cx,odata,cdata,mean,bmean,okernel,bkernel)
        realisations = postpred.rvs(num_posterior_pred_realisations)
        realisations_list.append(realisations)
    return(np.array(realisations_list))

def singleprocess_posterior_predictive_realisations(nx,x,idata,num_parameter_realisations,num_posterior_pred_realisations):
    data = idata.observed_data.data.data
    posterior = idata.posterior
    #parameter realisations 
    kern_vars = posterior['kern_var'].data[0,:]
    lengthscales = posterior['lengthscale'].data[0,:]
    means = posterior['mean'].data[0,:]

    realisations_list = []
    for kern_var,lengthscale,mean in tqdm(list(zip(kern_vars,lengthscales,means))[:num_parameter_realisations]):
        kernel = kern_var * kernels.ExpSquared(lengthscale)
        gp = GaussianProcess(kernel, x, diag=1e-5, mean=mean)
        gp_cond = gp.condition(data, nx).gp
        realisations = gp_cond.sample(rng_key,(num_posterior_pred_realisations,))
        realisations_list.append(realisations)
    return(np.array(realisations_list))