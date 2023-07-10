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

def tinygp_model(x,data=None,noise=None):
    """
    Example model where the data is generated from a GP.
    Args:x (jax device array): array of coordinates for data, shape [#points,dimcoords]
    data (jax device array): array of data values, shape [#points,]
    """
    kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
    # lengthscale = numpyro.sample("lengthscale", dist.Gamma(0.001,0.001))
    lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
    gp = GaussianProcess(kernel, x, diag=noise, mean=mean)
    numpyro.sample("data", gp.numpyro_dist(),obs=data)

# class Noise(kernels.Kernel):
#     def __init__(self, noise):
#         self.noise = noise

#     def evaluate(self, X1, X2):
#         return(jnp.where((X1==X2).all(),self.noise,0))

def tinygp_2process_model(cx,ox=None,cdata=None,odata=None,onoise=None,cnoise=None):
    """
   Example model where the climate data is generated from 2 GPs, one of which also generates the observations and one of which generates bias in the climate model.
    Args:
        cx (jax device array): array of coordinates for climate model, shape [#gridcells,dimcoords]
        ox (jax device array): array of coordinates for observations, shape [#sites,dimcoords]
        cdata (jax device array): array of data values for climate model, shape [#gridcells,]
        odata (jax device array): array of data values for climateervations, shape [#sites,]
    """

    kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
    # lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
    lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.2))
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
    gp = GaussianProcess(kernel, ox, diag=onoise, mean=mean)
    numpyro.sample("obs_temperature", gp.numpyro_dist(),obs=odata)

    bkern_var = numpyro.sample("bkern_var", dist.Gamma(3.0,0.5))
    # blengthscale = numpyro.sample("blengthscale", dist.Gamma(3.0,0.5))
    blengthscale = numpyro.sample("blengthscale", dist.Gamma(3.0,0.2))
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", dist.Normal(0.0, 2.0))

    ckernel = kernel+bkernel
    cgp = GaussianProcess(ckernel, cx, diag=cnoise, mean=mean+bmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=cdata)

def hierarchical_model(cx,ox=None,cdata=None,odata=None,
                       om_noise=None,cm_noise=None,olv_noise=None,clv_noise=None):
    """
   Example model where the climate data is generated from 2 GPs, one of which also generates the observations and one of which generates bias in the climate model.
    Args:
        cx (jax device array): array of coordinates for climate model, shape [#gridcells,dimcoords]
        ox (jax device array): array of coordinates for observations, shape [#sites,dimcoords]
        cdata (jax device array): array of data values for climate model, shape [#gridcells,]
        odata (jax device array): array of data values for observations, shape [#sites,]
        om_noise (float): value to allow noise in the GP fit for a realisation of the mean for the observations 
        cm_noise (float): value to allow noise in the GP fit for a realisation of the mean for the climate data 
        olv_noise (float): value to allow noise in the GP fit for a realisation of the log variance for the observations 
        clv_noise (float): value to allow noise in the GP fit for a realisation of the log variance for the climate data 
    Variable name meanings:
        t = true underlying data
        mt = the mean of the true underlying data
        vt = the variance of the true underlying data
        lvt = the log variance of the true underlying data
        ... same but for b = ... bias data
        ... same but for c = ... climate data
    """

    #GP that generates the mean of the truth 
    mt_kern_var = numpyro.sample("mt_kern_var", dist.Gamma(3.0,0.5))
    mt_lengthscale = numpyro.sample("mt_lengthscale", dist.Gamma(3.0,0.5))
    mt_kernel = mt_kern_var * kernels.ExpSquared(mt_lengthscale)
    mt_mean = numpyro.sample("mt_mean", dist.Normal(0.0, 2.0))
    mt_gp = GaussianProcess(mt_kernel, ox, diag=om_noise, mean=mt_mean)
    mt = numpyro.sample("mt", mt_gp.numpyro_dist())

    mb_kern_var = numpyro.sample("mb_kern_var", dist.Gamma(3.0,0.5))
    mb_lengthscale = numpyro.sample("mb_lengthscale", dist.Gamma(3.0,0.1))
    mb_kernel = mb_kern_var * kernels.ExpSquared(mb_lengthscale)
    mb_mean = numpyro.sample("mb_mean", dist.Normal(0.0, 2.0))

    mc_kernel = mt_kernel+mb_kernel
    mc_gp = GaussianProcess(mc_kernel, cx, diag=cm_noise, mean=mt_mean+mb_mean)
    mc = numpyro.sample("mc", mc_gp.numpyro_dist())

    lvt_kern_var = numpyro.sample("lvt_kern_var", dist.Gamma(3.0,0.5))
    lvt_lengthscale = numpyro.sample("lvt_lengthscale", dist.Gamma(3.0,0.5))
    lvt_kernel = lvt_kern_var * kernels.ExpSquared(lvt_lengthscale)
    lvt_mean = numpyro.sample("lvt_mean", dist.Normal(0.0, 2.0))
    lvt_gp = GaussianProcess(lvt_kernel, ox, diag=olv_noise, mean=lvt_mean)
    lvt = numpyro.sample("lvt", lvt_gp.numpyro_dist())

    lvb_kern_var = numpyro.sample("lvb_kern_var", dist.Gamma(3.0,0.5))
    lvb_lengthscale = numpyro.sample("lvb_lengthscale", dist.Gamma(3.0,0.1))
    lvb_kernel = lvb_kern_var * kernels.ExpSquared(lvb_lengthscale)
    lvb_mean = numpyro.sample("lvb_mean", dist.Normal(0.0, 2.0))

    lvc_kernel = lvt_kernel+lvb_kernel
    lvc_gp = GaussianProcess(lvc_kernel, cx, diag=clv_noise, mean=lvt_mean+lvb_mean)
    lvc = numpyro.sample("lvc", lvc_gp.numpyro_dist())

    vt = jnp.exp(lvt)
    vc = jnp.exp(lvc)

    numpyro.sample("t", dist.Normal(mt, jnp.sqrt(vt)),obs=odata)
    numpyro.sample("c", dist.Normal(mc, jnp.sqrt(vc)),obs=cdata)

def hierarchical_model_mask(cx,ox=None,cdata=None,odata=None,obs_mask=None,
                       om_noise=None,cm_noise=None,olv_noise=None,clv_noise=None):
    """
   Example model where the climate data is generated from 2 GPs, one of which also generates the observations and one of which generates bias in the climate model.
    Args:
        cx (jax device array): array of coordinates for climate model, shape [#gridcells,dimcoords]
        ox (jax device array): array of coordinates for observations, shape [#sites,dimcoords]
        cdata (jax device array): array of data values for climate model, shape [#gridcells,]
        odata (jax device array): array of data values for observations, shape [#sites,]
        om_noise (float): value to allow noise in the GP fit for a realisation of the mean for the observations 
        cm_noise (float): value to allow noise in the GP fit for a realisation of the mean for the climate data 
        olv_noise (float): value to allow noise in the GP fit for a realisation of the log variance for the observations 
        clv_noise (float): value to allow noise in the GP fit for a realisation of the log variance for the climate data 
    Variable name meanings:
        t = true underlying data
        mt = the mean of the true underlying data
        vt = the variance of the true underlying data
        lvt = the log variance of the true underlying data
        ... same but for b = ... bias data
        ... same but for c = ... climate data
    """

    #GP that generates the mean of the truth 
    mt_kern_var = numpyro.sample("mt_kern_var", dist.Gamma(3.0,1.0))
    mt_lengthscale = numpyro.sample("mt_lengthscale", dist.Gamma(3.0,0.5))
    mt_kernel = mt_kern_var * kernels.ExpSquared(mt_lengthscale) 
    mt_mean = numpyro.sample("mt_mean", dist.Normal(-7.0, 3.0))
    mt_gp = GaussianProcess(mt_kernel, ox, diag=om_noise, mean=mt_mean)
    mt = numpyro.sample("mt", mt_gp.numpyro_dist())

    mb_kern_var = numpyro.sample("mb_kern_var", dist.Gamma(3.0,1.0))
    mb_lengthscale = numpyro.sample("mb_lengthscale", dist.Gamma(3.0,0.5))
    mb_kernel = mb_kern_var * kernels.ExpSquared(mb_lengthscale)
    mb_mean = numpyro.sample("mb_mean", dist.Normal(-2.0, 2.0))

    mc_kernel = mt_kernel+mb_kernel
    mc_gp = GaussianProcess(mc_kernel, cx, diag=cm_noise, mean=mt_mean+mb_mean)
    mc = numpyro.sample("mc", mc_gp.numpyro_dist())

    lvt_kern_var = numpyro.sample("lvt_kern_var", dist.Gamma(3.0,1.0))
    lvt_lengthscale = numpyro.sample("lvt_lengthscale", dist.Gamma(3.0,0.5))
    lvt_kernel = lvt_kern_var * kernels.ExpSquared(lvt_lengthscale)
    lvt_mean = numpyro.sample("lvt_mean", dist.Normal(2.0, 1.0))
    lvt_gp = GaussianProcess(lvt_kernel, ox, diag=olv_noise, mean=lvt_mean)
    lvt = numpyro.sample("lvt", lvt_gp.numpyro_dist())

    lvb_kern_var = numpyro.sample("lvb_kern_var", dist.Gamma(3.0,1.0))
    lvb_lengthscale = numpyro.sample("lvb_lengthscale", dist.Gamma(3.0,0.5))
    lvb_kernel = lvb_kern_var * kernels.ExpSquared(lvb_lengthscale)
    lvb_mean = numpyro.sample("lvb_mean", dist.Normal(0.0, 1.0))

    lvc_kernel = lvt_kernel+lvb_kernel
    lvc_gp = GaussianProcess(lvc_kernel, cx, diag=clv_noise, mean=lvt_mean+lvb_mean)
    lvc = numpyro.sample("lvc", lvc_gp.numpyro_dist())

    vt = jnp.exp(lvt)
    vc = jnp.exp(lvc)

    t_masked_dist = dist.Normal(mt, jnp.sqrt(vt)).mask(obs_mask)
    numpyro.sample("t", t_masked_dist,obs=odata)
    numpyro.sample("c", dist.Normal(mc, jnp.sqrt(vc)),obs=cdata)

# def hierarchical_model_mask(cx,ox=None,cdata=None,odata=None,noise=None,obs_mask=None):
#     """
#    Example model where the climate data is generated from 2 GPs, one of which also generates the observations and one of which generates bias in the climate model.
#     Args:
#         cx (jax device array): array of coordinates for climate model, shape [#gridcells,dimcoords]
#         ox (jax device array): array of coordinates for observations, shape [#sites,dimcoords]
#         cdata (jax device array): array of data values for climate model, shape [#gridcells,]
#         odata (jax device array): array of data values for observations, shape [#sites,]
#     Variable name meanings:
#         t = true underlying data
#         mt = the mean of the true underlying data
#         vt = the variance of the true underlying data
#         lvt = the log variance of the true underlying data
#         ... same but for b = ... bias data
#         ... same but for c = ... climate data
#     """

#     #GP that generates the mean of the truth 
#     mt_kern_var = numpyro.sample("mt_kern_var", dist.Gamma(0.001,0.001))
#     mt_lengthscale = numpyro.sample("mt_lengthscale", dist.Gamma(0.001,0.001))
#     mt_kernel = mt_kern_var * kernels.ExpSquared(mt_lengthscale) + Noise(noise)
#     mt_mean = numpyro.sample("mt_mean", dist.Normal(-6.0, 10.0))
#     mt_gp = GaussianProcess(mt_kernel, ox, diag=1e-5, mean=mt_mean)
#     mt = numpyro.sample("mt", mt_gp.numpyro_dist())

#     mb_kern_var = numpyro.sample("mb_kern_var", dist.Gamma(0.001,0.001))
#     mb_lengthscale = numpyro.sample("mb_lengthscale", dist.Gamma(0.001,0.001))
#     mb_kernel = mb_kern_var * kernels.ExpSquared(mb_lengthscale) + Noise(noise)
#     mb_mean = numpyro.sample("mb_mean", dist.Normal(0.0, 10.0))

#     mc_kernel = mt_kernel+mb_kernel
#     mc_gp = GaussianProcess(mc_kernel, cx, mean=mt_mean+mb_mean)
#     mc = numpyro.sample("mc", mc_gp.numpyro_dist())

#     lvt_kern_var = numpyro.sample("lvt_kern_var", dist.Gamma(0.001,0.001))
#     lvt_lengthscale = numpyro.sample("lvt_lengthscale", dist.Gamma(0.001,0.001))
#     lvt_kernel = lvt_kern_var * kernels.ExpSquared(lvt_lengthscale) + Noise(noise)
#     lvt_mean = numpyro.sample("lvt_mean", dist.Normal(2.0, 5.0))
#     lvt_gp = GaussianProcess(lvt_kernel, ox, diag=1e-5, mean=lvt_mean)
#     lvt = numpyro.sample("lvt", lvt_gp.numpyro_dist())

#     lvb_kern_var = numpyro.sample("lvb_kern_var", dist.Gamma(0.001,0.001))
#     lvb_lengthscale = numpyro.sample("lvb_lengthscale", dist.Gamma(0.001,0.001))
#     lvb_kernel = lvb_kern_var * kernels.ExpSquared(lvb_lengthscale) + Noise(noise)
#     lvb_mean = numpyro.sample("lvb_mean", dist.Normal(0.0, 5.0))

#     lvc_kernel = lvt_kernel+lvb_kernel
#     lvc_gp = GaussianProcess(lvc_kernel, cx, mean=lvt_mean+lvb_mean)
#     lvc = numpyro.sample("lvc", lvc_gp.numpyro_dist())

#     vt = jnp.exp(lvt)
#     vc = jnp.exp(lvc)

#     t_masked_dist = dist.Normal(mt, jnp.sqrt(vt)).mask(obs_mask)
#     numpyro.sample("t", t_masked_dist,obs=odata)
#     numpyro.sample("c", dist.Normal(mc, jnp.sqrt(vc)),obs=cdata)

def diagonal_noise(coord,noise):
    return(np.diag(np.full(coord.shape[0],noise)))

def truth_posterior_predictive(nx,ox,cx,odata,cdata,omean,bmean,kernelo,kernelb,onoise,cnoise,jitter):
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

def bias_posterior_predictive(nx,ox,cx,odata,cdata,omean,bmean,kernelo,kernelb,onoise,cnoise,jitter):
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
        posterior_pred_func,nx,ox,cx,odata,cdata,idata,onoise,cnoise,jitter,
        num_parameter_realisations,num_posterior_pred_realisations):
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
        postpred = posterior_pred_func(nx,ox,cx,odata,cdata,mean,bmean,okernel,bkernel,onoise,cnoise,jitter)
        realisations = postpred.rvs(num_posterior_pred_realisations)
        realisations_list.append(realisations)
    return(np.array(realisations_list))

def realisations_2process(nx,ox,cx,odata,cdata,idata_2process,onoise,cnoise,jitter,npr,nppr):
    # npr = num_parameter_realisations
    # nppr = num_posterior_pred_realisations 

    truth_realisations = posterior_predictive_realisations(
        truth_posterior_predictive,
        nx,ox,cx,odata,cdata,idata_2process,
        onoise,cnoise,jitter,
        npr,nppr)

    bias_realisations = posterior_predictive_realisations(
        bias_posterior_predictive,
        nx,ox,cx,odata,cdata,idata_2process,
        onoise,cnoise,jitter,
        npr,nppr)
      
    truth_realisations = truth_realisations.reshape(-1,truth_realisations.shape[-1])
    bias_realisations = bias_realisations.reshape(-1,bias_realisations.shape[-1])

    return(truth_realisations,bias_realisations)

def posterior_predictive_realisations_hierarchical_mean(
        posterior_pred_func,nx,ox,cx,idata,
        om_noise,cm_noise,jitter,
        num_parameter_realisations,num_posterior_pred_realisations):
    posterior = idata.posterior
    #parameter realisations for process W
    mt_kern_vars = posterior['mt_kern_var'].data[0,:]
    mt_lengthscales = posterior['mt_lengthscale'].data[0,:]
    mt_means = posterior['mt_mean'].data[0,:]
    #parameter realisations for process D
    mb_kern_vars = posterior['mb_kern_var'].data[0,:]
    mb_lengthscales = posterior['mb_lengthscale'].data[0,:]
    mb_means = posterior['mb_mean'].data[0,:]

    mts = posterior['mt'].data[0,:]
    mcs = posterior['mc'].data[0,:]

    realisations_list = []
    for mt_kern_var,mt_lengthscale,mt_mean,mb_kern_var,mb_lengthscale,mb_mean,mt,mc in tqdm(list(zip(mt_kern_vars,mt_lengthscales,mt_means,mb_kern_vars,mb_lengthscales,mb_means,mts,mcs))[:num_parameter_realisations]):
        mt_kernel = mt_kern_var * kernels.ExpSquared(mt_lengthscale)
        mb_kernel = mb_kern_var * kernels.ExpSquared(mb_lengthscale)
        postpred = posterior_pred_func(
            nx,ox,cx,mt,mc,mt_mean,mb_mean,mt_kernel,mb_kernel,om_noise,cm_noise,jitter)
        realisations = postpred.rvs(num_posterior_pred_realisations)
        realisations_list.append(realisations)
    return(np.array(realisations_list))

def posterior_predictive_realisations_hierarchical_var(
        posterior_pred_func,nx,ox,cx,idata,
        olv_noise,clv_noise,jitter,
        num_parameter_realisations,num_posterior_pred_realisations):
    posterior = idata.posterior
    #parameter realisations for process W
    lvt_kern_vars = posterior['lvt_kern_var'].data[0,:]
    lvt_lengthscales = posterior['lvt_lengthscale'].data[0,:]
    lvt_means = posterior['lvt_mean'].data[0,:]
    #parameter realisations for process D
    lvb_kern_vars = posterior['lvb_kern_var'].data[0,:]
    lvb_lengthscales = posterior['lvb_lengthscale'].data[0,:]
    lvb_means = posterior['lvb_mean'].data[0,:]

    lvts = posterior['lvt'].data[0,:]
    lvcs = posterior['lvc'].data[0,:]

    realisations_list = []
    for lvt_kern_var,lvt_lengthscale,lvt_mean,lvb_kern_var,lvb_lengthscale,lvb_mean,lvt,lvc in tqdm(list(zip(lvt_kern_vars,lvt_lengthscales,lvt_means,lvb_kern_vars,lvb_lengthscales,lvb_means,lvts,lvcs))[:num_parameter_realisations]):
        lvt_kernel = lvt_kern_var * kernels.ExpSquared(lvt_lengthscale) 
        lvb_kernel = lvb_kern_var * kernels.ExpSquared(lvb_lengthscale)
        postpred = posterior_pred_func(
            nx,ox,cx,lvt,lvc,lvt_mean,lvb_mean,lvt_kernel,lvb_kernel,olv_noise,clv_noise,jitter)
        realisations = postpred.rvs(num_posterior_pred_realisations)
        realisations_list.append(realisations)
    return(np.array(realisations_list))

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

def plot_underlying_data(X,Y,Y2,ox,odata,cx,cdata,fs,ms):
    plt.figure(figsize=fs)
    plt.plot(X,Y,label='Truth',alpha=0.6)
    plt.plot(X,Y2,label='Bias',alpha=0.6)
    plt.plot(X,Y+Y2,label='Bias+Truth',alpha=0.6)

    plt.scatter(ox,odata,label='Observations',alpha=0.8,s=ms)
    plt.scatter(cx,cdata,color='g',label='Climate data',alpha=0.8,s=ms)

    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.legend()

def plotting_output_2process(X,Y,Y2,ox,odata,cx,cdata,truth_realisations,bias_realisations,nx,
                             fs=None,ms=None,ylims=None):
    plt.figure(figsize=fs)
    plt.plot(X,Y,label='Truth',alpha=1.0,linewidth=2)
    plt.plot(X,Y2,label='Bias',alpha=1.0,linewidth=2)
    plt.plot(X,Y+Y2,label='Bias+Truth',alpha=1.0,linewidth=2)
    plt.scatter(ox,odata,color='b',label='Observational data',alpha=0.7,s=ms)
    plt.scatter(cx,cdata,color='g',label='Climate data',alpha=0.7,s=ms)

    truth_pred_mean = truth_realisations.mean(axis=0)
    truth_pred_std = truth_realisations.std(axis=0)
    plt.plot(nx,truth_pred_mean,label='Truth Mean',color='m',alpha=1.0,linewidth=2)
    plt.fill_between(nx,truth_pred_mean+truth_pred_std,truth_pred_mean-truth_pred_std,label='Truth StdDev',color='m',alpha=0.3)
    bias_pred_mean = bias_realisations.mean(axis=0)
    bias_pred_std = bias_realisations.std(axis=0)
    plt.plot(nx,bias_pred_mean,label='Bias Mean',color='r',alpha=1.0,linewidth=2)
    plt.fill_between(nx,bias_pred_mean+bias_pred_std,bias_pred_mean-bias_pred_std,label='Bias StdDev',color='r',alpha=0.3)

    plt.ylim(ylims[0],ylims[1])
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.legend()

def create_levels(ds,sep,rounding,center=None):
    vmin = min(ds.min().data_vars.values())
    vmax = max(ds.max().data_vars.values())
    abs_max_rounded = max(np.abs(vmin),vmax).round(rounding)

    if center != None:
        levels = np.arange(-abs_max_rounded, abs_max_rounded+sep, sep)
    else:
        levels = np.arange(vmin.round(rounding), vmax.round(rounding)+sep, sep)
    return(levels)