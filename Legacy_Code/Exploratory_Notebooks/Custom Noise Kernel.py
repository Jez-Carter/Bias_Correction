# %%
#Importing Packages
import numpy as np
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import arviz as az
from src.model_fitting_functions import run_inference
from src.examples_functions import plot_underlying_data,tinygp_2process_model

import numpyro
import numpyro.distributions as dist

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
#Generating Underlying Process Data 
X = jnp.arange(0,120,0.1)
#Truth
GP = GaussianProcess(1 * kernels.ExpSquared(1),X,diag=1e-5,mean=1.0)
#Bias
GP2 = GaussianProcess(1 * kernels.ExpSquared(5),X,diag=1e-5,mean=-1.0)

mask = np.ones(len(X),dtype='bool')
mask[slice(int(len(X)/3),int(len(X)*2/3))]=False

# %%

Y = GP.sample(rng_key)
Y2 = GP2.sample(rng_key)

ox = X[mask][::20]
odata = Y[mask][::20]
cx = X[::10] 
cdata = (Y+Y2)[::10] 

plot_underlying_data(X,Y,Y2,ox,odata,cx,cdata,(20,10))
plt.show()

# %%
import tinygp

class Noise(tinygp.kernels.Kernel):
    def __init__(self, noise):
        self.noise = noise

    def evaluate(self, X1, X2):
        return(jnp.where(X1==X2,self.noise,0))

# %%

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
    kernel = kern_var * kernels.ExpSquared(lengthscale) + Noise(1e-5)
    mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
    gp = GaussianProcess(kernel, ox, mean=mean)
    numpyro.sample("obs_temperature", gp.numpyro_dist(),obs=odata)

    bkern_var = numpyro.sample("bkern_var", dist.Gamma(3.0,0.5))
    blengthscale = numpyro.sample("blengthscale", dist.Gamma(3.0,0.5))
    bkernel = bkern_var * kernels.ExpSquared(blengthscale) + Noise(1e-5)
    bmean = numpyro.sample("bmean", dist.Normal(0.0, 2.0))

    ckernel = kernel+bkernel
    cgp = GaussianProcess(ckernel, cx, mean=mean+bmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=cdata)

# %%

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,ox=None,cdata=cdata,odata=None)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

