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
from src.examples_functions import tinygp_model

import tinygp
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
#Climate
GP3 = GaussianProcess(1 * kernels.ExpSquared(1)+kernels.ExpSquared(5),X,diag=2e-5,mean=0.0)

mask = np.ones(len(X),dtype='bool')
mask[slice(int(len(X)/3),int(len(X)*2/3))]=False

# %%

Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y3 = GP3.sample(rng_key)

cx = X[::10]  

# %%
# plt.plot(X,Y,label='Y')
# plt.plot(X,Y2,label='Y2')
plt.plot(X,Y3,label='Y3')
plt.plot(X,Y+Y2,label='Y+Y2')
plt.legend()

# %%
class Noise(tinygp.kernels.Kernel):
    def __init__(self, noise):
        self.noise = noise

    def evaluate(self, X1, X2):
        return(jnp.where(X1==X2,self.noise,0))

def tinygp_2process_model(cx,cdata=None):
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

    bkern_var = numpyro.sample("bkern_var", dist.Gamma(3.0,0.5))
    blengthscale = numpyro.sample("blengthscale", dist.Gamma(3.0,0.5))
    bkernel = bkern_var * kernels.ExpSquared(blengthscale) + Noise(1e-5)
    bmean = numpyro.sample("bmean", dist.Normal(0.0, 2.0))

    ckernel = kernel+bkernel
    cgp = GaussianProcess(ckernel, cx, mean=mean+bmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=cdata)

# %%
cdata = (Y+Y2)[::10] 
mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)
# az.plot_trace(idata_test)

# %%
cdata_y3 = Y3[::10]
mcmc_2process_y3 = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata_y3)
idata_test_y3 = az.from_numpyro(mcmc_2process_y3)
az.summary(idata_test_y3,hdi_prob=0.95)

# %%
#Test on just bias data:
bx = X[::10] 
bdata = Y2[::10] 

mcmc_1process = run_inference(tinygp_model, rng_key_, 1000, 2000, bx,data=bdata)
idata_test = az.from_numpyro(mcmc_1process)
az.summary(idata_test,hdi_prob=0.95)


# %%
bdata = (Y2)[::10] 
mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, bx,cdata=bdata)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)