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
mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,ox=ox,cdata=cdata,odata=odata)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
rng_key, rng_key_ = random.split(rng_key)
Y = GP.sample(rng_key)
Y2 = GP2.sample(rng_key)

ox = X[mask][::20]
odata = Y[mask][::20]
cx = X[::10] 
cdata = (Y+Y2)[::10] 

plot_underlying_data(X,Y,Y2,ox,odata,cx,cdata,(20,10))
plt.show()
mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,ox=ox,cdata=cdata,odata=odata)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
rng_key, rng_key_ = random.split(rng_key)
Y = GP.sample(rng_key)
Y2 = GP2.sample(rng_key)

ox = X[mask][::20]
odata = Y[mask][::20]
cx = X[::10] 
cdata = (Y+Y2)[::10] 

plot_underlying_data(X,Y,Y2,ox,odata,cx,cdata,(20,10))
plt.show()
mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,ox=ox,cdata=cdata,odata=odata)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)