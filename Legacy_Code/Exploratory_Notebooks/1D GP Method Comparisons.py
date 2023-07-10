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
from src.examples_functions import tinygp_model,tinygp_2process_model
from src.examples_functions import realisations_2process
from src.examples_functions import singleprocess_posterior_predictive_realisations
from src.examples_functions import plot_underlying_data,plotting_output_2process

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
def lima_method(ox,odata,nx,npr,nppr):
    # npr = num_parameter_realisations
    # nppr = num_posterior_pred_realisations 
    mcmc = run_inference(
        tinygp_model, rng_key_, 1000, 2000, ox,data=odata,noise=onoise)
    idata = az.from_numpyro(mcmc)
    truth_realisations = singleprocess_posterior_predictive_realisations(
        nx,ox,idata,onoise,npr,nppr)
    truth_realisations = truth_realisations.reshape(-1,truth_realisations.shape[-1])
    return(truth_realisations)

def step_by_step_method(ox,odata,cx,cdata,cnoise,nx,npr,nppr):
    mcmc_climate = run_inference(
        tinygp_model, rng_key_, 1000, 2000, cx,data=cdata,noise=cnoise)
    idata_climate = az.from_numpyro(mcmc_climate)

    climate_realisations = singleprocess_posterior_predictive_realisations(
        ox,cx,idata_climate,cnoise,npr,nppr)
    bdata_realisations = climate_realisations - odata
    bdata_expecation = bdata_realisations.mean(axis=(0,1))
    bnoise = bdata_realisations.std(axis=(0,1)).mean()

    mcmc_bias = run_inference(
        tinygp_model, rng_key_, 1000, 2000, ox,data=bdata_expecation,noise=bnoise)
    idata_bias = az.from_numpyro(mcmc_bias)
    bias_realisations = singleprocess_posterior_predictive_realisations(
        nx,ox,idata_bias,bnoise,npr,nppr)
    
    climate_realisations = singleprocess_posterior_predictive_realisations(
        nx,cx,idata_climate,cnoise,npr,nppr)
    truth_realisations = climate_realisations - bias_realisations

    truth_realisations = truth_realisations.reshape(-1,truth_realisations.shape[-1])
    bias_realisations = bias_realisations.reshape(-1,bias_realisations.shape[-1])

    return(truth_realisations,bias_realisations)

# %%
X = jnp.arange(0,100,0.1)

# %%
# Scenario: Similar Lengthscales, Sparse Observations
onoise=1e-5
bnoise=1e-5
cnoise=1e-5
jitter=1e-5

GP = GaussianProcess(1 * kernels.ExpSquared(3),X,diag=onoise,mean=1.0)
Y = GP.sample(rng_key)

GP2 = GaussianProcess(1 * kernels.ExpSquared(3),X,diag=bnoise,mean=-1.0)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

cx = X[::10] 
cdata = (Y+Y2)[::10] 
osample = np.random.choice(range(X.size), size=int(0.2*cx.size), replace=False)
ox = X[osample]
odata = Y[osample]

plot_underlying_data(X,Y,Y2,ox,odata,cx,cdata,fs=(20,10),ms=20)

mcmc_2process = run_inference(
    tinygp_2process_model, rng_key_, 1000, 2000,
    cx,ox=ox,cdata=cdata,odata=odata,onoise=onoise,cnoise=cnoise)
idata_2process = az.from_numpyro(mcmc_2process)
nx = X[::5] # locations where predictions will be made

truth_realisations,bias_realisations = realisations_2process(
    nx,ox,cx,odata,cdata,idata_2process,onoise,cnoise,jitter,20,20)

plotting_output_2process(X,Y,Y2,ox,odata,cx,cdata,
                         truth_realisations,bias_realisations,nx,
                         fs=(20,10),ms=20,ylims=[-5,5])
plt.title('Full Method')

sbs_truth_realisations,sbs_bias_realisations = step_by_step_method(
    ox,odata,cx,cdata,cnoise,nx,20,20)

plotting_output_2process(X,Y,Y2,ox,odata,cx,cdata,
                         sbs_truth_realisations,sbs_bias_realisations,nx,
                         fs=(20,10),ms=20,ylims=[-5,5])
plt.title('Step-by-step Method')

# %%
# Scenario: Different Lengthscales, Clustered Observations
onoise=1e-5
bnoise=1e-5
cnoise=1e-5
jitter=1e-5

GP = GaussianProcess(1 * kernels.ExpSquared(1),X,diag=onoise,mean=1.0)
Y = GP.sample(rng_key)

GP2 = GaussianProcess(1 * kernels.ExpSquared(5),X,diag=bnoise,mean=-1.0)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

cx = X[::10] 
cdata = (Y+Y2)[::10] 
osample = np.random.choice(range(int(X.size *0.1)), size=int(0.2*cx.size), replace=False)
ox = X[osample]
odata = Y[osample]

plot_underlying_data(X,Y,Y2,ox,odata,cx,cdata,fs=(20,10),ms=20)

mcmc_2process = run_inference(
    tinygp_2process_model, rng_key_, 1000, 2000,
    cx,ox=ox,cdata=cdata,odata=odata,onoise=onoise,cnoise=cnoise)
idata_2process = az.from_numpyro(mcmc_2process)
nx = X[::5] # locations where predictions will be made

truth_realisations,bias_realisations = realisations_2process(
    nx,ox,cx,odata,cdata,idata_2process,onoise,cnoise,jitter,20,20)

plotting_output_2process(X,Y,Y2,ox,odata,cx,cdata,
                         truth_realisations,bias_realisations,nx,
                         fs=(20,10),ms=20,ylims=[-5,5])
plt.title('Full Method')

sbs_truth_realisations,sbs_bias_realisations = step_by_step_method(
    ox,odata,cx,cdata,cnoise,nx,20,20)

plotting_output_2process(X,Y,Y2,ox,odata,cx,cdata,
                         sbs_truth_realisations,sbs_bias_realisations,nx,
                         fs=(20,10),ms=20,ylims=[-5,5])
plt.title('Step-by-step Method')