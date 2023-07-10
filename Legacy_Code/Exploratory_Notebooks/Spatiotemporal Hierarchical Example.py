# %%
#Importing Packages
import numpy as np
import xarray as xr
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import arviz as az
from src.model_fitting_functions import run_inference
from src.examples_functions import hierarchical_model,truth_posterior_predictive,bias_posterior_predictive
from src.examples_functions import posterior_predictive_realisations_hierarchical_mean, posterior_predictive_realisations_hierarchical_var

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(5)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
#Creating Underlying Process Data 

X1 = np.arange(0,105,2)
X2 = np.arange(0,105,2)
D = np.arange(0,100,1)

ds = xr.Dataset(
    coords=dict(
        X1=("X1", X1),
        X2=("X2", X2),
        D=("D", D),
    ),
)

ds_stacked = ds.stack(X=('X1', 'X2'))
X = np.array(list(map(np.array, ds_stacked.X.data)))

# %%

#Latent mean and variance for truth and bias processes
GP_T_MEAN = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5,mean=1.0)
GP_T_LOGVAR = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5,mean=-1.0)
GP_B_MEAN = GaussianProcess(1 * kernels.ExpSquared(40),X,diag=1e-5,mean=-1.0)
GP_B_LOGVAR = GaussianProcess(1 * kernels.ExpSquared(40),X,diag=1e-5,mean=-1.0)

MEAN_T = GP_T_MEAN.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
LOGVAR_T = GP_T_LOGVAR.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
MEAN_B = GP_B_MEAN.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
LOGVAR_B = GP_B_LOGVAR.sample(rng_key)
MEAN_C = MEAN_T + MEAN_B
LOGVAR_C = LOGVAR_T + LOGVAR_B

ds_stacked["MEAN_T"]=(['X'],  MEAN_T)
ds_stacked["LOGVAR_T"]=(['X'],  LOGVAR_T)
ds_stacked["VAR_T"]=(['X'],  jnp.exp(LOGVAR_T))
ds_stacked["MEAN_B"]=(['X'],  MEAN_B)
ds_stacked["LOGVAR_B"]=(['X'],  LOGVAR_B)
ds_stacked["VAR_B"]=(['X'],  jnp.exp(LOGVAR_B))
ds_stacked["MEAN_C"]=(['X'],  MEAN_C)
ds_stacked["LOGVAR_C"]=(['X'],  LOGVAR_C)
ds_stacked["VAR_C"]=(['X'],  jnp.exp(LOGVAR_C))
ds = ds_stacked.unstack()

# %%
ds_stacked['MEAN_T'].data.reshape(-1,1).shape

# %%

def temporal_GP_samples(mean,kern_var,rng_key):
    rng_key, rng_key_ = random.split(rng_key)
    GP_T = GaussianProcess(kern_var * kernels.ExpSquared(10),D,diag=1e-5,mean=mean)
    T = GP_T.sample(rng_key)
    return(T)

# %%
temporal_GP_samples(1.0,1.0,rng_key)

# %%
from jax import vmap,pmap

# %%
a = np.array([[-2.0,2.0],[-2.0,2.0]])
b = np.array([[0.0,0.0],[0.0,0.0]])
vmap(temporal_GP_samples,(0,0,None),0)(a,b,rng_key).shape
pmap(temporal_GP_samples,(0,0),0)(a,b,rng_key).shape
# vmap(temporal_GP_samples)(a,b,rng_key).shape

#Could use pmap or could stick to vmap and use stacked array


# %%

GP_T = GaussianProcess(ds_stacked['VAR_T'].data[0] * kernels.ExpSquared(10),X,diag=1e-5,mean=1.0)



# GP_T = GaussianProcess(ds_stacked['VAR_T'].data * kernels.ExpSquared(10),X,diag=1e-5,mean=ds_stacked['MEAN_T'].data.ravel())


# Normal_T = dist.Normal(ds.MEAN_T.data, jnp.sqrt(ds.VAR_T.data))
# T = Normal_T.sample(rng_key,ds.D.shape)

# Normal_C = dist.Normal(ds.MEAN_C.data, jnp.sqrt(ds.VAR_C.data))
# C = Normal_C.sample(rng_key,ds.D.shape)

# ds["T"]=(['D','X1','X2'],  T)
# ds["C"]=(['D','X1','X2'],  C)