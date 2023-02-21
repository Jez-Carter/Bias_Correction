# %%
import jax.numpy as jnp
from jax import random
from tinygp import kernels, GaussianProcess
from src.model_fitting_functions import run_inference
import numpyro
import numpyro.distributions as dist
import arviz as az
import matplotlib.pyplot as plt
from src.examples_functions import tinygp_model

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(1)
rng_key, rng_key_ = random.split(rng_key)

# %%
def tinygp_model(x,data=None,noise=None):
	"""
	Example model where the data is generated from a GP.
	Args:x (jax device array): array of coordinates for data, shape [#points,dimcoords]
	data (jax device array): array of data values, shape [#points,]
	"""
	kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
	lengthscale = numpyro.sample("lengthscale", dist.Gamma(0.001,0.001))
	kernel = kern_var * kernels.ExpSquared(lengthscale)
	mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
	gp = GaussianProcess(kernel, x, diag=noise, mean=mean)
	numpyro.sample("data", gp.numpyro_dist(),obs=data)

# %%
x = jnp.arange(0,500,1)
GP = GaussianProcess(1 * kernels.ExpSquared(40),x,diag=1e-5,mean=1.0)
y = GP.sample(rng_key)

plt.plot(x,y)

# %%
mcmc_single = run_inference(tinygp_model, rng_key_, 1000, 2000, x,data=y)
idata_test = az.from_numpyro(mcmc_single)
az.summary(idata_test,hdi_prob=0.95)