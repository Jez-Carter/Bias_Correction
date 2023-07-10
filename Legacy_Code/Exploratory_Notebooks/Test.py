
# %%
#Importing Packages

import jax
from jax import random
import numpyro
import numpyro.distributions as dist
from src.model_fitting_functions import run_inference

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
prior_dist = dist.Normal(3.0,2.0)
data = dist.Normal(3.0,0.5).sample(rng_key,(10000,))
def model(data=None,prior_dist=None):
	mean = numpyro.sample("mean", prior_dist)
	numpyro.sample("obs", dist.Normal(mean,0.5),obs=data)
mcmc = run_inference(model, rng_key, 1000, 2000,data=data,prior_dist=prior_dist)
# %%timeit
data = dist.Normal(3.0,0.5).sample(rng_key,(10000,))
def model(data=None):
	mean = numpyro.sample("mean", dist.Normal(3.0,2.0))
	numpyro.sample("obs", dist.Normal(mean,0.5),obs=data)
mcmc = run_inference(model, rng_key, 1000, 2000,data=data)

