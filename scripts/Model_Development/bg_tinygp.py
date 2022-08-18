'''

'''

import timeit
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from tinygp import kernels, GaussianProcess, transforms
import arviz as az
from src.helper_functions import standardise
from src.model_fitting_functions import run_inference_tinygp
from src.model_fitting_functions import bg_tinygp_model
from src.model_fitting_functions import BernoulliGamma

numpyro.enable_x64()

base_path = '/data/notebooks/jupyterlab-biascorrlab/data/'

ds = xr.open_dataset(f'{base_path}ProcessedData/AP_Daily_Snowfall_Land_Only_Distributed_Observations_100_Reformatted.nc') 

# Create coordinate and data arrays in the right shape for using in Numpyro model
X = jnp.vstack(
    [
        ds.grid_longitude_standardised.data,
        ds.grid_latitude_standardised.data,
    ]
).T  # shape = [sites,2]

j_data = jnp.array(ds.prsn.data)  # JAX array is needed for Numpyro
Y = j_data[:,0,:]*10**5

# Running the GP Model
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
mcmc = run_inference_tinygp(bg_tinygp_model, rng_key_, 1000, 2000, Y, X=X)

bggp_idata = az.from_numpyro(
    mcmc,
    coords={"sites": np.arange(0, 100, 1)},
    dims={"alpha": ["sites"], "beta": ["sites"],"p": ["sites"]},
)

bggp_idata.posterior['alpha']=np.exp(bggp_idata.posterior['log_alpha'])

bggp_idata.posterior['p']=(('chain','draw','sites'),expit(bggp_idata.posterior['logit_p']))

# Saving output
outfile_path = f'{base_path}ProcessedData/AP_Daily_Snowfall_Land_Only_Distributed_Observations_100_BGGP_Fit.nc'
bggp_idata.to_netcdf(outfile_path)