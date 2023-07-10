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

idata = az.from_numpyro(
    mcmc,
    coords={"days": np.arange(0, 570, 1),"sites": np.arange(0, 100, 1)},
    dims={"obs": ["days","sites"],"log_alpha": ["sites"], "beta": ["sites"],"logit_p": ["sites"]},
)

parameters = list(idata.posterior.keys())
for param in parameters:
    if 'log_alpha' in param:
        new_param = param.replace("log_alpha", "alpha")
        idata.posterior[new_param]=np.exp(idata.posterior[param])
    elif 'logit_p' in param:
        new_param = param.replace("logit_p", "p")
        data = expit(idata.posterior[param].data)
        dimensions = list(idata.posterior[param].coords)
        idata.posterior[new_param]=(dimensions,data)
        
# Reassigning coordinates to observed data
idata.observed_data = idata.observed_data.assign_coords(
    grid_latitude=("sites", ds["grid_latitude"].data),
    grid_longitude=("sites", ds["grid_longitude"].data),
    latitude=("sites", ds["latitude"].data),
    longitude=("sites", ds["longitude"].data),
    grid_latitude_standardised=(
        "sites",
        ds["grid_latitude_standardised"].data,
    ),
    grid_longitude_standardised=(
        "sites",
        ds["grid_longitude_standardised"].data,
    ),
)

idata.posterior = idata.posterior.assign_coords(
    grid_latitude=("sites", ds["grid_latitude"].data),
    grid_longitude=("sites", ds["grid_longitude"].data),
    latitude=("sites", ds["latitude"].data),
    longitude=("sites", ds["longitude"].data),
    grid_latitude_standardised=(
        "sites",
        ds["grid_latitude_standardised"].data,
    ),
    grid_longitude_standardised=(
        "sites",
        ds["grid_longitude_standardised"].data,
    ),
)

# Saving output
outfile_path = f'{base_path}ProcessedData/AP_Daily_Snowfall_Land_Only_Distributed_Observations_100_BGGP_Fit.nc'
idata.to_netcdf(outfile_path)