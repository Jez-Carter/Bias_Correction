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
import arviz as az
from src.helper_functions import standardise
from src.model_fitting_functions import run_inference
from src.model_fitting_functions import bg_model

numpyro.enable_x64()

base_path = '/content/Bias_Correction/data/'

# ds = xr.open_dataset(f'{base_path}ProcessedData/AP_Daily_Snowfall_Land_Only_Distributed_Observations_100_Reformatted.nc')
ds = xr.open_dataset(f'{base_path}ProcessedData/AP_Daily_Snowfall_All_Observations_Reformatted.nc')

# Create coordinate and data arrays in the right shape for using in Numpyro model
X = jnp.vstack(
    [
        ds.grid_longitude_standardised.data,
        ds.grid_latitude_standardised.data,
    ]
).T  # shape = [sites,2]

j_data = jnp.array(ds.prsn.data)  # JAX array is needed for Numpyro
Y = j_data[:,0,:]*10**5
Y = np.expand_dims(Y, axis=1)

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

mcmc = run_inference(bg_model, rng_key_, 1000, 2000, Y)

idata = az.from_numpyro(
    mcmc,
    coords={
        "months": np.arange(0, 1, 1),
        "sites": np.arange(0, 1050, 1),
        "days": np.arange(0, 570, 1),
    },
    dims={
        "alpha": ["months", "sites"],
        "beta": ["months", "sites"],
        "p": ["months", "sites"],
        "obs": ["days", "months", "sites"],
    },
)

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
outfile_path = f'{base_path}ProcessedData/AP_Daily_Snowfall_All_Observations_BG_Fit.nc'
idata.to_netcdf(outfile_path)