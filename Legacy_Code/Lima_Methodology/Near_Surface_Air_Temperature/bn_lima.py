"""
This script takes daily 'observational' near-surface air temperature data from AWS's and uses numpyro to fit a Standard Normal distribution to the data using MCMC. 
The output of the script is samples from the posterior distribution of the parameters of the Standard Normal distribution.
Sites and months are treated as independent in this initial model. 
"""

import numpy as np
import xarray as xr
import jax.numpy as jnp
from jax import random
import numpyro
numpyro.enable_x64()
import arviz as az

from src.model_fitting_functions import normal_model
from src.model_fitting_functions import run_inference

base_path = '/home/jez/Bias_Correction/'
in_path = f'{base_path}data/ProcessedData/NST_Observations_Subset.nc'
out_path = f'{base_path}data/Lima2021/Daily_Temperature_NFit.nc'

ds = xr.open_dataset(in_path)
ds = ds.transpose("Day", "Station_Lower")

da_temp = ds['Temperature()']
j_data = jnp.array(da_temp.data)  # JAX array is needed for Numpyro
mask = jnp.isnan(j_data)
numpyro_mask = jnp.logical_not(mask) # True indicates a site and False not a site.

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

mcmc = run_inference(normal_model, rng_key_, 1000, 2000, j_data, numpyro_mask)

idata = az.from_numpyro(
    mcmc,
    coords={
        "sites": np.arange(0, len(ds.Station_Lower), 1),
        "days": np.arange(0, len(ds.Day), 1),
    },
    dims={
        "loc": ["sites"],
        "scale": ["sites"],
        "obs": ["days", "sites"],
    },
)

# Reassigning coordinates to observed data
idata.observed_data = idata.observed_data.assign_coords(
    Station_Lower=("sites", da_temp["Station_Lower"].data),
    Lat=("sites", da_temp["Lat(°C)"].data),
    Lon=("sites", da_temp["Lon(°C)"].data),
    Elevation=("sites", da_temp["Elevation(m)"].data),
    Institution=("sites", da_temp["Institution"].data),)

idata.posterior = idata.posterior.assign_coords(
    Station_Lower=("sites", da_temp["Station_Lower"].data),
    Lat=("sites", da_temp["Lat(°C)"].data),
    Lon=("sites", da_temp["Lon(°C)"].data),
    Elevation=("sites", da_temp["Elevation(m)"].data),
    Institution=("sites", da_temp["Institution"].data),)

# Saving output
idata.to_netcdf(out_path)