"""
This script takes estimates of the Bernoulli-Gamma parameters alpha and p at all 'observation' locations and fits a Gaussian Process to the spatial distribution of alpha and p seperately. 
The Gaussian Process has parameters kernel lengthscale, kernel variance and likelihood variance.
The values of the coordinates and alpha and p are all standardised before fitting.
"""

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
from src.model_fitting_functions import tinygp_model

jax.config.update("jax_enable_x64", True)

# Load the inference data from the Bernoulli-Gamma MCMC fit
infile_path = f"/data/notebooks/jupyterlab-biascorrlab/data/Lima2021/AP_Daily_Snowfall_044_BGFit.nc"
idata = az.from_netcdf(infile_path)

# Calculate expection values of the Bernoulli-Gamma parameters
idata_expectations = idata.posterior.mean(dim=["draw", "chain"])

# Create new standardised grid_latitude and grid_longitude coordinates
grid_latitude_standardised = standardise(idata_expectations.grid_latitude)
grid_longitude_standardised = standardise(idata_expectations.grid_longitude)
idata_expectations = idata_expectations.assign_coords(
    grid_latitude_standardised=("sites", grid_latitude_standardised.data)
)
idata_expectations = idata_expectations.assign_coords(
    grid_longitude_standardised=("sites", grid_longitude_standardised.data)
)

# Create coordinate and data arrays in the right shape for using in another GP Numpyro model
X = np.vstack(
    [
        idata_expectations.grid_longitude_standardised.data,
        idata_expectations.grid_latitude_standardised.data,
    ]
).T  # shape = [sites,2]
Y = np.vstack(
    [
        idata_expectations.sel(months=1)["alpha"].data,
        idata_expectations.sel(months=1)["p"].data,
    ]
)  # shape = [parameters,sites]
X, Y = jnp.array(X), jnp.array(Y)  # converting to jax arrays

# Running the GP Model
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
mcmc = run_inference_tinygp(tinygp_model, rng_key_, 1000, 2000, Y, X=X)

prior = Predictive(tinygp_model, num_samples=1000)(rng_key_,*(Y,X))
del prior['alpha']
del prior['p']

# Creating inference data object with results from GP fitting
gp_idata = az.from_numpyro(
    mcmc,
    prior=prior,
    coords={"sites": np.arange(0, 100, 1)},
    dims={"alpha": ["sites"], "p": ["sites"]},
)

# Reassigning coordinates to observed data
gp_idata.observed_data = gp_idata.observed_data.assign_coords(
    grid_latitude=("sites", idata_expectations["grid_latitude"].data),
    grid_longitude=("sites", idata_expectations["grid_longitude"].data),
    latitude=("sites", idata_expectations["latitude"].data),
    longitude=("sites", idata_expectations["longitude"].data),
    grid_latitude_standardised=(
        "sites",
        idata_expectations["grid_latitude_standardised"].data,
    ),
    grid_longitude_standardised=(
        "sites",
        idata_expectations["grid_longitude_standardised"].data,
    ),
)

# Saving output
outfile_path = f"/data/notebooks/jupyterlab-biascorrlab/data/Lima2021/AP_Daily_Snowfall_044_GPFit.nc"
gp_idata.to_netcdf(outfile_path)

