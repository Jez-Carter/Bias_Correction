"""
This script takes daily 'observational' snowfall data and uses numpyro to fit a Bernoulli-Gamma distribution to the data using MCMC. 
The output of the script is samples from the posterior distribution of the parameters of our Bernoulli-Gamma model.
Sites and months are treated as independent in this initial model. 
A relationship between alpha and beta is preserved by across all sites&months by coding in the following to the model: (log(beta)~N(a0+a1*alpha,betavar)).
"""

# Loading daily snowfall data and fitting a Bernoulli-Gamma distribution using MCMC
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
import arviz as az
from jax import random, vmap, jit
from src.model_fitting_functions import bg_model
from src.model_fitting_functions import run_inference

folder_path = sys.argv[1]

df_sample = pd.read_csv(f"{folder_path}AP_Daily_Snowfall_044_Sample.csv")
sample_data = df_sample["prsn"].to_numpy()
sample_data = sample_data.reshape(
    len(df_sample["month"].unique()), len(df_sample["lonlat"].unique()), -1
)  # ordered [months,sites,days]
sample_data = np.moveaxis(
    sample_data, -1, 0
)  # adjusting the axes so that it's [days,months,sites]
jsample_data = jnp.array(sample_data)  # converting to JAX array (needed for Numpyro)

# Saving Input Data in Format Ingested into Model
np.save(f"{folder_path}AP_Daily_Snowfall_044_Sample.npy", jsample_data)

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

mcmc = run_inference(bg_model, rng_key_, 1000, 2000, jsample_data)

df_coords = df_sample[
    ["grid_latitude", "grid_longitude", "latitude", "longitude"]
].drop_duplicates()
ds = az.from_numpyro(
    mcmc,
    coords={
        "months": df_sample["month"].drop_duplicates(),
        "sites": np.arange(0, 100, 1),
        "days": np.arange(0, 570, 1),
    },
    dims={
        "alpha": ["months", "sites"],
        "beta": ["months", "sites"],
        "p": ["months", "sites"],
        "obs": ["days", "months", "sites"],
    },
)
del ds.log_likelihood
del ds.sample_stats
ds = ds.assign_coords(
    grid_latitude=("sites", df_coords["grid_latitude"]),
    grid_longitude=("sites", df_coords["grid_longitude"]),
    latitude=("sites", df_coords["latitude"]),
    longitude=("sites", df_coords["longitude"]),
)

outfile_path = f"/data/notebooks/jupyterlab-biascorrlab/data/Lima2021/AP_Daily_Snowfall_044_BGFit.nc"
ds.to_netcdf(outfile_path)

