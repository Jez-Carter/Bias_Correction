#Loading daily snowfall data and fitting a Bernoulli-Gamma distribution using MCMC
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random, vmap, jit
from src.model_fitting_functions import bg_model 
from src.model_fitting_functions import run_inference

folder_path = sys.argv[1]

df_sample = pd.read_csv(f'{folder_path}AP_Daily_Snowfall_044_Sample.csv')
sample_data = df_sample['prsn'].to_numpy()
sample_data = sample_data.reshape(len(df_sample['month'].unique()),len(df_sample['lonlat'].unique()),-1) # ordered [months,sites,days]
sample_data = np.moveaxis(sample_data, -1, 0) # adjusting the axes so that it's [days,months,sites]
jsample_data = jnp.array(sample_data) #converting to JAX array (needed for Numpyro)

#Saving Input Data in Format Ingested into Model
np.save(f'{folder_path}AP_Daily_Snowfall_044_Sample.npy',jsample_data)

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

samples = run_inference(bg_model,rng_key_,1000,2000,jsample_data,coords=None)

samples_array = np.array(list(samples.items()),dtype=object)
np.save(f'{folder_path}AP_BGLima_Snowfall_044_Sample.npy',samples_array)

alpha_mean_estimates = samples['alpha'].mean(axis=(0))
p_mean_estimates = samples['p'].mean(axis=(0))
df_estimates = df_sample[['grid_latitude','grid_longitude','latitude','longitude','month']].drop_duplicates()
df_estimates['alpha']=alpha_mean_estimates.reshape(-1)
df_estimates['p']=p_mean_estimates.reshape(-1)
df_estimates.to_csv(f'{folder_path}AP_BGLima_Snowfall_044_Mean_Estimates.csv',index=True)
