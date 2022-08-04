'''
This script takes daily 'observational' snowfall data and uses numpyro to fit a hierarchial MVN - Bernoulli-Gamma model to the data using MCMC. 
The output of the script is samples from the posterior distribution of the parameters of both distributions:
['a0',
 'a1',
 'beta',
 'betavar',
 'kernel_length',
 'kernel_noise',
 'kernel_var',
 'log_alpha',
 'p']
At the moment only alpha is modelled as coming from a MVN distribution. 
'''

import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random, vmap, jit
import numpyro
import xarray as xr
from scipy.spatial import distance_matrix
from src.model_fitting_functions import bg_gp_model 
from src.model_fitting_functions import run_inference

numpyro.enable_x64()

folder_path = sys.argv[1]

df_sample = pd.read_csv(f'{folder_path}AP_Daily_Snowfall_044_Sample.csv')
sample_data = df_sample['prsn'].to_numpy()
sample_data = sample_data.reshape(len(df_sample['month'].unique()),len(df_sample['lonlat'].unique()),-1) # ordered [months,sites,days]
sample_data = np.moveaxis(sample_data, -1, 0) # adjusting the axes so that it's [days,months,sites]
jsample_data = jnp.array(sample_data) #converting to JAX array (needed for Numpyro)

df = pd.read_csv(f'{folder_path}AP_BGLima_Snowfall_044_Mean_Alpha_stand.csv', index_col=0)
X = jnp.array(df[['grid_latitude_stand','grid_longitude_stand']].values)
distance_matrix_values = jnp.array(distance_matrix(X,X))

#Saving Input Data in Format Ingested into Model
np.save(f'{folder_path}AP_Daily_Snowfall_044_Sample.npy',jsample_data)

#Saving BG Model Plate Diagram
numpyro.render_model(bg_gp_model, model_args=(distance_matrix_values,jsample_data), render_distributions=True,filename='BGGP_Model_Plate_Diagram.png')

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

samples = run_inference(bg_gp_model,rng_key_,1000,2000,jsample_data,distance_matrix=distance_matrix_values)

samples_array = np.array(list(samples.items()),dtype=object)
np.save(f'{folder_path}AP_BGGPLima_Snowfall_044_Sample.npy',samples_array)

keys = list(samples.keys())
samples_coord = np.arange(0,2000,1)
month_coord = np.arange(0,2,1)
sites_coord = np.arange(0,100,1)
coords = dict(samples_coord=(["samples"], samples_coord),month_coord=(["month"], month_coord),sites_coord=(["sites"], sites_coord))
data = dict(p=(["samples", "month","sites"], samples['p']))
ds = xr.Dataset(data_vars=data,coords=coords)
for key in keys:
    if key in ['a0','a1','betavar','kernel_length','kernel_noise','kernel_var']:
        ds[f'{key}']=(["samples"],samples[f'{key}'])
    elif key=='beta':
        ds[f'{key}']=(["samples","month","sites"],samples[f'{key}'])
    elif key=='log_alpha':
        ds[f'{key}']=(["samples","month","sites"],samples[f'{key}'][:,:,0,:])

ds.to_netcdf(f'{folder_path}AP_BGGPLima_Snowfall_044_Sample.nc', 'w')

alpha_mean_estimates = jnp.exp(samples['log_alpha'].mean(axis=(0)))
beta_mean_estimates = samples['beta'].mean(axis=(0))
p_mean_estimates = samples['p'].mean(axis=(0))

df_estimates = df_sample[['grid_latitude','grid_longitude','latitude','longitude','month']].drop_duplicates()
df_estimates['alpha']=alpha_mean_estimates.reshape(-1)
df_estimates['beta']=beta_mean_estimates.reshape(-1)
df_estimates['p']=p_mean_estimates.reshape(-1)
df_estimates.to_csv(f'{folder_path}AP_BGGPLima_Snowfall_044_Mean_Estimates.csv',index=True)
