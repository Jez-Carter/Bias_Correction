
import xarray as xr
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys

base_path = '/data/notebooks/jupyterlab-biascorrlab/data/Lima2021/'
lsm_path = 'http://192.171.173.134/thredds/fileServer/dsnefiles/Jez/MetUM_Data/CORDEX_044_Fixed/Antarctic_CORDEX_MetUM_0p44deg_lsm.nc'

#loading land-sea mask
ds_lsm = xr.open_dataset(f'{lsm_path}#mode=bytes')
ds_lsm = ds_lsm.drop('rotated_latitude_longitude')
da_lsm = ds_lsm.to_array()[0,40:70,5:40] # (grid_lat,grid_lon) #[40:70,5:40] selects Antarctic Peninsula
df_lsm = da_lsm.to_dataframe(name='lsm').reset_index().drop(columns='variable')

#loading daily snowfall data
ds = xr.open_dataset('/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_044.nc') # Note this netCDF data is already filtered to just the antarctic peninsula
ds = ds.drop('rotated_latitude_longitude')
ds = ds.drop_dims('bnds')
ds = ds*10**5 # Note I found that for some reason the fitting procedure struggles with very small values of snowfall, so I multiply by 10^5
df = ds.to_dataframe().reset_index()

#merging to include land-sea-mask column
df = df.merge(df_lsm)

#including month and latlon combined columns
time_values = df['time']
month_values = np.array([i.month for i in time_values])
df['month']=month_values
df['lonlat']=df[['latitude', 'longitude']].apply(tuple, axis=1)

#filtering so all sites have the same number of days of data
min_days = df.groupby(['month','lonlat']).count().min()[0] #calculating the minimum days for any month and site
df = df.groupby(['month','lonlat']).sample(min_days) # resampling so all months and sites have the same number of days

######################################
#Randomly_Distributed_Observations_100
######################################
print('Randomly_Distributed_Observations_100')

np.random.seed(0)
sample_size = 100
sample_indicies = np.random.choice(len(df['lonlat'].unique()),sample_size,replace=False)
df_sample = df[df['lonlat'].isin(df['lonlat'].unique()[sample_indicies])]
df_sample = df_sample.sort_values(['month','lonlat','time'])
df_sample.to_csv(f'{base_path}Randomly_Distributed_Observations_100/AP_Daily_Snowfall_044_Sample.csv')

######################################
#Land_Only_Distributed_Observations_100
######################################
print('Land_Only_Distributed_Observations_100')

np.random.seed(0)
sample_size = 100
df_land_only = df[df['lsm']==1]
sample_indicies = np.random.choice(len(df_land_only['lonlat'].unique()),sample_size,replace=False)
df_sample = df_land_only[df_land_only['lonlat'].isin(df_land_only['lonlat'].unique()[sample_indicies])]
df_sample = df_sample.sort_values(['month','lonlat','time'])
df_sample.to_csv(f'{base_path}Land_Only_Distributed_Observations_100/AP_Daily_Snowfall_044_Sample.csv')
