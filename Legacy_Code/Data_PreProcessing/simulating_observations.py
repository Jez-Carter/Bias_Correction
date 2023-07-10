'''
This script takes daily snowfall data from the climate model and samples to particular sites, which we treat as 'observations'.
The 'observation' data is stored as NetCDF file that can be loaded directly into an xarray dataset.
'''

import xarray as xr
import numpy as np

base_path = '/data/notebooks/jupyterlab-biascorrlab/data/'
lsm_path = 'RawData/Antarctic_CORDEX_MetUM_0p44deg_lsm.nc'

#loading land-sea mask
ds_lsm = xr.open_dataset(f'{base_path}{lsm_path}')
da_lsm = ds_lsm['lsm'][40:70,5:40] # (grid_lat,grid_lon) #[40:70,5:40] selects Antarctic Peninsula

#loading daily snowfall data and adding land sea mask variable
ds = xr.open_dataset(f'{base_path}ProcessedData/AP_Daily_Snowfall_044.nc') # Note this netCDF data is already filtered to just the antarctic peninsula
ds = ds.assign(lsm=da_lsm)

#Stacking dataset dimensions so sites can be randomly selected
ds_stacked = ds.stack(sites=['grid_latitude', 'grid_longitude'])
ds_stacked = ds_stacked.reset_index('sites') # needed for saving to net_cdf at end as there's no support currently for saving multiindecies to netcdf

######################################
#All
######################################
print('All_Observations')

outfile_path = f'{base_path}ProcessedData/AP_Daily_Snowfall_All_Observations.nc'
ds_stacked.to_netcdf(outfile_path)

######################################
#Randomly_Distributed_Observations_100
######################################
print('Randomly_Distributed_Observations_100')

np.random.seed(0)
sample_size = 100
sample_indicies = np.random.choice(len(ds_stacked.sites),sample_size,replace=False)
ds_sample = ds_stacked.isel(sites=sample_indicies)

outfile_path = f'{base_path}ProcessedData/AP_Daily_Snowfall_Randomly_Distributed_Observations_100.nc'
ds_sample.to_netcdf(outfile_path)

######################################
#Land_Only_Distributed_Observations_100
######################################
print('Land_Only_Distributed_Observations_100')

np.random.seed(0)
sample_size = 100
ds_stacked_filtered = ds_stacked.where(ds_stacked.lsm == 1,drop=True)
sample_indicies = np.random.choice(len(ds_stacked_filtered.sites),sample_size,replace=False)

ds_sample = ds_stacked_filtered.isel(sites=sample_indicies)

outfile_path = f'{base_path}ProcessedData/AP_Daily_Snowfall_Land_Only_Distributed_Observations_100.nc'
ds_sample.to_netcdf(outfile_path)