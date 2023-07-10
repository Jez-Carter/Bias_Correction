'''
This script loads 3-Hourly climate model data for Near-Surface Air Temperature over Antarctica and aggregates to daily. 
Additionally some adjustments are made to the coordinates such as adding Lat-Lon fields.
'''

#Importing Packages
from tqdm import tqdm
import xarray as xr
import numpy as np
from src.helper_functions import grid_coords_to_2d_latlon_coords

#Loading raw data
base_path = '/home/jez/'
metum_path = f'{base_path}DSNE_ice_sheets/Antarctic_CORDEX/MetUM/044_3hourly/'
out_path = f'{base_path}Bias_Correction/data/ProcessedData/MetUM_Daily_TAS.nc'

years = np.arange(1981,2019,1)

ds_list = []
for year in tqdm(years):
    print(year)
    filename = f'Antarctic_CORDEX_MetUM_0p44deg_3-hourly_tas_{year}.nc'
    path = f'{metum_path}{filename}'
    ds = xr.open_dataset(path)
    ds = ds.resample(time='D').mean()
    ds_list.append(ds)

ds = xr.concat(ds_list,'time')

#Adding lat,lon coordinates
ref_filename = f'Antarctic_CORDEX_MetUM_0p44deg_3-hourly_tas_2000.nc'
ref_path = f'{metum_path}{filename}'
ds = grid_coords_to_2d_latlon_coords(ds,ref_path)

#Saving
ds.to_netcdf(path)

    