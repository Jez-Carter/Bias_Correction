"""
This script takes daily 'observational' snowfall data and reformats the data such that it's in a form where the data can easily be queried and ingested into the numpyro model.
"""

import sys
import xarray as xr
import numpy as np
import jax.numpy as jnp

from src.helper_functions import standardise

path = sys.argv[1]
outpath = sys.argv[2]

ds_sample = xr.open_dataset(path) 

months = ds_sample.time.dt.month
ds_sample = ds_sample.assign_coords(month=("time", months.data))
ds_sample = ds_sample.sortby('month') #sorting the dataset to the order needed to assign the day coordinate
unique_count_by_month = ds_sample.time.groupby('month').count() # counting the number of days in each month
days = np.concatenate([np.arange(0,i,1) for i in unique_count_by_month]) # creating an ordered index that restarts for each different month 
ds = ds_sample.assign_coords(day=("time", days)) 
ds = ds.set_index(time=("month", "day")).unstack("time") #turning the time coordinate into a ordered index and a month index 
ds = ds.transpose('day','month','sites','bnds') #reordering data
ds = ds.isel(day=range(unique_count_by_month.min().data)) #filtering to consistent number of days for each month (gets rid of Nans)

# Create new standardised grid_latitude and grid_longitude coordinates
grid_latitude_standardised = standardise(ds.grid_latitude)
grid_longitude_standardised = standardise(ds.grid_longitude)
ds = ds.assign_coords(
    grid_latitude_standardised=("sites", grid_latitude_standardised.data),
    grid_longitude_standardised=("sites", grid_longitude_standardised.data)
)

ds.to_netcdf(outpath)