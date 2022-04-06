#Loading daily snowfall data, adding a month auxilliary coordinate and sampling to produce some test data.

import iris
import numpy as np
from src.netcdf_functions import aggregate_to_daily
from src.netcdf_functions import add_2d_latlon_aux_coords
from src.netcdf_functions import concatenate_cubes

data_directory = '/data/climatedata/'
metum_cube = iris.load(f'{data_directory}metum_cube_lres.nc')[0]
iris.coord_categorisation.add_month_number(metum_cube, metum_cube.coord('time'), name='month_number')

# Single Site, Single Month (Low Snowfall Site)
grid_lat,grid_lon = 50,50
single_site_cube = metum_cube[:,grid_lat,grid_lon] 
single_site_jan_cube = single_site_cube[single_site_cube.coord('month_number').points==1]
single_site_jan_data = single_site_jan_cube.data.data * 10**5 # Note I found that for some reason the fitting procedure struggles with very small values of rainfall, so I multiply by 10^5
np.save('/data/notebooks/jupyterlab-biascorrlab/data/single_site_jan_precip.npy',single_site_jan_data)

# Single Site, Single Month (High Snowfall Site)
grid_lat,grid_lon = 60,9
single_site_cube = metum_cube[:,grid_lat,grid_lon] 
single_site_jan_cube = single_site_cube[single_site_cube.coord('month_number').points==1]
single_site_jan_data = single_site_jan_cube.data.data * 10**5 # Note I found that for some reason the fitting procedure struggles with very small values of rainfall, so I multiply by 10^5
np.save('/data/notebooks/jupyterlab-biascorrlab/data/single_site_jan_precip_high_snowfall.npy',single_site_jan_data)

# Multiple Sites, Multiple Months

# Filtering the MetUM Cube:
metum_cube = metum_cube[:,40:70,5:40]

jan_metum_cube = metum_cube[metum_cube.coord('month_number').points==1]
feb_metum_cube = metum_cube[metum_cube.coord('month_number').points==2]

combine_lat_lon_size = metum_cube.shape[1]*metum_cube.shape[2]
jan_data = jan_metum_cube.data.data.reshape(jan_metum_cube.shape[0],combine_lat_lon_size)* 10**5 # Note I found that for some reason the fitting procedure struggles with very small values of rainfall, so I multiply by 10^5
feb_data = feb_metum_cube.data.data.reshape(feb_metum_cube.shape[0],combine_lat_lon_size)* 10**5 # Note I found that for some reason the fitting procedure struggles with very small values of rainfall, so I multiply by 10^5

samples = 50
sample_locations = np.random.choice(np.arange(0,combine_lat_lon_size,1),samples) 
jan_sample_data = jan_data[:,sample_locations]
feb_sample_data = feb_data[:,sample_locations]

np.save('/data/notebooks/jupyterlab-biascorrlab/data/AP_multi_site_jan_precip.npy',jan_sample_data)
np.save('/data/notebooks/jupyterlab-biascorrlab/data/AP_multi_site_feb_precip.npy',feb_sample_data)