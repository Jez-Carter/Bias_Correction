'''
This script loads 3-Hourly climate model data for Near-Surface Air Temperature over Antarctica and aggregates to daily. 
Additionally some adjustments are made to the coordinates such as adding Lat-Lon fields.
'''

from tqdm import tqdm
import numpy as np
import pandas as pd
import iris
import iris.fileformats
from src.netcdf_functions import aggregate_to_daily
from src.netcdf_functions import add_2d_latlon_aux_coords
from src.netcdf_functions import concatenate_cubes

years = np.arange(1981,2000,1)
# base_path = 'http://192.171.173.134/thredds/fileServer/dsnefiles/Jez/MetUM_Data/CORDEX_044_6Hourly/'
base_path = '/data/notebooks/jupyterlab-biascorrlab/data/RawData/'
months = [1,6]
months_constraint = iris.Constraint(time=lambda cell: cell.point.month in months)
coords_to_remove = ['day_of_year','forecast_period','forecast_reference_time','year']

cubes = []
for year in tqdm(years):
    # file_path = f'{base_path}Antarctic_CORDEX_MetUM_0p44deg_6_hourly_mean_prsn_{year}.nc#mode=bytes'
    file_path = f'{base_path}Antarctic_CORDEX_MetUM_0p44deg_6-hourly_mean_prsn_{year}.nc'
    cube = iris.load(file_path,months_constraint)[0][:,40:70,5:40] # (time,grid_lat,grid_lon) #[:,40:70,5:40] selects Antarctic Peninsula
    cube = aggregate_to_daily(cube)  # Note after aggregating this cube has shape 365,98,126
    add_2d_latlon_aux_coords(cube)
    cubes.append(cube)

cube = concatenate_cubes(iris.cube.CubeList(cubes))
for aux_coord in cube.aux_coords:
    if any([name in coords_to_remove for name in [aux_coord.var_name,aux_coord.long_name,aux_coord.standard_name]]):
        cube.remove_coord(aux_coord)
iris.save(cube,f'/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_044.nc') 

    