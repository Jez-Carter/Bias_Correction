#Loading 6hourly snowfall data and aggregating to daily as well as adding lat,lon coordinates

import numpy as np
import pandas as pd
import iris
import iris.fileformats
from src.netcdf_functions import aggregate_to_daily
from src.netcdf_functions import add_2d_latlon_aux_coords
from src.netcdf_functions import concatenate_cubes

years = np.arange(1981,1985,1)
base_path = 'http://192.171.173.134/thredds/fileServer/dsnefiles/Jez/MetUM_Data/'
resolutions = ['44','11']
months = [1,6]
months_constraint = iris.Constraint(time=lambda cell: cell.point.month in months)
total_days = 28*len(years)
samples = 100
np.random.seed(0)

for res in resolutions:
    cubes = []
    if res=='44':
        path_start = f'{base_path}/CORDEX_0{res}_6Hourly/Antarctic_CORDEX_MetUM_0p{res}deg_6_hourly_mean_prsn_'
    elif res=='11':
        path_start = f'{base_path}/CORDEX_0{res}_6Hourly/Antarctic_CORDEX_MetUM_0p{res}deg_6-hourly_mean_prsn_'
    for year in years:
        print(year)
        url = f'{path_start}{year}.nc#mode=bytes'
        if res=='44':
            cube = iris.load(url,months_constraint)[0][:,40:70,5:40] # (time,grid_lat,grid_lon) #[:,40:70,5:40] selects Antarctic Peninsula
        elif res=='11':
            cube = iris.load(url,months_constraint)[0][:,160:280,20:158] # (time,grid_lat,grid_lon) #[:,160:280,20:158] selects Antarctic Peninsula        
        cube = aggregate_to_daily(cube)  # Note after aggregating this cube has shape 365,98,126
        add_2d_latlon_aux_coords(cube)
        iris.coord_categorisation.add_month_number(cube, cube.coord('time'), name='month_number')
        cubes.append(cube)
    
    res_cube = concatenate_cubes(iris.cube.CubeList(cubes))
    iris.save(res_cube,f'/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_0{res}.nc') 
    
    data = []
    for month_number in months:
        month_cube = res_cube[res_cube.coord('month_number').points==month_number]
        month_data = res_cube.data.data #shape [#times,#lats,#lons]
        month_data = month_data.reshape(res_cube.shape[0],-1) # reshaping to [#times,#lats*#lons]
        data.append(month_data)
        
    #data has shape (#months, #days, #sites)
    
    data = np.array(data)[:,:total_days,:]# JAX doesn't like object arrays so data for each month needs to be the same shape
    data = data*10**5 # Note I found that for some reason the fitting procedure struggles with very small values of snowfall, so I multiply by 10^5
    data = np.moveaxis(data, 1, 0) # adjusting the axes so that it's [days,months,sites]
    
    if res=='11':
        sample = np.random.choice(np.arange(0,data.shape[-1],1), samples) 
        sample_data = data[:,:,sample]
        np.save(f'/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_0{res}_sample.npy',sample_data)
        grid_lat,grid_lon = cube.coord('grid_latitude').points.astype('float64'), cube.coord('grid_longitude').points.astype('float64') # shape [#lats,],[#lons,]
        GLAT, GLON = np.meshgrid(grid_lat, grid_lon) # shapes [#lats,#lons],[#lats,#lons]
        GLAT, GLON = GLAT.reshape(1,-1), GLON.reshape(1,-1) # shapes [#lats*#lons,],[#lats*#lons,]
        GLATLON = np.vstack([GLAT, GLON]) # shape [2,#lats*#lons]
        df_coords = pd.DataFrame(GLATLON.T,columns=['Grid_Latitude','Grid_Longitude'])
        df_sample_coords = df_coords.iloc[sample]
        LAT,LON = cube.coord('latitude').points.reshape(-1)[sample],cube.coord('longitude').points.reshape(-1)[sample]
        df_sample_coords['Latitude'],df_sample_coords['Longitude'] = LAT,LON
        df_coords.to_csv(f'/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_0{res}_coordinates.csv')
        df_sample_coords.to_csv(f'/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_0{res}_sample_coordinates.csv')
    
    else:
        np.save(f'/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_0{res}.npy',data) 
    