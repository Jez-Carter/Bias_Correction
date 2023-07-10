# %%

import xarray as xr
import cartopy.crs as ccrs
from pyproj import Transformer

# %%

base_path = '/home/jez/Bias_Correction/'
data_folder = f'{base_path}data/ProcessedData/'

obs_path = f'{data_folder}NST_Observations.nc'
out_path = f'{data_folder}NST_Observations_Reformatted.nc'

# obs_path = f'{data_folder}NST_Observations_Ross_Subset.nc'
# out_path = f'{data_folder}NST_Observations_Ross_Subset_Reformatted.nc'

ds_obs_jan = xr.open_dataset(obs_path).isel(Month=0)

# %%
ds_obs_jan_stacked = ds_obs_jan.stack(D=('Year','Day')).dropna('D','all')

# %%
ds_obs_jan_stacked

# %%
# Create grid coordinates that are Euclidean
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)

climate_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=0.0,
    globe=None,
)

lon=ds_obs_jan_stacked['Lon(°C)']
lat=ds_obs_jan_stacked['Lat(°C)']

transformer = Transformer.from_crs("epsg:4326", rotated_coord_system)
#NOTE epsg:4326 takes (latitude,longitude) whereas rotated_coord_system takes (rotated long,rotated lat)
#The change in order of the axes is important and reflected in the below line
glon,glat = transformer.transform(lat,lon)

ds_obs_jan_stacked = ds_obs_jan_stacked.assign_coords(glon=("Station_Lower", glon))
ds_obs_jan_stacked = ds_obs_jan_stacked.assign_coords(glat=("Station_Lower", glat))

# %% 
def convert_180_to_360(long_value):
    if long_value<0:
        return(long_value+360)
    else:
        return(long_value)
    
#Including climate coordinate system useful for finding nearest points etc
transformer = Transformer.from_crs("epsg:4326", climate_coord_system)
#NOTE epsg:4326 takes (latitude,longitude) whereas rotated_coord_system takes (rotated long,rotated lat)
#The change in order of the axes is important and reflected in the below line
grid_longitude,grid_latitude = transformer.transform(lat,lon)

grid_longitude_360 = [convert_180_to_360(i) for i in grid_longitude]

ds_obs_jan_stacked = ds_obs_jan_stacked.assign_coords(grid_longitude=("Station_Lower", grid_longitude_360))
ds_obs_jan_stacked = ds_obs_jan_stacked.assign_coords(grid_latitude=("Station_Lower", grid_latitude))

# %%
# ds_obs_jan_stacked.mean('D').plot.scatter(x="glat",y="glon")
# %%
ds_obs_jan = ds_obs_jan_stacked.unstack()
ds_obs_jan.to_netcdf(out_path)

# %%
