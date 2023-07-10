
# %%

import xarray as xr
import cartopy.crs as ccrs
from pyproj import Transformer

# %%

base_path = '/home/jez/Bias_Correction/'
climate_path = f'{base_path}data/ProcessedData/MetUM_Daily_TAS.nc'
mask_path = f'{base_path}data/ProcessedData/MetUM_044_Masks.nc'
out_folder = f'{base_path}data/ProcessedData/'

ds_climate = xr.open_dataset(climate_path)
ds_mask = xr.open_dataset(mask_path)

ds_climate = xr.merge([ds_climate,ds_mask])
ds_climate = ds_climate.isel(time=(ds_climate.time.dt.month == 1))
# ds_climate = ds_climate.where(ds_climate.ross_mask)
ds_climate_stacked = ds_climate.stack(X=(('grid_longitude','grid_latitude')))#.dropna('X','all')

# %%
climate_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=0.0,
    globe=None,
)

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
grid_lon=ds_climate_stacked['grid_longitude']
grid_lat=ds_climate_stacked['grid_latitude']

transformer = Transformer.from_crs(climate_coord_system, rotated_coord_system)

glon,glat = transformer.transform(grid_lon,grid_lat)

ds_climate_stacked = ds_climate_stacked.assign_coords(glon=("X", glon))
ds_climate_stacked = ds_climate_stacked.assign_coords(glat=("X", glat))

# %%

ds_climate = ds_climate_stacked.unstack()
ds_climate.to_netcdf(f'{out_folder}MetUM_Reformatted.nc')
# ds_climate.to_netcdf(f'{out_folder}MetUM_Ross_Subset_Reformatted.nc')
