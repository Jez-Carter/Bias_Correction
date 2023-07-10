'''
This script creates a mask dataset based on shapefiles and the land-sea mask from the MetUM 044 resolution model.
'''

import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
from src.helper_functions import grid_coords_to_2d_latlon_coords
from src.helper_functions import create_mask

#Loading land-sea mask data
base_path = '/home/jez/'
lsm_path = f'{base_path}DSNE_ice_sheets/Antarctic_CORDEX/MetUM/044_fixed/Antarctic_CORDEX_MetUM_0p44deg_lsm.nc'
out_path = f'{base_path}Bias_Correction/data/ProcessedData/MetUM_044_Masks.nc'

ds = xr.open_dataset(lsm_path)
ds = grid_coords_to_2d_latlon_coords(ds,lsm_path)

#Loading shapefile data
antarctica_shapefile_path = f'{base_path}Bias_Correction/data/Antarctica_Shapefile/antarctica_shapefile.shp'
antarctica_gdf = gpd.read_file(antarctica_shapefile_path)
ice_shelves_gdf = antarctica_gdf[antarctica_gdf['Id_text']=='Ice shelf'].reset_index()
ross_ice_shelf_gdf = ice_shelves_gdf.sort_values(by=['Area_km2']).iloc[[-1]].reset_index()

region_shapefile_path = f'{base_path}Bias_Correction/data/Ross_Region_Shapefile/ross_region.shp'
region_gdf = gpd.read_file(region_shapefile_path)

#Creating region mask
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
region_mask = create_mask(ds,region_gdf,map_proj)
ds['region_mask'] = (('grid_latitude', 'grid_longitude'), region_mask)

#Creating ross ice shelf mask
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
ross_mask = create_mask(ds,ross_ice_shelf_gdf,map_proj)
ds['ross_mask'] = (('grid_latitude', 'grid_longitude'), ross_mask)

#Saving
ds.to_netcdf(out_path)
