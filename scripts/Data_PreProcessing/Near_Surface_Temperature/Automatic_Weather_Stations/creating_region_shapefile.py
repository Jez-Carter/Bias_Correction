"""
This script uses the MetUM(011) climate data to create the boundary of a region to study and then creates a shapefile from that boundary.
"""

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
import geopandas as gpd
from src.helper_functions import grid_coords_to_2d_latlon_coords

base_path = '/home/jez/Bias_Correction/'
climate_data_path = f'{base_path}data/MetUM011_LandSeaMask.nc'
out_path = f'{base_path}data/Ross_Region_Shapefile/ross_region.shp'

ds_climate = xr.open_dataset(climate_data_path)
ds_climate = grid_coords_to_2d_latlon_coords(ds_climate,climate_data_path)

###Ross Ice Shelf Region ###
glat_lower,glat_upper = 15,40
glon_lower,glon_upper = 50,80
############################

line_1_lons = ds_climate.isel(grid_latitude=slice(glat_lower,glat_upper),grid_longitude=glon_lower).longitude.data
line_1_lats = ds_climate.isel(grid_latitude=slice(glat_lower,glat_upper),grid_longitude=glon_lower).latitude.data
line_2_lons = ds_climate.isel(grid_latitude=glat_upper,grid_longitude=slice(glon_lower,glon_upper)).longitude.data
line_2_lats = ds_climate.isel(grid_latitude=glat_upper,grid_longitude=slice(glon_lower,glon_upper)).latitude.data
line_3_lons = ds_climate.isel(grid_latitude=slice(glat_upper,glat_lower,-1),grid_longitude=glon_upper).longitude.data
line_3_lats = ds_climate.isel(grid_latitude=slice(glat_upper,glat_lower,-1),grid_longitude=glon_upper).latitude.data
line_4_lons = ds_climate.isel(grid_latitude=glat_lower,grid_longitude=slice(glon_upper,glon_lower,-1)).longitude.data
line_4_lats = ds_climate.isel(grid_latitude=glat_lower,grid_longitude=slice(glon_upper,glon_lower,-1)).latitude.data

box_lons = list(line_1_lons) + list(line_2_lons) + list(line_3_lons) + list(line_4_lons)
box_lats = list(line_1_lats) + list(line_2_lats) + list(line_3_lats) + list(line_4_lats)
box_lons_lats = [box_lons,box_lats]

polygon = Polygon(np.column_stack((box_lons_lats[0], box_lons_lats[1])))
gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])

gdf.to_file(out_path)  
