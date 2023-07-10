


import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs

base_path = '/home/jez/Bias_Correction/'
observations_path = f'{base_path}data/ProcessedData/NST_Observations.nc'
region_shapefile_path = f'{base_path}data/Ross_Region_Shapefile/ross_region.shp'
antarctica_shapefile_path = f'{base_path}data/Antarctica_Shapefile/antarctica_shapefile.shp'
out_folder = f'{base_path}data/ProcessedData/'

ds = xr.open_dataset(observations_path)
region_gdf = gpd.read_file(region_shapefile_path)
antarctica_gdf = gpd.read_file(antarctica_shapefile_path)
ice_shelves_gdf = antarctica_gdf[antarctica_gdf['Id_text']=='Ice shelf'].reset_index()
ross_ice_shelf_gdf = ice_shelves_gdf.sort_values(by=['Area_km2']).iloc[[-1]].reset_index()
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)

points = [Point(lon,lat) for lon,lat in zip(ds['Lon(°C)'],ds['Lat(°C)'])]
points_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=points)

region_mask = [region_gdf.to_crs(map_proj).contains(points_gdf[i:i+1].reset_index().to_crs(map_proj)) for i in range(points_gdf.shape[0])]
ross_mask = [ross_ice_shelf_gdf.to_crs(map_proj).contains(points_gdf[i:i+1].reset_index().to_crs(map_proj)) for i in range(points_gdf.shape[0])]
region_mask = np.array(region_mask).reshape(-1)
ross_mask = np.array(ross_mask).reshape(-1)

region_subset_ds = ds.sel(Station_Lower=region_mask)
ross_subset_ds = ds.sel(Station_Lower=ross_mask)

region_subset_ds['Institution'] = region_subset_ds.Institution.astype(str)
ross_subset_ds['Institution'] = ross_subset_ds.Institution.astype(str)

region_subset_ds.to_netcdf(f'{out_folder}NST_Observations_Region_Subset.nc')
ross_subset_ds.to_netcdf(f'{out_folder}NST_Observations_Ross_Subset.nc')
