


import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs

base_path = '/home/jez/Bias_Correction/'
observations_path = f'{base_path}data/ProcessedData/NST_Observations.nc'
region_shapefile_path = f'{base_path}data/Ross_Region_Shapefile/ross_region.shp'
antarctica_shapefile_path = f'{base_path}data/Antarctica_Shapefile/antarctica_shapefile.shp'
out_path = f'{base_path}data/ProcessedData/NST_Observations_Subset.nc'

ds = xr.open_dataset(observations_path).isel(Month=0)
region_gdf = gpd.read_file(region_shapefile_path)
antarctica_gdf = gpd.read_file(antarctica_shapefile_path)
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)

points = [Point(lon,lat) for lon,lat in zip(ds['Lon(°C)'],ds['Lat(°C)'])]
points_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=points)

mask = [region_gdf.to_crs(map_proj).contains(points_gdf[i:i+1].reset_index().to_crs(map_proj)) for i in range(points_gdf.shape[0])]
mask = np.array(mask).reshape(-1)

subset_ds = ds.sel(Station_Lower=mask)

subset_ds['Institution'] = subset_ds.Institution.astype(str)
subset_ds.to_netcdf(out_path)
