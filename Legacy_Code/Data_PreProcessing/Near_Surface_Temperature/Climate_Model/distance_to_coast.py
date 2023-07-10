# %%
#Importing Packages
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd

# %%

#Loading climate data
base_path = '/home/jez/Bias_Correction/'
nst_climate_path = f'{base_path}data/ProcessedData/MetUM_Reformatted.nc'
ds_nst_climate = xr.open_dataset(nst_climate_path)
ds = ds_nst_climate.isel(time=0).drop_vars(list(ds_nst_climate.keys()))

# %%
# %%
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
antarctica_shapefile_path = f'{base_path}data/Antarctica_Shapefile/antarctica_shapefile.shp'
antarctica_gdf = gpd.read_file(antarctica_shapefile_path)
antarctica_gdf = antarctica_gdf.to_crs(rotated_coord_system)

# %%
antarctica_gdf.plot()

# %%
ds_nst_climate.isel(time=0).drop_vars(list(ds_nst_climate.keys()))

# %%

# ds_climate = xr.merge([ds_nst_climate,ds_ele_climate,ds_mask])
# ds_climate = ds_climate.isel(time=(ds_climate.time.dt.month == 1))
# ds_climate['tas']=ds_climate['tas']-273.15

# ds_climate['Mean Jan Temperature'] = ds_climate['tas'].mean(['time'])
# ds_climate['Variance Jan Temperature'] = ds_climate['tas'].var(['time'])

# ds_climate_stacked = ds_climate.stack(X=('grid_latitude', 'grid_longitude'))

# %%


# import geopandas as gpd
# from shapely.geometry import Point, box
# from random import uniform
# from concurrent.futures import ThreadPoolExecutor
# from tqdm.notebook import tqdm
# import cartopy
# import matplotlib.pyplot as plt
# import numpy as np
# import xarray as xr
# import pandas as pd
# import shapely


# lon = np.arange(129.4, 153.75+0.05, 0.25)
# lat = np.arange(-43.75, -10.1+0.05, 0.25)

# precip = 10 * np.random.rand(len(lat), len(lon))


# ds = xr.Dataset({"precip": (["lat", "lon"], precip)},coords={"lon": lon,"lat": lat})

# ds['precip'].plot()

# def hv(lonlat1, lonlat2):
#     AVG_EARTH_RADIUS = 6371000. # Earth radius in meter

#     # Get array data; convert to radians to simulate 'map(radians,...)' part
#     coords_arr = np.deg2rad(lonlat1)
#     a = np.deg2rad(lonlat2)

#     # Get the differentiations
#     lat = coords_arr[:,1] - a[:,1,None]
#     lng = coords_arr[:,0] - a[:,0,None]

#     # Compute the "cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2" part.
#     # Add into "sin(lat * 0.5) ** 2" part.
#     add0 = np.cos(a[:,1,None])*np.cos(coords_arr[:,1])* np.sin(lng * 0.5) ** 2
#     d = np.sin(lat * 0.5) ** 2 +  add0

#     # Get h and assign into dataframe
#     h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
#     return {'dist_to_coastline': h.min(1), 'lonlat':lonlat2}

# def get_distance_to_coast(arr, country, resolution='50m'):

#     print('Get shape file...')
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#     #single geom for country
#     geom = world[world["name"]==country].dissolve(by='name').iloc[0].geometry

#     #single geom for the coastline
#     c = cartopy.io.shapereader.natural_earth(resolution=resolution, category='physical', name='coastline')

#     c     = gpd.read_file(c)
#     c.crs = 'EPSG:4326'

#     print('Group lat/lon points...')
#     points = []
#     i = 0
#     for ilat in arr['lat'].values:
#         for ilon in arr['lon'].values:
#                 points.append([ilon, ilat])
#                 i+=1

#     xlist = []
#     gdpclip = gpd.clip(c.to_crs('EPSG:4326'), geom.buffer(1))
#     for icoast in range(len(gdpclip)):
#         print('Get coastline ({}/{})...'.format(icoast+1, len(gdpclip)))
#         coastline = gdpclip.iloc[icoast].geometry #< This is a linestring

#         if type(coastline) is shapely.geometry.linestring.LineString:
#             coastline = [list(i) for i in coastline.coords]
#         elif type(coastline) is shapely.geometry.multilinestring.MultiLineString:
#             dummy = []
#             for line in coastline:
#                 dummy.extend([list(i) for i in line.coords])
#             coastline = dummy
#         else:
#             print('In function: get_distance_to_coast')
#             print('Type: {} not found'.format(type(type(coastline))))
#             exit()

#         print('Computing distances...')
#         result = hv(coastline, points)

#         print('Convert to xarray...')
#         gdf = gpd.GeoDataFrame.from_records(result)
#         lon = [i[0] for i in gdf['lonlat']]
#         lat = [i[1] for i in gdf['lonlat']]
#         df1 = pd.DataFrame(gdf)
#         df1['lat'] = lat
#         df1['lon'] = lon
#         df1 = df1.set_index(['lat', 'lon'])
#         xlist.append(df1.to_xarray())

#     xarr = xr.concat(xlist, dim='icoast').min('icoast')
#     xarr = xarr.drop('lonlat')

#     return xr.merge([arr, xarr])

# dist = get_distance_to_coast(ds['precip'], 'Australia')

# plt.figure()
# dist['dist_to_coastline'].plot()
# plt.show()