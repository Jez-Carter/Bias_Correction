# %%
#Importing Packages
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import mpl_scatter_density
import numpy as np
import skgstat as skg
from sklearn import preprocessing

from src.helper_functions import grid_coords_to_2d_latlon_coords
from src.helper_functions import create_mask

# %%

#Loading climate data
base_path = '/home/jez/Bias_Correction/'
nst_climate_path = f'{base_path}data/ProcessedData/MetUM_Daily_TAS.nc'
ele_climate_path = f'{base_path}data/Antarctic_CORDEX_MetUM_0p44deg_orog.nc'
mask_path = f'{base_path}data/ProcessedData/MetUM_044_Masks.nc'

ds_nst_climate = xr.open_dataset(nst_climate_path)
ds_ele_climate = xr.open_dataset(ele_climate_path)
ds_mask = xr.open_dataset(mask_path)

ds_climate = xr.merge([ds_nst_climate,ds_ele_climate,ds_mask])
ds_climate = ds_climate.isel(time=(ds_climate.time.dt.month == 1))

ds_climate['Mean Jan Temperature'] = ds_climate['tas'].mean(['time'])
ds_climate['Variance Jan Temperature'] = ds_climate['tas'].var(['time'])

ds_climate_stacked = ds_climate.stack(X=('grid_latitude', 'grid_longitude'))

# %%
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
da = ds_climate['Mean Jan Temperature'].where(ds_climate.lsm)
da.plot(x='longitude',y='latitude',
    subplot_kws={"projection": map_proj},
    transform=ccrs.PlateCarree())
plt.gca().coastlines()

# %%
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
da = ds_climate['Variance Jan Temperature'].where(ds_climate.lsm)
da.plot(x='longitude',y='latitude',
    subplot_kws={"projection": map_proj},
    transform=ccrs.PlateCarree())
plt.gca().coastlines()

# %%
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
da = ds_climate['orog'].where(ds_climate.lsm)
da.plot(x='longitude',y='latitude',
    subplot_kws={"projection": map_proj},
    transform=ccrs.PlateCarree(),
    cmap="viridis")
plt.gca().coastlines()

# %%
# Semivariogram just considering Grid Coordinates & LSM
fig, axs = plt.subplots(2,2,figsize=(10,5))
params = ['Mean Jan Temperature','Variance Jan Temperature']
ds = ds_climate.isel(grid_latitude=slice(None,None,4),grid_longitude=slice(None,None,4))
ds_stacked = ds.stack(X=('grid_latitude', 'grid_longitude'))
ds_stacked = ds_stacked.where(ds_stacked.lsm,drop=True)

for param in params:
    da = ds_stacked[param]
    x,y = da.grid_latitude,da.grid_longitude
    coords = np.dstack([x,y]).reshape(-1,2)
    semivariogram = skg.Variogram(coords, da.data,n_lags=40)
    semivariogram.plot(axes=axs[:,params.index(param)][::-1])
    axs[:,params.index(param)][0].set_title(param)
plt.show()

# %%
# Semivariogram considering Grid Coordinates & Elevation & LSM
fig, axs = plt.subplots(2,2,figsize=(10,5))
params = ['Mean Jan Temperature','Variance Jan Temperature']
ds = ds_climate.isel(grid_latitude=slice(None,None,4),grid_longitude=slice(None,None,4))
ds_stacked = ds.stack(X=('grid_latitude', 'grid_longitude'))
ds_stacked = ds_stacked.where(ds_stacked.lsm,drop=True)
ds_stacked = ds_stacked.set_coords(("orog"))

for param in params:
    da = ds_stacked[param]
    x,y,z = da.grid_latitude,da.grid_longitude,da.orog
    coords = np.dstack([x,y,z]).reshape(-1,3)
    semivariogram = skg.Variogram(coords, da.data,n_lags=40)
    semivariogram.plot(axes=axs[:,params.index(param)][::-1])
    axs[:,params.index(param)][0].set_title(param)
plt.show()

# %%
# Semivariogram considering Grid Coordinates & Elevation & LSM & Normalising
fig, axs = plt.subplots(2,2,figsize=(10,5))
params = ['Mean Jan Temperature','Variance Jan Temperature']
ds = ds_climate.isel(grid_latitude=slice(None,None,4),grid_longitude=slice(None,None,4))
ds_stacked = ds.stack(X=('grid_latitude', 'grid_longitude'))
ds_stacked = ds_stacked.where(ds_stacked.lsm,drop=True)
ds_stacked = ds_stacked.set_coords(("orog"))

for param in params:
    da = ds_stacked[param]
    x,y,z = da.grid_latitude,da.grid_longitude,da.orog
    x = (x-x.mean())/x.std()
    y = (y-y.mean())/y.std()
    z = (z-z.mean())/z.std()
    coords = np.dstack([x,y,z]).reshape(-1,3)
    semivariogram = skg.Variogram(coords, da.data,n_lags=40)
    semivariogram.plot(axes=axs[:,params.index(param)][::-1])
    axs[:,params.index(param)][0].set_title(param)
plt.show()

# %%
# Semivariogram considering Grid Coordinates & Elevation & Latitude & LSM & Normalising
fig, axs = plt.subplots(2,2,figsize=(10,5))
params = ['Mean Jan Temperature','Variance Jan Temperature']
ds = ds_climate.isel(grid_latitude=slice(None,None,4),grid_longitude=slice(None,None,4))
ds_stacked = ds.stack(X=('grid_latitude', 'grid_longitude'))
ds_stacked = ds_stacked.where(ds_stacked.lsm,drop=True)
ds_stacked = ds_stacked.set_coords(("orog"))

for param in params:
    da = ds_stacked[param]
    x,y,z,w = da.grid_latitude,da.grid_longitude,da.orog,da.latitude
    x = (x-x.mean())/x.std()
    y = (y-y.mean())/y.std()
    z = (z-z.mean())/z.std()
    w = (w-w.mean())/w.std()
    coords = np.dstack([x,y,z,w]).reshape(-1,4)
    semivariogram = skg.Variogram(coords, da.data,n_lags=40)
    semivariogram.plot(axes=axs[:,params.index(param)][::-1])
    axs[:,params.index(param)][0].set_title(param)
plt.show()

# %%
# Semivariogram just considering Grid Coordinates & LSM & Normalised
fig, axs = plt.subplots(2,2,figsize=(10,5))
params = ['Mean Jan Temperature','Variance Jan Temperature']
ds = ds_climate.isel(grid_latitude=slice(None,None,4),grid_longitude=slice(None,None,4))
ds_stacked = ds.stack(X=('grid_latitude', 'grid_longitude'))
ds_stacked = ds_stacked.where(ds_stacked.lsm,drop=True)

for param in params:
    da = ds_stacked[param]
    x,y = da.grid_latitude,da.grid_longitude
    x = (x-x.mean())/x.std()
    y = (y-y.mean())/y.std()
    coords = np.dstack([x,y]).reshape(-1,2)
    semivariogram = skg.Variogram(coords, da.data,n_lags=40)
    semivariogram.plot(axes=axs[:,params.index(param)][::-1])
    axs[:,params.index(param)][0].set_title(param)
plt.show()

# %%
ds_climate.mean('time').stack(x=('grid_latitude','grid_longitude'))
# %%
ds = ds_climate.mean('time').stack(x=('grid_latitude','grid_longitude'))
# ds.plot.scatter('tas','orog',marker='x',alpha=0.2)#,c='b')
ds.plot.scatter('orog','latitude',hue='tas',marker='x',alpha=0.2)#,c='b')

# %%
ds = ds_climate.mean('time').stack(x=('grid_latitude','grid_longitude'))
ds = ds.where(ds.orog>10,drop=True)
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

ax.scatter_density(ds.tas.data, ds.orog.data)

# %%
# xarray.where(cond, x, y, keep_attrs=None)
ds

# %%
normalize