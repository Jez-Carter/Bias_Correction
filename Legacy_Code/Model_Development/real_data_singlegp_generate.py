# %%
#Importing Packages
import xarray as xr
from tinygp import kernels
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import arviz as az
import geopandas as gpd
import cartopy.crs as ccrs
import arviz as az
from src.model_fitting_functions import run_inference
from src.residuals_functions import singleGP_model

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
# Loading residual data
base_path = '/home/jez/Bias_Correction/'
nst_climate_residual_path = f'{base_path}data/ProcessedData/MetUM_Reformatted_Residual.nc'
obs_residual_path = f'{base_path}data/ProcessedData/NST_Observations_Reformatted_Residual.nc'

ds_obs_stacked = xr.open_dataset(obs_residual_path)
ds_climate = xr.open_dataset(nst_climate_residual_path)
ds_climate_stacked = ds_climate.stack(X=(('grid_longitude','grid_latitude'))).dropna('X','all')
ds_climate_stacked = ds_climate_stacked.where(ds_climate_stacked.lsm,drop=True)

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
antarctica_shapefile_path = f'{base_path}data/Antarctica_Shapefile/antarctica_shapefile.shp'
antarctica_gdf = gpd.read_file(antarctica_shapefile_path)
antarctica_gdf = antarctica_gdf.to_crs(rotated_coord_system)

obs_grid_longitudes = ds_obs_stacked.grid_longitude
obs_grid_latitudes = ds_obs_stacked.grid_latitude
ds_climate_nearest = ds_climate.sel(grid_longitude=obs_grid_longitudes,grid_latitude=obs_grid_latitudes,method='nearest')
da_difference_residual = ds_climate_nearest['Mean Jan Temperature Residual'] - ds_obs_stacked['Temperature Residual']
ds_obs_stacked['Difference Nearest Residual'] = (('Station_Lower'), da_difference_residual.data)

ox = jnp.vstack(
    [
        ds_obs_stacked.glon.data,
        ds_obs_stacked.glat.data,
    ]
).T  # shape = [sites,2]
odata = jnp.array(ds_obs_stacked['Temperature Residual'].data).T

# %%
fig, axs = plt.subplots(1, 1, figsize=(6, 5))
plot = axs.scatter(
    x=ds_obs_stacked.glon,y=ds_obs_stacked.glat,  
    c=odata) 
plt.colorbar(plot)
antarctica_gdf.boundary.plot(ax=axs, color="k", linewidth=0.1)
# %%   
mcmc_prior = run_inference(
    singleGP_model, rng_key_, 1000, 5000,
    ox[0],kern=kernels.Matern32)
idata_prior = az.from_numpyro(mcmc_prior).posterior.drop_vars('data').drop_dims('data_dim_0')
outfile_dir = '/home/jez/Bias_Correction/data/Residual_Model/'
idata_prior.to_netcdf(f'{outfile_dir}idata_residual_singlegp_prior.nc')

# %%
onoise = 1
mcmc_matern32 = run_inference(
    singleGP_model, rng_key_, 1000, 2000,
    ox,odata,onoise,kern=kernels.Matern32)
idata_matern32 = az.from_numpyro(mcmc_matern32)
outfile_dir = '/home/jez/Bias_Correction/data/Residual_Model/'
idata_matern32.to_netcdf(f'{outfile_dir}idata_residual_singlegp_matern32.nc')
# %%
onoise = 1
mcmc_expsq = run_inference(
    singleGP_model, rng_key_, 1000, 2000,
    ox,odata,onoise,kern=kernels.ExpSquared)
idata_expsq = az.from_numpyro(mcmc_expsq)
outfile_dir = '/home/jez/Bias_Correction/data/Residual_Model/'
idata_expsq.to_netcdf(f'{outfile_dir}idata_residual_singlegp_expsq.nc')
