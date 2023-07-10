# %%
#Importing Packages
import numpy as np
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
from src.residuals_functions import singleprocess_posterior_predictive_realisations

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
base_path = '/home/jez/Bias_Correction/'

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
antarctica_shapefile_path = f'{base_path}data/Antarctica_Shapefile/antarctica_shapefile.shp'
antarctica_gdf = gpd.read_file(antarctica_shapefile_path)
antarctica_gdf = antarctica_gdf.to_crs(rotated_coord_system)

#Loading observations
obs_residual_path = f'{base_path}data/ProcessedData/NST_Observations_Reformatted_Residual.nc'
ds_obs_stacked = xr.open_dataset(obs_residual_path)
ox = jnp.vstack(
    [
        ds_obs_stacked.glon.data,
        ds_obs_stacked.glat.data,
    ]
).T  # shape = [sites,2]
odata = jnp.array(ds_obs_stacked['Temperature Residual'].data).T
onoise = 1

#Loading output from MCMC
outfile_dir = f'{base_path}data/Residual_Model/'
ds_prior = xr.open_dataset(f'{outfile_dir}idata_residual_singlegp_prior.nc')
idata_matern32 = az.from_netcdf(f'{outfile_dir}idata_residual_singlegp_matern32.nc')
idata_expsq = az.from_netcdf(f'{outfile_dir}idata_residual_singlegp_expsq.nc')

#Loading grid for predictions
nst_climate_residual_path = f'{base_path}data/ProcessedData/MetUM_Reformatted_Residual.nc'
ds_predictions = xr.open_dataset(nst_climate_residual_path).drop_vars('Mean Jan Temperature Residual')
ds_predictions_stacked = ds_predictions.stack(X=(('grid_longitude','grid_latitude')))

nx = np.vstack(
    [
        ds_predictions_stacked.glon.data,
        ds_predictions_stacked.glat.data,
    ]
).T  # shape = [sites,2]

# %%
az.summary(ds_prior,hdi_prob=0.95)
# %%
az.summary(idata_matern32,hdi_prob=0.95)
# %%
az.summary(idata_expsq,hdi_prob=0.95)

# %%

realisations_matern32_obs = singleprocess_posterior_predictive_realisations(
        ox,ox,idata_matern32,onoise,10,10)
realisations_expsq_obs = singleprocess_posterior_predictive_realisations(
        ox,ox,idata_expsq,onoise,10,10)

difference_matern32 = realisations_matern32_obs.mean(axis=(0,1))-odata
difference_expsq = realisations_expsq_obs.mean(axis=(0,1))-odata

realisations_matern32 = singleprocess_posterior_predictive_realisations(
        nx,ox,idata_matern32,onoise,10,10)
realisations_expsq = singleprocess_posterior_predictive_realisations(
        nx,ox,idata_expsq,onoise,10,10)

# %%
ds_predictions_stacked["mean_matern32"]=(['X'],  realisations_matern32.mean(axis=(0,1)))
ds_predictions_stacked["std_matern32"]=(['X'],  realisations_matern32.std(axis=(0,1)))
ds_predictions_stacked["mean_expsq"]=(['X'],  realisations_expsq.mean(axis=(0,1)))
ds_predictions_stacked["std_expsq"]=(['X'],  realisations_expsq.std(axis=(0,1)))
ds_predictions = ds_predictions_stacked.unstack()

# %%
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

vars = ['mean_matern32','std_matern32','mean_expsq','std_expsq']

for i,var in zip(range(4),vars):
    plt.subplot(2, 2, i+1)
    ds_predictions[var].plot.pcolormesh(
        x='glon',y='glat')#,vmin=vmin,vmax=vmax)
    plt.title(var)

min_glon, max_glon = ds_predictions.glon.min()-0.5,ds_predictions.glon.max()+0.5
min_glat, max_glat = ds_predictions.glat.min()-0.5,ds_predictions.glat.max()+0.5
for ax in axs.ravel():
    ax.set_xlim([min_glon, max_glon])
    ax.set_ylim([min_glat, max_glat])
    antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)

for row in axs:
    for ax in row:
        ax.scatter(
            x=ds_obs_stacked.glon,y=ds_obs_stacked.glat,
            c=ds_obs_stacked['Temperature Residual'],
            cmap = 'RdBu_r',
            vmin=-6,
            vmax=6,
            edgecolors='k',
            linewidths=0.1)  

plt.tight_layout()

# %%

matern32_means = idata_matern32.posterior.mean()
matern32_kern = matern32_means.kern_var.data * kernels.Matern32(matern32_means.lengthscale.data)
expsq_means = idata_expsq.posterior.mean()
expsq_kern = expsq_means.kern_var.data * kernels.Matern32(expsq_means.lengthscale.data)

x0 = np.array([0])
x1 = np.arange(0,10,0.1)
kerns = [matern32_kern,expsq_kern]
labels = ['matern32_kern','expsq_kern']
for kern,label in zip(kerns,labels):
    plt.plot(x1,kern(x0,x1)[0],label=label)
plt.legend()

# %%

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
datasets = [difference_matern32,difference_expsq]
labels = ['difference_matern32','difference_expsq']
plots = []

for ax,data,label in zip(axs,datasets,labels):
    plots.append(
            ax.scatter(
            x=ds_obs_stacked.glon,y=ds_obs_stacked.glat,
            c=data,
            cmap = 'RdBu_r',
            vmin=-4,
            vmax=4,
            label=label)
    )
    ax.legend()

for plot in plots:
    plt.colorbar(plot)

min_glon, max_glon = ds_predictions.glon.min()-0.5,ds_predictions.glon.max()+0.5
min_glat, max_glat = ds_predictions.glat.min()-0.5,ds_predictions.glat.max()+0.5
for ax in axs.ravel():
    ax.set_xlim([min_glon, max_glon])
    ax.set_ylim([min_glat, max_glat])
    antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.1)

