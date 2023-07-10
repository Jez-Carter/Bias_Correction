# %%

import numpy as np
import xarray as xr
import jax.numpy as jnp
from jax import random
import numpyro
numpyro.enable_x64()
import arviz as az
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs

from src.model_fitting_functions import run_inference
from src.examples_functions import hierarchical_model_mask
from src.helper_functions import remove_outliers
from src.examples_functions import posterior_predictive_realisations_hierarchical_mean, posterior_predictive_realisations_hierarchical_var
from src.examples_functions import create_levels,hierarchical_model,truth_posterior_predictive,bias_posterior_predictive

rng_key = random.PRNGKey(5)
rng_key, rng_key_ = random.split(rng_key)

# %%

base_path = '/home/jez/Bias_Correction/'
obs_path = f'{base_path}data/ProcessedData/NST_Observations_Ross_Subset_Reformatted.nc'
climate_path = f'{base_path}data/ProcessedData/MetUM_Ross_Subset_Reformatted.nc'
mask_path = f'{base_path}data/ProcessedData/MetUM_044_Masks.nc'

ds_obs = xr.open_dataset(obs_path)
ds_climate = xr.open_dataset(climate_path)

ds_obs_stacked = ds_obs.stack(D=('Year','Day'))
sorting_axis = ds_obs_stacked["Temperature()"].dims.index('D')
ds_obs_stacked["Temperature()"].data = np.sort(ds_obs_stacked["Temperature()"].data, axis=sorting_axis) # sorting Nans to back
ds_obs_stacked = ds_obs_stacked.dropna('D','all').dropna('Station_Lower','all')
min_days_data_condition = (ds_obs_stacked.isnull()==False).sum('D')>100
ds_obs_stacked = ds_obs_stacked.where(min_days_data_condition,drop=True)
white_island_condition = (ds_obs_stacked.Station_Lower=='white-island')==False
ds_obs_stacked = ds_obs_stacked.where(white_island_condition,drop=True)
ds_obs_stacked['non_null_values'] = (ds_obs_stacked['Temperature()'].isnull()==False).sum('D')

ds_climate['tas']=ds_climate['tas']-273.15
ds_climate = remove_outliers(ds_climate['tas'].mean('time'),ds_climate, dims=['grid_latitude','grid_longitude'], perc=[0.02,0.98])
ds_climate = ds_climate.dropna('grid_latitude','all').dropna('grid_longitude','all')
ds_climate_stacked = ds_climate.stack(X=(('grid_longitude','grid_latitude'))).dropna('X','all')

obs_grid_longitudes = ds_obs_stacked.isel(D=0).grid_longitude
obs_grid_latitudes = ds_obs_stacked.isel(D=0).grid_latitude
ds_climate_nearest = ds_climate.sel(grid_longitude=obs_grid_longitudes,grid_latitude=obs_grid_latitudes,method='nearest')
da_difference_mean = ds_climate_nearest.tas.mean('time') - ds_obs_stacked['Temperature()'].mean('D')
da_difference_var = ds_climate_nearest.tas.var('time') - ds_obs_stacked['Temperature()'].var('D')
da_difference_logvar = np.log(ds_climate_nearest.tas.var('time')) - np.log(ds_obs_stacked['Temperature()'].var('D'))
ds_obs_stacked['Difference Nearest Mean'] = (('Station_Lower'), da_difference_mean.data)
ds_obs_stacked['Difference Nearest Var'] = (('Station_Lower'), da_difference_var.data)
ds_obs_stacked['Difference Nearest LogVar'] = (('Station_Lower'), da_difference_logvar.data)

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
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
da = ds_obs_stacked['non_null_values']
s=10
plot = ax.scatter(x=da.glon,y=da.glat,c=da,s=s)

min_glon, max_glon = da.glon.min()-0.5,da.glon.max()+0.5
min_glat, max_glat = da.glat.min()-0.5,da.glat.max()+0.5
ax.set_xlim([min_glon, max_glon])
ax.set_ylim([min_glat, max_glat])
antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.colorbar(plot)
plt.title('Days of Data')
plt.tight_layout()

# %%
ox = np.vstack(
    [
        ds_obs_stacked.glon.data,
        ds_obs_stacked.glat.data,
    ]
).T  # shape = [sites,2]
odata = jnp.array(ds_obs_stacked['Temperature()'].data).T
omask = jnp.logical_not(jnp.isnan(odata)) 

cx = np.vstack(
    [
        ds_climate_stacked.glon.data,
        ds_climate_stacked.glat.data,
    ]
).T  # shape = [sites,2]
cdata = ds_climate_stacked.tas.data

print(f'ox shape = {ox.shape}')
print(f'cx shape = {cx.shape}')
print(f'odata shape = {odata.shape}')
print(f'cdata shape = {cdata.shape}')
print(f'ox mean = {ox.mean()}')
print(f'cx mean = {cx.mean()}')
print(f'odata mean = {odata[omask==1].mean()}')
print(f'cdata mean = {cdata.mean()}')
print(f'odata std = {odata[omask==1].std()}')
print(f'cdata std = {cdata.std()}')


# %%
# vmin_mean = da.mean('time').min()
# vmax_mean = da.mean('time').max()
# vmin_var = da.var('time').min()
# vmax_var = da.var('time').max()
# vmin_logvar = np.log(da.var('time')).min()
# vmax_logvar = np.log(da.var('time')).max()

# %%
fig, axs = plt.subplots(3, 1, figsize=(5, 10))
da = ds_climate['tas']
s=10
vmin_mean = min(np.nanmean(odata,axis=0))
vmax_mean = max(np.nanmean(odata,axis=0))
vmin_var = min(np.nanvar(odata,axis=0))
vmax_var = max(np.nanvar(odata,axis=0))
vmin_logvar = min(np.log(np.nanvar(odata,axis=0)))
vmax_logvar = max(np.log(np.nanvar(odata,axis=0)))

da.mean('time').plot.pcolormesh(x='glon',y='glat',ax=axs[0],alpha=0.8,vmin=vmin_mean,vmax=vmax_mean)
axs[0].scatter(x=ox[:,0],y=ox[:,1],c=np.nanmean(odata,axis=0),vmin=vmin_mean,vmax=vmax_mean,s=s)

da.var('time').plot.pcolormesh(x='glon',y='glat',ax=axs[1],alpha=0.8,vmin=vmin_var,vmax=vmax_var)
axs[1].scatter(x=ox[:,0],y=ox[:,1],c=np.nanvar(odata,axis=0),vmin=vmin_var,vmax=vmax_var,s=s)

np.log(da.var('time')).plot.pcolormesh(x='glon',y='glat',ax=axs[2],alpha=0.8,vmin=vmin_logvar,vmax=vmax_logvar)
axs[2].scatter(x=ox[:,0],y=ox[:,1],c=np.log(np.nanvar(odata,axis=0)),vmin=vmin_logvar,vmax=vmax_logvar,s=s)

min_glon, max_glon = da.glon.min()-0.5,da.glon.max()+0.5
min_glat, max_glat = da.glat.min()-0.5,da.glat.max()+0.5
for ax in axs.ravel():
    ax.set_xlim([min_glon, max_glon])
    ax.set_ylim([min_glat, max_glat])
    antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

# %%
fig, axs = plt.subplots(3, 1, figsize=(5, 10))
s=10
plots=[]
plots.append(axs[0].scatter(x=ds_obs_stacked.glon,y=ds_obs_stacked.glat,c=ds_obs_stacked['Difference Nearest Mean'],s=s))
plots.append(axs[1].scatter(x=ds_obs_stacked.glon,y=ds_obs_stacked.glat,c=ds_obs_stacked['Difference Nearest Var'],s=s))
plots.append(axs[2].scatter(x=ds_obs_stacked.glon,y=ds_obs_stacked.glat,c=ds_obs_stacked['Difference Nearest LogVar'],s=s))

min_glon, max_glon = da.glon.min()-0.5,da.glon.max()+0.5
min_glat, max_glat = da.glat.min()-0.5,da.glat.max()+0.5
for ax in axs.ravel():
    ax.set_xlim([min_glon, max_glon])
    ax.set_ylim([min_glat, max_glat])
    antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)

for plot in plots:
    plt.colorbar(plot)
plt.tight_layout()

# %%
cdata_sample = cdata[::5]
om_noise = 1e-4
cm_noise = 1e-4
olv_noise = 1e-4
clv_noise = 1e-4

mcmc_hierarchical = run_inference(
    hierarchical_model_mask, rng_key_, 1000, 2000,
    cx,ox=ox,cdata=cdata_sample,odata=odata,obs_mask=omask,
    om_noise=om_noise,cm_noise=cm_noise,olv_noise=olv_noise,clv_noise=clv_noise)

# %%

mcmc_hierarchical = run_inference(hierarchical_model_mask, rng_key_, 1000, 2000, cx,cdata=cdata_sample,ox=ox,odata=odata,noise=1e-4,obs_mask=omask)

# %%
idata_hierarchical = az.from_numpyro(
    mcmc_hierarchical,
    coords={"cx": np.arange(len(cx)),"ox": np.arange(len(ox))},
    dims={"lvc": ["cx"],"mc": ["cx"],"lvt": ["ox"],"mt": ["ox"]},
)
# %%
#Saving Output from MCMC
outfile_dir = '/home/jez/Bias_Correction/data/Hierarchical_Model/'
idata_hierarchical.to_netcdf(f'{outfile_dir}idata_hierarchical_nst.nc')
#Loading Output from MCMC
outfile_dir = '/home/jez/Bias_Correction/data/Hierarchical_Model/'
idata_hierarchical = az.from_netcdf(f'{outfile_dir}idata_hierarchical_nst.nc')

# %%
om_noise = 1e-5
cm_noise = 1e-5
olv_noise = 1e-5
clv_noise = 1e-5
jitter = 1e-5

ds_predictions = xr.open_dataset(climate_path)[['tas']].dropna('grid_latitude','all').dropna('grid_longitude','all')
ds_predictions = ds_predictions.drop_vars('tas')
ds_predictions = ds_predictions.isel(time=0).drop_vars('time')
ds_predictions_stacked = ds_predictions.stack(X=(('grid_longitude','grid_latitude')))

nx = np.vstack(
    [
        ds_predictions_stacked.glon.data,
        ds_predictions_stacked.glat.data,
    ]
).T  # shape = [sites,2]

num_parameter_realisations = 20
num_posterior_pred_realisations = 20
noise=1e-4

truth_realisations_mean = posterior_predictive_realisations_hierarchical_mean(
    truth_posterior_predictive,nx,ox,cx,idata_hierarchical,
    om_noise,cm_noise,jitter,
    num_parameter_realisations,num_posterior_pred_realisations)
bias_realisations_mean = posterior_predictive_realisations_hierarchical_mean(
    bias_posterior_predictive,nx,ox,cx,idata_hierarchical,
    om_noise,cm_noise,jitter,
    num_parameter_realisations,num_posterior_pred_realisations)

truth_realisations_logvar = posterior_predictive_realisations_hierarchical_var(
    truth_posterior_predictive,nx,ox,cx,idata_hierarchical,
    olv_noise,clv_noise,jitter,
    num_parameter_realisations,num_posterior_pred_realisations)
bias_realisations_logvar = posterior_predictive_realisations_hierarchical_var(
    bias_posterior_predictive,nx,ox,cx,idata_hierarchical,
    olv_noise,clv_noise,jitter,
    num_parameter_realisations,num_posterior_pred_realisations)
truth_realisations_var = np.exp(truth_realisations_logvar)
bias_realisations_var = np.exp(bias_realisations_logvar)

ds_predictions_stacked["mt_mean"]=(['X'],  truth_realisations_mean.mean(axis=(0,1)))
ds_predictions_stacked["mt_std"]=(['X'],  truth_realisations_mean.std(axis=(0,1)))
ds_predictions_stacked["mb_mean"]=(['X'],  bias_realisations_mean.mean(axis=(0,1)))
ds_predictions_stacked["mb_std"]=(['X'],  bias_realisations_mean.std(axis=(0,1)))
ds_predictions_stacked["mc_mean"]=(['X'],  (truth_realisations_mean+bias_realisations_mean).mean(axis=(0,1)))
ds_predictions_stacked["mc_std"]=(['X'],  (truth_realisations_mean+bias_realisations_mean).std(axis=(0,1)))

ds_predictions_stacked["vt_mean"]=(['X'],  truth_realisations_var.mean(axis=(0,1)))
ds_predictions_stacked["vt_std"]=(['X'],  truth_realisations_var.std(axis=(0,1)))
ds_predictions_stacked["vb_mean"]=(['X'],  bias_realisations_var.mean(axis=(0,1)))
ds_predictions_stacked["vb_std"]=(['X'],  bias_realisations_var.std(axis=(0,1)))
ds_predictions_stacked["vc_mean"]=(['X'],  (truth_realisations_var+bias_realisations_var).mean(axis=(0,1)))
ds_predictions_stacked["vc_std"]=(['X'],  (truth_realisations_var+bias_realisations_var).std(axis=(0,1)))

ds_predictions_stacked["lvt_mean"]=(['X'],  truth_realisations_logvar.mean(axis=(0,1)))
ds_predictions_stacked["lvt_std"]=(['X'],  truth_realisations_logvar.std(axis=(0,1)))
ds_predictions_stacked["lvb_mean"]=(['X'],  bias_realisations_logvar.mean(axis=(0,1)))
ds_predictions_stacked["lvb_std"]=(['X'],  bias_realisations_logvar.std(axis=(0,1)))
ds_predictions_stacked["lvc_mean"]=(['X'],  (truth_realisations_logvar+bias_realisations_logvar).mean(axis=(0,1)))
ds_predictions_stacked["lvc_std"]=(['X'],  (truth_realisations_logvar+bias_realisations_logvar).std(axis=(0,1)))

ds_predictions = ds_predictions_stacked.unstack()

# %%
# ds_predictions['mc_mean'].plot.pcolormesh(x='glon',y='glat')
# ds_predictions['mt_mean'].plot.pcolormesh(x='glon',y='glat')
# ds_predictions['mt_std'].plot.pcolormesh(x='glon',y='glat')
# ds_predictions
ds_predictions['mt_mean'].min()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
da = ds_predictions['mt_mean']
vmin = da.min()
vmax = da.max()
da.plot.pcolormesh(x='glon',y='glat',ax=ax,alpha=0.8,vmin=vmin,vmax=vmax)
ax.scatter(x=ox[:,0],y=ox[:,1],c=np.nanmean(odata,axis=0),vmin=vmin,vmax=vmax)

ax.set_xlim([min_glon, max_glon])
ax.set_ylim([min_glat, max_glat])
antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)
plt.tight_layout()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
da = ds_predictions['mt_std']
vmin = da.min()
vmax = da.max()
da.plot.pcolormesh(x='glon',y='glat',ax=ax,alpha=0.8,vmin=vmin,vmax=vmax)

ax.set_xlim([min_glon, max_glon])
ax.set_ylim([min_glat, max_glat])
antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)
plt.tight_layout()

# %%
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
da = ds_climate['tas']
# da = da.isel(time=slice(None,None,10))

vmin_mean = da.mean('time').min()
vmax_mean = da.mean('time').max()
vmin_var = da.var('time').min()
vmax_var = da.var('time').max()

da.mean('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[0],alpha=0.8,vmin=vmin_mean,vmax=vmax_mean)
axs.ravel()[0].scatter(x=ox[:,0],y=ox[:,1],c=np.nanmean(odata,axis=0),vmin=vmin_mean,vmax=vmax_mean)

da.var('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[1],alpha=0.8,vmin=vmin_var,vmax=vmax_var)
axs.ravel()[1].scatter(x=ox[:,0],y=ox[:,1],c=np.nanvar(odata,axis=0),vmin=vmin_var,vmax=vmax_var)

for i,var,title in zip([3,5], ['mt_mean','mt_std'], ['Mean Truth','Uncertainty in Mean Truth']):
    plt.subplot(3, 2, i)
    if var=='mt_mean':
        vmin,vmax = vmin_mean,vmax_mean
    else:
        vmin,vmax = None,None
    # ds_predictions[var].plot.contourf(x='glon',y='glat',vmin=vmin,vmax=vmax)
    ds_predictions[var].plot.pcolormesh(x='glon',y='glat',vmin=vmin,vmax=vmax)
    plt.title(title)

for i,var,title in zip([4,6], ['vt_mean','vt_std'], ['Variance Truth','Uncertainty in Variance Truth']):
    plt.subplot(3, 2, i)
    if var=='vt_mean':
        vmin,vmax = vmin_var,vmax_var
    else:
        vmin,vmax = None,None
    # ds_predictions[var].plot.contourf(x='glon',y='glat',vmin=vmin,vmax=vmax)
    ds_predictions[var].plot.pcolormesh(x='glon',y='glat',vmin=vmin,vmax=vmax)
    plt.title(title)


min_glon, max_glon = da.glon.min()-0.5,da.glon.max()+0.5
min_glat, max_glat = da.glat.min()-0.5,da.glat.max()+0.5
for ax in axs.ravel():
    ax.set_xlim([min_glon, max_glon])
    ax.set_ylim([min_glat, max_glat])
    antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

# %%
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
da = ds_climate['tas']
# da = da.isel(time=slice(None,None,10))

vmin_mean = da.mean('time').min()
vmax_mean = da.mean('time').max()
vmin_var = da.var('time').min()
vmax_var = da.var('time').max()

da.mean('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[0],alpha=0.8,vmin=vmin_mean,vmax=vmax_mean)
axs.ravel()[0].scatter(x=ox[:,0],y=ox[:,1],c=np.nanmean(odata,axis=0),vmin=vmin_mean,vmax=vmax_mean)

da.var('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[1],alpha=0.8,vmin=vmin_var,vmax=vmax_var)
axs.ravel()[1].scatter(x=ox[:,0],y=ox[:,1],c=np.nanvar(odata,axis=0),vmin=vmin_var,vmax=vmax_var)

for i,var,title in zip([3,5], ['mb_mean','mb_std'], ['Mean Bias','Uncertainty in Mean Bias']):
    plt.subplot(3, 2, i)
    ds_predictions[var].plot.contourf(x='glon',y='glat')
    plt.title(title)

for i,var,title in zip([4,6], ['vb_mean','vb_std'], ['Variance Bias','Uncertainty in Variance Bias']):
    plt.subplot(3, 2, i)
    ds_predictions[var].plot.contourf(x='glon',y='glat')
    plt.title(title)

min_glon, max_glon = da.glon.min()-0.5,da.glon.max()+0.5
min_glat, max_glat = da.glat.min()-0.5,da.glat.max()+0.5
for ax in axs.ravel():
    ax.set_xlim([min_glon, max_glon])
    ax.set_ylim([min_glat, max_glat])
    antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

# %%
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
da = ds_climate['tas']
# da = da.isel(time=slice(None,None,10))

vmin_mean = da.mean('time').min()
vmax_mean = da.mean('time').max()
vmin_var = da.var('time').min()
vmax_var = da.var('time').max()

da.mean('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[0],alpha=0.8,vmin=vmin_mean,vmax=vmax_mean)
axs.ravel()[0].scatter(x=ox[:,0],y=ox[:,1],c=np.nanmean(odata,axis=0),vmin=vmin_mean,vmax=vmax_mean)

da.var('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[1],alpha=0.8,vmin=vmin_var,vmax=vmax_var)
axs.ravel()[1].scatter(x=ox[:,0],y=ox[:,1],c=np.nanvar(odata,axis=0),vmin=vmin_var,vmax=vmax_var)

for i,var,title in zip([3,5], ['mc_mean','mc_std'], ['Mean Climate','Uncertainty in Mean Climate']):
    plt.subplot(3, 2, i)
    ds_predictions[var].plot.contourf(x='glon',y='glat')
    plt.title(title)

for i,var,title in zip([4,6], ['vc_mean','vc_std'], ['Variance Climate','Uncertainty in Variance Climate']):
    plt.subplot(3, 2, i)
    ds_predictions[var].plot.contourf(x='glon',y='glat')
    plt.title(title)

min_glon, max_glon = da.glon.min()-0.5,da.glon.max()+0.5
min_glat, max_glat = da.glat.min()-0.5,da.glat.max()+0.5
for ax in axs.ravel():
    ax.set_xlim([min_glon, max_glon])
    ax.set_ylim([min_glat, max_glat])
    antarctica_gdf.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

# %%
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
da = ds_climate['tas']
# da = da.isel(time=slice(None,None,10))

vmin_mean = da.mean('time').min()
vmax_mean = da.mean('time').max()
vmin_var = da.var('time').min()
vmax_var = da.var('time').max()

da.mean('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[0],alpha=0.8,vmin=vmin_mean,vmax=vmax_mean)
axs.ravel()[0].scatter(x=ox[:,0],y=ox[:,1],c=np.nanmean(odata_sample,axis=0),vmin=vmin_mean,vmax=vmax_mean)

da.var('time').plot.pcolormesh(x='glon',y='glat',ax=axs.ravel()[1],alpha=0.8,vmin=vmin_var,vmax=vmax_var)
axs.ravel()[1].scatter(x=ox[:,0],y=ox[:,1],c=np.nanvar(odata_sample,axis=0),vmin=vmin_var,vmax=vmax_var)

for i,var,title in zip([3,5], ['mc_mean','mc_std'], ['Mean Climate','Uncertainty in Mean Climate']):
    plt.subplot(3, 2, i)
    ds_predictions[var].plot.contourf(x='glon',y='glat')
    plt.title(title)

# %%
cx_sample = cx[::5]
cdata_sample = cdata[::5]
odata_sample = odata[:100]
omask_sample = omask[:100]

# %%
odata_sample.shape