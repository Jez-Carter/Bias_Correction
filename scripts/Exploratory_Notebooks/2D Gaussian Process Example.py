# %%
#Importing Packages
import numpy as np
import xarray as xr
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import matplotlib.pyplot as plt
import arviz as az
from src.model_fitting_functions import run_inference
from src.examples_functions import create_levels,tinygp_2process_model,truth_posterior_predictive,bias_posterior_predictive,posterior_predictive_realisations

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
#Creating Underlying Process Data 

X1 = np.arange(0,105,2)
X2 = np.arange(0,105,2)

ds = xr.Dataset(
    coords=dict(
        X1=("X1", X1),
        X2=("X2", X2),
    ),
)

ds_stacked = ds.stack(X=('X1', 'X2'))
X = np.array(list(map(np.array, ds_stacked.X.data)))

# #Truth
GP = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5,mean=1.0)
Y = GP.sample(rng_key)

#Bias
GP2 = GaussianProcess(1 * kernels.ExpSquared(40),X,diag=1e-5,mean=-1.0)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)

ds = ds_stacked.unstack()

# %%
#Observations
da_obs = ds.Y.isel(X1=slice(1,None,5),X2=slice(1,None,5))
X1_condition = (ds.X1<25) | (ds.X1>75) 
X2_condition = (ds.X2<25) | (ds.X2>75) 
da_obs = da_obs.where(X1_condition|X2_condition,drop=True)
da_obs_stacked = da_obs.stack(X=('X1', 'X2'))

#Climate model
da_climate = ds.Y3.isel(X1=slice(None,None,5),X2=slice(None,None,5))
da_climate_stacked = da_climate.stack(X=('X1', 'X2'))

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plots = []
variables = ['Y','Y2','Y3']
titles = ['Truth','Bias','Climate']
levels = create_levels(ds,0.25,0,center=True)

for i,var,title in zip(range(1, 4), variables, titles):
    plt.subplot(1, 3, i)
    plots.append(
        ds[f'{var}'].plot.contourf(x='X1',y='X2',levels=levels,center=0)
    )
    plt.title(title)

axs[0].scatter(da_obs_stacked.dropna('X').X1, da_obs_stacked.dropna('X').X2, s=30, marker='x', c='k')
axs[2].scatter(da_climate_stacked.X1, da_obs_stacked.X2, s=30, marker='+',c='k')

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(da_obs_stacked.dropna('X').X1, da_obs_stacked.dropna('X').X2, s=30, marker='x', c='k')
ax.scatter(da_climate_stacked.X1, da_obs_stacked.X2, s=30, marker='+',c='k')

# %%
val = 52
cdc = 'tab:orange' #climate_data_colour
odc = 'tab:blue' #observation_data_colour
bdc = 'tab:green'#bias_data_colour

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

da_obs_slice=da_obs.sel(X2=val)
da_climate_slice=da_climate.sel(X2=val,method="nearest")
ax.scatter(da_obs_slice.X1, da_obs_slice.data, s=30, marker='x',c=odc,label = f'Truth Observations X={da_obs_slice.X2.data}')
ax.scatter(da_climate_slice.X1, da_climate_slice.data, s=30, marker='+',c=cdc,label = f'Climate Output X={da_climate_slice.X2.data}')

ds_slice = ds.sel(X2=val)
ds_slice['Y'].plot(label = f'Underlying Truth X={val}',c=odc)
ds_slice['Y2'].plot(label = f'Underlying Bias X={val}',c=bdc)
ds_slice['Y3'].plot(label = f'Underlying Climate X={val}',c=cdc)

plt.legend()
plt.tight_layout()

# %%
odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

# %%

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)

# %%
#Saving Output from MCMC
outfile_dir = '/home/jez/Bias_Correction/data/Examples_Output/'
idata_2process = az.from_numpyro(mcmc_2process)
idata_2process.to_netcdf(f'{outfile_dir}idata_2process_2d.nc')

# %%
#Loading Output from MCMC
outfile_dir = '/home/jez/Bias_Correction/data/Examples_Output/'
idata_2process = az.from_netcdf(f'{outfile_dir}idata_2process_2d.nc')

# %%
#Realisations
ds_predictions = ds.isel(X1=slice(None,None,2),X2=slice(None,None,2))
ds_predictions_stacked = ds_predictions.stack(X=('X1', 'X2'))
nx = np.array(list(map(np.array, ds_predictions_stacked.X.data)))

num_parameter_realisations = 20
num_posterior_pred_realisations = 20

truth_realisations = posterior_predictive_realisations(truth_posterior_predictive,nx,ox,cx,odata,cdata,idata_2process,num_parameter_realisations,num_posterior_pred_realisations)
bias_realisations = posterior_predictive_realisations(bias_posterior_predictive,nx,ox,cx,odata,cdata,idata_2process,num_parameter_realisations,num_posterior_pred_realisations)


ds_predictions_stacked["mean_truth"]=(['X'],  truth_realisations.mean(axis=(0,1)))
ds_predictions_stacked["std_truth"]=(['X'],  truth_realisations.std(axis=(0,1)))
ds_predictions_stacked["mean_bias"]=(['X'],  bias_realisations.mean(axis=(0,1)))
ds_predictions_stacked["std_bias"]=(['X'],  bias_realisations.std(axis=(0,1)))

ds_predictions = ds_predictions_stacked.unstack()

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plots = []
variables = ['mean_truth','std_truth']
titles = ['Mean Pred. - Truth','Stddev Pred. - Truth']
truth_levels = create_levels(ds[['Y']],0.25,0,center=True)
uncertainty_levels = create_levels(ds_predictions[['std_truth']],0.1,1)

plt.subplot(1, 3, 1)
plots.append(ds['Y'].plot.contourf(x='X1',y='X2',levels=truth_levels,ax=axs.flatten()[0]))
plt.title('Truth')

for i,var,title,levels in zip(range(2, 4), variables, titles,[truth_levels,uncertainty_levels]):
    plt.subplot(1, 3, i)
    plots.append(
        ds_predictions[f'{var}'].plot.contourf(x='X1',y='X2',levels=levels)
    )
    plt.title(title)

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plots = []
variables = ['mean_bias','std_bias']
titles = ['Mean Pred. - Bias','Stddev Pred. - Bias']
truth_levels = create_levels(ds[['Y2']],0.25,0,center=True)
uncertainty_levels = create_levels(ds_predictions[['std_bias']],0.1,1)

plt.subplot(1, 3, 1)
plots.append(ds['Y2'].plot.contourf(x='X1',y='X2',levels=truth_levels,ax=axs.flatten()[0]))
plt.title('Truth')

for i,var,title,levels in zip(range(2, 4), variables, titles,[truth_levels,uncertainty_levels]):
    plt.subplot(1, 3, i)
    plots.append(
        ds_predictions[f'{var}'].plot.contourf(x='X1',y='X2',levels=levels)
    )
    plt.title(title)

# %%
val = 52
cdc = 'tab:orange' #climate_data_colour
odc = 'tab:blue' #observation_data_colour
bdc = 'tab:green'#bias_data_colour
tpdc = 'tab:purple'#truth_prediction_data_colour
bpdc = 'tab:olive'#bias_prediction_data_colour

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

da_obs_slice=da_obs.sel(X2=val)
da_climate_slice=da_climate.sel(X2=val,method="nearest")
ax.scatter(da_obs_slice.X1, da_obs_slice.data, s=30, marker='x',c=odc,label = f'Truth Observations X={da_obs_slice.X2.data}')
ax.scatter(da_climate_slice.X1, da_climate_slice.data, s=30, marker='+',c=cdc,label = f'Climate Output X={da_climate_slice.X2.data}')

ds_slice = ds.sel(X2=val)
ds_slice['Y'].plot(label = f'Underlying Truth X={val}',c=odc)
ds_slice['Y2'].plot(label = f'Underlying Bias X={val}',c=bdc)
ds_slice['Y3'].plot(label = f'Underlying Climate X={val}',c=cdc)

ds_predictions_slice = ds_predictions.sel(X2=val)
truth_mean_pred = ds_predictions_slice['mean_truth'].data
truth_std_pred = ds_predictions_slice['std_truth'].data
bias_mean_pred = ds_predictions_slice['mean_bias'].data
bias_std_pred = ds_predictions_slice['std_bias'].data

ax.plot(ds_predictions_slice.X1, truth_mean_pred, c=tpdc,label = f'Truth Predictions X={val}')
ax.fill_between(ds_predictions_slice.X1,truth_mean_pred+truth_std_pred,truth_mean_pred-truth_std_pred,label=f'Truth StdDev Pred X={val}',color=tpdc,alpha=0.3)
ax.plot(ds_predictions_slice.X1, bias_mean_pred, c=bpdc,label = f'Bias Predictions X={val}')
ax.fill_between(ds_predictions_slice.X1,bias_mean_pred+bias_std_pred,bias_mean_pred-bias_std_pred,label=f'Bias StdDev Pred X={val}',color=bpdc,alpha=0.3)

plt.legend()
plt.tight_layout()
