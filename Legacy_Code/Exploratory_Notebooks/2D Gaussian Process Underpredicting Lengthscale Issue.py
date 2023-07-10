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
from src.examples_functions import tinygp_2process_model,truth_posterior_predictive,bias_posterior_predictive,posterior_predictive_realisations

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %%
#Generating Underlying Process Data 
X1 = np.arange(0,100,2)
X2 = np.arange(0,100,2)

ds = xr.Dataset(
    coords=dict(
        X1=("X1", X1),
        X2=("X2", X2),
    ),
)

ds_stacked = ds.stack(X=('X1', 'X2'))
X = np.array(list(map(np.array, ds_stacked.X.data)))

GP = GaussianProcess(1 * kernels.ExpSquared(5),X,diag=1e-5,mean=1.0)
GP2 = GaussianProcess(1 * kernels.ExpSquared(20),X,diag=1e-5,mean=-1.0)
GP3 = GaussianProcess(1 * kernels.ExpSquared(5)+kernels.ExpSquared(20),X,diag=2e-5,mean=0.0)

Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y3 = GP3.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
# ds_stacked["Y3"]=(['X'],  Y+Y2)
ds_stacked["Y3"]=(['X'],  Y3)
ds = ds_stacked.unstack()

obs_sample = np.random.choice(ds_stacked.X.size, size=100, replace=False)
da_obs_stacked = ds_stacked.Y[obs_sample]
climate_sample = np.random.choice(ds_stacked.X.size, size=200, replace=False)
da_climate_stacked = ds_stacked.Y3[climate_sample]

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plots = []
variables = ['Y','Y2','Y3']
titles = ['Truth','Bias','Climate']
# vmin = min(ds.min().data_vars.values())
# vmax = max(ds.max().data_vars.values())

for i,var,title in zip(range(1, 4), variables, titles):
    plt.subplot(1, 3, i)
    plots.append(
        ds[f'{var}'].plot.contourf(x='X1',y='X2',levels=50)
    )
    plt.title(title)

for i,var,da in zip([0,2],['Y','Y3'],[da_obs_stacked,da_climate_stacked]):
    ax = axs.flatten()[i]
    da.to_dataset().plot.scatter(x='X1',y='X2',hue=f'{var}',s=30, edgecolors='k',add_colorbar=False, ax=ax, add_title=False)
# https://docs.xarray.dev/en/stable/generated/xarray.plot.scatter.html

# %%
odata.shape

# %%
mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)





# %%
#Generating Model Inference Data
Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)
ds = ds_stacked.unstack()

#Observations
da_obs = ds.Y.isel(X1=slice(1,None,5),X2=slice(1,None,5))
# X1_condition = (ds.X1<25) | (ds.X1>75) 
# X2_condition = (ds.X2<25) | (ds.X2>75) 
# da_obs = da_obs.where(X1_condition|X2_condition,drop=True)
da_obs_stacked = da_obs.stack(X=('X1', 'X2'))

#Climate model
da_climate = ds.Y3.isel(X1=slice(None,None,5),X2=slice(None,None,5))
da_climate_stacked = da_climate.stack(X=('X1', 'X2'))

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
#Random Locations Inference
Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)
ds = ds_stacked.unstack()

obs_sample = np.random.choice(ds_stacked.X.size, size=100, replace=False)
da_obs = ds_stacked.Y[obs_sample]
climate_sample = np.random.choice(ds_stacked.X.size, size=100, replace=False)
da_climate = ds_stacked.Y3[climate_sample]

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
#Generating Model Inference Data
rng_key, rng_key_ = random.split(rng_key)
Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)
ds = ds_stacked.unstack()

#Observations
da_obs = ds.Y.isel(X1=slice(1,None,5),X2=slice(1,None,5))
X1_condition = (ds.X1<25) | (ds.X1>75) 
X2_condition = (ds.X2<25) | (ds.X2>75) 
da_obs = da_obs.where(X1_condition|X2_condition,drop=True)
da_obs_stacked = da_obs.stack(X=('X1', 'X2'))

#Climate model
da_climate = ds.Y3.isel(X1=slice(None,None,5),X2=slice(None,None,5))
da_climate_stacked = da_climate.stack(X=('X1', 'X2'))

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
#Generating Model Inference Data
rng_key, rng_key_ = random.split(rng_key)
Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)
ds = ds_stacked.unstack()

#Observations
da_obs = ds.Y.isel(X1=slice(1,None,5),X2=slice(1,None,5))
X1_condition = (ds.X1<25) | (ds.X1>75) 
X2_condition = (ds.X2<25) | (ds.X2>75) 
da_obs = da_obs.where(X1_condition|X2_condition,drop=True)
da_obs_stacked = da_obs.stack(X=('X1', 'X2'))

#Climate model
da_climate = ds.Y3.isel(X1=slice(None,None,5),X2=slice(None,None,5))
da_climate_stacked = da_climate.stack(X=('X1', 'X2'))

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
#Generating Model Inference Data
rng_key, rng_key_ = random.split(rng_key)
Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)
ds = ds_stacked.unstack()

#Observations
da_obs = ds.Y.isel(X1=slice(1,None,5),X2=slice(1,None,5))
# X1_condition = (ds.X1<25) | (ds.X1>75) 
# X2_condition = (ds.X2<25) | (ds.X2>75) 
# da_obs = da_obs.where(X1_condition|X2_condition,drop=True)
da_obs_stacked = da_obs.stack(X=('X1', 'X2'))

#Climate model
da_climate = ds.Y3.isel(X1=slice(None,None,5),X2=slice(None,None,5))
da_climate_stacked = da_climate.stack(X=('X1', 'X2'))

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
#Generating Model Inference Data
rng_key, rng_key_ = random.split(rng_key)
Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)
ds = ds_stacked.unstack()

#Observations
da_obs = ds.Y.isel(X1=slice(1,None,10),X2=slice(1,None,10))
# X1_condition = (ds.X1<25) | (ds.X1>75) 
# X2_condition = (ds.X2<25) | (ds.X2>75) 
# da_obs = da_obs.where(X1_condition|X2_condition,drop=True)
da_obs_stacked = da_obs.stack(X=('X1', 'X2'))

#Climate model
da_climate = ds.Y3.isel(X1=slice(None,None,10),X2=slice(None,None,10))
da_climate_stacked = da_climate.stack(X=('X1', 'X2'))

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)

# %%
#Generating Model Inference Data
rng_key, rng_key_ = random.split(rng_key)
Y = GP.sample(rng_key)
rng_key, rng_key_ = random.split(rng_key)
Y2 = GP2.sample(rng_key)

ds_stacked["Y"]=(['X'],  Y)
ds_stacked["Y2"]=(['X'],  Y2)
ds_stacked["Y3"]=(['X'],  Y+Y2)
ds = ds_stacked.unstack()

#Observations
da_obs = ds.Y.isel(X1=slice(1,None,10),X2=slice(1,None,10))
# X1_condition = (ds.X1<25) | (ds.X1>75) 
# X2_condition = (ds.X2<25) | (ds.X2>75) 
# da_obs = da_obs.where(X1_condition|X2_condition,drop=True)
da_obs_stacked = da_obs.stack(X=('X1', 'X2'))

#Climate model
da_climate = ds.Y3.isel(X1=slice(None,None,10),X2=slice(None,None,10))
da_climate_stacked = da_climate.stack(X=('X1', 'X2'))

odata = da_obs_stacked.dropna('X').data
ox = np.array(list(map(np.array, da_obs_stacked.dropna('X').X.data)))
cdata = da_climate_stacked.data
cx = np.array(list(map(np.array, da_climate_stacked.X.data)))

rng_key, rng_key_ = random.split(rng_key)
mcmc_2process = run_inference(tinygp_2process_model, rng_key_, 1000, 2000, cx,cdata=cdata,ox=ox,odata=odata,noise=1e-5)
idata_test = az.from_numpyro(mcmc_2process)
az.summary(idata_test,hdi_prob=0.95)