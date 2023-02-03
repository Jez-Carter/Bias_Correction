'''
 
'''

import xarray as xr
import arviz as az
import numpy as np
import jax
from scipy import stats
from src.helper_functions import standardise
from src.helper_functions import build_gp
from src.helper_functions import empirical_cdf
from src.helper_functions import quantile_mapping

#Loading climate model data
climate_ds = xr.open_dataset(f'/data/notebooks/jupyterlab-biascorrlab/data/ProcessedData/AP_Daily_Snowfall_044.nc')
climate_ds_jan = climate_ds.where((climate_ds['time.month'] == 1), drop=True)

#Loading BG inference data
bg_idata_path = f'/data/notebooks/jupyterlab-biascorrlab/data/Lima2021/AP_Daily_Snowfall_044_BGFit.nc'
bg_idata = az.from_netcdf(bg_idata_path)

#Loading GP inference data 
gp_idata_path = f'/data/notebooks/jupyterlab-biascorrlab/data/Lima2021/AP_Daily_Snowfall_044_GPFit.nc'
gp_idata = az.from_netcdf(gp_idata_path)

#Standardising climate model coordinates by the same transform used with the reference data that the model was fit to
grid_longitude_standardised = standardise(climate_ds.grid_longitude,refdata=gp_idata.observed_data.grid_longitude)
grid_latitude_standardised = standardise(climate_ds.grid_latitude,refdata=gp_idata.observed_data.grid_latitude)

#Reshaping coordinates for shape that tinygp requires
x_grid_latitude_standardised,x_grid_longitude_standardised = np.meshgrid(grid_latitude_standardised,grid_longitude_standardised)
X = np.vstack(
    [
        x_grid_longitude_standardised.ravel(),
        x_grid_latitude_standardised.ravel(),
    ]
).T  # shape = [sites,2]

#Constructing the GP objects from the inference data - Note the build_gp function takes the expectations of the parameters for the GP
alpha_gp = build_gp(gp_idata,'alpha')
p_gp = build_gp(gp_idata,'p')

#Conditioning based on observed data and on new coordinates
_, alpha_gpcond = alpha_gp.condition(gp_idata.observed_data['alpha'].data, X)
_, p_gpcond = p_gp.condition(gp_idata.observed_data['p'].data, X)

#Returning expectations of the a0, a1 and betavar parameters
bg_idata_means = bg_idata.sel(months=1).posterior.mean(['chain','draw'])
a0_mean = bg_idata_means['a0']
a1_mean = bg_idata_means['a1']
betavar_mean = bg_idata_means['betavar']

#Returning the expected values of the mean and variance for the parameters p,alpha and beta at the new coordinates
alpha_pred_mean = alpha_gpcond.mean.reshape(35,30)
beta_pred_mean = np.exp(a0_mean.data+a1_mean.data*alpha_pred_mean+betavar_mean.data/2)
p_pred_mean = p_gpcond.mean.reshape(35,30)

alpha_pred_var = alpha_gpcond.variance.reshape(35,30)
beta_pred_var = (np.exp(betavar_mean.data)-1)*np.exp(2*a0_mean.data+a1_mean.data*alpha_pred_mean+betavar_mean.data)
p_pred_var = p_gpcond.variance.reshape(35,30)

#############################################################
samples_number = 20
alpha_pred_sample = alpha_gpcond.sample(jax.random.PRNGKey(1), shape=(samples_number,)).reshape(samples_number,35,30)
beta_pred_sample = np.exp(a0_mean.data+a1_mean.data*alpha_pred_sample+betavar_mean.data/2)
p_pred_sample = p_gpcond.sample(jax.random.PRNGKey(1), shape=(samples_number,)).reshape(samples_number,35,30)
#############################################################

#Updating the climate dataset to include these expected values for p, alpha and beta - Derived just from extrapolating the 'observations'
climate_ds_jan["alpha_pred_mean"]=(['grid_latitude', 'grid_longitude'],  alpha_pred_mean.T)
climate_ds_jan["beta_pred_mean"]=(['grid_latitude', 'grid_longitude'],  beta_pred_mean.T)
climate_ds_jan["p_pred_mean"]=(['grid_latitude', 'grid_longitude'],  p_pred_mean.T)

climate_ds_jan["alpha_pred_var"]=(['grid_latitude', 'grid_longitude'],  alpha_pred_var.T)
climate_ds_jan["beta_pred_var"]=(['grid_latitude', 'grid_longitude'],  beta_pred_var.T)
climate_ds_jan["p_pred_var"]=(['grid_latitude', 'grid_longitude'],  p_pred_var.T)

###################################################################
climate_ds_jan["alpha_pred_sample"]=(['draw','grid_longitude','grid_latitude'],  alpha_pred_sample)
climate_ds_jan["beta_pred_sample"]=(['draw','grid_longitude','grid_latitude'],  beta_pred_sample)
climate_ds_jan["p_pred_sample"]=(['draw','grid_longitude','grid_latitude'],  p_pred_sample)

climate_ds_jan = climate_ds_jan.transpose('time','grid_latitude','grid_longitude','bnds','draw')
###################################################################

###########################
#Applying Quantile Mapping#
###########################

#Calculating the empirical cumulative density function values from the climate model values of daily snowfall
ecdf = xr.apply_ufunc(
    empirical_cdf,
    climate_ds_jan.prsn,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    exclude_dims=set(("time",)),
    vectorize = True,
)
climate_ds_jan['ecdf']=ecdf.transpose('time','grid_latitude','grid_longitude')

climate_ds_jan['diff_p'] = (1-climate_ds_jan.ecdf.min('time')) - climate_ds_jan.p_pred_mean

###################################################################

#!!!!!!!!!! WARNING BODGE !!!!!!!!!!!!!!!!!#
climate_ds_jan['p_pred_sample'] = xarray.where(1<climate_ds_jan.p_pred_sample, 0.999, climate_ds_jan.p_pred_sample)
climate_ds_jan['p_pred_sample'] = xarray.where(0>climate_ds_jan.p_pred_sample, 0.001, climate_ds_jan.p_pred_sample)
#!!!!!!!!!! WARNING BODGE !!!!!!!!!!!!!!!!!#

climate_ds_jan['diff_p_sample'] = (1-climate_ds_jan.ecdf.min('time')) - climate_ds_jan.p_pred_sample
###################################################################

#Creating simple variable names for values needed in the quantile mapping function
p = climate_ds_jan.p_pred_mean
alpha = climate_ds_jan.alpha_pred_mean
beta = climate_ds_jan.beta_pred_mean
ecdf = climate_ds_jan.ecdf
prsn = climate_ds_jan.prsn
diff_p = climate_ds_jan.diff_p

###################################################################
p_sample = climate_ds_jan.p_pred_sample
alpha_sample = climate_ds_jan.alpha_pred_sample
beta_sample = climate_ds_jan.beta_pred_sample
diff_p_sample = climate_ds_jan.diff_p_sample
###################################################################

#Applying the quantile mapping function that works on one set of prsn values over every grid cell of our xarray dataset
corrected_prsn = xr.apply_ufunc(
    quantile_mapping,
    prsn,
    ecdf,
    diff_p,
    p,
    alpha,
    beta,
    input_core_dims=[["time"],["time"],[],[],[],[]],
    output_core_dims=[["time"]],
    exclude_dims=set(("time",)),
    vectorize = True,
)
climate_ds_jan['corrected_prsn']=corrected_prsn.transpose('time','grid_latitude','grid_longitude')

prsn_samples,_ = xr.broadcast(prsn, diff_p_sample)
ecdf_samples,_ = xr.broadcast(ecdf, diff_p_sample)

###################################################################
corrected_prsn_sample = xr.apply_ufunc(
    quantile_mapping,
    prsn,
    ecdf,
    diff_p_sample,
    p_sample,
    alpha_sample,
    beta_sample,
    input_core_dims=[["time"],["time"],[],[],[],[]],
    output_core_dims=[["time"]],
    exclude_dims=set(("time",)),
    vectorize = True,
)
###################################################################

climate_ds_jan['corrected_prsn']=corrected_prsn.transpose('time','grid_latitude','grid_longitude')

corrected_ecdf = xr.apply_ufunc(
    empirical_cdf,
    climate_ds_jan.corrected_prsn,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    exclude_dims=set(("time",)),
    vectorize = True,
)
climate_ds_jan['corrected_ecdf']=corrected_ecdf.transpose('time','grid_latitude','grid_longitude')

############################################
climate_ds_jan['corrected_prsn_sample']=corrected_prsn_sample.transpose('time','grid_latitude','grid_longitude','draw')
######################################

##################################################
corrected_ecdf_sample = xr.apply_ufunc(
    empirical_cdf,
    climate_ds_jan.corrected_prsn_sample,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    exclude_dims=set(("time",)),
    vectorize = True,
)
climate_ds_jan['corrected_ecdf_sample']=corrected_ecdf_sample.transpose('time','grid_latitude','grid_longitude','draw')
#####################################################

###########################################
#Saving the updated data with corrected values of daily snowfall
outfile_path = f"/data/notebooks/jupyterlab-biascorrlab/data/Lima2021/AP_Daily_Snowfall_044_QMCorrected_Samples.nc"
climate_ds_jan.to_netcdf(outfile_path)
############################################