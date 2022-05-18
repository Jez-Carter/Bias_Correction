import xarray as xr
from tqdm import tqdm
import numpy as np
import pandas as pd
import iris
from sklearn.preprocessing import StandardScaler
import gpflow 
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import sys

folder_path = sys.argv[1]

ds = xr.open_dataset('/data/notebooks/jupyterlab-biascorrlab/data/AP_Daily_Snowfall_044.nc') # open the daily snowfall netCDF file as a DataArray
ds = ds.drop('rotated_latitude_longitude') # drop rotated_latitude_longitude varaible
ds = ds.drop_dims('bnds') #drop time bounds dimension
ds = ds.isel(time=(ds.time.dt.month == 1))
df_all_daily = ds.to_dataframe().reset_index() #dataframe of all values of daily snowfall at every site
ds = ds.drop_dims('time')
df_all = ds.to_dataframe().reset_index() #dataframe of coordinates of all sites

df_stand = pd.read_csv(f'{folder_path}AP_BGLima_Snowfall_044_Mean_Alpha_stand.csv', index_col=0) # dataframe of mean estimated values of p and alpha at the 100 sample (observation) sites 

#converting coordinates to standardised forms for every site
df_all_stand = df_all.copy()
for i in ['grid_latitude','grid_longitude']:
    scale =  StandardScaler().fit(df_stand[[i]])
    df_all_stand[f'{i}_stand'] = scale.transform(df_all_stand[[i]])
    
#loading in estimated values at each site from bernoulli-gamma fit
bg_samples_obj = np.load(f'{folder_path}AP_BGLima_Snowfall_044_Sample.npy',allow_pickle=True)
bg_samples_dict = dict(zip(bg_samples_obj[:,0], bg_samples_obj[:,1]))
df_all_stand['a0']=bg_samples_dict['a0'].mean()
df_all_stand['a1']=bg_samples_dict['a1'].mean()
df_all_stand['betavar']=bg_samples_dict['betavar'].mean()

#making predictions of standardised forms of alpha and p at every site based on MVN distribution estimated using Gaussian Process and observations
for parameter in ['p','alpha']:
    gp_samples_obj = np.load(f'{folder_path}AP_GPLima_MCMC_{parameter.capitalize()}_Model_Summary.npy',allow_pickle=True)
    gp_samples_dict = dict(zip(gp_samples_obj[:,0], gp_samples_obj[:,1]))
    kern = gpflow.kernels.Exponential(lengthscales=[gp_samples_dict['kernel_lengthscales'].mean()],variance=[gp_samples_dict['kernel_variance'].mean()])
    model = gpflow.models.GPR(data=(df_stand[['grid_latitude_stand','grid_longitude_stand']], df_stand[[f'{parameter}_stand']]), kernel=kern, mean_function=None, noise_variance=gp_samples_dict['likelihood_variance'].mean())
    df_all_stand[f'mcmc_{parameter}_predictions_stand'] = model.predict_f(df_all_stand[['grid_latitude_stand','grid_longitude_stand']].to_numpy())[0]
    
#adjusting predictions to non-standardised forms
for parameter in ['p','alpha']:
    scale = StandardScaler().fit(df_stand[[f'{parameter}']])
    df_all_stand[f'mcmc_{parameter}_predictions']= scale.inverse_transform(df_all_stand[[f'mcmc_{parameter}_predictions_stand']])
    
beta = np.exp(stats.norm(loc=df_all_stand['a0']+df_all_stand['a1']*df_all_stand['mcmc_alpha_predictions'],scale=df_all_stand['betavar']).rvs())
df_all_stand['beta_predictions']=beta

df_all_stand.to_csv(f'{folder_path}AP_BGLima_Snowfall_044_Mean_Alpha_stand_All.csv',index=True)

#########################
#Applying Quantile Mapping
#########################

#creating columns in dataframe that will be updated during the following code
df_all_daily[['ecdf','adjusted_ecdf','corrected_prsn']]=np.nan

#looping through each site in turn 
for i in tqdm(df_all.index):
    
    #filtering to site i from df_all dataframe
    df_all_daily_test = df_all_daily[(df_all_daily['grid_latitude']==df_all['grid_latitude'][i]	) & (df_all_daily['grid_longitude']==df_all['grid_longitude'][i])]
    df_all_stand_test = df_all_stand[(df_all_stand['grid_latitude']==df_all['grid_latitude'][i]	) & (df_all_stand['grid_longitude']==df_all['grid_longitude'][i])]

    #using the ECDF function from statsmodels to estimate the empirical cumulative density function
    data = df_all_daily_test['prsn'] 
    ecdf = ECDF(data)
    df_all_daily_test['ecdf']=ecdf(data) #updating ecdf column 
    
    #returning parameter values needed for quantile mapping method
    climate_p = 1-ecdf(0)
    obs_p = df_all_stand_test['mcmc_p_predictions'].to_numpy()
    obs_alpha = df_all_stand_test['mcmc_alpha_predictions'].to_numpy()
    obs_scale = np.reciprocal(df_all_stand_test['beta_predictions']).to_numpy()
    difference_p = climate_p-obs_p # difference in p value from climate data and interpolated p value from observations
    
    #creating adjusted_ecdf column which will be filled based on the value of difference_p (needed for method of dealing with discreet cdfs while applying quantile mapping) 
    df_all_daily_test['adjusted_ecdf']=df_all_daily_test['ecdf']

    #sectioning data into zero and nonzero days of snowfall 
    zeros = data[data==0]
    nonzeros = data[data!=0]

    #applying the mapping, with 2 methods depending on the sign of difference_p
    if difference_p>0: #if difference_p = +ve then we need to convert some of the nonzeros from the climate data to zero values 
        nonzero_values_to_convert = df_all_daily_test[((1-climate_p)<df_all_daily_test['ecdf']) & (df_all_daily_test['ecdf']<(1-obs_p)[0])]
        nonzero_values_to_convert['corrected_prsn']=0
        zeros_df = zeros.to_frame()
        zeros_df['corrected_prsn']=0
        nonzeros_df = df_all_daily_test[df_all_daily_test['prsn']!=0]
        nonzeros_not_converted = nonzeros_df[~nonzeros_df.index.isin(nonzero_values_to_convert.index)]
        nonzeros_not_converted['corrected_prsn']=stats.gamma.ppf((nonzeros_not_converted['ecdf']-(1-obs_p))/(obs_p), a=obs_alpha,loc=0, scale=obs_scale)

        df_all_daily_test.update(nonzero_values_to_convert)
        df_all_daily_test.update(zeros_df)
        df_all_daily_test.update(nonzeros_not_converted)

    elif difference_p<0:  #if difference_p = -ve then we need to convert some of the zeros from the climate data to nonzero values 
        number_to_convert = int(len(data)*np.abs(difference_p))
        sample = zeros.sample(number_to_convert,random_state=1)
        cdf_sample_values = np.linspace((1-obs_p),(1-climate_p),number_to_convert)#np.linspace(climate_p,obs_p,number_to_convert)
        sample_df = sample.to_frame()
        sample_df['adjusted_ecdf']=cdf_sample_values
        df_all_daily_test.update(sample_df)

        zeros_notinsample = zeros[~zeros.index.isin(sample.index)].to_frame()
        zeros_notinsample['corrected_prsn']=0
        sample_df['corrected_prsn']=stats.gamma.ppf((sample_df['adjusted_ecdf']-(1-obs_p))/(obs_p), a=obs_alpha,loc=0, scale=obs_scale)
        nonzeros_df = df_all_daily_test[df_all_daily_test['prsn']!=0]
        nonzeros_df['corrected_prsn']=stats.gamma.ppf((nonzeros_df['ecdf']-(1-obs_p))/(obs_p), a=obs_alpha,loc=0, scale=obs_scale)

        df_all_daily_test.update(zeros_notinsample)
        df_all_daily_test.update(sample_df)
        df_all_daily_test.update(nonzeros_df)
    
    #adjusting the corrected snowfall values onto the proper scale
    df_all_daily_test['corrected_prsn']=df_all_daily_test['corrected_prsn']/10**5
    #updating the dataframe containing daily snowfall for all sites 
    df_all_daily.update(df_all_daily_test)

#saving the dataframe with the corrected daily snowfall values after quantile mapping
df_all_daily.to_csv(f'{folder_path}AP_BiasAdjusted_Snowfall_044.csv',index=True)