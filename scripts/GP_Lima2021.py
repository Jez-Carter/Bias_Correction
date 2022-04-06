#Loading alpha values for daily snowfall data and fitting a Gaussian Process distribution using MCMC
import numpy as np
import pandas as pd
import iris
import gpflow 
from gpflow.utilities import print_summary
from gpflow.utilities import read_values
from gpflow.ci_utils import ci_niter
from gpflow.kernels import Matern32
from gpflow.models import SGPR
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from scipy.cluster.vq import kmeans2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.model_fitting_functions import train_gp

data_directory = '/data/climatedata/'
metum_cube = iris.load(f'{data_directory}metum_cube_lres.nc')[0]
metum_cube = metum_cube[:,40:70,5:40] # (time,grid_lat,grid_lon) #[:,40:70,5:40] selects Antarctic Peninsula
combine_lat_lon_size = metum_cube.shape[1]*metum_cube.shape[2]

grid_lat,grid_lon = metum_cube.coord('grid_latitude').points.astype('float64'), metum_cube.coord('grid_longitude').points.astype('float64') # shape [30,],[35,]
jan_alphas_all = np.array(np.load('/data/notebooks/jupyterlab-biascorrlab/data/AP_all_sites_MCMC_samples.npy',allow_pickle=True)[-2])

#meshgrid grid_lats & grid_lons and then sample:
GLAT, GLON = np.meshgrid(grid_lat, grid_lon) # shapes [30,35],[30,35]
GLAT, GLON = GLAT.reshape(combine_lat_lon_size), GLON.reshape(combine_lat_lon_size) # shapes [1050,],[1050,]
GLATLON = np.column_stack([[GLAT, GLON]]).T # shape [1050,2]
df = pd.DataFrame(GLATLON,columns=['Grid_Latitude','Grid_Longitude'])

#including alpha values for january
jan_alphas = np.array(jan_alphas_all[:,0].mean(axis=0).reshape(-1,1)).astype('float64')
df['Alpha']=jan_alphas

#splitting training and test data
df_train, df_test = train_test_split(df,test_size=0.5,random_state=27)

#standardising based on test data
df_stand,df_train_stand,df_test_stand = df.copy(),df_train.copy(),df_test.copy() 
for i in ['Grid_Latitude','Grid_Longitude','Alpha']:
    scale =  StandardScaler().fit(df_train[[i]])
    df_stand[i],df_train_stand[i],df_test_stand[i] = scale.transform(df[[i]]),scale.transform(df_train[[i]]),scale.transform(df_test[[i]])

kern = gpflow.kernels.Matern32(lengthscales=[1,1],variance=0.005)
model = gpflow.models.GPR(data=(df_train_stand[['Grid_Latitude','Grid_Longitude']], df_train_stand[['Alpha']]), kernel=kern, mean_function=None, noise_variance=0.005)
print_summary(kern),print_summary(model)

opt = tf.optimizers.Adam(learning_rate=0.01)
m, logfs = train_gp(model, 2000, opt, verbose=True)

print_summary(kern),print_summary(model)

model_summary = np.array(list(read_values(model).items()),dtype=object)
np.save('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_GP_Model.npy',model_summary)

df_stand['Predictions'] = m.predict_f(df_stand[['Grid_Latitude','Grid_Longitude']].to_numpy())[0]
df_train_stand['Predictions'] = m.predict_f(df_train_stand[['Grid_Latitude','Grid_Longitude']].to_numpy())[0]
df_test_stand['Predictions'] = m.predict_f(df_test_stand[['Grid_Latitude','Grid_Longitude']].to_numpy())[0]
alpha_scale = StandardScaler().fit(df_train[['Alpha']])
df['Predictions']=alpha_scale.inverse_transform(df_stand[['Predictions']])
df_train['Predictions']=alpha_scale.inverse_transform(df_train_stand[['Predictions']])
df_test['Predictions']=alpha_scale.inverse_transform(df_test_stand[['Predictions']])

df.to_csv('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_GP_Model_Alpha_df.csv')
df_stand.to_csv('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_GP_Model_Alpha_df_stand.csv')
df_test.to_csv('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_GP_Model_Alpha_df_test.csv')
df_train.to_csv('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_GP_Model_Alpha_df_train.csv')
df_test_stand.to_csv('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_GP_Model_Alpha_df_test_stand.csv')
df_train_stand.to_csv('/data/notebooks/jupyterlab-biascorrlab/data/AP_Lima_GP_Model_Alpha_df_train_stand.csv')

