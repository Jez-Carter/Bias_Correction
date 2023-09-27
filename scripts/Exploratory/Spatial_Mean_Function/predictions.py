
# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions_meanfunction import plot_underlying_data_1d
from src.simulated_data_functions_meanfunction import generate_posterior_predictive_realisations
from src.simulated_data_functions_meanfunction import plot_predictions_1d
from src.simulated_data_functions_meanfunction import plot_predictions_2d

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
# scenario_ampledata = np.load(f'{inpath}scenario_ampledata_meanfunction.npy',allow_pickle='TRUE').item()
# scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth_meanfunction.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex_meanfunction.npy',allow_pickle='TRUE').item()

# %%
generate_posterior_predictive_realisations(scenario_sparse_complex,20,20)

# %% Saving Dictionaries
np.save(f'{inpath}scenario_base.npy', scenario_base) 
np.save(f'{inpath}scenario_ampledata.npy', scenario_ampledata) 
np.save(f'{inpath}scenario_sparse_smooth.npy', scenario_sparse_smooth) 
np.save(f'{inpath}scenario_sparse_complex.npy', scenario_sparse_complex) 
# np.save(f'{inpath}scenario_2d.npy', scenario_2d) 

# %%
def generate_mean_samples(scenario):
    posterior = scenario['mcmc'].posterior
    beta0 = posterior['beta0'].values
    beta1 = posterior['beta1'].values
    beta2 = posterior['beta2'].values
    xs = scenario['nx'].reshape(-1,1)
    samples = beta0 + beta1*(xs-50) + beta2*(xs-50)**2
    return(samples)

# %%
mean_samples = generate_mean_samples(scenario)
mean_samples_mean = mean_samples.mean(axis=1)
mean_samples_std = mean_samples.std(axis=1)

# %%
scenario = scenario_sparse_complex
fig, axs = plt.subplots(2, 1, figsize=(15, 15))

plot_underlying_data_1d(scenario,axs[0],ms=20)

axs[1].plot(scenario['X'],scenario['T'],label='Truth',alpha=1.0,color='tab:blue')
axs[1].plot(scenario['X'],scenario['B'],label='Truth',alpha=1.0,color='tab:orange')
plot_predictions_1d(scenario,'truth_posterior_predictive_realisations',axs[1],ms=20,color='tab:blue')
plot_predictions_1d(scenario,'bias_posterior_predictive_realisations',axs[1],ms=20,color='tab:orange')

axs[1].plot(scenario['nx'],mean_samples_mean,label='Mean Function',color='r',alpha=0.5)
axs[1].fill_between(scenario['nx'],mean_samples_mean+mean_samples_std,mean_samples_mean-mean_samples_std,label='$1\sigma$ Uncertainty',color='r',alpha=0.3)


# %% 
scenario['nx'].shape

# %%
scenario['mcmc'].posterior['beta0'].values.shape

# %%
(scenario['mcmc'].posterior['beta0'].values * scenario['nx'].reshape(-1,1)).shape


# %%
scenario['nx'].shape

# %%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
plot_predictions_2d(scenario_2d,axs.ravel())

titles = ['a. True Field: Unbiased Process',
          'b. Prediction Mean: Unbiased Process',
          'c. Prediction Standard Deviation: Unbiased Process',
          'd. True Field: Biased Process',
          'e. Prediction Mean: Biased Process',
          'f. Prediction Standard Deviation: Biased Process']

for ax,title in zip(axs.ravel(),titles):
    ax.set_title(title,pad=3,loc='left',fontsize=10)
