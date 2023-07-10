
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

from src.simulated_data_functions import plot_underlying_data_1d
from src.simulated_data_functions import generate_posterior_predictive_realisations
from src.simulated_data_functions import plot_predictions_1d
from src.simulated_data_functions import plot_predictions_2d

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d.npy',allow_pickle='TRUE').item()

# %%
generate_posterior_predictive_realisations(scenario_base,20,20)
generate_posterior_predictive_realisations(scenario_ampledata,20,20)
generate_posterior_predictive_realisations(scenario_sparse_smooth,20,20)
generate_posterior_predictive_realisations(scenario_sparse_complex,20,20)
# generate_posterior_predictive_realisations(scenario_2d,20,20)

# %% Saving Dictionaries
np.save(f'{inpath}scenario_base.npy', scenario_base) 
np.save(f'{inpath}scenario_ampledata.npy', scenario_ampledata) 
np.save(f'{inpath}scenario_sparse_smooth.npy', scenario_sparse_smooth) 
np.save(f'{inpath}scenario_sparse_complex.npy', scenario_sparse_complex) 
# np.save(f'{inpath}scenario_2d.npy', scenario_2d) 

# %%
scenario = scenario_sparse_complex
fig, axs = plt.subplots(2, 1, figsize=(15, 15))

plot_underlying_data_1d(scenario,axs[0],ms=20)

axs[1].plot(scenario['X'],scenario['T'],label='Truth',alpha=1.0,color='tab:blue')
axs[1].plot(scenario['X'],scenario['B'],label='Truth',alpha=1.0,color='tab:orange')
plot_predictions_1d(scenario,'truth_posterior_predictive_realisations',axs[1],ms=20,color='tab:blue')
plot_predictions_1d(scenario,'bias_posterior_predictive_realisations',axs[1],ms=20,color='tab:orange')

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
