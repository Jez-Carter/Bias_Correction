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

from src.simulated_data_functions_hierarchical import plot_underlying_data_mean_1d
from src.simulated_data_functions_hierarchical import generate_posterior_predictive_realisations_hierarchical_mean
from src.simulated_data_functions_hierarchical import generate_posterior_predictive_realisations_hierarchical_std
from src.simulated_data_functions_hierarchical import plot_predictions_1d_mean_hierarchical
from src.simulated_data_functions import plot_predictions_2d

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
scenario_base = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()

# %%
generate_posterior_predictive_realisations_hierarchical_mean(scenario_base,20,20)
generate_posterior_predictive_realisations_hierarchical_std(scenario_base,20,20)

# %%
outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
np.save(f'{outpath}scenario_base_hierarchical.npy', scenario_base) 

# %%
scenario = scenario_base
fig, axs = plt.subplots(2, 1, figsize=(15, 15))

plot_underlying_data_mean_1d(scenario,axs[0])

axs[1].plot(scenario['X'],scenario['MEAN_T'],label='Truth',alpha=1.0,color='tab:blue')
axs[1].plot(scenario['X'],scenario['MEAN_B'],label='Bias',alpha=1.0,color='tab:orange')
plot_predictions_1d_mean_hierarchical(scenario,'mean_truth_posterior_predictive_realisations',axs[1],ms=20,color='tab:blue')
plot_predictions_1d_mean_hierarchical(scenario,'mean_bias_posterior_predictive_realisations',axs[1],ms=20,color='tab:orange')
