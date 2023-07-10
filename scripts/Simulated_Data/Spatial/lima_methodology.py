# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams.update({'font.size': 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions import generate_posterior_lima
from src.simulated_data_functions import generate_posterior_predictive_realisations_lima
from src.simulated_data_functions import plot_predictions_1d

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()

# %%
scenarios = [scenario_ampledata,scenario_sparse_smooth,scenario_sparse_complex]
for scenario in scenarios:
    generate_posterior_lima(scenario,rng_key,1000,2000,1)

# %%
for scenario in scenarios:
    generate_posterior_predictive_realisations_lima(scenario,20,20,rng_key)

# %%
np.save(f'{inpath}scenario_ampledata_lima.npy', scenario_ampledata) 
np.save(f'{inpath}scenario_sparse_smooth_lima.npy', scenario_sparse_smooth) 
np.save(f'{inpath}scenario_sparse_complex_lima.npy', scenario_sparse_complex) 
# %%
legend_fontsize = 8
cm = 1/2.54
fig, axs = plt.subplots(3,1,figsize=(16*cm, 15.0*cm),dpi= 300)
scenarios = [scenario_ampledata,scenario_sparse_smooth,scenario_sparse_complex]
for scenario,ax in zip(scenarios,axs):
    ax.plot(scenario['X'],scenario['T'],label='Complete Realisation Truth',alpha=1.0,color='tab:blue')
    ax.plot(scenario['X'],scenario['B'],label='Complete Realisation Bias',alpha=1.0,color='tab:orange')
    plot_predictions_1d(scenario,'truth_posterior_predictive_realisations_lima',ax,ms=20,color='tab:blue')

for ax in axs:
    ax.set_xlabel('s')
    ax.set_ylabel('Value')
    ax.get_legend().remove()

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_xlabel('')

axs[0].annotate('a. Scenario 1',xy=(0.01,0.94),xycoords='axes fraction')
axs[1].annotate('b. Scenario 2',xy=(0.01,0.94),xycoords='axes fraction')
axs[2].annotate('c. Scenario 3',xy=(0.01,0.94),xycoords='axes fraction')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, 0),
           ncols=6,
           loc=10)
plt.tight_layout()
plt.show()