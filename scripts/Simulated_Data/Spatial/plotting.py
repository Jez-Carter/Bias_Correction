# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import arviz as az
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams.update({'font.size': 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions import plot_underlying_data_1d
from src.simulated_data_functions import plot_underlying_data_2d
from src.simulated_data_functions import plot_priors
from src.simulated_data_functions import plot_posteriors
from src.simulated_data_functions import plot_predictions_1d
from src.simulated_data_functions import plot_predictions_2d

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Loading Data
# scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d.npy',allow_pickle='TRUE').item()
outpath = '/home/jez/Bias_Correction/results/Paper_Images/'

# %% Plotting Underlying Data 1D
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plot_underlying_data_1d(scenario_ampledata,ax,ms=20)
# plt.savefig(f'{outpath}Underlying_Data_1D')

# %% Plotting Underlying Data 2D
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plot_underlying_data_2d(scenario_2d,axs,1)
# plt.savefig(f'{outpath}Underlying_Data_2D')

# %% Posterior Table Summary

az.summary(scenario_ampledata['mcmc'].posterior,hdi_prob=0.95)

# %% Plotting Priors & Posteriors 1D/2D

prior_keys = ['t_variance_prior','t_lengthscale_prior',
              't_mean_prior','b_variance_prior',
              'b_lengthscale_prior','b_mean_prior',
              'onoise_prior']
posterior_keys = ['kern_var','lengthscale','mean',
                  'bkern_var','blengthscale','bmean',
                  'onoise']
titles = ['a. Truth Kernel Variance',
          'b. Truth Kernel Lengthscale',
          'c. Truth Mean',
          'd. Bias Kernel Variance',
          'e. Bias Kernel Lengthscale',
          'f. Bias Mean',
          'g. Observation Noise']

scenario = scenario_2d

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.8)
gs.update(hspace=0.3)

axs = [plt.subplot(gs[0, :2]),
       plt.subplot(gs[0, 2:4]),
       plt.subplot(gs[0, 4:6]),
       plt.subplot(gs[1, :2]),
       plt.subplot(gs[1, 2:4]),
       plt.subplot(gs[1, 4:6]),
       plt.subplot(gs[2, 2:4])]

rng_key = random.PRNGKey(5)
plot_priors(scenario,prior_keys,axs,rng_key)
plot_posteriors(scenario['mcmc'].posterior,posterior_keys,axs)

for ax,title in zip(axs,titles):
    ax.set_title(title,pad=3,loc='left',fontsize=8)

axs[-1].legend(fontsize=8,labels=['Actual','Prior','Posterior'],loc=[1.1,0.7])

# plt.savefig(f'{outpath}Priors_and_Posteriors')


# %% Plotting Predictions 1D
scenario = scenario_ampledata
fig, axs = plt.subplots(2, 1, figsize=(15, 15))

plot_underlying_data_1d(scenario,axs[0],ms=20)

axs[1].plot(scenario['X'],scenario['T'],label='Truth',alpha=1.0,color='tab:blue')
axs[1].plot(scenario['X'],scenario['B'],label='Truth',alpha=1.0,color='tab:orange')
plot_predictions_1d(scenario,'truth_posterior_predictive_realisations',axs[1],ms=20,color='tab:blue')
plot_predictions_1d(scenario,'bias_posterior_predictive_realisations',axs[1],ms=20,color='tab:orange')

# plt.savefig(f'{outpath}Predictions_1D')

# %% Plotting Predictions 2D
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

# plt.savefig(f'{outpath}Predictions_2D')
