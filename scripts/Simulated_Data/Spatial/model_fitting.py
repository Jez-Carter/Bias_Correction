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

from src.simulated_data_functions import generate_posterior
from src.simulated_data_functions import plot_underlying_data_1d
from src.simulated_data_functions import plot_underlying_data_2d
from src.simulated_data_functions import plot_priors
from src.simulated_data_functions import plot_posteriors

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d.npy',allow_pickle='TRUE').item()

# %%
generate_posterior(scenario_ampledata,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)
# %%
generate_posterior(scenario_sparse_smooth,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)
# %%
generate_posterior(scenario_sparse_complex,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)
# %%
# generate_posterior(scenario_2d,rng_key,1000,2000,2)

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plot_underlying_data_1d(scenario_ampledata,ax,ms=20)

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plot_underlying_data_2d(scenario_2d,axs,1)

# %% Saving Dictionaries
# np.save(f'{inpath}scenario_base.npy', scenario_base) 
np.save(f'{inpath}scenario_ampledata.npy', scenario_ampledata) 
np.save(f'{inpath}scenario_sparse_smooth.npy', scenario_sparse_smooth) 
np.save(f'{inpath}scenario_sparse_complex.npy', scenario_sparse_complex) 
# np.save(f'{inpath}scenario_2d.npy', scenario_2d)

# %%
az.summary(scenario_ampledata['mcmc'].posterior,hdi_prob=0.95)

# %%
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
# %%

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

# %%
az.summary(scenario_2d['mcmc'].posterior,hdi_prob=0.95)
