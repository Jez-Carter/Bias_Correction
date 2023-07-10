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

from src.simulated_data_functions_hierarchical import generate_posterior_hierarchical
from src.simulated_data_functions import plot_underlying_data_1d
from src.simulated_data_functions import plot_underlying_data_2d
from src.simulated_data_functions import plot_priors
from src.simulated_data_functions import plot_posteriors

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
scenario_base = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d_hierarchical.npy',allow_pickle='TRUE').item()

# %%
scenario_base['mcmc'].posterior


# %%
generate_posterior_hierarchical(scenario_base,rng_key,1000,2000,2)
rng_key, rng_key_ = random.split(rng_key)
# %%
outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
np.save(f'{outpath}scenario_base_hierarchical.npy', scenario_base) 

# %%
scenario_base['mcmc'].posterior
# %%
list(scenario_base.keys())[30:]
# %%
prior_keys = ['MEAN_T_variance_prior','MEAN_T_lengthscale_prior',
              'MEAN_T_mean_prior','LOGVAR_T_variance_prior',
              'LOGVAR_T_lengthscale_prior','LOGVAR_T_mean_prior',
              'MEAN_B_variance_prior','MEAN_B_lengthscale_prior',
              'MEAN_B_mean_prior','LOGVAR_B_variance_prior',
              'LOGVAR_B_lengthscale_prior','LOGVAR_B_mean_prior']
posterior_keys = ['mt_kern_var','mt_lengthscale',
                  'mt_mean','lvt_kern_var',
                  'lvt_lengthscale','lvt_mean',
                  'mb_kern_var','mb_lengthscale',
                  'mb_mean','lvb_kern_var',
                  'lvb_lengthscale','lvb_mean']
titles = ['a. $\mu_Y$ Kernel Variance',
          'b. $\mu_Y$ Kernel Lengthscale',
          'c. $\mu_Y$ Mean',
          'd. $log(var_Y)$ Kernel Variance',
          'e. $log(var_Y)$ Kernel Lengthscale',
          'f. $log(var_Y)$ Mean',
          'g. $\mu_B$ Kernel Variance',
          'h. $\mu_B$ Kernel Lengthscale',
          'i. $\mu_B$ Mean',
          'j. $log(var_B)$ Kernel Variance',
          'k. $log(var_B)$ Kernel Lengthscale',
          'l. $log(var_B)$ Mean']
# %%

scenario = scenario_base

fig = plt.figure(figsize=(7.5, 10))
gs = gridspec.GridSpec(4, 3)
gs.update(wspace=0.8)
gs.update(hspace=0.3)

axs = [plt.subplot(gs[0, 0]),
       plt.subplot(gs[0, 1]),
       plt.subplot(gs[0, 2]),
       plt.subplot(gs[1, 0]),
       plt.subplot(gs[1, 1]),
       plt.subplot(gs[1, 2]),
       plt.subplot(gs[2, 0]),
       plt.subplot(gs[2, 1]),
       plt.subplot(gs[2, 2]),
       plt.subplot(gs[3, 0]),
       plt.subplot(gs[3, 1]),
       plt.subplot(gs[3, 2])]

rng_key = random.PRNGKey(5)
plot_priors(scenario,prior_keys,axs,rng_key)
plot_posteriors(scenario['mcmc'].posterior,posterior_keys,axs)

for ax,title in zip(axs,titles):
    ax.set_title(title,pad=3,loc='left',fontsize=8)

axs[-1].legend(fontsize=8,labels=['Actual','Prior','Posterior'],loc=[1.1,0.7])
plt.tight_layout()
plt.show()