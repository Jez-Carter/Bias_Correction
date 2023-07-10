# %% Importing packages
import seaborn as sns
import numpyro.distributions as dist
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import jax
from jax import random
import matplotlib.patches as patches
from tinygp import kernels, GaussianProcess,transforms
from src.simulated_data_functions import generate_underlying_data
from src.simulated_data_functions import plot_latent_data_1d
from src.simulated_data_functions import plot_underlying_data_1d
from src.simulated_data_functions import plot_underlying_data_1d_lima
from src.simulated_data_functions import plot_priors
from src.simulated_data_functions import plot_posteriors
from src.simulated_data_functions import plot_predictions_1d

from src.simulated_data_functions_hierarchical import plot_underlying_data_mean_1d
from src.simulated_data_functions_hierarchical import plot_underlying_data_std_1d
from src.simulated_data_functions_hierarchical import plot_predictions_1d_mean_hierarchical

from numpy.random import RandomState
import jax.numpy as jnp
import pandas as pd
import arviz as az
import matplotlib.gridspec as gridspec

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

out_path = '/home/jez/Bias_Correction/results/Paper_Images/'

jax.config.update("jax_enable_x64", True)

########### Methodology ###########

# %% Figure __
fig, ax = plt.subplots(figsize=(13*cm, 8.0*cm),dpi= 300)

xs = np.linspace(-1,5,100)
ys = norm.pdf(xs, 3, 0.5)
zs = norm.pdf(xs, 2, 1.0)
plot = ax.plot(xs, ys, lw=2,linestyle='dashed')
plt.hist(dist.Normal(3.0,0.5).sample(rng_key,(10000,)),density=True,bins=100,color=plot[0].get_color(),alpha=0.7,label='$Y\sim \mathcal{N}(3,0.5)$')
plot2 = ax.plot(xs, zs, lw=2,linestyle='dashed')
plt.hist(dist.Normal(2.0,1.0).sample(rng_key,(10000,)),density=True,bins=100,color=plot2[0].get_color(),alpha=0.7,label='$Z\sim \mathcal{N}(2,1)$')

y_percentile = np.percentile(dist.Normal(3.0,0.5).sample(rng_key,(10000,)), 20)
z_percentile = np.percentile(dist.Normal(2.0,1.0).sample(rng_key,(10000,)), 20)
plt.vlines(y_percentile,0,0.8,color='k',linestyle='dotted',label='20th Percentiles')
plt.vlines(z_percentile,0,0.8,color='k',linestyle='dotted')
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")
arrow = patches.FancyArrowPatch((z_percentile, 0.8), (y_percentile, 0.8),**kw,
                             connectionstyle="arc3,rad=-.2")
plt.gca().add_patch(arrow)

ax.set_ylabel('Probability Density')
ax.set_xlabel('Value Measured')
plt.legend(fontsize=legend_fontsize)
plt.show()
fig.savefig(f'{out_path}fig00.png',dpi=300,bbox_inches='tight')

# %% Figure __

kern = 1 * kernels.ExpSquared(20)
x = np.arange(0,100,0.1)
x_samples = np.array([10,50,70])
y_samples = np.array([0.5,-0.3,0.8])
gp = GaussianProcess(kern,x_samples,diag=1e-10).condition(y_samples, x).gp
gp_samples = gp.sample(rng_key,(5,))

fig, axs = plt.subplots(1,2,figsize=(17*cm,7*cm))
axs[0].plot(x,kern(x,x)[0],label='kernel variance=1, lengthscale=20')
axs[0].set_xlabel(r"$d(s,s')$")
axs[0].set_ylabel(r"$RBF(s,s')$")

axs[1].plot(x,gp_samples[0],color='b',alpha=0.3,label='Realisations')
for sample in gp_samples[1:]:
    axs[1].plot(x,sample,color='b',alpha=0.3)
axs[1].fill_between(x,
                    gp_samples.mean(axis=0)-gp_samples.std(axis=0),
                    gp_samples.mean(axis=0)+gp_samples.std(axis=0),
                    color='k',
                    alpha=0.1,
                    label='$1\sigma$ Uncertainty')
axs[1].plot(x,gp_samples.mean(axis=0),color='k',alpha=0.5,linestyle='--',label='Expectation')
axs[1].scatter(x_samples,y_samples,color='r',s=30,label='Measurements')
axs[1].set_xlabel(r"$s$")
axs[1].set_ylabel(r"$Value$")

axs[0].annotate('a.',xy=(-0.08,-0.15),xycoords='axes fraction')
axs[1].annotate('b.',xy=(-0.08,-0.15),xycoords='axes fraction')

plt.legend(fontsize=legend_fontsize)
# plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig01.png',dpi=300,bbox_inches='tight')

# %% Figure __

min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()

fig, ax = plt.subplots(figsize=(17*cm, 8.0*cm),dpi= 300)
plot_latent_data_1d(scenario_base,ax,ms=20)
ax.set_xlabel('s')
ax.set_ylabel('Value')
ax.get_legend().remove()
labels= ["$\phi_Y \sim \mathcal{GP}(m=1,k_{RBF}(s,s'|v=1,l=3))$",
         "$\phi_B \sim \mathcal{GP}(m=-1,k_{RBF}(s,s'|v=1,l=10))$",
         '$\phi_Z = \phi_Y+\phi_B$']
fig.legend(labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.83, 0.85),
           ncols=1,
           loc=10)
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig02.png',dpi=300,bbox_inches='tight')

# %% Figure __
inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()

fig, axs = plt.subplots(3,1,figsize=(16*cm, 15.0*cm),dpi= 300)
plot_underlying_data_1d(scenario_ampledata,axs[0],ms=20)
plot_underlying_data_1d(scenario_sparse_smooth,axs[1],ms=20)
plot_underlying_data_1d(scenario_sparse_complex,axs[2],ms=20)

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
           ncols=5,
           loc=10)
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig03.png',dpi=300,bbox_inches='tight')

# %% Table __
inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()

parameters = ['Truth Variance $v_Y$',
              'Truth Lengthscale $l_Y$',
              'Truth Mean $m_Y$',
              'Bias Variance $v_B$',
              'Bias Lengthscale $l_B$',
              'Bias Mean $m_B$',
              '\# Observations',
              'Observation Noise',
              '\# Climate Model Predictions'

]
parameters_shorthand = ['t_variance',
                        't_lengthscale',
                        't_mean',
                        'b_variance',
                        'b_lengthscale',
                        'b_mean',
                        'ox',
                        'onoise',
                        'cx'
                        ]

values_list = []
for scenario in [scenario_ampledata,scenario_sparse_smooth,scenario_sparse_complex]:
    values = []
    for i in parameters_shorthand:
        if i=='ox' or i=='cx':
            value = len(scenario[i])
        else:
            value = scenario[i]
        values.append(value)
    values_list.append(values)
values = np.array(values_list)

scenarios = ['Scenario 1','Scenario 2','Scenario 3']
df = pd.DataFrame(data=values.T, index=parameters, columns=scenarios)
caption = 'A table showing the parameters used to generate the realisations and data for 3 scenarios.'
print(df.to_latex(escape=False,caption=caption))

# %% Table __

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()

parameters = ['Truth Variance $v_Y$',
              'Truth Lengthscale $l_Y$',
              'Truth Mean $m_Y$',
              'Bias Variance $v_B$',
              'Bias Lengthscale $l_B$',
              'Bias Mean $m_B$',
              'Observation Noise']

desired_index_order = ['kern_var',
                       'lengthscale',
                       'mean',
                       'bkern_var',
                       'blengthscale',
                       'bmean',
                       'onoise']

columns = ['Expectation',
           'Standard Dev.',
           '95\% C.I. Lower Bound',
           '95\% C.I. Upper Bound']

desired_columns = ['mean',
                   'sd',
                   'hdi_2.5%',
                   'hdi_97.5%']

scenarios = [scenario_ampledata,scenario_sparse_smooth,scenario_sparse_complex]

for scenario,values in zip(scenarios,values_list):
    df = az.summary(scenario['mcmc'].posterior,hdi_prob=0.95)
    df=df.reindex(desired_index_order)
    df=df.set_index(np.array(parameters))
    df = df[desired_columns]
    df.columns = columns
    df.insert(0, 'Specified Value', np.array(values)[[0,1,2,3,4,5,7]])
    print(df.to_latex(escape=False))

# %% Figure __
prior_keys = ['t_variance_prior','t_lengthscale_prior',
              't_mean_prior','b_variance_prior',
              'b_lengthscale_prior','b_mean_prior',
              'onoise_prior']
posterior_keys = ['kern_var','lengthscale','mean',
                  'bkern_var','blengthscale','bmean',
                  'onoise']
titles = ['a. Truth Variance',
          'b. Truth Lengthscale',
          'c. Truth Mean',
          'd. Bias Variance',
          'e. Bias Lengthscale',
          'f. Bias Mean',
          'g. Observation Noise']

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario = scenario_ampledata
# scenario = scenario_sparse_complex

fig = plt.figure(figsize=(17*cm,16*cm),dpi= 300)
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

axs[-1].legend(fontsize=legend_fontsize,labels=['Specified','Prior','Posterior'],loc=[1.1,0.7])
axs[1].set_ylim(0,0.5)
axs[1].set_xlim(0,20)

plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig04.png',dpi=300,bbox_inches='tight')

# %% Figure __
inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth = np.load(f'{inpath}scenario_sparse_smooth.npy',allow_pickle='TRUE').item()
scenario_sparse_complex = np.load(f'{inpath}scenario_sparse_complex.npy',allow_pickle='TRUE').item()

fig, axs = plt.subplots(3,1,figsize=(16*cm, 15.0*cm),dpi= 300)
scenarios = [scenario_ampledata,scenario_sparse_smooth,scenario_sparse_complex]

for scenario,ax in zip(scenarios,axs):
    # ax.plot(scenario['X'],scenario['T'],label='Complete Realisation Truth',alpha=1.0,color='tab:cyan')
    # ax.plot(scenario['X'],scenario['B'],label='Complete Realisation Bias',alpha=1.0,color='tab:orange')
    plot_predictions_1d(scenario,'truth_posterior_predictive_realisations',ax,ms=20,color='tab:purple')
    plot_predictions_1d(scenario,'bias_posterior_predictive_realisations',ax,ms=20,color='tab:red')
    plot_underlying_data_1d(scenario,ax,ms=20)

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
           bbox_to_anchor=(0.5, -0.01),
           ncols=5,
           loc=10)
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig05.png',dpi=300,bbox_inches='tight')

# %% Figure __
inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_ampledata_lima = np.load(f'{inpath}scenario_ampledata_lima.npy',allow_pickle='TRUE').item()
scenario_sparse_smooth_lima = np.load(f'{inpath}scenario_sparse_smooth_lima.npy',allow_pickle='TRUE').item()
scenario_sparse_complex_lima = np.load(f'{inpath}scenario_sparse_complex_lima.npy',allow_pickle='TRUE').item()

fig, axs = plt.subplots(3,1,figsize=(16*cm, 15.0*cm),dpi= 300)
scenarios = [scenario_ampledata_lima,scenario_sparse_smooth_lima,scenario_sparse_complex_lima]
for scenario,ax in zip(scenarios,axs):
    plot_predictions_1d(scenario,'truth_posterior_predictive_realisations_lima',ax,ms=20,color='tab:purple')
    plot_underlying_data_1d_lima(scenario,ax,ms=20)

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
           ncols=7,
           loc=10)
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig06.png',dpi=300,bbox_inches='tight')

# %% Figure __

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_base_hierarchical = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()
scenario = scenario_base_hierarchical

# rng_key = random.PRNGKey(3)
# indecies = random.choice(rng_key,
#               jnp.arange(0,scenario_base_hierarchical['ox'].shape[0]),
#               shape=(3,),
#               replace=False,
#               p=None,
#               axis=0)
ox = scenario['ox'].copy()
ox.sort()
indecies = []
for i in [4,12,30]:
    indecies.append(np.argwhere(scenario['ox']==ox[i])[0][0])

fig, axs = plt.subplots(1,3,figsize=(17*cm, 6.0*cm),dpi= 300)

for ax,index in zip(axs,indecies):
    odata = scenario['odata'][:,index]
    omean = scenario['MEAN_T_obs'][index]
    ostdev = jnp.exp(scenario['LOGVAR_T_obs'][index])
    index_location = scenario['ox'][index]
    difference_array = np.absolute(scenario['cx']-index_location)
    nearest_index = difference_array.argmin()
    cdata = scenario['cdata'][:,nearest_index]
    cmean = scenario['MEAN_C_climate'][nearest_index]
    cstdev = jnp.exp(scenario['LOGVAR_C_climate'][nearest_index])
    ax.hist(odata,density=True,color='tab:blue',alpha=0.5,label='In-situ Observation')
    xs = np.linspace(odata.min()-1,odata.max()+1,100)
    ys = norm.pdf(xs, omean, ostdev)
    ax.plot(xs,ys,linestyle='dashed',color='tab:blue',label='In-situ Latent Distribution')
    ax.hist(cdata,density=True,color='tab:orange',alpha=0.5,label='Climate Model Output')
    xs = np.linspace(cdata.min()-1,cdata.max()+1,100)
    zs = norm.pdf(xs, cmean, cstdev)
    ax.plot(xs,zs,linestyle='dashed',color='tab:orange',label='Climate Model Latent Distribution')

for ax in axs:
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    # ax.get_legend().remove()
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, 0),
           ncols=7,
           loc=10)
axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')
axs[2].annotate('c.',xy=(0.01,1.01),xycoords='axes fraction')

plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig07.png',dpi=300,bbox_inches='tight')

# %% Figure __

min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_base_hierarchical = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()

fig, axs = plt.subplots(2,1,figsize=(17*cm, 15.0*cm),dpi= 300)
plot_underlying_data_mean_1d(scenario_base_hierarchical,axs[0],ms=20)
plot_underlying_data_std_1d(scenario_base_hierarchical,axs[1],ms=20)
axs[0].set_ylabel('Mean Value')
axs[1].set_ylabel('Standard Deviation Value')
for ax in axs:
    ax.set_xlabel('s')
    ax.get_legend().remove()
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, 0),
           ncols=7,
           loc=10)

for ax in axs:
    for index in indecies:
        index_location = scenario['ox'][index]
        ax.axvline(index_location,0,1,linestyle='dashed',color='k',alpha=0.3)
axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig08.png',dpi=300,bbox_inches='tight')

# %% Figure __
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
titles = ['a. $v_{\mu_Y}$',
          'b. $l_{\mu_Y}$',
          'c. $m_{\mu_Y}$',
          'd. $v_{log(\sigma^2_Y)}$',
          'e. $l_{log(\sigma^2_Y)}$',
          'f. $m_{log(\sigma^2_Y)}$',
          'g. $v_{\mu_B}$',
          'h. $l_{\mu_B}$',
          'i. $m_{\mu_B}$',
          'j. $v_{log(\sigma^2_B)}$',
          'k. $l_{log(\sigma^2_B)}$',
          'l. $m_{log(\sigma^2_B)}$']

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_base_hierarchical = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()
scenario = scenario_base_hierarchical

fig = plt.figure(figsize=(17*cm,16*cm),dpi= 300)
gs = gridspec.GridSpec(4, 3)
gs.update(wspace=0.5)
gs.update(hspace=0.4)

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

for ax in axs:
    ax.set_ylabel('Prob. Density')
labels=['Specified','Prior','Posterior']
fig.legend(labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, 0.05),
           ncols=7,
           loc=10)

plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig09.png',dpi=300,bbox_inches='tight')

# %% Figure __
inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_base_hierarchical = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()
scenario = scenario_base_hierarchical

fig, axs = plt.subplots(2,1,figsize=(16*cm, 10.0*cm),dpi= 300)

plot_underlying_data_mean_1d(scenario,axs[0],ms=20)
plot_underlying_data_std_1d(scenario,axs[1],ms=20)
plot_predictions_1d_mean_hierarchical(scenario,'mean_truth_posterior_predictive_realisations',axs[0],ms=20,color='tab:purple')
plot_predictions_1d_mean_hierarchical(scenario,'mean_bias_posterior_predictive_realisations',axs[0],ms=20,color='tab:red')
plot_predictions_1d_mean_hierarchical(scenario,'std_truth_posterior_predictive_realisations',axs[1],ms=20,color='tab:purple')
plot_predictions_1d_mean_hierarchical(scenario,'std_bias_posterior_predictive_realisations',axs[1],ms=20,color='tab:red')

for ax in axs:
    ax.set_xlabel('s')
    ax.get_legend().remove()

axs[0].set_ylabel('Mean Value')
axs[1].set_ylabel('Std. Dev. Value')

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_xlabel('')

axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, -0.01),
           ncols=5,
           loc=10)
axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig10.png',dpi=300,bbox_inches='tight')

# %%
scenario['ox'][indecies]

# %% Checking Shapes
for key in list(scenario.keys()):
    if 'prior' in key:
        None
    else:
        try:
            print(f'{key}:{scenario[key].shape}')
        except:
            None
