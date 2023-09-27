# %% Importing packages
import numpy as np
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt
import jax
from jax import random
from src.simulated_data_functions import plot_priors
from src.simulated_data_functions import plot_posteriors
from src.hierarchical.simulated_data_functions_hierarchical import plot_underlying_data_mean_1d
from src.hierarchical.simulated_data_functions_hierarchical import plot_underlying_data_std_1d
from src.hierarchical.simulated_data_functions_hierarchical import plot_predictions_1d_mean_hierarchical
import jax.numpy as jnp
import pandas as pd
import matplotlib.gridspec as gridspec
import arviz as az

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

pd.options.display.max_colwidth = 100

plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

out_path = '/home/jez/Bias_Correction/results/Paper_Images/'

jax.config.update("jax_enable_x64", True)

# %% Loading the data
inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_base_hierarchical = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()
scenario = scenario_base_hierarchical

# %% Table: Parameters used to generate the data

parameters_shorthand = ['MEAN_T_variance',
                        'MEAN_T_lengthscale',
                        'MEAN_T_mean',
                        'LOGVAR_T_variance',
                        'LOGVAR_T_lengthscale',
                        'LOGVAR_T_mean',
                        'MEAN_B_variance',
                        'MEAN_B_lengthscale',
                        'MEAN_B_mean',
                        'LOGVAR_B_variance',
                        'LOGVAR_B_lengthscale',
                        'LOGVAR_B_mean',
                        'ox',
                        'cx',
                        'osamples',
                        'csamples'
                        ]

parameters = ['In-Situ Mean, Kernel Variance ($v_{\mu_Y}$)',
              'In-Situ Mean, Kernel Lengthscale ($l_{\mu_Y}$)',
              'In-Situ Mean, Mean Constant ($m_{\mu_Y}$)',
              'In-Situ Transformed Variance, Kernel Variance ($v_{\tilde{\sigma}^2_Y}$)',
              'In-Situ Transformed Variance, Kernel Lengthscale ($l_{\tilde{\sigma}^2_Y}$)',
              'In-Situ Transformed Variance, Mean Constant ($m_{\tilde{\sigma}^2_Y}$)',
              'Bias Mean, Kernel Variance ($v_{\mu_B}$)',
              'Bias Mean, Kernel Lengthscale ($l_{\mu_B}$)',
              'Bias Mean, Mean Constant ($m_{\mu_B}$)',
              'Bias Transformed Variance, Kernel Variance ($v_{\tilde{\sigma}^2_B}$)',
              'Bias Transformed Variance, Kernel Lengthscale ($l_{\tilde{\sigma}^2_B}$)',
              'Bias Transformed Variance, Mean Constant ($m_{\tilde{\sigma}^2_B}$)',
              '\# Spatial Locations of In-Situ Observations',
              '\# Spatial Locations of Climate Model Predictions',
              '\# Samples per Location of In-Situ Observations',
              '\# Samples per Location of Climate Model Predictions']

values = []
for i in parameters_shorthand:
    if i=='ox' or i=='cx':
        value = len(scenario[i])
    else:
        value = scenario[i]
    values.append(value)

df = pd.DataFrame(data=np.array(values).T, index=parameters, columns=['Hierarchical Scenario'])
caption = 'A table showing the parameters used to generate the realisations and data for 3 scenarios.'
print(df.to_latex(escape=False,caption=caption))

# %% Figure: Visualising sample PDFs from 3 sites

ox = scenario['ox'].copy()
ox.sort()
indecies = []
for i in [4,12,30]:
    indecies.append(np.argwhere(scenario['ox']==ox[i])[0][0])

fig, axs = plt.subplots(1,3,figsize=(17*cm, 5.5*cm),dpi= 300)

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


# %% Figure: Visualising the data

min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

fig, axs = plt.subplots(2,1,figsize=(17*cm, 9.0*cm),dpi= 300)
plot_underlying_data_mean_1d(scenario_base_hierarchical,axs[0],ms=20)
plot_underlying_data_std_1d(scenario_base_hierarchical,axs[1],ms=20)
axs[0].set_ylabel('Mean Parameter')
axs[1].set_ylabel('Std. Dev. Parameter')
for ax in axs:
    ax.get_legend().remove()

axs[1].set_xlabel('s')
axs[0].set_xticklabels([])

handles, labels = axs[0].get_legend_handles_labels()
labels = ['Parameter Value: In-Situ Data',
          'Parameter Bias',
          'Parameter Value: Climate Model Output',
          'In-Situ Observations Sample',
          'Climate Model Output Sample']
# legend_order = [0,3,1,4,2]
# handles = [handles[i] for i in legend_order]
# labels = [labels[i] for i in legend_order]
fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, -0.01),
           ncols=3,
           loc=10)

for ax in axs:
    for index in indecies:
        index_location = scenario['ox'][index]
        ax.axvline(index_location,0,1,linestyle='dashed',color='k',alpha=0.3)
axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')
plt.tight_layout()
plt.subplots_adjust(hspace=0.08)
plt.show()
fig.savefig(f'{out_path}fig08.png',dpi=300,bbox_inches='tight')

# %% Table: Prior and posterior distribution statistics

parameters = ['In-Situ Mean, Kernel Variance $v_{\mu_Y}$',
              'In-Situ Mean, Kernel Lengthscale $l_{\mu_Y}$',
              'In-Situ Mean, Mean Constant $m_{\mu_Y}$',
              'In-Situ Transformed Variance, Kernel Variance $v_{\tilde{\sigma}^2_Y}$',
              'In-Situ Transformed Variance, Kernel Lengthscale $l_{\tilde{\sigma}^2_Y}$',
              'In-Situ Transformed Variance, Mean Constant $m_{\tilde{\sigma}^2_Y}$',
              'Bias Mean, Kernel Variance $v_{\mu_B}$ ',
              'Bias Mean, Kernel Lengthscale $l_{\mu_B}$',
              'Bias Mean, Mean Constant $m_{\mu_B}$',
              'Bias Transformed Variance, Kernel Variance $v_{\tilde{\sigma}^2_B}$',
              'Bias Transformed Variance, Kernel Lengthscale $l_{\tilde{\sigma}^2_B}$',
              'Bias Transformed Variance, Mean Constant $m_{\tilde{\sigma}^2_B}$']

desired_index_order = ['mt_kern_var',
                       'mt_lengthscale',
                       'mt_mean',
                       'lvt_kern_var',
                       'lvt_lengthscale',
                       'lvt_mean',
                       'mb_kern_var',
                       'mb_lengthscale',
                       'mb_mean',
                       'lvb_kern_var',
                       'lvb_lengthscale',
                       'lvb_mean']

columns = ['Exp.',
           'Std. Dev.',
           '95\% C.I. L.B.',
           '95\% C.I. U.B.']

desired_columns = ['mean',
                   'sd',
                   'hdi_2.5%',
                   'hdi_97.5%']

prior_keys = ['MEAN_T_variance_prior',
             'MEAN_T_lengthscale_prior',
             'MEAN_T_mean_prior',
             'LOGVAR_T_variance_prior',
             'LOGVAR_T_lengthscale_prior',
             'LOGVAR_T_mean_prior',
             'MEAN_B_variance_prior',
             'MEAN_B_lengthscale_prior',
             'MEAN_B_mean_prior',
             'LOGVAR_B_variance_prior',
             'LOGVAR_B_lengthscale_prior',
             'LOGVAR_B_mean_prior']

values

df = az.summary(scenario['mcmc'].posterior,hdi_prob=0.95)
df=df.reindex(desired_index_order)
df=df.set_index(np.array(parameters))
df = df[desired_columns]
df.columns = columns

expectations = []
standard_deviations = []
LB_CIs = []
UB_CIs = []
for key in prior_keys:
    distribution = scenario[key]
    expectation = distribution.mean
    variance = distribution.variance
    standard_deviation = jnp.sqrt(variance)
    if distribution.reparametrized_params == ['concentration', 'rate']:
        LB_CI = gamma.ppf(0.025, distribution.concentration, loc=0, scale=1/distribution.rate)
        UB_CI = gamma.ppf(0.975, distribution.concentration, loc=0, scale=1/distribution.rate)
    else:
        LB_CI = distribution.icdf(0.025)
        UB_CI = distribution.icdf(0.975)
    expectations.append(expectation)
    standard_deviations.append(standard_deviation)
    LB_CIs.append(LB_CI)
    UB_CIs.append(UB_CI)
d = {columns[0]:expectations,
    columns[1]:standard_deviations,
    columns[2]:LB_CIs,
    columns[3]:UB_CIs}
df_prior = pd.DataFrame(data=d, index=parameters)

df['Distribution']='Posterior'
df_prior['Distribution'] = 'Prior'

df_conc = pd.concat([df,df_prior])
df_conc = df_conc.set_index([df_conc.index,'Distribution'])
df_conc = df_conc.unstack()
df_conc = df_conc.reindex(parameters)
df_conc = df_conc.reindex(columns=['Prior', 'Posterior'], level='Distribution')

df_conc = df_conc.swaplevel(axis='columns')
df_conc = df_conc.reindex(columns=['Prior', 'Posterior'], level='Distribution')

df_conc.insert(0, 'Specified Value', np.array(values)[[0,1,2,3,4,5,6,7,8,9,10,11]])
df_conc = df_conc.astype(float)
df_conc = df_conc.round(2)
print(df_conc.to_latex(escape=False))


# %% Figure: Visualising posterior predictives 
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

axs[0].set_ylabel('Mean Parameter')
axs[1].set_ylabel('Std. Dev. Parameter')

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_xlabel('')

axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')

handles, labels = axs[0].get_legend_handles_labels()
# labels = ['$\phi_Y$ Expectation',
#           '$\phi_Y$ 1$\sigma$ Uncertainty',
#           '$\phi_B$ Expectation',
#           '$\phi_B$ 1$\sigma$ Uncertainty',
#           '$\phi_Y$ Underlying Field',
#           '$\phi_B$ Underlying Field',
#           '$\phi_Z$ Underlying Field',
#           'In-Situ Observations',
#           'Climate Model Output']

labels = ['Parameter Value: In-Situ Data',
          'Parameter Bias',
          'Parameter Value: Climate Model Output',
          'In-Situ Observations Sample',
          'Climate Model Output Sample',
          'Pred. Value: In-Situ Data',
          'Pred. Uncertainty: In-Situ Data',
          'Pred. Value: Bias',
          'Pred. Uncertainty: Bias']

fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, -0.03),
           ncols=3,
           loc=10)
axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig10.png',dpi=300,bbox_inches='tight')
# %%
