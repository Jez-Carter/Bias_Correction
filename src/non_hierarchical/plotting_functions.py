import numpy as np
import matplotlib.pyplot as plt

def create_levels(scenario,sep,rounding,center=None):
    data = np.array([scenario['T'],scenario['B'],scenario['C']])
    vmin = data.min()
    vmax = data.max()
    abs_max_rounded = max(np.abs(vmin),vmax).round(rounding)
    if center != None:
        levels = np.arange(-abs_max_rounded, abs_max_rounded+sep, sep)
    else:
        levels = np.arange(vmin.round(rounding), vmax.round(rounding)+sep, sep)
    return(levels)

def remove_outliers(array, perc=[0.001,0.99]):
  lower_threshold = np.quantile(array,perc[0])
  upper_threshold = np.quantile(array,perc[1])
  outlier_condition = (array > upper_threshold) | (array < lower_threshold)
  return(array[outlier_condition==False])

def plot_priors(scenario,prior_keys,axs,rng_key,vlinewidth):
    for key,ax in zip(prior_keys,axs):
        variable = key.split('_prior')[0]
        value = scenario[variable]
        prior_sample = scenario[key].sample(rng_key,(10000,))
        prior_sample = remove_outliers(prior_sample)
        ax.hist(prior_sample,density=True,bins=100,alpha=0.6)
        ax.axvline(x=value, ymin=0, ymax=1,linestyle='--',color='k',linewidth=vlinewidth)

def plot_posteriors(posterior,posterior_keys,axs):
    for key,ax in zip(posterior_keys,axs):
        posterior_sample = posterior[key].data.reshape(-1)
        posterior_sample = remove_outliers(posterior_sample)
        ax.hist(posterior_sample,density=True,bins=100,alpha=0.6)

def plot_prior_and_posteriors(posterior,posterior_keys,axs):
    for key,ax in zip(posterior_keys,axs):
        posterior_sample = posterior[key].data.reshape(-1)
        posterior_sample = remove_outliers(posterior_sample)
        ax.hist(posterior_sample,density=True,bins=100,alpha=0.8)

############# 1 Dimensional Data Figures #############

def plot_underlying_data_1d(scenario,ax,ms):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=0.6)
    ax.plot(scenario['X'],scenario['B'],label='Bias',alpha=0.6)
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=0.6)

    ax.scatter(scenario['ox'],scenario['odata'],label='Observations',alpha=0.8,s=ms,marker='x')
    ax.scatter(scenario['cx'],scenario['cdata'],color='g',label='Climate Model Output',alpha=0.8,s=ms,marker='+')

    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plot_latent_data_1d(scenario,ax,ms):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=0.6)
    ax.plot(scenario['X'],scenario['B'],label='Bias',alpha=0.6)
    ax.plot(scenario['X'],scenario['C'],label='Climate Model',alpha=0.6)

    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plot_predictions_1d(scenario,key,ax,ms=None,ylims=None,color=None):
    pred_mean = scenario[key].mean(axis=0)
    pred_std = scenario[key].std(axis=0)
    ax.plot(scenario['nx'],pred_mean,label='Expectation',color=color,alpha=0.5)
    ax.fill_between(scenario['nx'],pred_mean+pred_std,pred_mean-pred_std,label='$1\sigma$ Uncertainty',color=color,alpha=0.3)
    
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plot_underlying_data_1d_lima(scenario,ax,ms):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=0.6)
    ax.scatter(scenario['ox'],scenario['odata'],label='Observations',alpha=0.8,s=ms,marker='x')
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

############# 2 Dimensional Data Figures #############

def plot_underlying_data_2d(scenario,axs,ms):
    plots = []
    variables = ['T','B','C']
    titles = ['Truth','Bias','Climate Model Output']
    levels = create_levels(scenario,0.25,0,center=True)
    for ax,var,title in zip(axs, variables, titles):
        plots.append(ax.contourf(scenario['X1'],
                    scenario['X2'],
                    scenario[var].reshape(scenario['X1'].shape),
                    label=title,
                    alpha=0.6,
                    cmap='RdBu_r',
                    levels=levels
        ))
    for plot in plots:
        plt.colorbar(plot)
    axs[0].scatter(scenario['ox'][:,0],
                scenario['ox'][:,1],
                s=30, marker='o', c="None",edgecolor='k')
    axs[2].scatter(scenario['cx'][:,0],
                scenario['cx'][:,1],
                s=30, marker='x', c="k")
    CX1_min = scenario['CX1'].min()
    CX2_min = scenario['CX2'].min()
    CX1_max = scenario['CX1'].max()
    CX2_max = scenario['CX2'].max()
    sepCX1 = scenario['CX1'][0,1]-scenario['CX1'][0,0]
    sepCX2 = scenario['CX2'][1,0]-scenario['CX2'][0,0]
    x1_markers = scenario['CX1'][0,:]+sepCX1/2
    x2_markers = scenario['CX2'][:,0]+sepCX2/2
    for value in x1_markers[:-1]:
        axs[2].axvline(value,CX1_min,CX1_max,linestyle='--',color='k')
    for value in x2_markers[:-1]:
        axs[2].axhline(value,CX2_min,CX2_max,linestyle='--',color='k')

def plot_predictions_2d(scenario,axs):

    truth = scenario['truth_posterior_predictive_realisations']
    truth_mean = truth.mean(axis=0)
    truth_std = truth.std(axis=0)
    bias = scenario['bias_posterior_predictive_realisations']
    bias_mean = bias.mean(axis=0)
    bias_std = bias.std(axis=0)
    T = scenario['T']
    B = scenario['B']

    plots = []

    levels = create_levels(scenario,0.25,0,center=True)

    for ax,data in zip(axs[::3],[T,B]):
        plots.append(ax.contourf(scenario['X1'],
                    scenario['X2'],
                    data.reshape(scenario['X1'].shape),
                    cmap='RdBu_r',
                    levels=levels
        ))

    for ax,data in zip(axs[1::3],[truth_mean,bias_mean]):
        plots.append(ax.contourf(scenario['X1'],
                    scenario['X2'],
                    data.reshape(scenario['X1'].shape),
                    cmap='RdBu_r',
                    levels=levels
        ))

    for ax,data in zip(axs[2::3],[truth_std,bias_std]):
        plots.append(ax.contourf(scenario['X1'],
                    scenario['X2'],
                    data.reshape(scenario['X1'].shape),
                    cmap='viridis'
        ))

    for plot in plots:
        plt.colorbar(plot)

    for ax in axs:

        ax.scatter(scenario['ox'][:,0],
                    scenario['ox'][:,1],
                    s=30, marker='o', c="None",edgecolor='k',alpha=0.5)
        ax.scatter(scenario['cx'][:,0],
                    scenario['cx'][:,1],
                    s=30, marker='x', c="k",alpha=0.5)

############# Quantile Mapping Figures #############
def plotting_quantile_mapping(ax,scenario):
    ax.plot(scenario['t'],scenario['ocomplete_realisation'],label='True Underlying Field',alpha=0.6)
    ax.plot(scenario['t'],scenario['ccomplete_realisation'],label='Biased Underlying Field',alpha=0.6)
    ax.scatter(scenario['ot'],scenario['odata'],label='In-Situ Observations',alpha=1.0,s=10,marker='x')
    ax.scatter(scenario['ct'],scenario['cdata'],label='Climate Model Output',alpha=1.0,s=10,marker='x')

    ax.plot(scenario['ct'],scenario['c_corrected'].mean(axis=0),
            label='Bias Corrected Output Expectation',
            color='k',alpha=0.6,linestyle='dotted')

    ax.fill_between(scenario['ct'],
                    scenario['c_corrected'].mean(axis=0)+3*scenario['c_corrected'].std(axis=0),
                    scenario['c_corrected'].mean(axis=0)-3*scenario['c_corrected'].std(axis=0),
                    interpolate=True, color='k',alpha=0.2,
                    label='Bias Corrected Output Uncertainty 3$\sigma$')