import timeit
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import multivariate_normal

jax.config.update("jax_enable_x64", True)

############# DATA GENERATION #############
def generate_underlying_data(scenario,rng_key):
    rng_key, rng_key_ = random.split(rng_key)
    GP_T = GaussianProcess(
        scenario['t_variance'] * kernels.ExpSquared(scenario['t_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['t_mean'])
    GP_B = GaussianProcess(
        scenario['b_variance'] * kernels.ExpSquared(scenario['b_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['b_mean'])
    
    scenario['T'] = GP_T.sample(rng_key)
    scenario['B'] = GP_B.sample(rng_key_)
    scenario['C'] = scenario['T']+scenario['B']

    rng_key, rng_key_ = random.split(rng_key)
    scenario['odata'] = GP_T.condition(scenario['T'],scenario['ox']).gp.sample(rng_key)
    odata_noise = dist.Normal(0.0,scenario['onoise']).sample(rng_key_,scenario['odata'].shape)
    scenario['odata'] = scenario['odata'] + odata_noise

    rng_key, rng_key_ = random.split(rng_key)
    scenario['cdata_o'] = GP_T.condition(scenario['T'],scenario['cx']).gp.sample(rng_key)
    scenario['cdata_b'] = GP_B.condition(scenario['B'],scenario['cx']).gp.sample(rng_key_)
    scenario['cdata'] = scenario['cdata_o']+scenario['cdata_b']
    rng_key, rng_key_ = random.split(rng_key)
    cdata_noise = dist.Normal(0.0,scenario['cnoise']).sample(rng_key,scenario['cdata'].shape)
    scenario['cdata'] = scenario['cdata'] + cdata_noise

############# MODEL FITTING #############
def run_inference(model,rng_key,num_warmup,num_samples,num_chains,*args,**kwargs):#data,distance_matrix=None):
    """
    Helper function for doing MCMC inference
    Args:
        model (python function): function that follows numpyros syntax
        rng_key (np array): PRNGKey for reproducible results
        num_warmup (int): Number of MCMC steps for warmup
        num_samples (int): Number of MCMC samples to take of parameters after warmup
        data (jax device array): data in shape [#days,#months,#sites]
        distance_matrix_values(jax device array): matrix of distances between sites, shape [#sites,#sites]
    Returns:
        MCMC numpyro instance (class object): An MCMC class object with functions such as .get_samples() and .run()
    """
    starttime = timeit.default_timer()

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains
    )
    
    mcmc.run(rng_key, *args,**kwargs)

    mcmc.print_summary()
    print("Time Taken:", timeit.default_timer() - starttime)
    return mcmc

def diagonal_noise(coord,noise):
    return(jnp.diag(jnp.full(coord.shape[0],noise)))

def generate_obs_conditional_climate_dist(scenario,ckernel,cmean,cnoise_var,okernel,omean,onoise_var):
    ox = scenario['ox']
    cx = scenario['cx']
    cdata = scenario['cdata']
    y2 = cdata
    u1 = jnp.full(ox.shape[0], omean)
    u2 = jnp.full(cx.shape[0], cmean)
    k11 = okernel(ox,ox) + diagonal_noise(ox,onoise_var)
    k12 = okernel(ox,cx)
    k21 = okernel(cx,ox) 
    k22 = ckernel(cx,cx) + diagonal_noise(cx,cnoise_var)
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn_dist = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn_dist)

def tinygp_2process_model(scenario):
    """
   Example model where the climate data is generated from 2 GPs,
   one of which also generates the observations and one of
   which generates bias in the climate model.
    """
    kern_var = numpyro.sample("kern_var", scenario['t_variance_prior'])
    lengthscale = numpyro.sample("lengthscale", scenario['t_lengthscale_prior'])
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", scenario['t_mean_prior'])

    bkern_var = numpyro.sample("bkern_var", scenario['b_variance_prior'])
    blengthscale = numpyro.sample("blengthscale", scenario['b_lengthscale_prior'])
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", scenario['b_mean_prior'])

    ckernel = kernel+bkernel
    cmean = mean+bmean
    cnoise_var = scenario['cnoise']**2
    cgp = GaussianProcess(ckernel, scenario['cx'], diag=cnoise_var, mean=cmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(),obs=scenario['cdata'])

    onoise_var = numpyro.sample("onoise", scenario['onoise_prior'])**2
    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(scenario,ckernel,cmean,cnoise_var,kernel,mean,onoise_var)
    numpyro.sample("obs_temperature", obs_conditional_climate_dist,obs=scenario['odata'])

def generate_posterior(scenario,rng_key,num_warmup,num_samples,num_chains):    
    mcmc_2process = run_inference(
        tinygp_2process_model, rng_key, num_warmup, num_samples,num_chains,scenario)
    idata_2process = az.from_numpyro(mcmc_2process)
    scenario['mcmc'] = idata_2process
    scenario['mcmc_samples']=mcmc_2process.get_samples()

############# Predictions #############
def generate_truth_predictive_dist(scenario,
                                   posterior_param_realisation):
    t_variance_realisation = posterior_param_realisation['t_variance_realisation']
    t_lengthscale_realisation = posterior_param_realisation['t_lengthscale_realisation']
    t_mean_realisation = posterior_param_realisation['t_mean_realisation']
    b_variance_realisation = posterior_param_realisation['b_variance_realisation']
    b_lengthscale_realisation = posterior_param_realisation['b_lengthscale_realisation']
    b_mean_realisation = posterior_param_realisation['b_mean_realisation']
    onoise_realisation = posterior_param_realisation['onoise_realisation']
    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    cnoise_var = scenario['cnoise']**2
    odata = scenario['odata']
    cdata = scenario['cdata']
    omean = t_mean_realisation
    bmean = b_mean_realisation
    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)
    onoise_var = onoise_realisation**2

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], omean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelo(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([kernelo(nx,ox),kernelo(nx,cx)])
    k21 = jnp.vstack([kernelo(ox,nx),kernelo(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,onoise_var),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,cnoise_var)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    # k1g2 = k11 - jnp.matmul(p21.T,p21)
    k1g2 = k11 - jnp.matmul(jnp.matmul(k12,k22i),k21)
    k1g2 = k1g2
    # mvn = multivariate_normal(u1g2,k1g2)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_bias_predictive_dist(scenario,
                                   posterior_param_realisation):
    t_variance_realisation = posterior_param_realisation['t_variance_realisation']
    t_lengthscale_realisation = posterior_param_realisation['t_lengthscale_realisation']
    t_mean_realisation = posterior_param_realisation['t_mean_realisation']
    b_variance_realisation = posterior_param_realisation['b_variance_realisation']
    b_lengthscale_realisation = posterior_param_realisation['b_lengthscale_realisation']
    b_mean_realisation = posterior_param_realisation['b_mean_realisation']
    onoise_realisation = posterior_param_realisation['onoise_realisation']
    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    cnoise_var = scenario['cnoise']**2
    odata = scenario['odata']
    cdata = scenario['cdata']
    omean = t_mean_realisation
    bmean = b_mean_realisation
    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)
    onoise_var = onoise_realisation**2

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], bmean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelb(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([jnp.full((len(nx),len(ox)),0),kernelb(nx,cx)])
    k21 = jnp.vstack([jnp.full((len(ox),len(nx)),0),kernelb(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,onoise_var),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,cnoise_var)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    # mvn = multivariate_normal(u1g2,k1g2)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_posterior_predictive_realisations(
        scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior = scenario['mcmc'].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration=0
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            'iteration':iteration,
            't_variance_realisation': posterior['kern_var'].data[0,:][i],
            't_lengthscale_realisation': posterior['lengthscale'].data[0,:][i],
            't_mean_realisation': posterior['mean'].data[0,:][i],
            'b_variance_realisation': posterior['bkern_var'].data[0,:][i],
            'b_lengthscale_realisation': posterior['blengthscale'].data[0,:][i],
            'b_mean_realisation': posterior['bmean'].data[0,:][i],
            'onoise_realisation': posterior['onoise'].data[0,:][i]
        }
        
        truth_predictive_dist = generate_truth_predictive_dist(scenario,
                                   posterior_param_realisation)
        bias_predictive_dist = generate_bias_predictive_dist(scenario,
                                   posterior_param_realisation)
        iteration+=1

        # truth_predictive_realisations = truth_predictive_dist.rvs(num_posterior_pred_realisations)
        # bias_predictive_realisations = bias_predictive_dist.rvs(num_posterior_pred_realisations)
        rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        bias_predictive_realisations = bias_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(truth_posterior_predictive_realisations)
    bias_posterior_predictive_realisations = jnp.array(bias_posterior_predictive_realisations)
    truth_posterior_predictive_realisations = truth_posterior_predictive_realisations.reshape(-1,truth_posterior_predictive_realisations.shape[-1])
    bias_posterior_predictive_realisations = bias_posterior_predictive_realisations.reshape(-1,bias_posterior_predictive_realisations.shape[-1])
    scenario['truth_posterior_predictive_realisations'] = truth_posterior_predictive_realisations
    scenario['bias_posterior_predictive_realisations'] = bias_posterior_predictive_realisations


############# Plotting #############
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

def plot_priors(scenario,prior_keys,axs,rng_key):
    for key,ax in zip(prior_keys,axs):
        variable = key.split('_prior')[0]
        value = scenario[variable]
        prior_sample = scenario[key].sample(rng_key,(10000,))
        prior_sample = remove_outliers(prior_sample)
        ax.hist(prior_sample,density=True,bins=100,alpha=0.8)
        ax.axvline(x=value, ymin=0, ymax=1,linestyle='--',color='k')

def plot_posteriors(posterior,posterior_keys,axs):
    for key,ax in zip(posterior_keys,axs):
        posterior_sample = posterior[key].data.reshape(-1)
        posterior_sample = remove_outliers(posterior_sample)
        ax.hist(posterior_sample,density=True,bins=100,alpha=0.8)

def plot_prior_and_posteriors(posterior,posterior_keys,axs):
    for key,ax in zip(posterior_keys,axs):
        posterior_sample = posterior[key].data.reshape(-1)
        posterior_sample = remove_outliers(posterior_sample)
        ax.hist(posterior_sample,density=True,bins=100,alpha=0.8)

def plot_predictions_1d(scenario,key,ax,ms=None,ylims=None,color=None):
    pred_mean = scenario[key].mean(axis=0)
    pred_std = scenario[key].std(axis=0)
    ax.plot(scenario['nx'],pred_mean,label='Expectation',color=color,alpha=0.5)
    ax.fill_between(scenario['nx'],pred_mean+pred_std,pred_mean-pred_std,label='$1\sigma$ Uncertainty',color=color,alpha=0.3)
    
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

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

############# Lima Method #############
def lima_model(scenario):
    """
    Example model where the truth is modelled just using the 
    observational data, which is generated from a GP
    """
    kern_var = numpyro.sample("kern_var", scenario['t_variance_prior'])
    lengthscale = numpyro.sample("lengthscale", scenario['t_lengthscale_prior'])
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", scenario['t_mean_prior'])

    noise_var = scenario['onoise']**2
    gp = GaussianProcess(kernel, scenario['ox'], diag=noise_var, mean=mean)
    numpyro.sample("observations", gp.numpyro_dist(),obs=scenario['odata'])

def generate_posterior_lima(scenario,rng_key,num_warmup,num_samples,num_chains):    
    mcmc = run_inference(
        lima_model, rng_key, num_warmup, num_samples,num_chains,scenario)
    idata = az.from_numpyro(mcmc)
    scenario['mcmc_lima'] = idata
    scenario['mcmc_lima_samples']=mcmc.get_samples()

def posterior_predictive_dist_lima(scenario,
                                   posterior_param_realisation):
    t_variance_realisation = posterior_param_realisation['t_variance_realisation']
    t_lengthscale_realisation = posterior_param_realisation['t_lengthscale_realisation']
    t_mean_realisation = posterior_param_realisation['t_mean_realisation']
    onoise_realisation = posterior_param_realisation['onoise_realisation']
    nx = scenario['nx']
    ox = scenario['ox']
    odata = scenario['odata']
    onoise_var = onoise_realisation**2
    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    gp = GaussianProcess(kernelo, ox, diag=onoise_var, mean=t_mean_realisation)
    gp_cond = gp.condition(odata, nx).gp
    return(gp_cond.numpyro_dist())

def generate_posterior_predictive_realisations_lima(
        scenario,
        num_parameter_realisations,
        num_posterior_pred_realisations,
        rng_key):
    
    posterior = scenario['mcmc'].posterior
    truth_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            't_variance_realisation': posterior['kern_var'].data[0,:][i],
            't_lengthscale_realisation': posterior['lengthscale'].data[0,:][i],
            't_mean_realisation': posterior['mean'].data[0,:][i],
            'onoise_realisation': posterior['onoise'].data[0,:][i]
        }
        truth_predictive_dist = posterior_predictive_dist_lima(scenario,
                                   posterior_param_realisation)

        truth_predictive_realisations = truth_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(truth_posterior_predictive_realisations)
    truth_posterior_predictive_realisations = truth_posterior_predictive_realisations.reshape(-1,truth_posterior_predictive_realisations.shape[-1])
    scenario['truth_posterior_predictive_realisations_lima'] = truth_posterior_predictive_realisations

def plot_underlying_data_1d_lima(scenario,ax,ms):
    ax.plot(scenario['X'],scenario['T'],label='Truth',alpha=0.6)

    ax.scatter(scenario['ox'],scenario['odata'],label='Observations',alpha=0.8,s=ms,marker='x')

    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()


############# LEGACY CODE #############
# cnoise = numpyro.sample("cnoise", scenario['cnoise_prior'])**2
