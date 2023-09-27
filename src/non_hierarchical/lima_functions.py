import timeit
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from tinygp import kernels, GaussianProcess
import jax
import jax.numpy as jnp
import arviz as az
from scipy.stats import multivariate_normal

jax.config.update("jax_enable_x64", True)

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
