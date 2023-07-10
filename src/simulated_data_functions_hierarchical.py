import timeit
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.stats import norm
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import arviz as az

jax.config.update("jax_enable_x64", True)

############# DATA GENERATION #############
def generate_underlying_data_hierarchical(scenario,rng_key):
    rng_key, rng_key_ = random.split(rng_key)

    GP_MEAN_T = GaussianProcess(
        scenario['MEAN_T_variance'] * kernels.ExpSquared(scenario['MEAN_T_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['MEAN_T_mean'])
    GP_LOGVAR_T = GaussianProcess(
        scenario['LOGVAR_T_variance'] * kernels.ExpSquared(scenario['LOGVAR_T_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['LOGVAR_T_mean'])
    GP_MEAN_B = GaussianProcess(
        scenario['MEAN_B_variance'] * kernels.ExpSquared(scenario['MEAN_B_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['MEAN_B_mean'])
    GP_LOGVAR_B = GaussianProcess(
        scenario['LOGVAR_B_variance'] * kernels.ExpSquared(scenario['LOGVAR_B_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['LOGVAR_B_mean'])
    
    scenario['MEAN_T'] = GP_MEAN_T.sample(rng_key)
    scenario['LOGVAR_T'] = GP_LOGVAR_T.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_B'] = GP_MEAN_B.sample(rng_key)
    scenario['LOGVAR_B'] = GP_LOGVAR_B.sample(rng_key_)
    scenario['MEAN_C'] = scenario['MEAN_T']+scenario['MEAN_B']
    scenario['LOGVAR_C'] = scenario['LOGVAR_T']+scenario['LOGVAR_B']

    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_T_obs'] = GP_MEAN_T.condition(scenario['MEAN_T'],scenario['ox']).gp.sample(rng_key)
    scenario['LOGVAR_T_obs'] = GP_MEAN_T.condition(scenario['LOGVAR_T'],scenario['ox']).gp.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    N_T_obs = dist.Normal(scenario['MEAN_T_obs'],jnp.sqrt(jnp.exp(scenario['LOGVAR_T_obs'])))
    scenario['odata'] = N_T_obs.sample(rng_key,(scenario['osamples'],))

    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_T_climate'] = GP_MEAN_T.condition(scenario['MEAN_T'],scenario['cx']).gp.sample(rng_key)
    scenario['LOGVAR_T_climate'] = GP_LOGVAR_T.condition(scenario['LOGVAR_T'],scenario['cx']).gp.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_B_climate'] = GP_MEAN_B.condition(scenario['MEAN_B'],scenario['cx']).gp.sample(rng_key)
    scenario['LOGVAR_B_climate'] = GP_LOGVAR_B.condition(scenario['LOGVAR_B'],scenario['cx']).gp.sample(rng_key_)
    scenario['MEAN_C_climate'] = scenario['MEAN_T_climate']+scenario['MEAN_B_climate']
    scenario['LOGVAR_C_climate'] = scenario['LOGVAR_T_climate']+scenario['LOGVAR_B_climate']

    rng_key, rng_key_ = random.split(rng_key)
    N_C_climate = dist.Normal(scenario['MEAN_C_climate'],jnp.sqrt(jnp.exp(scenario['LOGVAR_C_climate'])))
    scenario['cdata'] = N_C_climate.sample(rng_key,(scenario['csamples'],))                 

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

def generate_mt_conditional_mc_dist(scenario,
                                    mc_kernel,
                                    mc_mean,
                                    mt_kernel,
                                    mt_mean,
                                    mc):
    ox = scenario['ox']
    cx = scenario['cx']
    y2 = mc
    u1 = jnp.full(ox.shape[0], mt_mean)
    u2 = jnp.full(cx.shape[0], mc_mean)
    k11 = mt_kernel(ox,ox) + diagonal_noise(ox,scenario['jitter'])
    k12 = mt_kernel(ox,cx)
    k21 = mt_kernel(cx,ox) 
    k22 = mc_kernel(cx,cx) + diagonal_noise(cx,scenario['jitter'])
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn_dist = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn_dist)

def generate_lvt_conditional_lvc_dist(scenario,
                                    lvc_kernel,
                                    lvc_mean,
                                    lvt_kernel,
                                    lvt_mean,
                                    lvc):
    ox = scenario['ox']
    cx = scenario['cx']
    y2 = lvc
    u1 = jnp.full(ox.shape[0], lvt_mean)
    u2 = jnp.full(cx.shape[0], lvc_mean)
    k11 = lvt_kernel(ox,ox) + diagonal_noise(ox,scenario['jitter'])
    k12 = lvt_kernel(ox,cx)
    k21 = lvt_kernel(cx,ox) 
    k22 = lvc_kernel(cx,cx) + diagonal_noise(cx,scenario['jitter'])
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn_dist = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn_dist)

def hierarchical_model(scenario):
    mt_kern_var = numpyro.sample("mt_kern_var", scenario['MEAN_T_variance_prior'])
    mt_lengthscale = numpyro.sample("mt_lengthscale", scenario['MEAN_T_lengthscale_prior'])
    mt_kernel = mt_kern_var * kernels.ExpSquared(mt_lengthscale)
    mt_mean = numpyro.sample("mt_mean", scenario['MEAN_T_mean_prior'])
    mb_kern_var = numpyro.sample("mb_kern_var", scenario['MEAN_B_variance_prior'])
    mb_lengthscale = numpyro.sample("mb_lengthscale", scenario['MEAN_B_lengthscale_prior'])
    mb_kernel = mb_kern_var * kernels.ExpSquared(mb_lengthscale)
    mb_mean = numpyro.sample("mb_mean", scenario['MEAN_B_mean_prior'])

    mc_kernel = mt_kernel+mb_kernel
    mc_mean = mt_mean+mb_mean
    mc_gp = GaussianProcess(mc_kernel, scenario['cx'], diag=scenario['jitter'], mean=mc_mean)
    mc = numpyro.sample("mc", mc_gp.numpyro_dist())
    mt_conditional_mc_dist = generate_mt_conditional_mc_dist(scenario,
                                                             mc_kernel,
                                                             mc_mean,
                                                             mt_kernel,
                                                             mt_mean,
                                                             mc)
    mt = numpyro.sample("mt", mt_conditional_mc_dist)

    lvt_kern_var = numpyro.sample("lvt_kern_var", scenario['LOGVAR_T_variance_prior'])
    lvt_lengthscale = numpyro.sample("lvt_lengthscale", scenario['LOGVAR_T_lengthscale_prior'])
    lvt_kernel = lvt_kern_var * kernels.ExpSquared(lvt_lengthscale)
    lvt_mean = numpyro.sample("lvt_mean", scenario['LOGVAR_T_mean_prior'])
    lvb_kern_var = numpyro.sample("lvb_kern_var", scenario['LOGVAR_B_variance_prior'])
    lvb_lengthscale = numpyro.sample("lvb_lengthscale", scenario['LOGVAR_B_lengthscale_prior'])
    lvb_kernel = lvb_kern_var * kernels.ExpSquared(lvb_lengthscale)
    lvb_mean = numpyro.sample("lvb_mean", scenario['LOGVAR_B_mean_prior'])

    lvc_kernel = lvt_kernel+lvb_kernel
    lvc_mean = lvt_mean+lvb_mean
    lvc_gp = GaussianProcess(lvc_kernel, scenario['cx'], diag=scenario['jitter'], mean=lvc_mean)
    lvc = numpyro.sample("lvc", lvc_gp.numpyro_dist())
    lvt_conditional_lvc_dist = generate_lvt_conditional_lvc_dist(scenario,
                                                             lvc_kernel,
                                                             lvc_mean,
                                                             lvt_kernel,
                                                             lvt_mean,
                                                             lvc)
    lvt = numpyro.sample("lvt", lvt_conditional_lvc_dist)

    vt = jnp.exp(lvt)
    vc = jnp.exp(lvc)
    numpyro.sample("t", dist.Normal(mt, jnp.sqrt(vt)),obs=scenario['odata'])
    numpyro.sample("c", dist.Normal(mc, jnp.sqrt(vc)),obs=scenario['cdata'])

def generate_posterior_hierarchical(scenario,rng_key,num_warmup,num_samples,num_chains):    
    mcmc = run_inference(
        hierarchical_model, rng_key, num_warmup, num_samples,num_chains,scenario)
    idata = az.from_numpyro(mcmc)
    scenario['mcmc'] = idata
    scenario['mcmc_samples']=mcmc.get_samples()

############# Predictions #############

def generate_mean_truth_predictive_dist(scenario,
                                   posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation['mt_variance_realisation']
    mt_lengthscale_realisation = posterior_param_realisation['mt_lengthscale_realisation']
    mt_mean_realisation = posterior_param_realisation['mt_mean_realisation']
    mb_variance_realisation = posterior_param_realisation['mb_variance_realisation']
    mb_lengthscale_realisation = posterior_param_realisation['mb_lengthscale_realisation']
    mb_mean_realisation = posterior_param_realisation['mb_mean_realisation']

    mt_realisation = posterior_param_realisation['mt_realisation']
    mc_realisation = posterior_param_realisation['mc_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    odata = mt_realisation
    cdata = mc_realisation
    omean = mt_mean_realisation
    bmean = mb_mean_realisation
    kernelo = mt_variance_realisation * kernels.ExpSquared(mt_lengthscale_realisation)
    kernelb = mb_variance_realisation * kernels.ExpSquared(mb_lengthscale_realisation)

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], omean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelo(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([kernelo(nx,ox),kernelo(nx,cx)])
    k21 = jnp.vstack([kernelo(ox,nx),kernelo(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,jitter),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,jitter)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_logvar_truth_predictive_dist(scenario,
                                   posterior_param_realisation):
    lvt_variance_realisation = posterior_param_realisation['lvt_variance_realisation']
    lvt_lengthscale_realisation = posterior_param_realisation['lvt_lengthscale_realisation']
    lvt_mean_realisation = posterior_param_realisation['lvt_mean_realisation']
    lvb_variance_realisation = posterior_param_realisation['lvb_variance_realisation']
    lvb_lengthscale_realisation = posterior_param_realisation['lvb_lengthscale_realisation']
    lvb_mean_realisation = posterior_param_realisation['lvb_mean_realisation']

    lvt_realisation = posterior_param_realisation['lvt_realisation']
    lvc_realisation = posterior_param_realisation['lvc_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    odata = lvt_realisation
    cdata = lvc_realisation
    omean = lvt_mean_realisation
    bmean = lvb_mean_realisation
    kernelo = lvt_variance_realisation * kernels.ExpSquared(lvt_lengthscale_realisation)
    kernelb = lvb_variance_realisation * kernels.ExpSquared(lvb_lengthscale_realisation)

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], omean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelo(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([kernelo(nx,ox),kernelo(nx,cx)])
    k21 = jnp.vstack([kernelo(ox,nx),kernelo(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,jitter),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,jitter)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_mean_bias_predictive_dist(scenario,
                                   posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation['mt_variance_realisation']
    mt_lengthscale_realisation = posterior_param_realisation['mt_lengthscale_realisation']
    mt_mean_realisation = posterior_param_realisation['mt_mean_realisation']
    mb_variance_realisation = posterior_param_realisation['mb_variance_realisation']
    mb_lengthscale_realisation = posterior_param_realisation['mb_lengthscale_realisation']
    mb_mean_realisation = posterior_param_realisation['mb_mean_realisation']

    mt_realisation = posterior_param_realisation['mt_realisation']
    mc_realisation = posterior_param_realisation['mc_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    odata = mt_realisation
    cdata = mc_realisation
    omean = mt_mean_realisation
    bmean = mb_mean_realisation
    kernelo = mt_variance_realisation * kernels.ExpSquared(mt_lengthscale_realisation)
    kernelb = mb_variance_realisation * kernels.ExpSquared(mb_lengthscale_realisation)

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], bmean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelb(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([jnp.full((len(nx),len(ox)),0),kernelb(nx,cx)])
    k21 = jnp.vstack([jnp.full((len(ox),len(nx)),0),kernelb(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,jitter),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,jitter)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_logvar_bias_predictive_dist(scenario,
                                   posterior_param_realisation):
    lvt_variance_realisation = posterior_param_realisation['lvt_variance_realisation']
    lvt_lengthscale_realisation = posterior_param_realisation['lvt_lengthscale_realisation']
    lvt_mean_realisation = posterior_param_realisation['lvt_mean_realisation']
    lvb_variance_realisation = posterior_param_realisation['lvb_variance_realisation']
    lvb_lengthscale_realisation = posterior_param_realisation['lvb_lengthscale_realisation']
    lvb_mean_realisation = posterior_param_realisation['lvb_mean_realisation']

    lvt_realisation = posterior_param_realisation['lvt_realisation']
    lvc_realisation = posterior_param_realisation['lvc_realisation']

    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']
    odata = lvt_realisation
    cdata = lvc_realisation
    omean = lvt_mean_realisation
    bmean = lvb_mean_realisation
    kernelo = lvt_variance_realisation * kernels.ExpSquared(lvt_lengthscale_realisation)
    kernelb = lvb_variance_realisation * kernels.ExpSquared(lvb_lengthscale_realisation)

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], bmean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelb(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([jnp.full((len(nx),len(ox)),0),kernelb(nx,cx)])
    k21 = jnp.vstack([jnp.full((len(ox),len(nx)),0),kernelb(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,jitter),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,jitter)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_posterior_predictive_realisations_hierarchical_mean(
        scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior = scenario['mcmc'].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            'mt_variance_realisation': posterior['mt_kern_var'].data[0,:][i],
            'mt_lengthscale_realisation': posterior['mt_lengthscale'].data[0,:][i],
            'mt_mean_realisation': posterior['mt_mean'].data[0,:][i],
            'mb_variance_realisation': posterior['mb_kern_var'].data[0,:][i],
            'mb_lengthscale_realisation': posterior['mb_lengthscale'].data[0,:][i],
            'mb_mean_realisation': posterior['mb_mean'].data[0,:][i],
            'mt_realisation': posterior['mt'].data[0,:][i],
            'mc_realisation': posterior['mc'].data[0,:][i]
        }
        
        truth_predictive_dist = generate_mean_truth_predictive_dist(scenario,
                                   posterior_param_realisation)
        bias_predictive_dist = generate_mean_bias_predictive_dist(scenario,
                                   posterior_param_realisation)

        rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        bias_predictive_realisations = bias_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(truth_posterior_predictive_realisations)
    bias_posterior_predictive_realisations = jnp.array(bias_posterior_predictive_realisations)
    truth_posterior_predictive_realisations = truth_posterior_predictive_realisations.reshape(-1,truth_posterior_predictive_realisations.shape[-1])
    bias_posterior_predictive_realisations = bias_posterior_predictive_realisations.reshape(-1,bias_posterior_predictive_realisations.shape[-1])
    scenario['mean_truth_posterior_predictive_realisations'] = truth_posterior_predictive_realisations
    scenario['mean_bias_posterior_predictive_realisations'] = bias_posterior_predictive_realisations

def generate_posterior_predictive_realisations_hierarchical_std(
        scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior = scenario['mcmc'].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            'lvt_variance_realisation': posterior['lvt_kern_var'].data[0,:][i],
            'lvt_lengthscale_realisation': posterior['lvt_lengthscale'].data[0,:][i],
            'lvt_mean_realisation': posterior['lvt_mean'].data[0,:][i],
            'lvb_variance_realisation': posterior['lvb_kern_var'].data[0,:][i],
            'lvb_lengthscale_realisation': posterior['lvb_lengthscale'].data[0,:][i],
            'lvb_mean_realisation': posterior['lvb_mean'].data[0,:][i],
            'lvt_realisation': posterior['lvt'].data[0,:][i],
            'lvc_realisation': posterior['lvc'].data[0,:][i]
        }
        
        truth_predictive_dist = generate_logvar_truth_predictive_dist(scenario,
                                   posterior_param_realisation)
        bias_predictive_dist = generate_logvar_bias_predictive_dist(scenario,
                                   posterior_param_realisation)

        rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        bias_predictive_realisations = bias_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(truth_posterior_predictive_realisations)
    bias_posterior_predictive_realisations = jnp.array(bias_posterior_predictive_realisations)
    truth_posterior_predictive_realisations = truth_posterior_predictive_realisations.reshape(-1,truth_posterior_predictive_realisations.shape[-1])
    bias_posterior_predictive_realisations = bias_posterior_predictive_realisations.reshape(-1,bias_posterior_predictive_realisations.shape[-1])
    scenario['std_truth_posterior_predictive_realisations'] = jnp.sqrt(jnp.exp(truth_posterior_predictive_realisations))
    scenario['std_bias_posterior_predictive_realisations'] = jnp.sqrt(jnp.exp(bias_posterior_predictive_realisations))

############# Plotting #############

def plot_underlying_data_mean_1d(scenario,ax,ms):
    ax.plot(scenario['X'],scenario['MEAN_T'],label='Truth Mean',alpha=0.6)
    ax.plot(scenario['X'],scenario['MEAN_B'],label='Bias Mean',alpha=0.6)
    ax.plot(scenario['X'],scenario['MEAN_C'],label='Climate Model Mean',alpha=0.6)

    ax.scatter(scenario['ox'],scenario['odata'].mean(axis=0),label='Observations Mean',alpha=0.8,s=ms,marker='x')
    ax.scatter(scenario['cx'],scenario['cdata'].mean(axis=0),color='g',label='Climate Model Output Mean',alpha=0.8,s=ms,marker='+')

    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plot_underlying_data_std_1d(scenario,ax,ms):
    ax.plot(scenario['X'],jnp.sqrt(jnp.exp(scenario['LOGVAR_T'])),label='Truth Std',alpha=0.6)
    ax.plot(scenario['X'],jnp.sqrt(jnp.exp(scenario['LOGVAR_B'])),label='Bias Std',alpha=0.6)
    ax.plot(scenario['X'],jnp.sqrt(jnp.exp(scenario['LOGVAR_C'])),label='Climate Model Std',alpha=0.6)

    ax.scatter(scenario['ox'],scenario['odata'].std(axis=0),label='Observations Std',alpha=0.8,s=ms,marker='x')
    ax.scatter(scenario['cx'],scenario['cdata'].std(axis=0),color='g',label='Climate Model Output Std',alpha=0.8,s=ms,marker='+')
    
    ax.set_xlabel('time')
    ax.set_ylabel('temperature std')
    ax.legend()

def plot_pdfs_1d(scenario,ax,index):
    MEAN_T_sample = scenario['MEAN_T'][index]
    STDEV_T_sample = jnp.sqrt(jnp.exp(scenario['LOGVAR_T']))[index]
    MEAN_C_sample = scenario['MEAN_C'][index]
    STDEV_C_sample = jnp.sqrt(jnp.exp(scenario['LOGVAR_C']))[index]

    min_x = min(MEAN_T_sample-3*STDEV_T_sample,MEAN_C_sample-3*STDEV_C_sample)
    max_x = max(MEAN_T_sample+3*STDEV_T_sample,MEAN_C_sample+3*STDEV_C_sample)
    xs = np.linspace(min_x,max_x,100)
    yts = norm.pdf(xs, MEAN_T_sample, STDEV_T_sample)
    ycs = norm.pdf(xs, MEAN_C_sample, STDEV_C_sample)

    ax.plot(xs, yts, lw=2, label='In-situ observations',color='tab:blue')
    ax.plot(xs, ycs, lw=2, label='Climate model output',color='tab:green')
    ax.fill_between(xs, yts, interpolate=True, color='tab:blue',alpha=0.6)
    ax.fill_between(xs, ycs, interpolate=True, color='tab:green',alpha=0.6)
    ax.set_xlabel('temperature')
    ax.set_ylabel('probability density')
    ax.legend()

def create_levels(scenario,vars,sep,rounding,center=None):
    data_list = []
    for var in vars:
        data_list.append(scenario[var])
    data = np.array(data_list)
    vmin = data.min()
    vmax = data.max()
    abs_max_rounded = max(np.abs(vmin),vmax).round(rounding)
    if center != None:
        levels = np.arange(-abs_max_rounded, abs_max_rounded+sep, sep)
    else:
        levels = np.arange(vmin.round(rounding), vmax.round(rounding)+sep, sep)
    return(levels)

def plot_underlying_data_2d(scenario,variables,axs,ms,center,cmap):
    plots = []
    titles = ['Truth','Bias','Climate Model Output']
    levels = create_levels(scenario,variables,0.25,0,center=center)

    for ax,var,title in zip(axs, variables, titles):
        plots.append(ax.contourf(scenario['X1'],
                    scenario['X2'],
                    scenario[var].reshape(scenario['X1'].shape),
                    label=title,
                    alpha=0.6,
                    cmap=cmap,
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

def plot_predictions_1d_mean_hierarchical(scenario,key,ax,ms=None,ylims=None,color=None):
    pred_mean = scenario[key].mean(axis=0)
    pred_std = scenario[key].std(axis=0)
    ax.plot(scenario['nx'],pred_mean,label='Expectation',color=color,alpha=0.5)
    ax.fill_between(scenario['nx'],pred_mean+pred_std,pred_mean-pred_std,label='$1\sigma$ Uncertainty',color=color,alpha=0.3)
    
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()
