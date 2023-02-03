import timeit
import numpyro
import jax.numpy as jnp
import jax.scipy.stats.gamma as jgamma
from jax.scipy.special import expit
from jax import random, vmap, jit
from numpyro.distributions import constraints
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, BarkerMH
from numpy import ndarray
import numpy as np
from numpy import ndarray
# from tensorflow.keras.optimizers import Optimizer
# from gpflow.models import SGPR
from tinygp import kernels, GaussianProcess,transforms
from scipy.cluster.vq import kmeans2

def normal_model(jdata,mask):
    """
    Standard Normal model code for use with numpyro
    Args:
        jdata (jax device array): data in shape [days,months,sites]
    """
    # Number of Months and Sites
    sites = jdata.shape[1]
    
    with numpyro.plate("Sites", sites, dim=-1) as j:        
        loc = numpyro.sample("loc", dist.Uniform(-40, 0))
        scale = numpyro.sample("scale", dist.Gamma(0.001, 0.001)) 
        
        masked_dist = dist.Normal(loc, scale).mask(mask)
        numpyro.sample("obs", masked_dist, obs=jdata)

class BernoulliGamma(numpyro.distributions.Distribution):
    """
    Creates a Bernoulli-Gamma distribution class to use with numpyro
    Args:
        parameters (list): a list of parameters in the order of [p,alpha,beta]
    Returns:
        (numpyro.distributions.Distribution)
    """
    support = constraints.positive

    def __init__(self, params):
        self.p = params[0]
        self.alpha = params[1]
        self.scale = jnp.reciprocal(params[2])
        super().__init__(batch_shape=jnp.shape(params[0]), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        
        bernoulli = 1 - self.p
        gamma = self.p*jgamma.pdf(jnp.where(value == 0, 1, value),a=self.alpha,loc=0, scale=self.scale) # where statement prevents evaluations at values of zero and instead evaluates at 1
        
        log_bernoulli = jnp.log(bernoulli)
        log_gamma = jnp.log(jnp.where(gamma == 0, 1e-20 , gamma)) # where statement prevents log evaluations at values of zero and instead evaluates at 1e-20 (Log contributions from outliers e.g. negative values and very high snowfall get then capped at ~-46 (np.log(1e-20)))
        
        log_bernoulli_sum = jnp.sum(jnp.where(value == 0, log_bernoulli, 0), axis=0)
        log_gamma_sum = jnp.sum(jnp.where(value != 0, log_gamma, 0), axis=0)

        return log_bernoulli_sum+log_gamma_sum

def base_model(jdata):
    """
    Bernoulli-Gamma model code for use with numpyro
    Args:
        jdata (jax device array): data in shape [days,months,sites]
    """

    p = numpyro.sample("p", dist.Uniform(0.01, 0.99))
    alpha = numpyro.sample("alpha", dist.Gamma(0.001, 0.001))
    beta = numpyro.sample("beta", dist.Gamma(0.001, 0.001))

    numpyro.sample(
        "obs",
        BernoulliGamma([p, alpha, beta]),
        obs=jdata,
    )
    
def top_model(distance_matrix_values,jdata):
    """
    Gaussian Process model code for use with numpyro
    Args:
        distance_matrix_values (jax device array): matrix of distances between sites, shape [#sites,#sites]
        jdata (jax device array): data in shape [#days,#months,#sites]
    """
    
    #Hyper Params GP:
    var_alpha = numpyro.sample("kernel_var_alpha", dist.LogNormal(0.1, 10.0))
    noise_alpha = numpyro.sample("kernel_noise_alpha", dist.LogNormal(0.1, 10.0))
    length_alpha = numpyro.sample("kernel_length_alpha", dist.LogNormal(0.1, 10.0))
    
    kern_alpha = gpkernel(distance_matrix_values, var_alpha, length_alpha, noise_alpha)
    
    alpha = numpyro.sample("alpha", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern_alpha))
    
def full_model(distance_matrix_values,jdata):
    """
    Bernoulli-Gamma, Gaussian Process Hierarchical model code for use with numpyro
    Args:
        distance_matrix_values (jax device array): matrix of distances between sites, shape [#sites,#sites]
        jdata (jax device array): data in shape [#days,#months,#sites]
    """
    
    #Hyper Params GP:
    var_alpha = numpyro.sample("kernel_var_alpha", dist.LogNormal(0.1, 10.0))
    noise_alpha = numpyro.sample("kernel_noise_alpha", dist.LogNormal(0.1, 10.0))
    length_alpha = numpyro.sample("kernel_length_alpha", dist.LogNormal(0.1, 10.0))
    
    var_beta = numpyro.sample("kernel_var_beta", dist.LogNormal(0.1, 10.0))
    noise_beta = numpyro.sample("kernel_noise_beta", dist.LogNormal(0.1, 10.0))
    length_beta = numpyro.sample("kernel_length_beta", dist.LogNormal(0.1, 10.0))
    
    var_p = numpyro.sample("kernel_var_p", dist.LogNormal(0.1, 10.0))
    noise_p = numpyro.sample("kernel_noise_p", dist.LogNormal(0.1, 10.0))
    length_p = numpyro.sample("kernel_length_p", dist.LogNormal(0.1, 10.0))
    
    kern_alpha = gpkernel(distance_matrix_values, var_alpha, length_alpha, noise_alpha)
    kern_beta = gpkernel(distance_matrix_values, var_beta, length_beta, noise_beta)
    kern_p = gpkernel(distance_matrix_values, var_p, length_p, noise_p)
    
    alpha = numpyro.sample("alpha", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern_alpha))
    beta = numpyro.sample("beta", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern_beta))
    p = numpyro.sample("p", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern_p))
    
    # Number of Months and Sites
    months = jdata.shape[1]
    sites = jdata.shape[2]
    
    numpyro.sample(
        "obs",
        BernoulliGamma([p, alpha, beta]),
        obs=jdata,
    )

    
def bg_model(jdata):
    """
    Bernoulli-Gamma model code for use with numpyro
    Args:
        jdata (jax device array): data in shape [days,months,sites]
    """
    # Hyper-Params: (These are used to describe the relationship between alpha and the scale parameter from the Bernoulli-Gamma dist.)
    a0 = numpyro.sample("a0", dist.Uniform(-10, 10.0))
    a1 = numpyro.sample("a1", dist.Uniform(-10, 10.0))
    betavar = numpyro.sample("betavar", dist.InverseGamma(0.001, 0.001))
    
    # Number of Months and Sites
    months = jdata.shape[1]
    sites = jdata.shape[2]

    with numpyro.plate("Months", months, dim=-2) as k:
        with numpyro.plate("Sites", sites, dim=-1) as j:
            # Bernoulli-Gamma Params
            p = numpyro.sample("p", dist.Uniform(0.01, 0.99))
            alpha = numpyro.sample("alpha", dist.Gamma(0.001, 0.001))
            beta = numpyro.sample("beta",dist.LogNormal(a0+a1*alpha,betavar))

            numpyro.sample(
                "obs",
                BernoulliGamma([p, alpha, beta]),
                obs=jdata,
            )
    
def gpkernel(distance_matrix_values, var, length, noise, jitter=1.0e-6, include_noise=True):
    """
    Takes a distance matrix and using kernel length scales, variance and noise converts into a covariance matrix using an exponential squared kernel
    Args:
        distance_matrix_values(jax device array): matrix of distances between sites, shape [#sites,#sites]
        var(float): kernel variance
        length(float): kernel lengthscale
        noise(float): kernel noise
    Returns:
        k(jax device array): covariance matrix of shape [#sites,#sites]
    """
    deltaXsq = jnp.power(distance_matrix_values / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(distance_matrix_values.shape[0])
    return k
    

def bg_gp_model(distance_matrix_values,jdata):
    """
    Bernoulli-Gamma, Gaussian Process Hierarchical model code for use with numpyro
    Args:
        distance_matrix_values (jax device array): matrix of distances between sites, shape [#sites,#sites]
        jdata (jax device array): data in shape [#days,#months,#sites]
    """
    # Hyper-Params: (These are used to describe the relationship between alpha and the scale parameter from the Bernoulli-Gamma dist.)
    a0 = numpyro.sample("a0", dist.Uniform(-10, 10.0))
    a1 = numpyro.sample("a1", dist.Uniform(-10, 10.0))
    betavar = numpyro.sample("betavar", dist.InverseGamma(0.001, 0.001))
    # alphavar = numpyro.sample("alphavar", dist.InverseGamma(0.001, 0.001))
    
    #Hyper Params GP:
    var_alpha = numpyro.sample("kernel_var_alpha", dist.LogNormal(0.1, 10.0))
    noise_alpha = numpyro.sample("kernel_noise_alpha", dist.LogNormal(0.1, 10.0))
    length_alpha = numpyro.sample("kernel_length_alpha", dist.LogNormal(0.1, 10.0))
    var_p = numpyro.sample("kernel_var_p", dist.LogNormal(0.1, 10.0))
    noise_p = numpyro.sample("kernel_noise_p", dist.LogNormal(0.1, 10.0))
    length_p = numpyro.sample("kernel_length_p", dist.LogNormal(0.1, 10.0))
    
    kern_alpha = gpkernel(distance_matrix_values, var_alpha, length_alpha, noise_alpha)
    kern_p = gpkernel(distance_matrix_values, var_p, length_p, noise_p)

    # Number of Months and Sites
    months = jdata.shape[1]
    sites = jdata.shape[2]
    
    with numpyro.plate("Months", months, dim=-2) as k:
        
        log_alpha = numpyro.sample("log_alpha", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern_alpha))
        
#         log_alpha = numpyro.sample("log_alpha", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern_alpha))
        alpha = jnp.exp(log_alpha)
        alpha = alpha.reshape(months,sites)
        
        # alpha = numpyro.sample("alpha",dist.LogNormal(log_alpha,alphavar))
        # log_alpha = log_alpha.reshape(months,sites)
        logit_p = numpyro.sample("logit_p", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern_p))
        p = expit(logit_p) # dist.bernoulli(MNV)
        p = p.reshape(months,sites) 
        # with numpyro.plate("Sites", sites, dim=-1) as j:
        #     p = numpyro.sample("p", dist.Uniform(0.01, 0.99))
    
    # alpha = numpyro.sample("alpha",dist.LogNormal(log_alpha,alphavar))
    
    beta = numpyro.sample("beta",dist.LogNormal(a0+a1*alpha,betavar))

    numpyro.sample(
        "obs",
        BernoulliGamma([p, alpha, beta]),
        obs=jdata,
    )
     
def run_inference(model,rng_key,num_warmup,num_samples,*args,**kwargs):#data,distance_matrix=None):
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
        num_chains=1,
    )
    
    mcmc.run(rng_key, *args,**kwargs)

    # if distance_matrix==None:
    #     mcmc.run(rng_key, data)
    # else:
    #     mcmc.run(rng_key, distance_matrix, data)
    mcmc.print_summary()
    print("Time Taken:", timeit.default_timer() - starttime)
    return mcmc

def run_inference_tinygp(model,rng_key,num_warmup,num_samples,data,X=None):
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
        samples (dictionary): parameter MCMC samples
    """
    starttime = timeit.default_timer()

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
    )

    mcmc.run(rng_key, X, data)
    mcmc.print_summary()
    print("Time Taken:", timeit.default_timer() - starttime)
    return mcmc#.get_samples()

def tinygp_model(x,jdata=None):
    """
   Gaussian Process model code for use with numpyro
    Args:
        distance_matrix_values (jax device array): matrix of distances between sites, shape [#sites,#sites]
        jdata (jax device array): data in shape [#days,#months,#sites]
    """

    alpha = jdata[0]
    p = jdata[1]

    
    alpha_kern_var = numpyro.sample("alpha_kern_var", dist.HalfNormal(1.0))
    alpha_like_var = numpyro.sample("alpha_like_var", dist.HalfNormal(1.0))
    alpha_lengthscale = numpyro.sample("alpha_lengthscale", dist.HalfNormal(1))
    alpha_kernel = alpha_kern_var * kernels.Exp(alpha_lengthscale)
    alpha_mean = numpyro.sample("alpha_mean", dist.Normal(0.0, 2.0))
    alpha_gp = GaussianProcess(alpha_kernel, x, diag=alpha_like_var+1e-5, mean=alpha_mean)
    numpyro.sample("alpha", alpha_gp.numpyro_dist(),obs=alpha)

    p_kern_var = numpyro.sample("p_kern_var", dist.HalfNormal(1.0))
    p_like_var = numpyro.sample("p_like_var", dist.HalfNormal(1.0))
    p_lengthscale = numpyro.sample("p_lengthscale", dist.HalfNormal(1))
    p_kernel = p_kern_var * kernels.Exp(p_lengthscale)
    p_mean = numpyro.sample("p_mean", dist.Normal(0.0, 2.0))
    p_gp = GaussianProcess(p_kernel, x, diag=p_like_var+1e-5, mean=p_mean)
    numpyro.sample("p", p_gp.numpyro_dist(),obs=p)

def tinygp_model_nst(x,jdata=None):
    """
   Gaussian Process model code for use with numpyro
    Args:
        distance_matrix_values (jax device array): matrix of distances between sites, shape [#sites,#sites]
        jdata (jax device array): data in shape [#days,#months,#sites]
    """

    loc = jdata[0]
    scale = jdata[1]

    
    loc_kern_var = numpyro.sample("loc_kern_var", dist.HalfNormal(1.0))
    loc_like_var = numpyro.sample("loc_like_var", dist.HalfNormal(1.0))
    loc_lengthscale = numpyro.sample("loc_lengthscale", dist.HalfNormal(1))
    loc_kernel = loc_kern_var * kernels.Exp(loc_lengthscale)
    loc_mean = numpyro.sample("loc_mean", dist.Normal(0.0, 2.0))
    loc_gp = GaussianProcess(loc_kernel, x, diag=loc_like_var+1e-5, mean=loc_mean)
    numpyro.sample("loc", loc_gp.numpyro_dist(),obs=loc)

    scale_kern_var = numpyro.sample("scale_kern_var", dist.HalfNormal(1.0))
    scale_like_var = numpyro.sample("scale_like_var", dist.HalfNormal(1.0))
    scale_lengthscale = numpyro.sample("scale_lengthscale", dist.HalfNormal(1))
    scale_kernel = scale_kern_var * kernels.Exp(scale_lengthscale)
    scale_mean = numpyro.sample("scale_mean", dist.Normal(0.0, 2.0))
    scale_gp = GaussianProcess(scale_kernel, x, diag=scale_like_var+1e-5, mean=scale_mean)
    numpyro.sample("scale", scale_gp.numpyro_dist(),obs=scale)

        
# def train_gp(m, nits: int, opt: Optimizer, verbose: bool=False):
#     """
#     Helper function for training Gaussian Process in GPFlow using standard minimising algorithm
#     Args:
#         m (gpflow.models class): GPflow class that takes data, coords and kernel 
#         nits (int): Number of optimiser iterations
#         opt (tensorflow optimiser function): Minimising function e.g. adam
#     Returns:
#         m (gpflow.models class): GPFlow class with updated parameter estimates
#         logfs (?): ?Some sort of training loss?
#     """
#     logfs = []
#     for i in range(nits):
#         opt.minimize(m.training_loss, m.trainable_variables)
#         current_loss = -m.training_loss().numpy()
#         logfs.append(current_loss)
#         if verbose and i%50==0:
#             print(f"Step {i}: loss {current_loss}")
#     return m, logfs
    
def bg_tinygp_model(x,jdata=None):
    # Hyper-Params: (These are used to describe the relationship between alpha and the scale parameter from the Bernoulli-Gamma dist.)
    a0 = numpyro.sample("a0", dist.Uniform(-10, 10.0))
    a1 = numpyro.sample("a1", dist.Uniform(-10, 10.0))
    betavar = numpyro.sample("betavar", dist.InverseGamma(0.001, 0.001))
    
    log_alpha_kern_var = numpyro.sample("log_alpha_kern_var", dist.HalfNormal(1.0))
    log_alpha_like_var = numpyro.sample("log_alpha_like_var", dist.HalfNormal(1.0))
    log_alpha_lengthscale = numpyro.sample("log_alpha_lengthscale", dist.HalfNormal(1))
    log_alpha_kernel = log_alpha_kern_var * kernels.Exp(log_alpha_lengthscale)
    log_alpha_mean = numpyro.sample("log_alpha_mean", dist.Normal(0.0, 2.0))
    log_alpha_gp = GaussianProcess(log_alpha_kernel, x, diag=log_alpha_like_var+1e-5, mean=log_alpha_mean)
    # alpha = numpyro.sample("alpha", alpha_gp.numpyro_dist())
    log_alpha = numpyro.sample("log_alpha", log_alpha_gp.numpyro_dist())
    alpha = jnp.exp(log_alpha)

    logit_p_kern_var = numpyro.sample("logit_p_kern_var", dist.HalfNormal(1.0))
    logit_p_like_var = numpyro.sample("logit_p_like_var", dist.HalfNormal(1.0))
    logit_p_lengthscale = numpyro.sample("logit_p_lengthscale", dist.HalfNormal(1))
    logit_p_kernel = logit_p_kern_var * kernels.Exp(logit_p_lengthscale)
    logit_p_mean = numpyro.sample("logit_p_mean", dist.Normal(0.0, 2.0))
    logit_p_gp = GaussianProcess(logit_p_kernel, x, diag=logit_p_like_var+1e-5, mean=logit_p_mean)
    # p = numpyro.sample("p", p_gp.numpyro_dist())
    logit_p = numpyro.sample("logit_p", logit_p_gp.numpyro_dist())
    p = expit(logit_p)
    
    beta = numpyro.sample("beta",dist.LogNormal(a0+a1*alpha,betavar))
        
    numpyro.sample(
        "obs",
        BernoulliGamma([p, alpha, beta]),
        obs=jdata,
    )
    
def tinygp_example_model(x,jdata=None):
    """
   Gaussian Process model code for use with numpyro
    Args:
        distance_matrix_values (jax device array): matrix of distances between sites, shape [#sites,#sites]
        jdata (jax device array): data in shape [#days,#months,#sites]
    """

    kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
    # like_var = numpyro.sample("like_var", dist.HalfNormal(1.0))
    lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
    # gp = GaussianProcess(kernel, x, diag=like_var+1e-5, mean=mean)
    gp = GaussianProcess(kernel, x, mean=mean)

    truth = numpyro.sample("temperature", gp.numpyro_dist(),obs=jdata)
    print(truth)

def tinygp_2process_example_model(cx,ox=None,cdata=None,odata=None):
    """
   Example model where the climate data is generated from 2 GPs, one of which also generates the observations and one of which generates bias in the climate model.
    Args:
        cx (jax device array): array of coordinates for climate model, shape [#gridcells,dimcoords]
        ox (jax device array): array of coordinates for observations, shape [#sites,dimcoords]
        cdata (jax device array): array of data values for climate model, shape [#gridcells,]
        odata (jax device array): array of data values for observations, shape [#sites,]
    """

    #GP that generates the bias
    bkern_var = numpyro.sample("bkern_var", dist.Gamma(3.0,0.5))
    blengthscale = numpyro.sample("blengthscale", dist.Gamma(3.0,0.5))
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", dist.Normal(0.0, 2.0))
    bgp = GaussianProcess(bkernel, cx, mean=bmean)
    bias = numpyro.sample("bias_temperature", bgp.numpyro_dist()) 

    #GP that generates the truth (& so the observations directly)
    kern_var = numpyro.sample("kern_var", dist.Gamma(3.0,0.5))
    lengthscale = numpyro.sample("lengthscale", dist.Gamma(3.0,0.5))
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", dist.Normal(0.0, 2.0))
    gp = GaussianProcess(kernel, ox, mean=mean)
    obs = numpyro.sample("obs_temperature", gp.numpyro_dist(),obs=odata)

    #Conditional GP for evaluating the likelihood of the climate data
    gp_cond = gp.condition(odata, cx).gp
    climatetruth = numpyro.sample("climatetruth_temperature", gp_cond.numpyro_dist(),obs=cdata-bias)

# from numpyro.handlers import mask

# def normal_model(jdata,series_lengths):
#     """
#     Standard Normal model code for use with numpyro
#     Args:
#         jdata (jax device array): data in shape [days,months,sites]
#     """
#     # Number of Months and Sites
#     sites = jdata.shape[1]
    
#     max_length = jdata.shape[0]#series_lengths.max()
#     series = jnp.arange(max_length).broadcast([sites,]).T
    
#     with numpyro.plate("Sites", sites, dim=-1) as j:        
#         loc = numpyro.sample("loc", dist.Uniform(-40, 0))
#         scale = numpyro.sample("scale", dist.Gamma(0.001, 0.001)) 

#         with mask(mask=series < series_lengths):
#             numpyro.sample(
#                 "obs",
#                 dist.Normal(loc, scale),
#                 obs=jdata,
#             )

# ds = xr.open_dataset('/data/notebooks/jupyterlab-biascorrlab/data/ProcessedData/NST_Observations.nc')
# ds = ds.transpose("Day", "Month", "Station_Lower")

# da_temp = ds['Temperature()']
# dat_temp_nans_sorted = np.sort(ds['Temperature()'])
# j_data = jnp.array(dat_temp_nans_sorted.data)  # JAX array is needed for Numpyro
# Y = j_data[:,0,:]
# Y = np.expand_dims(Y, axis=1)
# Y_test = Y[:,0,:10]

# series_lengths = jnp.isnan(Y_test).sum(axis=0)

# numpyro.enable_x64()

# starttime = timeit.default_timer()

# kernel = NUTS(normal_model)
# mcmc = MCMC(
#     kernel,
#     num_warmup=1000,
#     num_samples=2000,
#     num_chains=1,
# )

# mcmc.run(rng_key, Y_test[:,:3], series_lengths)
# mcmc.print_summary()
# print("Time Taken:", timeit.default_timer() - starttime)