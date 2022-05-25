import timeit
import numpyro
import jax.numpy as jnp
import jax.scipy.stats.gamma as jgamma
from jax import random, vmap, jit
from numpyro.distributions import constraints
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, BarkerMH
from numpy import ndarray
import numpy as np
from numpy import ndarray
from tensorflow.keras.optimizers import Optimizer
from gpflow.models import SGPR
from scipy.cluster.vq import kmeans2
    
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
    
    #Hyper Params GP:
    var = numpyro.sample("kernel_var", dist.LogNormal(0.1, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.1, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.1, 10.0))
    
    kern = gpkernel(distance_matrix_values, var, length, noise)

    # Number of Months and Sites
    months = jdata.shape[1]
    sites = jdata.shape[2]
    
    with numpyro.plate("Months", months, dim=-2) as k:
        log_alpha = numpyro.sample("log_alpha", dist.MultivariateNormal(loc=jnp.zeros(distance_matrix_values.shape[0]), covariance_matrix=kern))
        alpha = jnp.exp(log_alpha)
        alpha = alpha.reshape(months,sites)
        with numpyro.plate("Sites", sites, dim=-1) as j:
            p = numpyro.sample("p", dist.Uniform(0.01, 0.99))
            
    beta = numpyro.sample("beta",dist.LogNormal(a0+a1*alpha,betavar))

    numpyro.sample(
        "obs",
        BernoulliGamma([p, alpha, beta]),
        obs=jdata,
    )
     
def run_inference(model,rng_key,num_warmup,num_samples,data,distance_matrix=None):
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
    if distance_matrix==None:
        mcmc.run(rng_key, data)
    else:
        mcmc.run(rng_key, distance_matrix, data)
    mcmc.print_summary()
    print("Time Taken:", timeit.default_timer() - starttime)
    return mcmc.get_samples()
        
def train_gp(m, nits: int, opt: Optimizer, verbose: bool=False):
    """
    Helper function for training Gaussian Process in GPFlow using standard minimising algorithm
    Args:
        m (gpflow.models class): GPflow class that takes data, coords and kernel 
        nits (int): Number of optimiser iterations
        opt (tensorflow optimiser function): Minimising function e.g. adam
    Returns:
        m (gpflow.models class): GPFlow class with updated parameter estimates
        logfs (?): ?Some sort of training loss?
    """
    logfs = []
    for i in range(nits):
        opt.minimize(m.training_loss, m.trainable_variables)
        current_loss = -m.training_loss().numpy()
        logfs.append(current_loss)
        if verbose and i%50==0:
            print(f"Step {i}: loss {current_loss}")
    return m, logfs
    

