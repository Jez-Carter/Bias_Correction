import numpyro
import jax.numpy as jnp
import jax.scipy.stats.gamma as jgamma
from numpyro.distributions import constraints
import numpyro.distributions as dist
from numpy import ndarray
import numpy as np
from numpy import ndarray

from tensorflow.keras.optimizers import Optimizer
from gpflow.models import SGPR
from scipy.cluster.vq import kmeans2

class BernoulliGamma(numpyro.distributions.Distribution):
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
    
def lima_model(jdata):

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
    
def train_gp(m, nits: int, opt: Optimizer, verbose: bool=False):
    logfs = []
    for i in range(nits):
        opt.minimize(m.training_loss, m.trainable_variables)
        current_loss = -m.training_loss().numpy()
        logfs.append(current_loss)
        if verbose and i%50==0:
            print(f"Step {i}: loss {current_loss}")
    return m, logfs
    
# def train_sgpr(m: SGPR, nits: int, opt: Optimizer, verbose: bool=False):
#     logfs = []
#     for i in range(nits):
#         opt.minimize(m.training_loss, m.trainable_variables)
#         current_loss = -m.training_loss().numpy()
#         logfs.append(current_loss)
#         if verbose and i%50==0:
#             print(f"Step {i}: loss {current_loss}")
#     return m, logfs

# def standardise(X):
#     """
#     Standardise a dataset column-wise to unary Gaussian.

#     Example
#     -------
#     >>> X = np.random.randn(10, 2)
#     >>> Xtransform, Xmean, Xstd = standardise(X)

#     """

#     mu = np.mean(X, axis=0)
#     sigma = np.std(X, axis=0)
#     return (X-mu)/sigma, mu, sigma

# def unstandardise(X, mu, sigma):
#     """
#     Reproject data back onto its original scale.

#     Example
#     -------
#     X = unstandardise(Xtransform, Xmean, Xstd)
#     """
#     return (X*sigma)+mu

# def transform_y(Y):
#     Ylog = np.log(Y)
#     Ymean = np.mean(Ylog)
#     centred = Ylog-Ymean
#     return (centred, Ymean)
  
# def untransform_y(y, ymean):
#     return np.exp(y + ymean)

# def get_inducing(X: ndarray, n_inducing: int):
#     Z = kmeans2(X, k=n_inducing, minit='points')[0]
#     print(f"{Z.shape[0]} inducing points initialised")
#     return Z