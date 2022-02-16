import numpyro
import jax.numpy as jnp
import jax.scipy.stats.gamma as jgamma
from numpyro.distributions import constraints

class BernoulliGamma(numpyro.distributions.Distribution):
    support = constraints.positive

    def __init__(self, params):
        self.p = params[0]
        self.alpha = params[1]
        self.scale = params[2]
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