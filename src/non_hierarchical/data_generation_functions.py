import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
import jax
from jax import random

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

