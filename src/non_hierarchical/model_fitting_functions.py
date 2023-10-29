import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
import jax.numpy as jnp
import arviz as az

from src.helper_functions import diagonal_noise
from src.helper_functions import run_inference


def generate_obs_conditional_climate_dist(
    scenario, ckernel, cmean, cnoise_var, okernel, omean, onoise_var
):
    ox = scenario["ox"]
    cx = scenario["cx"]
    cdata = scenario["cdata"]
    y2 = cdata
    u1 = jnp.full(ox.shape[0], omean)
    u2 = jnp.full(cx.shape[0], cmean)
    k11 = okernel(ox, ox) + diagonal_noise(ox, onoise_var)
    k12 = okernel(ox, cx)
    k21 = okernel(cx, ox)
    k22 = ckernel(cx, cx) + diagonal_noise(cx, cnoise_var)
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn_dist = dist.MultivariateNormal(u1g2, k1g2)
    return mvn_dist


def tinygp_2process_model(scenario):
    """
    Example model where the climate data is generated from 2 GPs,
    one of which also generates the observations and one of
    which generates bias in the climate model.
    """
    kern_var = numpyro.sample("kern_var", scenario["t_variance_prior"])
    lengthscale = numpyro.sample("lengthscale", scenario["t_lengthscale_prior"])
    kernel = kern_var * kernels.ExpSquared(lengthscale)
    mean = numpyro.sample("mean", scenario["t_mean_prior"])

    bkern_var = numpyro.sample("bkern_var", scenario["b_variance_prior"])
    blengthscale = numpyro.sample("blengthscale", scenario["b_lengthscale_prior"])
    bkernel = bkern_var * kernels.ExpSquared(blengthscale)
    bmean = numpyro.sample("bmean", scenario["b_mean_prior"])

    ckernel = kernel + bkernel
    cmean = mean + bmean
    cnoise_var = scenario["cnoise"] ** 2
    cgp = GaussianProcess(ckernel, scenario["cx"], diag=cnoise_var, mean=cmean)
    numpyro.sample("climate_temperature", cgp.numpyro_dist(), obs=scenario["cdata"])

    onoise_var = numpyro.sample("onoise", scenario["onoise_prior"]) ** 2
    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(
        scenario, ckernel, cmean, cnoise_var, kernel, mean, onoise_var
    )
    numpyro.sample(
        "obs_temperature", obs_conditional_climate_dist, obs=scenario["odata"]
    )


def generate_posterior(scenario, rng_key, num_warmup, num_samples, num_chains):
    mcmc_2process = run_inference(
        tinygp_2process_model, rng_key, num_warmup, num_samples, num_chains, scenario
    )
    idata_2process = az.from_numpyro(mcmc_2process)
    scenario["mcmc"] = idata_2process
    scenario["mcmc_samples"] = mcmc_2process.get_samples()
