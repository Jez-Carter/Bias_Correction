import numpy as np
import numpyro.distributions as dist
from jax import random
import jax.numpy as jnp
from tinygp import kernels
from scipy.stats import multivariate_normal

from src.helper_functions import diagonal_noise


def generate_truth_predictive_dist(nx, scenario, posterior_param_realisation):
    t_variance_realisation = posterior_param_realisation["t_variance_realisation"]
    t_lengthscale_realisation = posterior_param_realisation["t_lengthscale_realisation"]
    t_mean_realisation = posterior_param_realisation["t_mean_realisation"]
    b_variance_realisation = posterior_param_realisation["b_variance_realisation"]
    b_lengthscale_realisation = posterior_param_realisation["b_lengthscale_realisation"]
    b_mean_realisation = posterior_param_realisation["b_mean_realisation"]
    onoise_realisation = posterior_param_realisation["onoise_realisation"]
    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    cnoise_var = scenario["cnoise"] ** 2
    odata = scenario["odata"]
    cdata = scenario["cdata"]
    omean = t_mean_realisation
    bmean = b_mean_realisation
    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)
    onoise_var = onoise_realisation**2

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], omean)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], omean), jnp.full(cx.shape[0], omean + bmean)]
    )
    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, onoise_var), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise_var),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    # k1g2 = k11 - jnp.matmul(p21.T,p21)
    k1g2 = k11 - jnp.matmul(jnp.matmul(k12, k22i), k21)
    k1g2 = k1g2
    # mvn = multivariate_normal(u1g2,k1g2)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_bias_predictive_dist(nx, scenario, posterior_param_realisation):
    t_variance_realisation = posterior_param_realisation["t_variance_realisation"]
    t_lengthscale_realisation = posterior_param_realisation["t_lengthscale_realisation"]
    t_mean_realisation = posterior_param_realisation["t_mean_realisation"]
    b_variance_realisation = posterior_param_realisation["b_variance_realisation"]
    b_lengthscale_realisation = posterior_param_realisation["b_lengthscale_realisation"]
    b_mean_realisation = posterior_param_realisation["b_mean_realisation"]
    onoise_realisation = posterior_param_realisation["onoise_realisation"]
    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    cnoise_var = scenario["cnoise"] ** 2
    odata = scenario["odata"]
    cdata = scenario["cdata"]
    omean = t_mean_realisation
    bmean = b_mean_realisation
    kernelo = t_variance_realisation * kernels.ExpSquared(t_lengthscale_realisation)
    kernelb = b_variance_realisation * kernels.ExpSquared(b_lengthscale_realisation)
    onoise_var = onoise_realisation**2

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], bmean)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], omean), jnp.full(cx.shape[0], omean + bmean)]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, onoise_var), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise_var),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    # mvn = multivariate_normal(u1g2,k1g2)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_posterior_predictive_realisations(
    nx, scenario, num_parameter_realisations, num_posterior_pred_realisations
):
    posterior = scenario["mcmc"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration = 0
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            "iteration": iteration,
            "t_variance_realisation": posterior["kern_var"].data[0, :][i],
            "t_lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "t_mean_realisation": posterior["mean"].data[0, :][i],
            "b_variance_realisation": posterior["bkern_var"].data[0, :][i],
            "b_lengthscale_realisation": posterior["blengthscale"].data[0, :][i],
            "b_mean_realisation": posterior["bmean"].data[0, :][i],
            "onoise_realisation": posterior["onoise"].data[0, :][i],
        }

        truth_predictive_dist = generate_truth_predictive_dist(
            nx, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_bias_predictive_dist(
            nx, scenario, posterior_param_realisation
        )
        iteration += 1

        # truth_predictive_realisations = truth_predictive_dist.rvs(num_posterior_pred_realisations)
        # bias_predictive_realisations = bias_predictive_dist.rvs(num_posterior_pred_realisations)
        rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )
    truth_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations.reshape(
            -1, truth_posterior_predictive_realisations.shape[-1]
        )
    )
    bias_posterior_predictive_realisations = (
        bias_posterior_predictive_realisations.reshape(
            -1, bias_posterior_predictive_realisations.shape[-1]
        )
    )
    scenario[
        "truth_posterior_predictive_realisations"
    ] = truth_posterior_predictive_realisations
    scenario[
        "bias_posterior_predictive_realisations"
    ] = bias_posterior_predictive_realisations
