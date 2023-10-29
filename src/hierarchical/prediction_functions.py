import numpy as np
import numpyro.distributions as dist
from tinygp import kernels
import jax
from jax import random
import jax.numpy as jnp

from src.helper_functions import diagonal_noise

jax.config.update("jax_enable_x64", True)


def generate_mean_truth_predictive_dist(nx, scenario, posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation["mt_variance_realisation"]
    mt_lengthscale_realisation = posterior_param_realisation[
        "mt_lengthscale_realisation"
    ]
    mt_mean_realisation = posterior_param_realisation["mt_mean_realisation"]
    mb_variance_realisation = posterior_param_realisation["mb_variance_realisation"]
    mb_lengthscale_realisation = posterior_param_realisation[
        "mb_lengthscale_realisation"
    ]
    mb_mean_realisation = posterior_param_realisation["mb_mean_realisation"]

    mt_realisation = posterior_param_realisation["mt_realisation"]
    mc_realisation = posterior_param_realisation["mc_realisation"]

    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = mt_realisation
    cdata = mc_realisation
    omean = mt_mean_realisation
    bmean = mb_mean_realisation
    kernelo = mt_variance_realisation * kernels.ExpSquared(mt_lengthscale_realisation)
    kernelb = mb_variance_realisation * kernels.ExpSquared(mb_lengthscale_realisation)

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], omean)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], omean), jnp.full(cx.shape[0], omean + bmean)]
    )
    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
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
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_logvar_truth_predictive_dist(nx, scenario, posterior_param_realisation):
    lvt_variance_realisation = posterior_param_realisation["lvt_variance_realisation"]
    lvt_lengthscale_realisation = posterior_param_realisation[
        "lvt_lengthscale_realisation"
    ]
    lvt_mean_realisation = posterior_param_realisation["lvt_mean_realisation"]
    lvb_variance_realisation = posterior_param_realisation["lvb_variance_realisation"]
    lvb_lengthscale_realisation = posterior_param_realisation[
        "lvb_lengthscale_realisation"
    ]
    lvb_mean_realisation = posterior_param_realisation["lvb_mean_realisation"]

    lvt_realisation = posterior_param_realisation["lvt_realisation"]
    lvc_realisation = posterior_param_realisation["lvc_realisation"]

    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = lvt_realisation
    cdata = lvc_realisation
    omean = lvt_mean_realisation
    bmean = lvb_mean_realisation
    kernelo = lvt_variance_realisation * kernels.ExpSquared(lvt_lengthscale_realisation)
    kernelb = lvb_variance_realisation * kernels.ExpSquared(lvb_lengthscale_realisation)

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], omean)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], omean), jnp.full(cx.shape[0], omean + bmean)]
    )
    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
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
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_mean_bias_predictive_dist(nx, scenario, posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation["mt_variance_realisation"]
    mt_lengthscale_realisation = posterior_param_realisation[
        "mt_lengthscale_realisation"
    ]
    mt_mean_realisation = posterior_param_realisation["mt_mean_realisation"]
    mb_variance_realisation = posterior_param_realisation["mb_variance_realisation"]
    mb_lengthscale_realisation = posterior_param_realisation[
        "mb_lengthscale_realisation"
    ]
    mb_mean_realisation = posterior_param_realisation["mb_mean_realisation"]

    mt_realisation = posterior_param_realisation["mt_realisation"]
    mc_realisation = posterior_param_realisation["mc_realisation"]

    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = mt_realisation
    cdata = mc_realisation
    omean = mt_mean_realisation
    bmean = mb_mean_realisation
    kernelo = mt_variance_realisation * kernels.ExpSquared(mt_lengthscale_realisation)
    kernelb = mb_variance_realisation * kernels.ExpSquared(mb_lengthscale_realisation)

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], bmean)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], omean), jnp.full(cx.shape[0], omean + bmean)]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
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
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_logvar_bias_predictive_dist(nx, scenario, posterior_param_realisation):
    lvt_variance_realisation = posterior_param_realisation["lvt_variance_realisation"]
    lvt_lengthscale_realisation = posterior_param_realisation[
        "lvt_lengthscale_realisation"
    ]
    lvt_mean_realisation = posterior_param_realisation["lvt_mean_realisation"]
    lvb_variance_realisation = posterior_param_realisation["lvb_variance_realisation"]
    lvb_lengthscale_realisation = posterior_param_realisation[
        "lvb_lengthscale_realisation"
    ]
    lvb_mean_realisation = posterior_param_realisation["lvb_mean_realisation"]

    lvt_realisation = posterior_param_realisation["lvt_realisation"]
    lvc_realisation = posterior_param_realisation["lvc_realisation"]

    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = lvt_realisation
    cdata = lvc_realisation
    omean = lvt_mean_realisation
    bmean = lvb_mean_realisation
    kernelo = lvt_variance_realisation * kernels.ExpSquared(lvt_lengthscale_realisation)
    kernelb = lvb_variance_realisation * kernels.ExpSquared(lvb_lengthscale_realisation)

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], bmean)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], omean), jnp.full(cx.shape[0], omean + bmean)]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
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
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_posterior_predictive_realisations_hierarchical_mean(
    nx, scenario, num_parameter_realisations, num_posterior_pred_realisations
):
    posterior = scenario["mcmc"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            "mt_variance_realisation": posterior["mt_kern_var"].data[0, :][i],
            "mt_lengthscale_realisation": posterior["mt_lengthscale"].data[0, :][i],
            "mt_mean_realisation": posterior["mt_mean"].data[0, :][i],
            "mb_variance_realisation": posterior["mb_kern_var"].data[0, :][i],
            "mb_lengthscale_realisation": posterior["mb_lengthscale"].data[0, :][i],
            "mb_mean_realisation": posterior["mb_mean"].data[0, :][i],
            "mt_realisation": posterior["mt"].data[0, :][i],
            "mc_realisation": posterior["mc"].data[0, :][i],
        }

        truth_predictive_dist = generate_mean_truth_predictive_dist(
            nx, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_mean_bias_predictive_dist(
            nx, scenario, posterior_param_realisation
        )

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
    climate_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations + bias_posterior_predictive_realisations
    )
    scenario[
        "mean_truth_posterior_predictive_realisations"
    ] = truth_posterior_predictive_realisations
    scenario[
        "mean_bias_posterior_predictive_realisations"
    ] = bias_posterior_predictive_realisations
    scenario[
        "mean_climate_posterior_predictive_realisations"
    ] = climate_posterior_predictive_realisations


def generate_posterior_predictive_realisations_hierarchical_std(
    nx, scenario, num_parameter_realisations, num_posterior_pred_realisations
):
    posterior = scenario["mcmc"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            "lvt_variance_realisation": posterior["lvt_kern_var"].data[0, :][i],
            "lvt_lengthscale_realisation": posterior["lvt_lengthscale"].data[0, :][i],
            "lvt_mean_realisation": posterior["lvt_mean"].data[0, :][i],
            "lvb_variance_realisation": posterior["lvb_kern_var"].data[0, :][i],
            "lvb_lengthscale_realisation": posterior["lvb_lengthscale"].data[0, :][i],
            "lvb_mean_realisation": posterior["lvb_mean"].data[0, :][i],
            "lvt_realisation": posterior["lvt"].data[0, :][i],
            "lvc_realisation": posterior["lvc"].data[0, :][i],
        }

        truth_predictive_dist = generate_logvar_truth_predictive_dist(
            nx, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_logvar_bias_predictive_dist(
            nx, scenario, posterior_param_realisation
        )

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
    climate_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations + bias_posterior_predictive_realisations
    )

    scenario["std_truth_posterior_predictive_realisations"] = jnp.sqrt(
        jnp.exp(truth_posterior_predictive_realisations)
    )
    scenario["std_bias_posterior_predictive_realisations"] = jnp.sqrt(
        jnp.exp(bias_posterior_predictive_realisations)
    )
    scenario["std_climate_posterior_predictive_realisations"] = jnp.sqrt(
        jnp.exp(climate_posterior_predictive_realisations)
    )
