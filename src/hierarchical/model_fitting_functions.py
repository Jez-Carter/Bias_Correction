import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
import jax
import jax.numpy as jnp
import arviz as az

from src.helper_functions import diagonal_noise
from src.helper_functions import run_inference


jax.config.update("jax_enable_x64", True)


def generate_mt_conditional_mc_dist(
    scenario, mc_kernel, mc_mean, mt_kernel, mt_mean, mc
):
    ox = scenario["ox"]
    cx = scenario["cx"]
    y2 = mc
    u1 = jnp.full(ox.shape[0], mt_mean)
    u2 = jnp.full(cx.shape[0], mc_mean)
    k11 = mt_kernel(ox, ox) + diagonal_noise(ox, scenario["jitter"])
    k12 = mt_kernel(ox, cx)
    k21 = mt_kernel(cx, ox)
    k22 = mc_kernel(cx, cx) + diagonal_noise(cx, scenario["jitter"])
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn_dist = dist.MultivariateNormal(u1g2, k1g2)
    return mvn_dist


def generate_lvt_conditional_lvc_dist(
    scenario, lvc_kernel, lvc_mean, lvt_kernel, lvt_mean, lvc
):
    ox = scenario["ox"]
    cx = scenario["cx"]
    y2 = lvc
    u1 = jnp.full(ox.shape[0], lvt_mean)
    u2 = jnp.full(cx.shape[0], lvc_mean)
    k11 = lvt_kernel(ox, ox) + diagonal_noise(ox, scenario["jitter"])
    k12 = lvt_kernel(ox, cx)
    k21 = lvt_kernel(cx, ox)
    k22 = lvc_kernel(cx, cx) + diagonal_noise(cx, scenario["jitter"])
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn_dist = dist.MultivariateNormal(u1g2, k1g2)
    return mvn_dist


def hierarchical_model(scenario):
    mt_kern_var = numpyro.sample("mt_kern_var", scenario["MEAN_T_variance_prior"])
    mt_lengthscale = numpyro.sample(
        "mt_lengthscale", scenario["MEAN_T_lengthscale_prior"]
    )
    mt_kernel = mt_kern_var * kernels.ExpSquared(mt_lengthscale)
    mt_mean = numpyro.sample("mt_mean", scenario["MEAN_T_mean_prior"])
    mb_kern_var = numpyro.sample("mb_kern_var", scenario["MEAN_B_variance_prior"])
    mb_lengthscale = numpyro.sample(
        "mb_lengthscale", scenario["MEAN_B_lengthscale_prior"]
    )
    mb_kernel = mb_kern_var * kernels.ExpSquared(mb_lengthscale)
    mb_mean = numpyro.sample("mb_mean", scenario["MEAN_B_mean_prior"])

    mc_kernel = mt_kernel + mb_kernel
    mc_mean = mt_mean + mb_mean
    mc_gp = GaussianProcess(
        mc_kernel, scenario["cx"], diag=scenario["jitter"], mean=mc_mean
    )
    mc = numpyro.sample("mc", mc_gp.numpyro_dist())
    mt_conditional_mc_dist = generate_mt_conditional_mc_dist(
        scenario, mc_kernel, mc_mean, mt_kernel, mt_mean, mc
    )
    mt = numpyro.sample("mt", mt_conditional_mc_dist)

    lvt_kern_var = numpyro.sample("lvt_kern_var", scenario["LOGVAR_T_variance_prior"])
    lvt_lengthscale = numpyro.sample(
        "lvt_lengthscale", scenario["LOGVAR_T_lengthscale_prior"]
    )
    lvt_kernel = lvt_kern_var * kernels.ExpSquared(lvt_lengthscale)
    lvt_mean = numpyro.sample("lvt_mean", scenario["LOGVAR_T_mean_prior"])
    lvb_kern_var = numpyro.sample("lvb_kern_var", scenario["LOGVAR_B_variance_prior"])
    lvb_lengthscale = numpyro.sample(
        "lvb_lengthscale", scenario["LOGVAR_B_lengthscale_prior"]
    )
    lvb_kernel = lvb_kern_var * kernels.ExpSquared(lvb_lengthscale)
    lvb_mean = numpyro.sample("lvb_mean", scenario["LOGVAR_B_mean_prior"])

    lvc_kernel = lvt_kernel + lvb_kernel
    lvc_mean = lvt_mean + lvb_mean
    lvc_gp = GaussianProcess(
        lvc_kernel, scenario["cx"], diag=scenario["jitter"], mean=lvc_mean
    )
    lvc = numpyro.sample("lvc", lvc_gp.numpyro_dist())
    lvt_conditional_lvc_dist = generate_lvt_conditional_lvc_dist(
        scenario, lvc_kernel, lvc_mean, lvt_kernel, lvt_mean, lvc
    )
    lvt = numpyro.sample("lvt", lvt_conditional_lvc_dist)

    vt = jnp.exp(lvt)
    vc = jnp.exp(lvc)
    numpyro.sample("t", dist.Normal(mt, jnp.sqrt(vt)), obs=scenario["odata"])
    numpyro.sample("c", dist.Normal(mc, jnp.sqrt(vc)), obs=scenario["cdata"])


def generate_posterior_hierarchical(
    scenario, rng_key, num_warmup, num_samples, num_chains
):
    mcmc = run_inference(
        hierarchical_model, rng_key, num_warmup, num_samples, num_chains, scenario
    )
    idata = az.from_numpyro(mcmc)
    scenario["mcmc"] = idata
    scenario["mcmc_samples"] = mcmc.get_samples()
