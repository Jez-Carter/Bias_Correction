import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
import jax
from jax import random
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


############# DATA GENERATION #############
def generate_underlying_data_hierarchical(scenario, rng_key):
    rng_key, rng_key_ = random.split(rng_key)

    GP_MEAN_T = GaussianProcess(
        scenario["MEAN_T_variance"]
        * kernels.ExpSquared(scenario["MEAN_T_lengthscale"]),
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["MEAN_T_mean"],
    )
    GP_LOGVAR_T = GaussianProcess(
        scenario["LOGVAR_T_variance"]
        * kernels.ExpSquared(scenario["LOGVAR_T_lengthscale"]),
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["LOGVAR_T_mean"],
    )
    GP_MEAN_B = GaussianProcess(
        scenario["MEAN_B_variance"]
        * kernels.ExpSquared(scenario["MEAN_B_lengthscale"]),
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["MEAN_B_mean"],
    )
    GP_LOGVAR_B = GaussianProcess(
        scenario["LOGVAR_B_variance"]
        * kernels.ExpSquared(scenario["LOGVAR_B_lengthscale"]),
        scenario["X"],
        diag=scenario["jitter"],
        mean=scenario["LOGVAR_B_mean"],
    )

    scenario["MEAN_T"] = GP_MEAN_T.sample(rng_key)
    scenario["LOGVAR_T"] = GP_LOGVAR_T.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario["MEAN_B"] = GP_MEAN_B.sample(rng_key)
    scenario["LOGVAR_B"] = GP_LOGVAR_B.sample(rng_key_)
    scenario["MEAN_C"] = scenario["MEAN_T"] + scenario["MEAN_B"]
    scenario["LOGVAR_C"] = scenario["LOGVAR_T"] + scenario["LOGVAR_B"]

    rng_key, rng_key_ = random.split(rng_key)
    scenario["MEAN_T_obs"] = GP_MEAN_T.condition(
        scenario["MEAN_T"], scenario["ox"]
    ).gp.sample(rng_key)
    scenario["LOGVAR_T_obs"] = GP_MEAN_T.condition(
        scenario["LOGVAR_T"], scenario["ox"]
    ).gp.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    N_T_obs = dist.Normal(
        scenario["MEAN_T_obs"], jnp.sqrt(jnp.exp(scenario["LOGVAR_T_obs"]))
    )
    scenario["odata"] = N_T_obs.sample(rng_key, (scenario["osamples"],))

    rng_key, rng_key_ = random.split(rng_key)
    scenario["MEAN_T_climate"] = GP_MEAN_T.condition(
        scenario["MEAN_T"], scenario["cx"]
    ).gp.sample(rng_key)
    scenario["LOGVAR_T_climate"] = GP_LOGVAR_T.condition(
        scenario["LOGVAR_T"], scenario["cx"]
    ).gp.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario["MEAN_B_climate"] = GP_MEAN_B.condition(
        scenario["MEAN_B"], scenario["cx"]
    ).gp.sample(rng_key)
    scenario["LOGVAR_B_climate"] = GP_LOGVAR_B.condition(
        scenario["LOGVAR_B"], scenario["cx"]
    ).gp.sample(rng_key_)
    scenario["MEAN_C_climate"] = scenario["MEAN_T_climate"] + scenario["MEAN_B_climate"]
    scenario["LOGVAR_C_climate"] = (
        scenario["LOGVAR_T_climate"] + scenario["LOGVAR_B_climate"]
    )

    rng_key, rng_key_ = random.split(rng_key)
    N_C_climate = dist.Normal(
        scenario["MEAN_C_climate"], jnp.sqrt(jnp.exp(scenario["LOGVAR_C_climate"]))
    )
    scenario["cdata"] = N_C_climate.sample(rng_key, (scenario["csamples"],))
