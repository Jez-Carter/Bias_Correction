import timeit
import numpy as np
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def run_inference(
    model, rng_key, num_warmup, num_samples, num_chains, *args, **kwargs
):  # data,distance_matrix=None):
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
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )

    mcmc.run(rng_key, *args, **kwargs)

    mcmc.print_summary()
    print("Time Taken:", timeit.default_timer() - starttime)
    return mcmc


def diagonal_noise(coord, noise):
    return jnp.diag(jnp.full(coord.shape[0], noise))


def create_levels(scenario, sep, rounding, center=None):
    data = np.array([scenario["T"], scenario["B"], scenario["C"]])
    vmin = data.min()
    vmax = data.max()
    abs_max_rounded = max(np.abs(vmin), vmax).round(rounding)
    if center != None:
        levels = np.arange(-abs_max_rounded, abs_max_rounded + sep, sep)
    else:
        levels = np.arange(vmin.round(rounding), vmax.round(rounding) + sep, sep)
    return levels


def remove_outliers(array, perc=[0.001, 0.99]):
    lower_threshold = np.quantile(array, perc[0])
    upper_threshold = np.quantile(array, perc[1])
    outlier_condition = (array > upper_threshold) | (array < lower_threshold)
    return array[outlier_condition == False]
