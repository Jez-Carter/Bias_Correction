# %% Importing packages
import numpy as np
from scipy.stats import norm
import jax
from jax import random
from tinygp import kernels, GaussianProcess
from sklearn import preprocessing

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)


def quantile_mapping(index, scenario):
    index_location = scenario["cx"][index]
    difference_array = np.absolute(scenario["ox"] - index_location)
    nearest_index = difference_array.argmin()  # nearest in-situ observation site index

    # means and stdevs of climate model output and nearest in-situ observation site
    cmean = scenario["cdata"].mean(axis=0)[index]
    cstd = scenario["cdata"].std(axis=0)[index]
    omean = scenario["odata"].mean(axis=0)[nearest_index]
    ostd = scenario["odata"].std(axis=0)[nearest_index]

    # creating time coordinates to generate some example timeseries data
    t = np.arange(0, 100, 0.1)
    ct = np.linspace(0, 100, scenario["cdata"].shape[0])
    ot = np.random.choice(ct, scenario["odata"].shape[0], replace=False)
    ot.sort()

    # defining a GP (with mean=0 and std=1) and sampling at specific time coordinates
    kernel = 1 * kernels.Cosine(scale=20) * kernels.ExpSquared(scale=20)
    # tinygp.kernels.Product(kernel1: Kernel, kernel2: Kernel)
    # GP = GaussianProcess(1 * kernels.ExpSineSquared(scale=20,gamma=2),t)
    # GP = GaussianProcess(1 * kernels.Cosine(scale=20),t)
    GP = GaussianProcess(kernel, t)
    complete_realisation = GP.sample(rng_key)
    complete_realisation = preprocessing.scale(complete_realisation)

    c_realisation = GP.condition(complete_realisation, ct).gp.sample(rng_key)
    o_realisation = GP.condition(complete_realisation, ot).gp.sample(rng_key)

    # applying scaling for climate model output and in-situ observations
    c_complete_realisation = complete_realisation * cstd + cmean
    o_complete_realisation = complete_realisation * ostd + omean
    c_realisation = c_realisation * cstd + cmean
    o_realisation = o_realisation * ostd + omean

    # loading postpred and sampling at index
    mean_truth_postpred = scenario["mean_truth_posterior_predictive_realisations"][
        :, index
    ]
    std_truth_postpred = scenario["std_truth_posterior_predictive_realisations"][
        :, index
    ]
    mean_climate_postpred = scenario["mean_climate_posterior_predictive_realisations"][
        :, index
    ]
    std_climate_postpred = scenario["std_climate_posterior_predictive_realisations"][
        :, index
    ]
    # applying quantile mapping
    cdf_c = norm.cdf(
        c_realisation,
        mean_climate_postpred.reshape(-1, 1),
        std_climate_postpred.reshape(-1, 1),
    )

    c_corrected = norm.ppf(
        cdf_c, mean_truth_postpred.reshape(-1, 1), std_truth_postpred.reshape(-1, 1)
    )

    # returning output
    quantile_mapping_output = {
        "t": t,
        "complete_realisation": complete_realisation,
        "o_complete_realisation": o_complete_realisation,
        "c_complete_realisation": c_complete_realisation,
        "ot": ot,
        "ct": ct,
        "o_realisation": o_realisation,
        "c_realisation": c_realisation,
        "c_corrected": c_corrected,
        "cmean": cmean,
        "cstd": cstd,
        "omean": omean,
        "ostd": ostd,
    }
    return quantile_mapping_output


def plot_qm_output(qm_output, ax):
    ax.scatter(
        qm_output["ot"],
        qm_output["o_realisation"],
        label="In-Situ Observations",
        alpha=1.0,
        s=15,
        marker="X",
        edgecolors="w",
        linewidths=0.2,
        zorder=2,
    )
    ax.scatter(
        qm_output["ct"],
        qm_output["c_realisation"],
        label="Climate Model Output",
        alpha=1.0,
        s=15,
        marker="P",
        edgecolors="w",
        linewidths=0.2,
        zorder=2,
        color="tab:green",
    )

    ax.scatter(
        qm_output["ct"],
        qm_output["c_corrected"].mean(axis=0),
        label="Bias Corrected Output Expectation",
        alpha=0.9,
        s=5,
        marker=".",
        color="k",
    )
    ax.plot(
        qm_output["ct"],
        qm_output["c_corrected"].mean(axis=0),
        color="k",
        alpha=1.0,
        linestyle="-",
        linewidth=0.8,
        zorder=1,
    )

    ax.fill_between(
        qm_output["ct"],
        qm_output["c_corrected"].mean(axis=0)
        + 3 * qm_output["c_corrected"].std(axis=0),
        qm_output["c_corrected"].mean(axis=0)
        - 3 * qm_output["c_corrected"].std(axis=0),
        interpolate=True,
        color="k",
        alpha=0.5,
        label="Bias Corrected Output Uncertainty 3$\sigma$",
        linewidth=0.5,
        facecolor="none",
        edgecolor="k",
        linestyle=(0, (5, 2)),
    )
    for corrected_timeseries in qm_output["c_corrected"][::10]:
        ax.plot(
            qm_output["ct"],
            corrected_timeseries,
            color="k",
            alpha=0.2,
            linestyle="-",
            linewidth=0.2,
            zorder=1,
        )
