import numpy as np
from scipy.stats import norm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def create_levels(scenario, vars, sep, rounding, center=None):
    data_list = []
    for var in vars:
        data_list.append(scenario[var])
    data = np.array(data_list)
    vmin = data.min()
    vmax = data.max()
    abs_max_rounded = max(np.abs(vmin), vmax).round(rounding)
    if center != None:
        levels = np.arange(-abs_max_rounded, abs_max_rounded + sep, sep)
    else:
        levels = np.arange(vmin.round(rounding), vmax.round(rounding) + sep, sep)
    return levels


############# 1 Dimensional Data Figures #############


def plot_underlying_data_mean_1d(scenario, ax, ms):
    ax.plot(scenario["X"], scenario["MEAN_T"], label="Truth Mean", alpha=0.6)
    ax.plot(scenario["X"], scenario["MEAN_B"], label="Bias Mean", alpha=0.6)
    ax.plot(scenario["X"], scenario["MEAN_C"], label="Climate Model Mean", alpha=0.6)

    ax.scatter(
        scenario["ox"],
        scenario["odata"].mean(axis=0),
        label="Observations Mean",
        alpha=0.8,
        s=ms,
        marker="X",
        edgecolors="w",
        linewidths=0.2,
    )
    ax.scatter(
        scenario["cx"],
        scenario["cdata"].mean(axis=0),
        color="g",
        label="Climate Model Output Mean",
        alpha=0.8,
        s=ms,
        marker="P",
        edgecolors="w",
        linewidths=0.2,
    )

    ax.set_xlabel("time")
    ax.set_ylabel("temperature")
    ax.legend()


def plot_underlying_data_std_1d(scenario, ax, ms):
    ax.plot(
        scenario["X"],
        jnp.sqrt(jnp.exp(scenario["LOGVAR_T"])),
        label="Truth Std",
        alpha=0.6,
    )
    ax.plot(
        scenario["X"],
        jnp.sqrt(jnp.exp(scenario["LOGVAR_B"])),
        label="Bias Std",
        alpha=0.6,
    )
    ax.plot(
        scenario["X"],
        jnp.sqrt(jnp.exp(scenario["LOGVAR_C"])),
        label="Climate Model Std",
        alpha=0.6,
    )

    ax.scatter(
        scenario["ox"],
        scenario["odata"].std(axis=0),
        label="Observations Std",
        alpha=0.8,
        s=ms,
        marker="X",
        edgecolors="w",
        linewidths=0.2,
    )
    ax.scatter(
        scenario["cx"],
        scenario["cdata"].std(axis=0),
        color="g",
        label="Climate Model Output Std",
        alpha=0.8,
        s=ms,
        marker="P",
        edgecolors="w",
        linewidths=0.2,
    )

    ax.set_xlabel("time")
    ax.set_ylabel("temperature std")
    ax.legend()


def plot_pdfs_1d(scenario, ax, index):
    MEAN_T_sample = scenario["MEAN_T"][index]
    STDEV_T_sample = jnp.sqrt(jnp.exp(scenario["LOGVAR_T"]))[index]
    MEAN_C_sample = scenario["MEAN_C"][index]
    STDEV_C_sample = jnp.sqrt(jnp.exp(scenario["LOGVAR_C"]))[index]

    min_x = min(MEAN_T_sample - 3 * STDEV_T_sample, MEAN_C_sample - 3 * STDEV_C_sample)
    max_x = max(MEAN_T_sample + 3 * STDEV_T_sample, MEAN_C_sample + 3 * STDEV_C_sample)
    xs = np.linspace(min_x, max_x, 100)
    yts = norm.pdf(xs, MEAN_T_sample, STDEV_T_sample)
    ycs = norm.pdf(xs, MEAN_C_sample, STDEV_C_sample)

    ax.plot(xs, yts, lw=2, label="In-situ observations", color="tab:blue")
    ax.plot(xs, ycs, lw=2, label="Climate model output", color="tab:green")
    ax.fill_between(xs, yts, interpolate=True, color="tab:blue", alpha=0.6)
    ax.fill_between(xs, ycs, interpolate=True, color="tab:green", alpha=0.6)
    ax.set_xlabel("temperature")
    ax.set_ylabel("probability density")
    ax.legend()


def plot_predictions_1d_mean_hierarchical(
    scenario, key, ax, ms=None, ylims=None, color=None
):
    pred_mean = scenario[key].mean(axis=0)
    pred_std = scenario[key].std(axis=0)
    ax.plot(scenario["cx"], pred_mean, label="Expectation", color=color, alpha=0.5)
    ax.fill_between(
        scenario["cx"],
        pred_mean + pred_std,
        pred_mean - pred_std,
        label="$1\sigma$ Uncertainty",
        color=color,
        alpha=0.3,
    )

    ax.set_xlabel("time")
    ax.set_ylabel("temperature")
    ax.legend()


############# 2 Dimensional Data Figures #############


def plot_underlying_data_2d(scenario, variables, axs, ms, center, cmap):
    plots = []
    titles = ["Truth", "Bias", "Climate Model Output"]
    levels = create_levels(scenario, variables, 0.25, 0, center=center)

    for ax, var, title in zip(axs, variables, titles):
        plots.append(
            ax.contourf(
                scenario["X1"],
                scenario["X2"],
                scenario[var].reshape(scenario["X1"].shape),
                label=title,
                alpha=0.6,
                cmap=cmap,
                levels=levels,
            )
        )

    for plot in plots:
        plt.colorbar(plot)

    axs[0].scatter(
        scenario["ox"][:, 0],
        scenario["ox"][:, 1],
        s=30,
        marker="o",
        c="None",
        edgecolor="k",
    )
    axs[2].scatter(scenario["cx"][:, 0], scenario["cx"][:, 1], s=30, marker="x", c="k")

    CX1_min = scenario["CX1"].min()
    CX2_min = scenario["CX2"].min()
    CX1_max = scenario["CX1"].max()
    CX2_max = scenario["CX2"].max()
    sepCX1 = scenario["CX1"][0, 1] - scenario["CX1"][0, 0]
    sepCX2 = scenario["CX2"][1, 0] - scenario["CX2"][0, 0]
    x1_markers = scenario["CX1"][0, :] + sepCX1 / 2
    x2_markers = scenario["CX2"][:, 0] + sepCX2 / 2

    for value in x1_markers[:-1]:
        axs[2].axvline(value, CX1_min, CX1_max, linestyle="--", color="k")
    for value in x2_markers[:-1]:
        axs[2].axhline(value, CX2_min, CX2_max, linestyle="--", color="k")
