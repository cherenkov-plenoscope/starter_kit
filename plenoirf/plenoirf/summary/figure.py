import matplotlib
import matplotlib.pyplot as plt
import numpy as np


CONFIG_16_9 = {"rows": 1080, "cols": 1920, "fontsize": 2, "format": "jpg"}


def figure(config=CONFIG_16_9, dpi=120):
    sc = config["fontsize"]
    width = config["cols"] / dpi
    height = config["rows"] / dpi
    return plt.figure(figsize=(width / sc, height / sc), dpi=dpi * sc)


def ax_add_circle(
    ax,
    x,
    y,
    r,
    linewidth=1.0,
    linestyle="-",
    color="k",
    alpha=1,
    num_steps=1000,
):
    phi = np.linspace(0, 2 * np.pi, num_steps)
    xs = x + r * np.cos(phi)
    ys = y + r * np.sin(phi)
    ax.plot(
        xs,
        ys,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        color=color,
    )


def ax_add_slider(ax, start, stop, values, label, log=False):
    ax.set_ylim([start, stop])
    if log:
        ax.semilogy()
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(label)
    for value in values:
        ax.plot([0, 1], [value, value], "k", linewidth=5)


def ax_add_hist(
    ax,
    bin_edges,
    bincounts,
    linestyle="-",
    linecolor="k",
    linealpha=1.0,
    bincounts_upper=None,
    bincounts_lower=None,
    face_color=None,
    face_alpha=None,
):
    assert bin_edges.shape[0] == bincounts.shape[0] + 1
    for i, bincount in enumerate(bincounts):
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [bincount, bincount],
            linestyle=linestyle,
            color=linecolor,
            alpha=linealpha,
        )
        if bincounts_upper is not None and bincounts_lower is not None:
            ax.fill_between(
                x=[bin_edges[i], bin_edges[i + 1]],
                y1=[bincounts_lower[i], bincounts_lower[i]],
                y2=[bincounts_upper[i], bincounts_upper[i]],
                color=face_color,
                alpha=face_alpha,
                edgecolor="none",
            )


def ax_add_hatches(ax, ix, iy, x_bin_edges, y_bin_edges, alpha=0.1):
    x0 = x_bin_edges[ix]
    x1 = x_bin_edges[ix + 1]
    y0 = y_bin_edges[iy]
    y1 = y_bin_edges[iy + 1]
    ax.plot([x0, x1], [y0, y1], "-k", alpha=alpha)


def radial_histogram(
    radial2_bin_edges, radial2_values, energy_mask, pasttrigger_mask,
):
    num_cradial_bins = len(radial2_bin_edges) - 1
    num_thrown = np.histogram(
        radial2_values, weights=energy_mask, bins=radial2_bin_edges
    )[0]

    energy_pasttrigger_mask = np.logical_and(
        energy_mask, pasttrigger_mask
    ).astype(np.int)
    num_pasttrigger = np.histogram(
        radial2_values, weights=energy_pasttrigger_mask, bins=radial2_bin_edges
    )[0]

    num_pasttrigger_relunc = np.nan * np.ones(num_cradial_bins)
    for bb in range(num_cradial_bins):
        if num_pasttrigger[bb] > 0:
            num_pasttrigger_relunc[bb] = (
                np.sqrt(num_pasttrigger[bb]) / num_pasttrigger[bb]
            )

    trgprb = np.nan * np.ones(num_cradial_bins)
    for bb in range(num_cradial_bins):
        if num_thrown[bb] > 0:
            trgprb[bb] = num_pasttrigger[bb] / num_thrown[bb]
    trgprb_absunc = trgprb * num_pasttrigger_relunc

    _up = trgprb + trgprb_absunc
    _up_valid = np.logical_not(np.isnan(_up))
    _lo = trgprb - trgprb_absunc
    _lo_valid = np.logical_not(np.isnan(_lo))
    if np.sum(_lo_valid) and np.sum(_up_valid):
        ymin = np.min(_lo[_lo_valid])
        if ymin < 0.0:
            ymin = np.min(trgprb[_lo_valid])
        ylim = [ymin, np.max(_up[_up_valid])]
    else:
        ylim = None
    return {
        "num_thrown": num_thrown,
        "num_pasttrigger": num_pasttrigger,
        "num_pasttrigger_relunc": num_pasttrigger_relunc,
        "trgprb": trgprb,
        "trgprb_absunc": trgprb_absunc,
        "ylim": ylim,
        "num_energy_bin_pasttrigger": np.sum(energy_pasttrigger_mask),
    }


def write_radial_histogram_figure(
    path,
    radial2_bin_edges,
    trgprb,
    trgprb_absunc,
    ylim,
    figure_config,
    xlabel,
    ylabel,
    title,
):
    fig = figure(figure_config)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_add_hist(
        ax=ax,
        bin_edges=radial2_bin_edges,
        bincounts=trgprb,
        linestyle="k-",
        bincounts_upper=trgprb + trgprb_absunc,
        bincounts_lower=trgprb - trgprb_absunc,
        face_color="k",
        face_alpha=0.3,
    )
    ax.set_title(title)
    ax.semilogy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.set_xlim([0, np.max(radial2_bin_edges)])
    ax.set_ylim(ylim)
    fig.savefig(path)
    plt.close(fig)
