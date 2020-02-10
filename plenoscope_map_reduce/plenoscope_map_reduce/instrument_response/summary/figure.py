import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CONFIG_16_9 = {
    "rows": 1080,
    "cols": 1920,
    "fontsize": 2,
    "format": "jpg"
}


def figure(config=CONFIG_16_9, dpi=120):
    sc = config['fontsize']
    width = config['cols']/dpi
    height = config['rows']/dpi
    return plt.figure(
        figsize=(width/sc, height/sc),
        dpi=dpi*sc)


def ax_add_circle(
    ax,
    x,
    y,
    r,
    linewidth=1.,
    linestyle='-',
    color='k',
    alpha=1,
    num_steps=1000
):
    phi = np.linspace(0, 2*np.pi, num_steps)
    xs = x + r*np.cos(phi)
    ys = y + r*np.sin(phi)
    ax.plot(
        xs, ys,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color)


def ax_add_slider(ax, start, stop, values, label, log=False):
    ax.set_ylim([start, stop])
    if log:
        ax.semilogy()
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(label)
    for value in values:
        ax.plot(
            [0, 1],
            [value, value],
            "k",
            linewidth=5)


def ax_add_hist(ax, bin_edges, bincounts, linestyle, color, alpha):
    assert bin_edges.shape[0] == bincounts.shape[0] + 1
    for i, bincount in enumerate(bincounts):
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [bincount, bincount],
            linestyle)
        ax.fill_between(
            x=[bin_edges[i], bin_edges[i + 1]],
            y1=[bincount, bincount],
            color=color,
            alpha=alpha,
            edgecolor='none')


def ax_add_hatches(ax, ix, iy, x_bin_edges, y_bin_edges, alpha=0.1):
    x0 = x_bin_edges[ix]
    x1 = x_bin_edges[ix + 1]
    y0 = y_bin_edges[iy]
    y1 = y_bin_edges[iy + 1]
    ax.plot([x0, x1], [y0, y1], '-k', alpha=alpha)