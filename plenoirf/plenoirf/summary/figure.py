import numpy as np


def mark_ax_airshower_spectrum(ax, x=0.93, y=0.93, fontsize=42):
    ax.text(
        x=x,
        y=y,
        s=r"$\star$",
        color="k",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


def mark_ax_thrown_spectrum(ax, x=0.93, y=0.93, fontsize=42):
    ax.text(
        x=x,
        y=y,
        s=r"$\bullet$",
        color="k",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


def histogram_confusion_matrix_with_normalized_columns(
    x,
    y,
    x_bin_edges,
    y_bin_edges,
    weights_x=None,
    min_exposure_x=100,
    default_low_exposure=np.nan,
):
    assert len(x) == len(y)
    if weights_x is not None:
        assert len(x) == len(weights_x)

    num_bins_x = len(x_bin_edges) - 1
    assert num_bins_x >= 1

    num_bins_y = len(y_bin_edges) - 1
    assert num_bins_y >= 1

    confusion_bins = np.histogram2d(
        x, y, weights=weights_x, bins=[x_bin_edges, y_bin_edges],
    )[0]

    exposure_bins_no_weights = np.histogram2d(
        x, y, bins=[x_bin_edges, y_bin_edges],
    )[0]

    confusion_bins_normalized_columns = confusion_bins.copy()
    for col in range(num_bins_x):
        if np.sum(exposure_bins_no_weights[col, :]) >= min_exposure_x:
            confusion_bins_normalized_columns[col, :] /= np.sum(confusion_bins[col, :])
        else:
            confusion_bins_normalized_columns[col, :] = (
                np.ones(num_bins_y) * default_low_exposure
            )

    return {
        "x_bin_edges": x_bin_edges,
        "y_bin_edges": y_bin_edges,
        "confusion_bins": confusion_bins,
        "confusion_bins_normalized_columns": confusion_bins_normalized_columns,
        "exposure_bins_x_no_weights": np.sum(exposure_bins_no_weights, axis=1),
        "exposure_bins_x": np.sum(confusion_bins, axis=1),
        "min_exposure_x": min_exposure_x,
    }

def ax_add_cut_indicator(
    ax,
    x,
    y,
    cuts,
    fontsize=12,
    family="monospace",
    prefix="cuts:",
    color="black",
    alpha_off=0.1
):
    for sp in ["left", "bottom", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    cuts_on = str(prefix)
    cuts_off = " " * len(str(prefix))
    for cut_key in cuts:
        if cuts[cut_key]:
            cuts_on += " " + str.upper(cut_key[0:3])
            cuts_off += " " + "   "
        else:
            cuts_on += " " + "   "
            cuts_off += " " + str.upper(cut_key[0:3])
    ax.text(
        x=x,
        y=y,
        s=cuts_on,
        color=color,
        transform=ax.transAxes,
        fontsize=fontsize,
        family=family
    )
    ax.text(
        x=x,
        y=y,
        s=cuts_off,
        color=color,
        alpha=alpha_off,
        transform=ax.transAxes,
        fontsize=fontsize,
        family=family
    )
