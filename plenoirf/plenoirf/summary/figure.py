import numpy as np


FIGURE_STYLE = {"rows": 720, "cols": 1280, "fontsize": 1.0}
AX_SPAN = [0.2, 0.2, 0.75, 0.75]

SOURCES = {
    "diffuse": {
        "label": "area $\\times$ solid angle",
        "unit": "m$^{2}$ sr",
        "limits": [1e-1, 1e5],
    },
    "point": {"label": "area", "unit": "m$^{2}$", "limits": [1e1, 1e6],},
}

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
