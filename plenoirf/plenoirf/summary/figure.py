import numpy as np


FIGURE_STYLE = {"rows": 720, "cols": 1280, "fontsize": 1.0}
AX_SPAN = [0.2, 0.2, 0.75, 0.75]

SOURCES = {
    "diffuse": {
        "label": "area $\\times$ solid angle",
        "unit": "m$^{2}$ sr",
        "limits": {
            "passed_trigger": [1e-1, 1e5],
            "passed_all_cuts": [1e-1, 1e3],
        }
    },
    "point": {
        "label": "area",
        "unit": "m$^{2}$",
        "limits": {
            "passed_trigger": [1e1, 1e6],
            "passed_all_cuts": [1e1, 1e5],
        },
    }
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
    ax0_key,
    ax0_values,
    ax0_bin_edges,
    ax1_key,
    ax1_values,
    ax1_bin_edges,
    ax0_weights=None,
    min_exposure_ax0=100,
    default_low_exposure=np.nan,
):
    assert len(ax0_values) == len(ax1_values)
    if ax0_weights is not None:
        assert len(ax0_values) == len(ax0_weights)

    num_bins_ax0 = len(ax0_bin_edges) - 1
    assert num_bins_ax0 >= 1

    num_bins_ax1 = len(ax1_bin_edges) - 1
    assert num_bins_ax1 >= 1

    confusion_bins = np.histogram2d(
        ax0_values,
        ax1_values,
        weights=ax0_weights,
        bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    exposure_bins_no_weights = np.histogram2d(
        ax0_values, ax1_values, bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    confusion_bins_normalized_on_ax0 = confusion_bins.copy()
    for i0 in range(num_bins_ax0):
        if np.sum(exposure_bins_no_weights[i0, :]) >= min_exposure_ax0:
            confusion_bins_normalized_on_ax0[i0, :] /= np.sum(
                confusion_bins[i0, :]
            )
        else:
            confusion_bins_normalized_on_ax0[i0, :] = (
                np.ones(num_bins_ax1) * default_low_exposure
            )

    return {
        "ax0_key": ax0_key,
        "ax1_key": ax1_key,
        "ax0_bin_edges": ax0_bin_edges,
        "ax1_bin_edges": ax1_bin_edges,
        "confusion_bins": confusion_bins,
        "confusion_bins_normalized_on_ax0": confusion_bins_normalized_on_ax0,
        "exposure_bins_ax0_no_weights": np.sum(exposure_bins_no_weights, axis=1),
        "exposure_bins_ax0": np.sum(confusion_bins, axis=1),
        "min_exposure_ax0": min_exposure_ax0,
    }
