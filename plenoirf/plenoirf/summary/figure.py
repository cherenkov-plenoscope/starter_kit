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
