#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

onregion_config = sum_config["on_off_measuremnent"]["onregion"]

required_opening_angle_for_onregion = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0290_find_onregion_opening_angle")
)

# opening-angle VS reco. num. photons
# -----------------------------------
opening_angle_scaling = onregion_config["opening_angle_scaling"]

fig = seb.figure(seb.FIGURE_16_9)
ax = seb.add_axes(fig=fig, span=(0.1, 0.1, 0.8, 0.8))
_rnp = np.geomspace(
    np.min(opening_angle_scaling["reco_num_photons_pe"]),
    np.max(opening_angle_scaling["reco_num_photons_pe"]),
    1337,
)
ax.plot(
    opening_angle_scaling["reco_num_photons_pe"],
    opening_angle_scaling["scale"],
    "kx",
)
ax.plot(
    _rnp,
    np.interp(
        x=_rnp,
        xp=opening_angle_scaling["reco_num_photons_pe"],
        fp=opening_angle_scaling["scale"],
    ),
    "k-",
)
ax.set_xlabel("reco. num. photons / p.e.")
ax.set_ylabel(r"opening-angle / 1$^{\circ}$")
ax.semilogx()
fig.savefig(opj(pa["out_dir"], "opening_angle_vs_reco_num_photons.jpg"))
seb.close_figure(fig)


# ellipticity_scaling VS reco. core-radius
# ----------------------------------------
ellipticity_scaling = onregion_config["ellipticity_scaling"]

fig = seb.figure(seb.FIGURE_16_9)
ax = seb.add_axes(fig=fig, span=(0.1, 0.1, 0.8, 0.8))
_rnp = np.linspace(
    np.min(ellipticity_scaling["reco_core_radius_m"]),
    np.max(ellipticity_scaling["reco_core_radius_m"]),
    1337,
)
ax.plot(
    ellipticity_scaling["reco_core_radius_m"],
    ellipticity_scaling["scale"],
    "kx",
)
ax.plot(
    _rnp,
    np.interp(
        x=_rnp,
        xp=ellipticity_scaling["reco_core_radius_m"],
        fp=ellipticity_scaling["scale"],
    ),
    "k-",
)
ax.set_xlabel("reco. core-radius / m")
ax.set_ylabel("mayor/minor / 1")
fig.savefig(opj(pa["out_dir"], "ellipticity_vs_reco_core_radius.jpg"))
seb.close_figure(fig)
