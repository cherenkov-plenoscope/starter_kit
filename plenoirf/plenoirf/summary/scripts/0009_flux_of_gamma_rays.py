#!/usr/bin/python
import sys
import plenoirf as irf
import os
import json_numpy
import cosmic_fluxes

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

fine_energy_bin_edges, _ = irf.utils.power10space_bin_edges(
    binning=sum_config["energy_binning"],
    fine=sum_config["energy_binning"]["fine"]["interpolation"]
)
fine_energy_bin_centers = irf.utils.bin_centers(fine_energy_bin_edges)

# load catalog
# ------------
fermi_3fgl = cosmic_fluxes.fermi_3fgl_catalog()

# export catalog locally
# ----------------------
with open(os.path.join(pa["out_dir"], "fermi_3fgl_catalog.json"), "wt") as f:
    f.write(json_numpy.dumps(fermi_3fgl, indent=4))

# make reference source
# ---------------------
(
    differential_flux_per_m2_per_s_per_GeV,
    name,
) = irf.summary.make_gamma_ray_reference_flux(
    fermi_3fgl=fermi_3fgl,
    gamma_ray_reference_source=sum_config["gamma_ray_reference_source"],
    energy_supports_GeV=fine_energy_bin_centers,
)

json_numpy.write(
    os.path.join(os.path.join(pa["out_dir"], "reference_source.json")),
    {
        "name": name,
        "differential_flux": {
            "values": differential_flux_per_m2_per_s_per_GeV,
            "unit": "per_m2_per_s_per_GeV",
            "unit_tex": "m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$",
        },
        "energy": {
            "values": fine_energy_bin_centers,
            "unit": "GeV",
            "unit_tex": "GeV",
        },
        "energy_implicit": {
            "fine": "interpolation",
            "supports": "centers"
        },
    }
)
