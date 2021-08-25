#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_numpy
import magnetic_deflection

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

raw_cosmic_ray_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0010_flux_of_cosmic_rays")
)

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["interpolation"]

deflection_table = magnetic_deflection.read(
    work_dir=os.path.join(pa["run_dir"], "magnetic_deflection"),
)

SITES = irf_config["config"]["sites"]
COSMICS = list(irf_config["config"]["particles"].keys())
COSMICS.remove("gamma")

geomagnetic_cutoff_fraction = sum_config["airshower_flux"][
    "fraction_of_flux_below_geomagnetic_cutoff"
]


def _rigidity_to_total_energy(rigidity_GV):
    return rigidity_GV * 1.0


# interpolate
# -----------
cosmic_ray_fluxes = {}
for pk in COSMICS:
    cosmic_ray_fluxes[pk] = {}
    cosmic_ray_fluxes[pk]["differential_flux"] = np.interp(
        x=energy_bin["centers"],
        xp=raw_cosmic_ray_fluxes[pk]["energy"]["values"],
        fp=raw_cosmic_ray_fluxes[pk]["differential_flux"]["values"],
    )

# earth's geomagnetic cutoff
# --------------------------
air_shower_fluxes = {}
for sk in SITES:
    air_shower_fluxes[sk] = {}
    for pk in COSMICS:
        air_shower_fluxes[sk][pk] = {}
        cutoff_energy = _rigidity_to_total_energy(
            rigidity_GV=irf_config["config"]["sites"][sk][
                "geomagnetic_cutoff_rigidity_GV"
            ]
        )
        below_cutoff = energy_bin["centers"] < cutoff_energy
        air_shower_fluxes[sk][pk]["differential_flux"] = np.array(
            cosmic_ray_fluxes[pk]["differential_flux"]
        )
        air_shower_fluxes[sk][pk]["differential_flux"][
            below_cutoff
        ] *= geomagnetic_cutoff_fraction

# zenith compensation
# -------------------
air_shower_fluxes_zc = {}
for sk in SITES:
    air_shower_fluxes_zc[sk] = {}
    for pk in COSMICS:
        air_shower_fluxes_zc[sk][pk] = {}
        primary_zenith_deg = np.interp(
            x=energy_bin["centers"],
            xp=deflection_table[sk][pk]["energy_GeV"],
            fp=deflection_table[sk][pk]["primary_zenith_deg"],
        )
        scaling = np.cos(np.deg2rad(primary_zenith_deg))
        zc_flux = scaling * air_shower_fluxes[sk][pk]["differential_flux"]
        air_shower_fluxes_zc[sk][pk]["differential_flux"] = zc_flux


# export
# ------
for sk in SITES:
    for pk in COSMICS:
        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)
        json_numpy.write(
            os.path.join(sk_pk_dir, "differential_flux.json"),
            {
                "comment": (
                    "The flux of air-showers seen by/ relevant for the "
                    "instrument. Respects geomagnetic cutoff "
                    "and zenith-compensation when primary is "
                    "deflected in earth's magnetic-field."
                ),
                "values": air_shower_fluxes_zc[sk][pk]["differential_flux"],
                "unit": raw_cosmic_ray_fluxes[pk]["differential_flux"]["unit"],
            },
        )
