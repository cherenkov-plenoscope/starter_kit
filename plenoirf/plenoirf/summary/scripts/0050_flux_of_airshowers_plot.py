#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import pandas as pd
import json_numpy
import magnetic_deflection
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

raw_cosmic_ray_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0010_flux_of_cosmic_rays")
)

fine_energy_bin_edges, num_fine_energy_bins = irf.utils.power10space_bin_edges(
    binning=sum_config["energy_binning"],
    fine=sum_config["energy_binning"]["fine"]["interpolation"]
)

deflection_table = magnetic_deflection.read(
    work_dir=os.path.join(pa["run_dir"], "magnetic_deflection"),
)

energy_lower = fine_energy_bin_edges[0]
energy_upper = fine_energy_bin_edges[-1]
fine_energy_bin_centers = irf.utils.bin_centers(fine_energy_bin_edges)

particle_colors = sum_config["plot"]["particle_colors"]

SITES = irf_config["config"]["sites"]
COSMICS = list(irf_config["config"]["particles"].keys())
COSMICS.remove("gamma")

geomagnetic_cutoff_fraction = 0.05

def _rigidity_to_total_energy(rigidity_GV):
    return rigidity_GV * 1.0


# interpolate
# -----------
cosmic_ray_fluxes = {}
for pk in COSMICS:
    cosmic_ray_fluxes[pk] = {}
    cosmic_ray_fluxes[pk]["differential_flux"] = np.interp(
        x=fine_energy_bin_centers,
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
            rigidity_GV=irf_config["config"]["sites"][sk]["geomagnetic_cutoff_rigidity_GV"]
        )
        below_cutoff = fine_energy_bin_centers < cutoff_energy
        air_shower_fluxes[sk][pk]["differential_flux"] = np.array(
            cosmic_ray_fluxes[pk]["differential_flux"]
        )
        air_shower_fluxes[sk][pk]["differential_flux"][below_cutoff] *= geomagnetic_cutoff_fraction

# zenith compensation
# -------------------
air_shower_fluxes_zc = {}
for sk in SITES:
    air_shower_fluxes_zc[sk] = {}
    for pk in COSMICS:
        air_shower_fluxes_zc[sk][pk] = {}
        primary_zenith_deg = np.interp(
            x=fine_energy_bin_centers,
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
                "unit_tex": raw_cosmic_ray_fluxes[pk]["differential_flux"]["unit_tex"],
            },
        )



airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0050_flux_of_airshowers_plot")
)



for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in airshower_fluxes[sk]:
        ax.plot(
            fine_energy_bin_centers,
            airshower_fluxes[sk][pk]["differential_flux"]["values"],
            label=pk,
            color=particle_colors[pk],
        )
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(
        "differential flux of airshowers /\n"
        + "m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$"
    )
    ax.loglog()
    ax.set_xlim([energy_lower, energy_upper])
    ax.legend()
    # ax.set_title("compensated for zenith-distance w.r.t. observation-plane")
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "{:s}_airshower_differential_flux.jpg".format(sk),
        )
    )
    seb.close_figure(fig)


# gamma-ray-flux of reference source
# ----------------------------------
(
    gamma_differential_flux_per_m2_per_s_per_GeV,
    gamma_name,
) = irf.summary.make_gamma_ray_reference_flux(
    fermi_3fgl=raw_cosmic_ray_fluxes["gamma_sources"],
    gamma_ray_reference_source=sum_config["gamma_ray_reference_source"],
    energy_supports_GeV=fine_energy_bin_centers,
)
out_df = pd.DataFrame(
    {
        "energy / GeV": fine_energy_bin_centers,
        "differential flux / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$": (
            gamma_differential_flux_per_m2_per_s_per_GeV
        ),
    }
)
out_path = os.path.join(pa["out_dir"], "{:s}.csv".format(gamma_name))
with open(out_path, "wt") as f:
    f.write(out_df.to_csv(index=False))
