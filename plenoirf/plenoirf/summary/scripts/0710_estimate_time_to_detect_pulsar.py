#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis
import json_numpy
import cosmic_fluxes
import propagate_uncertainties
import copy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]
COSMIC_RAYS = copy.deepcopy(PARTICLES)
COSMIC_RAYS.pop("gamma")

onregion_rates = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)
onregion_acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]


# pulsars phase-o-gram
def ppog_init():
    return {
        "phase_bin_edges": [],
        "phase_bin_edges_unit": "rad",
        "energy_bin_edges": [],
        "energy_bin_edges_unit": "GeV",
        "differential_flux": [],
        "differential_flux_au": [],
        "differential_flux_unit": "m$^{-2}$ s^{-1} (GeV)$^{-1}$",
    }


fermi_3fgl = cosmic_fluxes.fermi_3fgl_catalog()

for source in fermi_3fgl:
    if source["source_name"] == "3FGL J2140.0+4715":
        PEAK_SOURCE_FERMI_STYLE = copy.deepcopy(source)
    if source["source_name"] == "3FGL J2139.5+3919":
        BASE_SOURCE_FERMI_STYLE = copy.deepcopy(source)


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def ppog_init_dummy(
    num_phase_bins=81,
    num_energy_bins=15,
    energy_start_GeV=1e-1,
    energy_stop_GeV=1e2,
    peak_source_fermi_style=PEAK_SOURCE_FERMI_STYLE,
    base_source_fermi_style=BASE_SOURCE_FERMI_STYLE,
    peak_std_rad=0.05,
    base_fraction_of_peak=0.1,
):
    ppog = ppog_init()

    ppog["phase_bin_edges"] = np.linspace(0, 2 * np.pi, num_phase_bins + 1)
    ppog["energy_bin_edges"] = np.geomspace(
        energy_start_GeV, energy_stop_GeV, num_energy_bins + 1
    )

    ppog["differential_flux"] = np.zeros(
        shape=(num_phase_bins, num_energy_bins)
    )
    ppog["differential_flux_au"] = np.zeros(
        shape=(num_phase_bins, num_energy_bins)
    )

    for p in range(num_phase_bins):
        phi_start = ppog["phase_bin_edges"][p]
        phi_stop = ppog["phase_bin_edges"][p + 1]
        phi = np.mean([phi_start, phi_stop])
        for e in range(num_energy_bins):
            E_start = ppog["energy_bin_edges"][e]
            E_stop = ppog["energy_bin_edges"][e + 1]
            E = np.mean([E_start, E_stop])
            peak_amplitude = gaussian(x=phi, mu=np.pi, sigma=peak_std_rad)
            dKdE_peak = peak_amplitude * cosmic_fluxes.flux_of_fermi_source(
                fermi_source=peak_source_fermi_style, energy=E,
            )
            base_1 = base_fraction_of_peak * (0.5 + 0.4 * np.sin(phi))
            base_2 = (
                base_fraction_of_peak * (1 / 2) * (0.5 + 0.4 * np.cos(2 * phi))
            )
            base_2 = (
                base_fraction_of_peak * (1 / 3) * (0.5 + 0.4 * np.cos(3 * phi))
            )
            base_amplitude = base_1 + base_2

            dKdE_base = base_amplitude * cosmic_fluxes.flux_of_fermi_source(
                fermi_source=base_source_fermi_style, energy=E,
            )

            ppog["differential_flux"][p, e] = dKdE_base + dKdE_peak
            ppog["differential_flux_au"][p, e] = (
                0.05 * ppog["differential_flux"][p, e]
            )
    return ppog


def ppog_integrate_flux_over_energy(ppog):
    num_phase_bins = len(ppog["phase_bin_edges"]) - 1
    num_energy_bins = len(ppog["energy_bin_edges"]) - 1
    assert num_phase_bins >= 1
    assert num_energy_bins >= 1
    F = np.zeros(num_phase_bins)
    F_au = np.zeros(num_phase_bins)
    for p in range(num_phase_bins):
        f = np.zeros(num_energy_bins)
        f_au = np.zeros(num_energy_bins)
        for e in range(num_energy_bins):
            E_start = ppog["energy_bin_edges"][e]
            E_stop = ppog["energy_bin_edges"][e + 1]
            E_width_GeV = E_stop - E_start
            f[e] = ppog["differential_flux"][p, e] * E_width_GeV
            f_au[e] = ppog["differential_flux_au"][p, e] * E_width_GeV
        F[p], F_au[p] = propagate_uncertainties.sum(x=f, x_au=f_au)
    return F, F_au


ppog = ppog_init_dummy()
Fint, Fint_au = ppog_integrate_flux_over_energy(ppog=ppog)

fig = seb.figure(seb.FIGURE_1_1)
ax_c = seb.add_axes(fig=fig, span=[0.2, 0.27, 0.55, 0.55])
ax_h = seb.add_axes(fig=fig, span=[0.2, 0.11, 0.55, 0.1])
ax_cb = seb.add_axes(fig=fig, span=[0.8, 0.27, 0.02, 0.55])
_pcm_confusion = ax_c.pcolormesh(
    ppog["phase_bin_edges"] / (2 * np.pi),
    ppog["energy_bin_edges"],
    np.transpose(ppog["differential_flux"]),
    cmap="Greys",
    norm=seb.plt_colors.PowerNorm(gamma=0.5),
)
ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
ax_cb.set_ylabel(r"dK/dE / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$")
ax_c.set_ylabel("E / GeV")
ax_c.semilogy()
ax_c.set_xticklabels([])

seb.ax_add_histogram(
    ax=ax_h,
    bin_edges=ppog["phase_bin_edges"] / (2 * np.pi),
    bincounts=Fint,
    linestyle="-",
    linecolor="k",
    draw_bin_walls=True,
)
ax_h.set_xlim([0, 1])
ax_h.set_xlabel(r"phase / 2$\pi$")
ax_h.set_ylabel(r"K / m$^{-2}$ s$^{-1}$")

fig.savefig(os.path.join(pa["out_dir"], "pulsar.jpg"))
seb.close(fig)


for sk in SITES:
    for ok in ONREGION_TYPES:
        rate_cosmic_rays = []
        rate_cosmic_rays_au = []
        for pk in COSMIC_RAYS:
            rate = onregion_rates[sk][ok][pk]["integral_rate"]["mean"]
            rate_au = onregion_rates[sk][ok][pk]["integral_rate"][
                "absolute_uncertainty"
            ]
        rate_cosmic_rays.append(rate)
        rate_cosmic_rays_au.append(rate_au)
        rate_cosmic_rays, rate_cosmic_rays_au = propagate_uncertainties.sum(
            x=rate_cosmic_rays, x_au=rate_cosmic_rays_au
        )

        effective_collection_area_gamma_rays = onregion_acceptance[sk][ok][
            "gamma"
        ]["point"]["mean"]
        effective_collection_area_gamma_rays_au = onregion_acceptance[sk][ok][
            "gamma"
        ]["point"]["absolute_uncertainty"]

        print(
            "Site: ",
            sk,
            ", Size of on-region: ",
            ok,
            ", Rate cosmic-rays: ",
            rate_cosmic_rays,
            "+-",
            rate_cosmic_rays_au,
            "s^{-1}",
            ", Area gamma-rays at 1GeV ",
            effective_collection_area_gamma_rays[2],
            "+-",
            effective_collection_area_gamma_rays_au[2],
            "m^{2}",
        )
