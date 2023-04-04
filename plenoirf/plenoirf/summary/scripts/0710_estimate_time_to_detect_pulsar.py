#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis
import json_numpy
import binning_utils
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
energy_fine_bin = energy_binning["interpolation"]


prng = np.random.Generator(
    np.random.generator.PCG64(sum_config["random_seed"])
)


# pulsars phase-o-gram
def ppog_init():
    return {
        "phase_bin_edges": [],
        "phase_bin_edges_unit": "rad",
        "energy_bin_edges": [],
        "energy_bin_edges_unit": "GeV",
        "relative_amplitude_vs_phase": [],
        "relative_amplitude_vs_phase_cdf": [],
        "differential_flux_vs_energy": [],
        "differential_flux_vs_energy_unit": "m$^{-2}$ s^{-1} (GeV)$^{-1}$",
    }


OUR_PULSAR = {
    "source_name": "Test Pulsar",
    "spectrum_type": "PLExpCutoff",
    "spectral_index": -1.5,
    "exp_index": 1.0,
    "pivot_energy_GeV": 0.5,
    "cutoff_energy_GeV": 2.5,
    "flux_density_per_m2_per_GeV_per_s": 3e-4,
}


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def ppog_init_dummy(
    num_phase_bins=1234,
    energy_bin_edges=np.geomspace(1e-1, 1e2, 15),
    energy_start_GeV=1e-1,
    energy_stop_GeV=1e2,
    peak_source_fermi_style=OUR_PULSAR,
    peak_std_rad=0.05,
    base_fraction_of_peak=0.1,
):
    ppog = ppog_init()
    assert num_phase_bins >= 1
    num_energy_bins = len(energy_bin_edges) - 1
    assert num_energy_bins >= 1

    ppog["phase_bin_edges"] = np.linspace(0, 2 * np.pi, num_phase_bins + 1)
    ppog["energy_bin_edges"] = np.array(energy_bin_edges)

    ppog["relative_amplitude_vs_phase"] = np.zeros(num_phase_bins)
    ppog["differential_flux_vs_energy"] = np.zeros(num_energy_bins)

    phase_normalized = np.zeros(num_phase_bins)
    for p in range(num_phase_bins):
        phi_start = ppog["phase_bin_edges"][p]
        phi_stop = ppog["phase_bin_edges"][p + 1]
        phi = np.mean([phi_start, phi_stop])

        peak_1_amplitude = gaussian(
            x=phi, mu=(2 * np.pi * 1 / 3), sigma=2 * peak_std_rad
        )
        peak_2_amplitude = gaussian(
            x=phi, mu=(2 * np.pi * 2 / 3), sigma=peak_std_rad
        )
        peak_amplitude = peak_1_amplitude + peak_2_amplitude

        base_1 = base_fraction_of_peak * (0.5 + 0.4 * np.sin(phi))
        base_2 = (
            base_fraction_of_peak * (1 / 2) * (0.5 + 0.4 * np.cos(2 * phi))
        )
        base_3 = (
            base_fraction_of_peak * (1 / 3) * (0.5 + 0.4 * np.cos(3 * phi))
        )
        base_amplitude = base_1 + base_2 + base_3
        phase_normalized[p] = base_amplitude + peak_amplitude

    ppog["relative_amplitude_vs_phase"] = phase_normalized / np.sum(
        phase_normalized
    )

    cdf = np.array(
        [0] + np.cumsum(ppog["relative_amplitude_vs_phase"]).tolist()
    )

    ppog["relative_amplitude_vs_phase_cdf"] = cdf

    for e in range(num_energy_bins):
        E_start = ppog["energy_bin_edges"][e]
        E_stop = ppog["energy_bin_edges"][e + 1]
        E = np.mean([E_start, E_stop])
        dKdE = cosmic_fluxes.flux_of_fermi_source(
            fermi_source=peak_source_fermi_style, energy=E,
        )
        ppog["differential_flux_vs_energy"][e] = dKdE

    return ppog


def ppog_draw_phase(ppog, prng, num=1):
    r = prng.uniform(low=0.0, high=1.0, size=num)
    phases = np.interp(
        x=r,
        xp=ppog["relative_amplitude_vs_phase_cdf"],
        fp=ppog["phase_bin_edges"],
    )
    phases = np.mod(phases, (2 * np.pi))
    return phases


pulsar = ppog_init_dummy(energy_bin_edges=energy_fine_bin["edges"])

arrival_phases = ppog_draw_phase(pulsar, prng, 100 * 1000)

arrival_phases_counts = np.histogram(
    arrival_phases, bins=pulsar["phase_bin_edges"],
)[0]
arrival_phases_counts = arrival_phases_counts / np.sum(arrival_phases_counts)

fig = seb.figure(seb.FIGURE_1_1)
ax_c = seb.add_axes(fig=fig, span=[0.2, 0.45, 0.7, 0.5])
ax_h = seb.add_axes(fig=fig, span=[0.2, 0.11, 0.7, 0.2])
ax_c.plot(
    pulsar["energy_bin_edges"][0:-1],
    pulsar["differential_flux_vs_energy"],
    "k-",
)
ax_c.loglog()
_ymax = np.max(pulsar["differential_flux_vs_energy"])
ax_c.set_ylim([1e-6 * _ymax, _ymax])
ax_c.set_xlabel("energy / GeV")
ax_c.set_ylabel(
    r"$\frac{\mathrm{d\,flux}}{\mathrm{d\,energy}}$ / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$"
)
seb.ax_add_histogram(
    ax=ax_h,
    bin_edges=pulsar["phase_bin_edges"] / (2 * np.pi),
    bincounts=pulsar["relative_amplitude_vs_phase"],
    linestyle="-",
    linecolor="k",
    draw_bin_walls=True,
)

seb.ax_add_histogram(
    ax=ax_h,
    bin_edges=pulsar["phase_bin_edges"] / (2 * np.pi),
    bincounts=arrival_phases_counts,
    linestyle=":",
    linecolor="r",
    draw_bin_walls=True,
)

ax_h.set_xlim([0, 1])
ax_h.set_xlabel(r"phase / 2$\pi$")
ax_h.set_ylabel(r"relative / 1")
fig.savefig(os.path.join(pa["out_dir"], "pulsar.jpg"))
seb.close(fig)


NUM_BINS_PER_PHASE = 137

c = {}
c["phase"] = {}
c["phase"]["bin"] = binning_utils.Binning(
    bin_edges=np.linspace(0, 2 * np.pi, NUM_BINS_PER_PHASE),
)
c["observation_time_block_s"] = 3600


def expose_to_background(
    observation_time_s, background_rate_per_s, phase_bin, prng,
):
    background_intensity = observation_time_s * background_rate_per_s
    background_intensity = int(np.round(background_intensity))
    assert background_intensity > 1e3, "intensity is low -> rounding error."
    background_arrival_times = prng.uniform(
        low=phase_bin["start"],
        high=phase_bin["stop"],
        size=background_intensity,
    )
    background = np.histogram(
        background_arrival_times, bins=phase_bin["edges"]
    )[0]
    return background


def draw_time_to_detect_next_gamma_ray(gamma_rate_per_s, prng, num):
    return -np.log(prng.uniform(size=num)) / gamma_rate_per_s


def expose_to_pulsar(
    observation_time_s, gamma_rate_per_s, phase_bin, pulsar, prng,
):
    tt = 0.0
    intensity = 0
    while tt < observation_time_s:
        tt += draw_time_to_detect_next_gamma_ray(
            gamma_rate_per_s=gamma_rate_per_s, prng=prng, num=1
        )
        intensity += 1

    gamma_ray_phases = ppog_draw_phase(ppog=pulsar, prng=prng, num=intensity)

    gamma_rays = np.histogram(gamma_ray_phases, bins=phase_bin["edges"])[0]

    return gamma_rays


for sk in ["chile"]: #SITES:
    for ok in ["large"]: #ONREGION_TYPES:

        # background
        # ----------
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

        # signal
        # ------
        A_gamma = onregion_acceptance[sk][ok]["gamma"]["point"]["mean"]
        A_gamma_fine_m2 = np.interp(
            x=energy_fine_bin["centers"], xp=energy_bin["centers"], fp=A_gamma,
        )

        dKdE_per_m2_per_s_per_GeV = pulsar["differential_flux_vs_energy"]

        rate_gamma_per_s = 0.0
        for ebin in range(energy_fine_bin["num_bins"]):
            E_start_GeV = energy_fine_bin["edges"][ebin]
            E_stop_GeV = energy_fine_bin["edges"][ebin + 1]
            E_width_GeV = E_stop_GeV - E_start_GeV
            rate_gamma_per_s += (
                A_gamma_fine_m2[ebin]
                * dKdE_per_m2_per_s_per_GeV[ebin]
                * E_width_GeV
            )

        res = {"gamma": [], "background": []}
        exposure_time = 0.0
        for h in range(100):

            gam = expose_to_pulsar(
                observation_time_s=c["observation_time_block_s"],
                gamma_rate_per_s=rate_gamma_per_s,
                phase_bin=c["phase"]["bin"],
                pulsar=pulsar,
                prng=prng,
            )

            bkg = expose_to_background(
                observation_time_s=c["observation_time_block_s"],
                background_rate_per_s=rate_cosmic_rays,
                phase_bin=c["phase"]["bin"],
                prng=prng,
            )
            exposure_time += c["observation_time_block_s"]
            res["gamma"].append(gam)
            res["background"].append(bkg)


            tot_background = np.sum(res["background"], axis=0)
            tot_background_mean = np.mean(tot_background)
            tot_background_au = np.sqrt(tot_background)

            tot_gamma = np.sum(res["gamma"], axis=0)

            tot = tot_background + tot_gamma
            tot_au = np.sqrt(tot)

            if np.mod(h, 10) == 0:

                fig = seb.figure(seb.FIGURE_1_1)
                ax = seb.add_axes(fig=fig, span=[0.2, 0.2, 0.7, 0.7])
                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=c["phase"]["bin"]["edges"] / (2 * np.pi),
                    bincounts=tot/exposure_time,
                    bincounts_lower=(tot-tot_au)/exposure_time,
                    bincounts_upper=(tot+tot_au)/exposure_time,
                    linestyle="-",
                    linecolor="k",
                    face_color="k",
                    face_alpha=0.2,
                    draw_bin_walls=False,
                )
                _ymin = rate_cosmic_rays / c["phase"]["bin"]["num"]
                ax.set_ylim([0.99 * _ymin, _ymin * 1.5])

                ax.set_xlabel(r"phase / 2$\pi$")
                ax.set_ylabel(r"rate / s$^{-1}$")
                fig.savefig(os.path.join(pa["out_dir"], "phase_{:d}.jpg".format(h)))
                seb.close(fig)






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
            A_gamma[2],
            "m^{2}",
        )
