#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import spectral_energy_distribution_units as sed
from plenoirf.analysis import spectral_energy_distribution as sed_styles
import cosmic_fluxes
import os
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)


onregion_acceptance = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)
onregion_rates = irf.json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)

energy_lower = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper = sum_config["energy_binning"]["upper_edge_GeV"]
energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config["energy_binning"]["num_bins"]["trigger_acceptance_onregion"]
    + 1,
)
energy_bin_centers = irf.utils.bin_centers(energy_bin_edges)

detection_threshold_std = sum_config["on_off_measuremnent"][
    "detection_threshold_std"
]
on_over_off_ratio = sum_config["on_off_measuremnent"]["on_over_off_ratio"]

cosmic_ray_keys = list(irf_config["config"]["particles"].keys())
cosmic_ray_keys.remove("gamma")

fermi_broadband = irf.analysis.fermi_lat_integral_spectral_exclusion_zone()
assert fermi_broadband["energy"]["unit_tex"] == "GeV"
assert (
    fermi_broadband["differential_flux"]["unit_tex"]
    == "m$^{-2}$ s$^{-1}$ GeV$^{-1}$"
)

# gamma-ray-flux of crab-nebula
# -----------------------------
crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

internal_sed_style = sed_styles.PLENOIRF_SED_STYLE

output_sed_styles = {
    "plenoirf": sed_styles.PLENOIRF_SED_STYLE,
    "science": sed_styles.SCIENCE_SED_STYLE,
    "fermi": sed_styles.FERMI_SED_STYLE,
}

instrument_systematic_uncertainty = 5e-3

loop_systematic_uncertainty = [0.0, 1e-2]
loop_systematic_uncertainty_line_style = ["-", ":"]

loop_observation_time = [60, 300, 1500]
loop_observation_time_line_color = ["red", "brown", "orange"]

oridx = 0
onregion_opening_angle_deg = sum_config["on_off_measuremnent"]["onregion"][
    "loop_opening_angle_deg"
][oridx]

# background rates
# ----------------
cosmic_ray_rate_onregion = {}
electron_rate_onregion = {}
for site_key in irf_config["config"]["sites"]:

    electron_rate_onregion[site_key] = onregion_rates[site_key]["electron"][
        "integral_rate"
    ]["mean"][oridx]

    cosmic_ray_rate_onregion[site_key] = 0
    for cosmic_ray_key in cosmic_ray_keys:
        cosmic_ray_rate_onregion[site_key] += onregion_rates[site_key][
            cosmic_ray_key
        ]["integral_rate"]["mean"][oridx]

x_lim_GeV = np.array([1e-1, 1e4])
y_lim_per_m2_per_s_per_GeV = np.array([1e-0, 1e-16])

PLOT_TANGENTIAL_POWERLAWS = False


for site_key in irf_config["config"]["sites"]:

    components = []

    # Crab reference fluxes
    # ---------------------
    for i in range(4):
        com = {}
        scale_factor = np.power(10.0, (-1) * i)
        com["energy"] = np.array(crab_flux["energy"]["values"])
        com["differential_flux"] = scale_factor * np.array(
            crab_flux["differential_flux"]["values"]
        )
        com["label"] = "{:1.1e} Crab".format(scale_factor)
        com["color"] = "k"
        com["alpha"] = 1.0 / (1.0 + i)
        com["linestyle"] = "--"
        components.append(com.copy())

    # Fermi-LAT broadband
    # -------------------
    com = {}
    com["energy"] = np.array(fermi_broadband["energy"]["values"])
    com["differential_flux"] = np.array(
        fermi_broadband["differential_flux"]["values"]
    )
    com["label"] = "Fermi-LAT 10y"
    com["color"] = "k"
    com["alpha"] = 1.0
    com["linestyle"] = "-"
    components.append(com)

    # plenoscope
    # ----------
    for obt in range(len(loop_observation_time)):
        observation_time_s = loop_observation_time[obt]

        for sys, systematic_uncertainty in enumerate(
            loop_systematic_uncertainty
        ):
            gamma_effective_area_m2 = np.array(
                np.array(
                    onregion_acceptance[site_key]["gamma"]["point"]["mean"]
                )[:, oridx]
            )

            critical_rate_per_s = irf.analysis.integral_sensitivity.estimate_critical_rate(
                background_rate_in_onregion_per_s=cosmic_ray_rate_onregion[
                    site_key
                ],
                onregion_over_offregion_ratio=on_over_off_ratio,
                observation_time_s=observation_time_s,
                instrument_systematic_uncertainty=systematic_uncertainty,
                detection_threshold_std=5.0,
                method="LiMa_eq17",
            )

            powlaws = irf.analysis.integral_sensitivity.estimate_critical_power_laws(
                effective_area_bins_m2=gamma_effective_area_m2,
                effective_area_energy_bin_edges_GeV=energy_bin_edges,
                critical_rate_per_s=critical_rate_per_s,
                power_law_spectral_indices=np.linspace(-6.0, 0.0, 100),
            )

            (
                tangent_E_GeV,
                tangent_dF_per_m2_per_GeV_per_s,
            ) = irf.analysis.integral_sensitivity.estimate_tangent_of_critical_power_laws(
                critical_power_laws=powlaws
            )
            com = {}
            com["energy"] = tangent_E_GeV
            com["differential_flux"] = tangent_dF_per_m2_per_GeV_per_s
            com["label"] = "Portal {:2.0f}s, trigger, sys. {:1.1e}".format(
                observation_time_s, systematic_uncertainty,
            )

            com["alpha"] = 1.0 / (1.0 + sys)
            com["color"] = loop_observation_time_line_color[obt]
            com["linestyle"] = loop_systematic_uncertainty_line_style[sys]
            components.append(com)

            if PLOT_TANGENTIAL_POWERLAWS:
                for powlaw in powlaws:
                    _E_GeV = np.geomspace(
                        np.min(energy_bin_edges),
                        np.max(energy_bin_edges),
                        1337,
                    )
                    _dF_per_m2_per_GeV_per_s = cosmic_fluxes._power_law(
                        energy=_E_GeV,
                        flux_density=powlaw[
                            "flux_density_per_m2_per_GeV_per_s"
                        ],
                        spectral_index=powlaw["spectral_index"],
                        pivot_energy=powlaw["pivot_energy_GeV"],
                    )
                    com = {}
                    com["energy"] = _E_GeV
                    com["differential_flux"] = _dF_per_m2_per_GeV_per_s
                    com["label"] = None
                    com["alpha"] = 0.01
                    com["color"] = "black"
                    com["linestyle"] = "-"
                    components.append(com)

    for sed_style_key in output_sed_styles:
        sed_style = output_sed_styles[sed_style_key]

        fig = seb.figure(seb.FIGURE_16_9)
        ax = seb.add_axes(fig, (0.1, 0.1, 0.8, 0.8))

        for com in components:

            _energy, _dFdE = sed.convert_units_with_style(
                x=com["energy"],
                y=com["differential_flux"],
                input_style=internal_sed_style,
                target_style=sed_style,
            )
            ax.plot(
                _energy,
                _dFdE,
                label=com["label"],
                color=com["color"],
                alpha=com["alpha"],
                linestyle=com["linestyle"],
            )

        _x_lim, _y_lim = sed.convert_units_with_style(
            x=x_lim_GeV,
            y=y_lim_per_m2_per_s_per_GeV,
            input_style=internal_sed_style,
            target_style=sed_style,
        )

        ax.set_xlim(np.sort(_x_lim))
        ax.set_ylim(np.sort(_y_lim))
        ax.loglog()
        ax.legend(loc="best", fontsize=10)
        ax.set_xlabel(sed_style["x_label"] + " / " + sed_style["x_unit"])
        ax.set_ylabel(sed_style["y_label"] + " / " + sed_style["y_unit"])
        ax.set_title(
            "onregion-opening-angle at 100p.e. {: 2.1f}".format(
                onregion_opening_angle_deg
            )
            + r"$^{\circ}$",
            family="monospace",
        )
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_integral_spectral_exclusion_zone_style_{:s}.jpg".format(
                    site_key, sed_style_key
                ),
            )
        )
        seb.close_figure(fig)
