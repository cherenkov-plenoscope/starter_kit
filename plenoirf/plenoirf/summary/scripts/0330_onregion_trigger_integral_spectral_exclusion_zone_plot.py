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
diff_sensitivity = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0327_differential_sensitivity_plot")
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

fermi = irf.other_instruments.fermi_lat
cta = irf.other_instruments.cherenkov_telescope_array_south


# gamma-ray-flux of crab-nebula
# -----------------------------
crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

internal_sed_style = sed_styles.PLENOIRF_SED_STYLE

output_sed_styles = {
    "portal": sed_styles.PLENOIRF_SED_STYLE,
    "science": sed_styles.SCIENCE_SED_STYLE,
    "fermi": sed_styles.FERMI_SED_STYLE,
    "cta": sed_styles.CHERENKOV_TELESCOPE_ARRAY_SED_STYLE,
}

observation_time = 30 * 60
observation_time_str = irf.utils.make_civil_time_str(
    time_s=observation_time,
    format_seconds="{:.0f}"
)


def find_observation_time_index(observation_times, observation_time, max_rel_error=0.1):
    observation_times = np.array(observation_times)
    obstidx = np.argmin(np.abs(observation_times - observation_time))
    assert (
        np.abs(observation_times[obstidx] - observation_time)
        < max_rel_error * observation_time
    )
    return obstidx


oridx = 1
sys_unc = sum_config["on_off_measuremnent"]["systematic_uncertainty"]
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
y_lim_per_m2_per_s_per_GeV = np.array([1e3, 1e-16])

PLOT_TANGENTIAL_POWERLAWS = False


for site_key in irf_config["config"]["sites"]:

    components = []

    # Crab reference fluxes
    # ---------------------
    for i in range(4):
        com = {}
        scale_factor = np.power(10.0, (-1) * i)
        com["energy"] = [np.array(crab_flux["energy"]["values"])]
        com["differential_flux"] = [
            scale_factor * np.array(crab_flux["differential_flux"]["values"])
        ]
        com["label"] = "{:1.1e} Crab".format(scale_factor) if i == 0 else None
        com["color"] = "k"
        com["alpha"] = 1.0 / (1.0 + i)
        com["linestyle"] = "--"
        components.append(com.copy())

    # Fermi-LAT diff
    # --------------
    fermi_diff = fermi.differential_sensitivity(l=0, b=90)
    com = {}
    com["energy"] = [np.array(fermi_diff["energy"]["values"])]
    com["differential_flux"] = [
        np.array(fermi_diff["differential_flux"]["values"])
    ]
    com["label"] = fermi.LABEL + ", 10y, (l=0, b=90), diff."
    com["color"] = fermi.COLOR
    com["alpha"] = 1.0
    com["linestyle"] = "-"
    components.append(com)

    # CTA South 30min
    # ---------------
    cta_diff = cta.differential_sensitivity(observation_time=observation_time)
    com = {}
    com["energy"] = [np.array(cta_diff["energy"]["values"])]
    com["differential_flux"] = [
        np.array(cta_diff["differential_flux"]["values"])
    ]
    com["label"] = cta.LABEL + ", " + observation_time_str +" , diff."
    com["color"] = cta.COLOR
    com["alpha"] = 1.0
    com["linestyle"] = "-"
    components.append(com)

    # Plenoscope diff
    # ---------------
    obstidx = find_observation_time_index(
        observation_times=diff_sensitivity[
            site_key]["differential_sensitivity"]["observation_times"],
        observation_time=observation_time
    )

    com = {}
    com["energy"] = []
    com["differential_flux"] = []

    for ii in range(len(energy_bin_edges) - 1):
        com["energy"].append([energy_bin_edges[ii], energy_bin_edges[ii + 1]])
        _dFdE_sens = np.array(
            diff_sensitivity[site_key]["differential_sensitivity"][
                "differential_flux"
            ]
        )[ii, oridx, obstidx]
        com["differential_flux"].append([_dFdE_sens, _dFdE_sens])
    com["label"] = "Portal, " + observation_time_str + ", diff., sys. {:.1e}".format(sys_unc)
    com["color"] = "black"
    com["alpha"] = 1.0
    com["linestyle"] = "-"
    components.append(com)

    for sed_style_key in output_sed_styles:
        sed_style = output_sed_styles[sed_style_key]

        fig = seb.figure(seb.FIGURE_16_9)
        ax = seb.add_axes(fig, (0.1, 0.1, 0.8, 0.8))

        for com in components:

            for ii in range(len(com["energy"])):
                _energy, _dFdE = sed.convert_units_with_style(
                    x=com["energy"][ii],
                    y=com["differential_flux"][ii],
                    input_style=internal_sed_style,
                    target_style=sed_style,
                )
                ax.plot(
                    _energy,
                    _dFdE,
                    label=com["label"] if ii == 0 else None,
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
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_integral_spectral_exclusion_zone_style_{:s}.jpg".format(
                    site_key, sed_style_key
                ),
            )
        )
        seb.close_figure(fig)
