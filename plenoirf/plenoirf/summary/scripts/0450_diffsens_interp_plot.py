#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import spectral_energy_distribution_units as sed
from plenoirf.analysis import spectral_energy_distribution as sed_styles
import cosmic_fluxes
import os
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

diff_sensitivity = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0440_diff_sens_estimate")
)

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

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
    time_s=observation_time, format_seconds="{:.0f}"
)


def find_observation_time_index(
    observation_times, observation_time, max_rel_error=0.1
):
    return irf.utils.find_closest_index_in_array_for_value(
        arr=observation_times,
        val=observation_time,
        max_rel_error=max_rel_error,
    )


DIFF_SENS_SCENARIOS = irf.analysis.differential_sensitivity.SCENARIOS

oridx = 1
sys_unc = sum_config["on_off_measuremnent"]["systematic_uncertainty"]
onregion_opening_angle_deg = sum_config["on_off_measuremnent"]["onregion"][
    "loop_opening_angle_deg"
][oridx]

x_lim_GeV = np.array([1e-1, 1e4])
y_lim_per_m2_per_s_per_GeV = np.array([1e3, 1e-16])


for sk in irf_config["config"]["sites"]:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)

    for dk in DIFF_SENS_SCENARIOS:

        components = []

        # Crab reference fluxes
        # ---------------------
        for i in range(4):
            com = {}
            scale_factor = np.power(10.0, (-1) * i)
            com["energy"] = [np.array(crab_flux["energy"]["values"])]
            com["differential_flux"] = [
                scale_factor
                * np.array(crab_flux["differential_flux"]["values"])
            ]
            com[
                "label"
            ] = None  # "{:1.1e} Crab".format(scale_factor) if i == 0 else None
            com["color"] = "k"
            com["alpha"] = 0.25 / (1.0 + i)
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
        com["label"] = fermi.LABEL + ", 10y, (l=0, b=90)"
        com["color"] = fermi.COLOR
        com["alpha"] = 1.0
        com["linestyle"] = "-"
        components.append(com)

        # CTA South 30min
        # ---------------
        cta_diff = cta.differential_sensitivity(
            observation_time=observation_time
        )
        com = {}
        com["energy"] = [np.array(cta_diff["energy"]["values"])]
        com["differential_flux"] = [
            np.array(cta_diff["differential_flux"]["values"])
        ]
        com["label"] = cta.LABEL + ", " + observation_time_str
        com["color"] = cta.COLOR
        com["alpha"] = 1.0
        com["linestyle"] = "-"
        components.append(com)

        # Plenoscope diff
        # ---------------
        obstidx = find_observation_time_index(
            observation_times=diff_sensitivity[sk][dk]["observation_times"],
            observation_time=observation_time,
        )

        com = {}
        com["energy"] = []
        com["differential_flux"] = []

        for ii in range(energy_bin["num_bins"]):
            com["energy"].append(
                [energy_bin["edges"][ii], energy_bin["edges"][ii + 1]]
            )
            _dFdE_sens = diff_sensitivity[sk][dk]["differential_flux"][
                ii, oridx, obstidx
            ]
            com["differential_flux"].append([_dFdE_sens, _dFdE_sens])
        com["label"] = (
            "Portal, " + observation_time_str + ", sys. {:.1e}".format(sys_unc)
        )
        com["color"] = "black"
        com["alpha"] = 1.0
        com["linestyle"] = "-"
        components.append(com)

        for sed_style_key in output_sed_styles:
            sed_style_dir = os.path.join(sk_dir, "sed_style_" + sed_style_key)
            os.makedirs(sed_style_dir, exist_ok=True)
            sed_style = output_sed_styles[sed_style_key]

            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

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
            etype = irf.analysis.differential_sensitivity.SCENARIOS[dk][
                "energy_axes_label"
            ]
            ax.set_xlabel(
                etype + " " + sed_style["x_label"] + " /" + sed_style["x_unit"]
            )
            ax.set_ylabel(sed_style["y_label"] + " /\n " + sed_style["y_unit"])
            fig.savefig(
                os.path.join(
                    sed_style_dir,
                    "{:s}_differential_sensitivity_sed_style_{:s}_scenario_{:s}.jpg".format(
                        sk, sed_style_key, dk
                    ),
                )
            )
            seb.close(fig)
