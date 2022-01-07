#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import sebastians_matplotlib_addons as seb
import pylatex as ltx
import warnings
import json_numpy
import io


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

production_dirname = irf.summary.production_name_from_run_dir(pa["run_dir"])
fname = os.path.join(pa["summary_dir"], "{:s}_ltx".format(production_dirname))

BIB_REFERENCES_PATH = os.path.join(
    os.getcwd(), "sebastians_references", "references"
)

sk = "namibia"
ok = "large"
dk = "bell_spectrum"

SED_STYLE_KEY = "portal"

geometry_options = {
    "paper": "a4paper",
    # "paperwidth": "18cm",
    # "paperheight": "32cm",
    "head": "0cm",
    "left": "2cm",
    "right": "2cm",
    "top": "0cm",
    "bottom": "2cm",
    "includehead": True,
    "includefoot": True,
}


def noesc(text):
    return ltx.utils.NoEscape(text)


def read_json_but_forgive(path, default={}):
    try:
        with open(path, "rt") as f:
            out = json_numpy.loads(f.read())
    except Exception as e:
        print(e)
        warnings.warn("Failed to load '{:s}'".format(path))
        out = default
    return out


def make_basic_version_str(
    production_dirname, production_provenance, analysis_provenance
):
    pp = production_provenance
    ap = analysis_provenance

    ver = ""
    ver += "Production\n"
    ver += "    dirname: {:s}\n".format(production_dirname)
    ver += "    date: {:s}\n".format(pp["time"]["iso"][0:16])
    ver += "    git-commit: {:s}\n".format(
        pp["starter_kit"]["git"]["commit"][0:9]
    )
    ver += "    hostname: {:s}\n".format(pp["hostname"])
    ver += "Analysis\n"
    ver += "    date: {:s}\n".format(ap["time"]["iso"][0:16])
    ver += "    git-commit:   {:s}\n".format(
        ap["starter_kit"]["git"]["commit"][0:9]
    )
    ver += "    hostname: {:s}\n".format(ap["hostname"])
    return ver


def make_trigger_modus_str(analysis_trigger, production_trigger):
    prdtrg = production_trigger
    anatrg = analysis_trigger
    s = ""
    s += "Modus\n"
    s += "    Accepting object-distance "
    s += "{:.1f}km, focus {:02d}\n".format(
        1e-3
        * prdtrg["object_distances_m"][anatrg["modus"]["accepting_focus"]],
        anatrg["modus"]["accepting_focus"],
    )
    s += "    Rejecting object-distance "
    if anatrg["modus"]["use_rejection_focus"]:
        s += "{:.1f}km, focus {:02d}\n".format(
            1e-3
            * prdtrg["object_distances_m"][anatrg["modus"]["rejecting_focus"]],
            anatrg["modus"]["rejecting_focus"],
        )
        s += "    Intensity-ratio between foci: {:.2f}\n".format(
            anatrg["modus"]["intensity_ratio_between_foci"]
        )
    else:
        s += "None\n"
        s += "\n"
    s += "Threshold\n"
    s += "    {:d}p.e. ".format(anatrg["threshold_pe"])
    s += "({:d}p.e. in production)\n".format(prdtrg["threshold_pe"])
    return s


def dict_to_pretty_str(dictionary):
    ss = json_numpy.dumps(dictionary, indent=2)
    ss = ss.replace('"', '')
    ss = ss.replace('{', '')
    ss = ss.replace('}', '')
    oss = io.StringIO()
    for line in ss.splitlines():
        if len(line) > 0:
            oss.write(line)
            oss.write("\n")
    oss.seek(0)
    return oss.read()


def ppath(*args):
    p1 = os.path.join(*args)
    p2 = os.path.normpath(p1)
    return os.path.abspath(p2)


def get_total_trigger_rate_at_analysis_threshold(sk):
    trigger_rates_by_origin = json_numpy.read_tree(
        ppath(pa["summary_dir"], "0131_trigger_rates_total")
    )[sk]["trigger_rates_by_origin"]

    idx = trigger_rates_by_origin["analysis_trigger_threshold_idx"]
    total_rate_per_s = 0.0
    for origin in trigger_rates_by_origin["origins"]:
        total_rate_per_s += trigger_rates_by_origin["origins"][origin][idx]
    return total_rate_per_s


production_provenance = read_json_but_forgive(
    path=os.path.join(pa["run_dir"], "event_table", "provenance.json")
)
analysis_provenance = read_json_but_forgive(
    path=os.path.join(pa["summary_dir"], "provenance.json")
)

energy_resolution_figure_path = ppath(
    pa["summary_dir"],
    "0066_energy_estimate_quality",
    sk + "_gamma_resolution.jpg",
)
assert os.path.exists(energy_resolution_figure_path)

energy_confusion_figure_path = ppath(
    pa["summary_dir"], "0066_energy_estimate_quality", sk + "_gamma.jpg",
)
assert os.path.exists(energy_confusion_figure_path)

angular_resolution_figure_path = ppath(
    pa["summary_dir"], "0230_point_spread_function", sk + "_gamma.jpg"
)
assert os.path.exists(angular_resolution_figure_path)

differential_sensitivity_figure_path = ppath(
    pa["summary_dir"],
    "0550_diffsens_plot",
    sk,
    ok,
    SED_STYLE_KEY,
    "{:s}_{:s}_{:s}_differential_sensitivity_sed_style_{:s}.jpg".format(
        sk, ok, dk, SED_STYLE_KEY
    ),
)
assert os.path.exists(differential_sensitivity_figure_path)

sens_vs_observation_time_figure_path = ppath(
    pa["summary_dir"],
    "0610_sensitivity_vs_observation_time",
    sk,
    ok,
    dk,
    "sensitivity_vs_obseravtion_time_{:s}.jpg".format(SED_STYLE_KEY),
)
assert os.path.exists(sens_vs_observation_time_figure_path)

ratescan_figure_path = ppath(
    pa["summary_dir"], "0130_trigger_ratescan_plot", sk + "_ratescan.jpg"
)
assert os.path.exists(ratescan_figure_path)

diff_trigger_rates_figure_path = ppath(
    pa["summary_dir"],
    "0106_trigger_rates_for_cosmic_particles_plot",
    sk + "_differential_trigger_rate.jpg",
)

total_trigger_rate_per_s = get_total_trigger_rate_at_analysis_threshold(sk)
total_trigger_rate_per_s_ltx = irf.utils.latex_scientific(
    real=total_trigger_rate_per_s, format_template="{:.3e}"
)

basic_version_str = make_basic_version_str(
    production_dirname=production_dirname,
    production_provenance=production_provenance,
    analysis_provenance=analysis_provenance,
)

doc = ltx.Document(
    default_filepath=fname,
    documentclass="article",
    document_options=[],
    geometry_options=geometry_options,
    font_size="small",
    page_numbers=True,
)


def Verbatim(string):
    return r"\begin{verbatim}" + r"{:s}".format(string) + r"\end{verbatim}"


doc.preamble.append(ltx.Package("multicol"))
doc.preamble.append(ltx.Package("lipsum"))
doc.preamble.append(ltx.Package("float"))
doc.preamble.append(ltx.Package("verbatim"))

doc.preamble.append(
    ltx.Command("title", noesc(r"Simulating the Cherenkov-Plenoscope"),)
)
doc.preamble.append(ltx.Command("author", "Sebastian A. Mueller"))
doc.preamble.append(ltx.Command("date", ""))

doc.append(noesc(r"\maketitle"))
doc.append(noesc(r"\begin{multicols}{2}"))


with doc.create(ltx.Section("Version", numbering=False)):
    doc.append(noesc(Verbatim(basic_version_str)))

with doc.create(ltx.Section("Performance", numbering=False)):

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            differential_sensitivity_figure_path,
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Differential sensitivity for a point-like source of gamma-rays. "
                r"Fermi-LAT \cite{wood2016fermiperformance} in orange. "
                r"CTA-south \cite{cta2018baseline} in blue. "
            )
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            sens_vs_observation_time_figure_path,
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Sensitivity vs. observation-time at 25\,GeV. "
                r"Fermi-LAT in orange and CTA-south in blue taken from "
                r"\cite{funk2013comparison}."
            )
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            angular_resolution_figure_path, width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Angular resolution. "
                r"Fermi-LAT \cite{wood2016fermiperformance} in orange. "
                r"CTA-south \cite{cta2018baseline} in blue. "
            )
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            energy_resolution_figure_path, width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Energy resolution. "
                r"Fermi-LAT \cite{wood2016fermiperformance} in orange. "
                r"CTA-south \cite{cta2018baseline} in blue. "
            )
        )

    doc.append(
        noesc(
            r"The Crab Nebula's gamma-ray-flux \cite{aleksic2015measurement} "
            r"\mbox{(100\%, 10\%, 1\%, and 0.1\%)} is shown in fading gray dashes. "
        )
    )


# doc.append(noesc(r"\columnbreak"))

with doc.create(ltx.Section("Site", numbering=False)):
    doc.append(sk)
    doc.append(
        noesc(
            Verbatim(
                dict_to_pretty_str(irf_config["config"]["sites"][sk])
            )
        )
    )
    doc.append(
        noesc(
            r"Flux of airshowers (not cosmic particles) are estimated "
            r"based on the "
            r"fluxes of cosmic protons \cite{aguilar2015precision}, "
            r"electrons and positrons \cite{aguilar2014precision}, and "
            r"helium \cite{patrignani2017helium}."
        )
    )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0016_flux_of_airshowers_plot",
                sk + "_airshower_differential_flux.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Flux of airshowers (not particles) at the site. "
            "This includes airshowers below the geomagnetic-cutoff created by secondary, terrestrial particles."
        )


trgstr = make_trigger_modus_str(
    analysis_trigger=sum_config["trigger"],
    production_trigger=irf_config["config"]["sum_trigger"],
)

with doc.create(ltx.Section("Trigger", numbering=False)):
    doc.append(noesc(Verbatim(trgstr)))
    doc.append(
        noesc(
            "Trigger-rate during observation is $\\approx{"
            + total_trigger_rate_per_s_ltx
            + r"}\,$s$^{-1}$"
        )
    )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(ratescan_figure_path, width=noesc(r"1.0\linewidth"))
        fig.add_caption(
            "Ratescan. For low thresholds the rates seem "
            "to saturate. This is because of limited statistics. "
            "The rates are expected to raise further."
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0071_trigger_probability_vs_cherenkov_size_plot",
                sk + "_trigger_probability_vs_cherenkov_size.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Trigger-probability vs. true Cherenkov-size.")

with doc.create(ltx.Section("Acceptance at Trigger", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0101_trigger_acceptance_for_cosmic_particles_plot",
                sk + "_diffuse.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective acceptance for a diffuse source.")
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0101_trigger_acceptance_for_cosmic_particles_plot",
                sk + "_point.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective area for a point like source.")

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            diff_trigger_rates_figure_path, width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Trigger-rate on gamma-ray-source {:s}".format(
                    sum_config["gamma_ray_reference_source"]["name_3fgl"]
                )
                + r"\cite{acero2015fermi3fglcatalog}."
            )
        )

with doc.create(
    ltx.Section("Cherenkov- and Night-Sky-Light", numbering=False)
):
    doc.append("Finding Cherenkov-photons in the pool of nigth-sky-light.")
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0060_cherenkov_photon_classification_plot",
                sk + "_gamma_confusion.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Size-confusion of Cherenkov-photons emitted in airshowers initiated by gamma-rays."
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0060_cherenkov_photon_classification_plot",
                sk + "_gamma_sensitivity_vs_true_energy.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Classification-power for Cherenkov-photons emitted in airshowers initiated by gamma-rays."
        )

with doc.create(ltx.Section("Energy", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            energy_confusion_figure_path, width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Energy-confusion for gamma-rays.")


with doc.create(ltx.Section("Acceptance after all Cuts", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0301_onregion_trigger_acceptance_plot",
                "{:s}_{:s}_diffuse.jpg".format(sk, ok),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective acceptance for a diffuse source.")

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0301_onregion_trigger_acceptance_plot",
                "{:s}_{:s}_point.jpg".format(sk, ok),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective area for a point like source.")

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                pa["summary_dir"],
                "0325_onregion_trigger_rates_for_cosmic_rays_plot",
                "{:s}_{:s}_differential_event_rates.jpg".format(sk, ok),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Final rates in on-region while observing {:s}".format(
                sum_config["gamma_ray_reference_source"]["name_3fgl"]
            )
        )

doc.append(noesc(r"\bibliographystyle{apalike}"))
doc.append(noesc(r"\bibliography{" + BIB_REFERENCES_PATH + "}"))

doc.append(noesc(r"\end{multicols}{2}"))
doc.generate_pdf(clean_tex=False)
