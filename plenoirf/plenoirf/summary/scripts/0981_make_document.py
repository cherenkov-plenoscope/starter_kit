#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import sebastians_matplotlib_addons as seb
import pylatex as ltx


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])


production_name = irf.summary.production_name_from_run_dir(pa["run_dir"])
fname = os.path.join(pa["summary_dir"], "{:s}_ltx".format(production_name))

site_key = "namibia"
SED_STYLE_KEY = "portal"

geometry_options = {
    "paperwidth": "21cm",
    "paperheight": "31cm",
    "head": "0cm",
    "left": "1cm",
    "right": "1cm",
    "top": "0cm",
    "bottom": "1cm",
    "includehead": True,
    "includefoot": True,
}
"""
production_provenance = read_json_but_forgive(
    path=opj(pa["run_dir"], "event_table", "provenance.json")
)
analysis_provenance = read_json_but_forgive(
    path=opj(pa["summary_dir"], "provenance.json")
)
"""

def ppath(*args):
    p1 = os.path.join(*args)
    p2 = os.path.normpath(p1)
    return os.path.abspath(p2)

energy_resolution_figure_path = ppath(
    pa["summary_dir"],
    "0066_energy_estimate_quality",
    site_key + "_gamma_resolution.jpg"
)

angular_resolution_figure_path = ppath(
    pa["summary_dir"],
    "0230_point_spread_function",
    site_key + "_gamma.jpg"
)

differential_sensitivity_figure_path = ppath(
    pa["summary_dir"],
    "0330_differential_sensitivity_plot",
    site_key + "_differential_sensitivity_sed-style-{:s}.jpg".format(SED_STYLE_KEY)
)

sens_vs_observation_time_figure_path = ppath(
    pa["summary_dir"],
    "0331_sensitivity_vs_observation_time",
    site_key + "_sensitivity_vs_obseravtion_time_{:s}.jpg".format(SED_STYLE_KEY)
)

ratescan_figure_path = ppath(
    pa["summary_dir"],
    "0130_trigger_ratescan_plot",
    site_key + "_ratescan.jpg"
)

diff_trigger_rates_figure_path = ppath(
    pa["summary_dir"],
    "0106_trigger_rates_for_cosmic_particles_plot",
    site_key + "_differential_trigger_rate.jpg"
)

VTIGHT = r"\vspace{-0.5cm}"


doc = ltx.Document(
    default_filepath=fname,
    documentclass="article",
    document_options=["twocolumn"],
    geometry_options=geometry_options,
    page_numbers=True,
)
doc.change_length(r"\TPHorizModule", "1cm")
doc.change_length(r"\TPVertModule", "1cm")


with doc.create(ltx.TextBlock(width=10, horizontal_pos=0, vertical_pos=0)) as blk:
    blk.append('Some regular text and some ')


with doc.create(ltx.Section('Performance', numbering=False)):
    with doc.create(ltx.Figure(position='h!')) as fig:
        fig.add_image(
            differential_sensitivity_figure_path,
            width=ltx.utils.NoEscape(r'1.0\linewidth')
        )
        fig.add_caption("Differential sensitivity.")

    doc.append(ltx.utils.NoEscape(VTIGHT))
    with doc.create(ltx.Figure(position='h!')) as fig:
        fig.add_image(
            sens_vs_observation_time_figure_path,
            width=ltx.utils.NoEscape(r'1.0\linewidth')
        )
        fig.add_caption(ltx.utils.NoEscape("Sensitivity vs. observation-time at 25\,GeV."))

    doc.append(ltx.utils.NoEscape(VTIGHT))
    with doc.create(ltx.Figure(position='h!')) as fig:
        fig.add_image(
            angular_resolution_figure_path,
            width=ltx.utils.NoEscape(r'1.0\linewidth')
        )
        fig.add_caption("Angular resolution.")

    doc.append(ltx.utils.NoEscape(VTIGHT))
    with doc.create(ltx.Figure(position='h!')) as fig:
        fig.add_image(
            energy_resolution_figure_path,
            width=ltx.utils.NoEscape(r'1.0\linewidth')
        )
        fig.add_caption("Energy resolution.")



# Add stuff to the document
with doc.create(ltx.Section('Trigger', numbering=False)):

    with doc.create(ltx.Figure(position='h!')) as fig:
        fig.add_image(
            ratescan_figure_path,
            width=ltx.utils.NoEscape(r'1.0\linewidth')
        )
        fig.add_caption("Ratescan.")

    with doc.create(ltx.Figure(position='h!')) as fig:
        fig.add_image(
            diff_trigger_rates_figure_path,
            width=ltx.utils.NoEscape(r'1.0\linewidth')
        )
        fig.add_caption("Trigger-rate on {:s}".format(sum_config['gamma_ray_reference_source']['name_3fgl']))



with doc.create(ltx.Section('Two Kittens')):
    with doc.create(ltx.Figure(position='h!')) as kittens:
        with doc.create(ltx.SubFigure(
                position='b',
                width=ltx.utils.NoEscape(r'0.45\linewidth'))) as left_kitten:

            left_kitten.add_image(
                ppath(pa["summary_dir"], "0066_energy_estimate_quality", site_key + "_gamma_resolution.jpg"),
                width=ltx.utils.NoEscape(r'\linewidth')
            )
            left_kitten.add_caption('Kitten on the left')
        with doc.create(ltx.SubFigure(
                position='b',
                width=ltx.utils.NoEscape(r'0.45\linewidth'))) as right_kitten:

            right_kitten.add_image(
                ppath(pa["summary_dir"], "0066_energy_estimate_quality", site_key + "_gamma_resolution.jpg"),
                width=ltx.utils.NoEscape(r'\linewidth')
            )
            right_kitten.add_caption('Kitten on the right')
        kittens.add_caption("Two kittens")


doc.generate_pdf(clean_tex=False)
