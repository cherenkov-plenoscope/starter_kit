#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import json
import dominate
import warnings

from plenoirf.summary import samtex as sam

import weasyprint

FIGURE_WIDTH_PIXEL = 80

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

energy_bin_edges_coarse = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins']['point_spread_function'] + 1
)

def get_value_by_key_but_forgive(dic, key):
    try:
        return dic[key]
    except KeyError:
        print("WARNING key '{:s}' does not exist!".format(key))
        return {}


def read_json_but_forgive(path, default={}):
    try:
        with open(path, 'rt') as f:
            out = json.loads(f.read())
    except Exception as e:
        print(e)
        warnings.warn("Failed to load '{:s}'".format(path))
        out = default
    return out


def make_site_table(
    sites,
    energy_bin_edges,
    wild_card='{site_key:s}_key_{energy_bin_index:06d}.jpg',
    site_width_px=FIGURE_WIDTH_PIXEL*4,
):
    """
    num columns = len(sites)
    num rows = len(energy_bin_edges) - 1
    """
    matrix = []

    for energy_bin_index in range(len(energy_bin_edges) - 1):
        row = []
        for site_key in sites:
            sub_row = []
            path = wild_card.format(
                site_key=site_key,
                energy_bin_index=energy_bin_index
            )
            _img = sam.img(
                src=path,
                width_px=site_width_px
            )
            sub_row.append(_img)
            row.append(
                sam.table(
                    [sub_row],
                    width_px=site_width_px
                )
            )
        matrix.append(row)

    return sam.table(
        matrix=matrix,
        width_px=len(sites)*site_width_px
    )


def make_site_particle_index_table(
    sites,
    particles,
    energy_bin_edges,
    wild_card='{site_key:s}_{particle_key:s}_key_{energy_bin_index:06d}.jpg',
    particle_width_px=FIGURE_WIDTH_PIXEL,
    header=True,
):
    site_width_px = len(particles)*particle_width_px

    matrix = []
    if header:
        side_head_row = []
        for site_key in sites:
            side_head_row.append(sam.h(site_key, level=5, text_align='center'))
        matrix.append(side_head_row)

        row = []
        for site_key in sites:
            sub_row = []
            for particle_key in particles:
                sub_row.append(
                    sam.h(particle_key, level=6, text_align='center')
                )
            row.append(
                sam.table(matrix=[sub_row], width_px=site_width_px)
            )
        matrix.append(row)

    for energy_bin_index in range(len(energy_bin_edges) - 1):
        row = []
        for site_key in sites:
            sub_row = []
            for particle_key in particles:
                path = wild_card.format(
                    site_key=site_key,
                    particle_key=particle_key,
                    energy_bin_index=energy_bin_index
                )
                _img = sam.img(src=path, width_px=particle_width_px)
                sub_row.append(_img)
            row.append(sam.table(matrix=[sub_row], width_px=site_width_px))
        matrix.append(row)

    return sam.table(
        matrix=matrix,
        width_px=len(sites)*site_width_px
    )


evt_provenance = read_json_but_forgive(
    path=opj(pa['run_dir'], 'event_table', 'provenance.json')
)
ana_provenance = read_json_but_forgive(
    path=opj(pa['summary_dir'], 'provenance.json')
)

_html = dominate.tags.html()
_bd = _html.add(dominate.tags.body())

_bd += sam.h('Cherenkov-plenoscope', level=1)
_bd += sam.p(
    "Summarizing the instrument's response.",
    font_size=75,
    text_align='justify',
    font_family='calibri',
)

_bd += sam.h('git-commit', level=4)
_bd += sam.code(
    get_value_by_key_but_forgive(
        dic=get_value_by_key_but_forgive(
            dic=get_value_by_key_but_forgive(
                dic=evt_provenance,
                key='starter_kit'
            ),
            key='git'
        ),
        key='commit'
    ),
    font_size=100,
    line_height=100,
),

site_matrix = []
_row = []
for site_key in irf_config['config']['sites']:
    _row.append(
        sam.h(site_key, level=5, text_align='center')
    )
site_matrix.append(_row)
_row = []
for site_key in irf_config['config']['sites']:
    _row.append(
        sam.code(
            json.dumps(irf_config['config']['sites'][site_key], indent=4),
            font_size=50,
            line_height=100,
        ),
    )
site_matrix.append(_row)

_bd += sam.table(
    site_matrix,
    width_px=FIGURE_WIDTH_PIXEL*8
)

_bd += sam.h('Light-field-trigger', level=2)
_trigger_config = sum_config['trigger'].copy()
_trigger_config.pop('ratescan_thresholds_pe')
_bd += sam.code(
    json.dumps(_trigger_config, indent=4),
    font_size=50,
    line_height=100,
)

_bd += sam.h('Random-seed', level=2)
_bd += sam.p(
    "The run-id and airshower-id form the random-seed for an airshower.",
    font_size=75,
    text_align='justify',
    font_family='calibri'
)
_runs_config = get_value_by_key_but_forgive(irf_config, 'runs').copy()
_bd += sam.code(
    json.dumps(_runs_config, indent=4),
    font_size=50,
    line_height=100,
)

_bd += sam.h('Effective area, ponit source, trigger-level', level=2)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0101_trigger_acceptance_for_cosmic_particles_plot',
        '{site_key:s}_point.jpg'
    )
)

_bd += sam.h('Effective acceptance, diffuse source, trigger-level', level=2)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0101_trigger_acceptance_for_cosmic_particles_plot',
        '{site_key:s}_diffuse.jpg'
    )
)

_bd += sam.h('Flux of airshowers', level=2)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0050_flux_of_airshowers_plot',
        '{site_key:s}_airshower_differential_flux.jpg'
    )
)

_bd += sam.h('Trigger-ratescan', level=2)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0130_trigger_ratescan_plot',
        '{site_key:s}_ratescan.jpg'
    )
)

_bd += sam.h('Differential trigger-rates, entire field-of-view', level=2)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0106_trigger_rates_for_cosmic_particles_plot',
        '{site_key:s}_differential_trigger_rate.jpg'
    )
)

_bd += sam.h('Directions of primaries, past trigger', level=2)
_bd += sam.p(
    "Primary particle's incidend direction color-coded "
    "with their probability to trigger the plenoscope. "
    "Hatched solid angles are unknown. ",
    font_size=75,
    text_align='justify',
    font_family='calibri'
)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        '0810_grid_direction_of_primaries_plot',
        '{site_key:s}_{particle_key:s}_'
        'grid_direction_pasttrigger_{energy_bin_index:06d}.jpg'
    )
)
_bd += sam.h('Cherenkov-intensity on observation-level, past trigger', level=2)
_bd += sam.p(
    "Areal intensity of Cherenkov-photons on the observation-level. "
    "Only showing airshowers which passed the plenoscope's trigger. "
    "Color-coding shows the average intensity of a single airshower. ",
    font_size=75,
    text_align='justify',
    font_family='calibri'
)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        '0815_grid_illumination_plot',
        '{site_key:s}_{particle_key:s}_'
        'grid_area_pasttrigger_{energy_bin_index:06d}.jpg'
    )
)
_bd += sam.h('Trigger-probability vs. true Cherenkov-size / p.e.', level=2)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0071_trigger_probability_vs_cherenkov_size_plot',
        '{site_key:s}_trigger_probability_vs_cherenkov_size.jpg'
    )
)
_bd += sam.h('Trigger-probability vs. offaxis-angle', level=2)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0075_trigger_probability_vs_offaxis',
        '{site_key:s}_{particle_key:s}_trigger_probability_vs_offaxis.jpg'
    )
)
_bd += sam.h('Trigger-probability vs. offaxis-angle vs. energy', level=2)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        '0075_trigger_probability_vs_offaxis',
        '{site_key:s}_{particle_key:s}_'
        'trigger_probability_vs_offaxis_{energy_bin_index:06d}.jpg'
    )
)
_bd += sam.h('Cherenkov and night-sky-background classification', level=2)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0060_cherenkov_photon_classification_plot',
        '{site_key:s}_{particle_key:s}_confusion.jpg'
    )
)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0060_cherenkov_photon_classification_plot',
        '{site_key:s}_{particle_key:s}_'
        'sensitivity_vs_true_energy.jpg'
    )
)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0060_cherenkov_photon_classification_plot',
        '{site_key:s}_{particle_key:s}_'
        'true_size_over_extracted_size_vs_true_energy.jpg'
    )
)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0060_cherenkov_photon_classification_plot',
        '{site_key:s}_{particle_key:s}_'
        'true_size_over_extracted_size_vs_true_size.jpg'
    )
)

_bd += sam.h('Direction-reconstruction', level=2)
_bd += sam.p(
    "Opening angle for 68% containment. "
    "Dashed line shows fix onregion openin angle.",
    text_align='justify',
    font_family='calibri'
)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0201_gamma_point_spread_function_plot',
        '{site_key:s}_gamma_psf_radial.jpg'
    )
)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        '0201_gamma_point_spread_function_plot',
        '{site_key:s}_gamma_{energy_bin_index:06d}_psf.jpg'
    )
)

_bd += sam.p(
    "Reconstructed directions in field-of-view:",
    text_align='justify',
    font_family='calibri'
)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0205_reconstructed_directions_in_field_of_view',
        '{site_key:s}_{particle_key:s}.jpg'
    )
)

_bd += sam.h('Effective area, trigger, fix onregion', level=2)
_bd += sam.p(
    "Direction reconstructed to be in an onregion of fixed solid angle. "
    "Fade lines show entire field-of-view.",
    font_size=75,
    text_align='justify',
    font_family='calibri')
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0301_onregion_trigger_acceptance_plot',
        '{site_key:s}_{particle_key:s}_point_fix_onregion.jpg'
    )
)

_bd += sam.h('Effective acceptance, trigger, fix onregion', level=2)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0301_onregion_trigger_acceptance_plot',
        '{site_key:s}_{particle_key:s}_diffuse_fix_onregion.jpg'
    )
)

_bd += sam.h('Differential trigger-rates, fix onregion', level=2)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0325_onregion_trigger_rates_for_cosmic_rays_plot',
        '{site_key:s}_differential_trigger_rates_in_onregion.jpg'
    )
)

_bd += sam.h('Braodband-sensitivity, trigger, fix onregion', level=2)
_bd += sam.p(
    "A.k.a integral spectral exclusion zone. Only on trigger-level.\n"
    "Figures have same information, "
    "but different styles of Spectral-Energy-Distributions (SEDs). ",
    font_size=75,
    text_align='justify',
    font_family='calibri'
)
_bd += sam.code(
    json.dumps(sum_config['on_off_measuremnent'], indent=4),
    font_size=50,
    line_height=100,
)
'''
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0330_onregion_trigger_integral_spectral_exclusion_zone_plot',
        '{site_key:s}_integral_spectral_exclusion_zone_style_plenoirf.jpg'
    )
)
'''
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0330_onregion_trigger_integral_spectral_exclusion_zone_plot',
        '{site_key:s}_integral_spectral_exclusion_zone_style_science.jpg'
    )
)
'''
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0330_onregion_trigger_integral_spectral_exclusion_zone_plot',
        '{site_key:s}_integral_spectral_exclusion_zone_style_fermi.jpg'
    )
)
'''

_bd += sam.h('Braodband-sensitivity, (phd-thesis)', level=2)
_bd += sam.p(
    "Gamma acceptance off all field-of-view multiplied by 0.68. "
    "Cosmic-rays are only leptons and proton. "
    "Cosmic-ray acceptance is taken from all field-of-view and scaled by "
    "the ratio "
    "of solid angles between all field-of-view and onregion. "
    "Onregion opening angle is 0.31 deg.",
    font_size=75,
    text_align='justify',
    font_family='calibri'
)
_bd += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0340_trigger_integral_spectral_exclusion_zone_as_in_phd',
        '{site_key:s}_integral_spectral_exclusion_zone_style_science.jpg'
    )
)

_bd += sam.h('Magnetic deflection in atmosphere', level=2)
_bd += sam.h('primary azimuth', level=4)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '..',
        'magnetic_deflection',
        'control_figures',
        '{site_key:s}_{particle_key:s}_primary_azimuth_deg.jpg'
    )
)
_bd += sam.h('primary zenith', level=4)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '..',
        'magnetic_deflection',
        'control_figures',
        '{site_key:s}_{particle_key:s}_primary_zenith_deg.jpg'
    )
)
_bd += sam.h('cherenkov-pool x', level=4)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        pa['run_dir'],
        'magnetic_deflection',
        'control_figures',
        '{site_key:s}_{particle_key:s}_cherenkov_pool_x_m.jpg'
    )
)
_bd += sam.h('cherenkov-pool y', level=4)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '..',
        'magnetic_deflection',
        'control_figures',
        '{site_key:s}_{particle_key:s}_cherenkov_pool_y_m.jpg'
    )
)


_bd += sam.h('Runtime', level=2)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0910_runtime',
        '{site_key:s}_{particle_key:s}_relative_runtime.jpg'
    )
)
_bd += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0910_runtime',
        '{site_key:s}_{particle_key:s}_speed_runtime.jpg'
    )
)

_bd += sam.h('Configurations', level=2)
_bd += sam.h('Plenoscope-scenery', level=3)
_bd += sam.code(
    json.dumps(irf_config['plenoscope_scenery'], indent=4),
    font_size=50,
    line_height=100
)
_bd += sam.h('Plenoscope read-out, and night-sky-background', level=3)
_bd += sam.code(
    json.dumps(irf_config['merlict_propagation_config'], indent=4),
    font_size=50,
    line_height=100
)
_bd += sam.h('Sites and particles', level=3)
_bd += sam.code(
    json.dumps(irf_config['config'], indent=4),
    font_size=50,
    line_height=100
)

_bd += sam.h('Provenance', level=2)
_bd += sam.h('Populating event-tables', level=3)
_bd += sam.code(
    json.dumps(evt_provenance, indent=4),
    font_size=50,
    line_height=100
)

_bd += sam.h('Running analysis', level=3)
_bd += sam.code(
    json.dumps(ana_provenance, indent=4),
    font_size=50,
    line_height=100
)

with open(opj(pa['summary_dir'], 'index.html'), 'wt') as fout:
    fout.write(_html.render())

production_name = pa['run_dir']
if production_name[-1] == '/':
    production_name = os.path.dirname(production_name)

weasyprint.HTML(opj(pa['summary_dir'], 'index.html')).write_pdf(
    opj(pa['summary_dir'], '{:s}.pdf'.format(production_name)))
