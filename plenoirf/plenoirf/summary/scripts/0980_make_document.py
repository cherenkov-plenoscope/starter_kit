#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import json
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

text_aligns = [
    'center',
    'left',
    'right',
    'justify',
]


def table(matrix, width=100, indent=4):
    off = ' '*indent
    out = ''
    out += '<table style="width:{:.1f}px;">\n'.format(float(width))
    for row in matrix:
        out += off+'<tr>\n'
        for column in row:
            out += 2*off+'<th>'+column+'</th>\n'
        out += off+'</tr>\n'
    out += '</table>'
    return out


def h(text, level=1, font_family='calibri', text_align='left'):
    return ''.join([
        '<h{:d} style="'.format(level),
        'font-family:{:s};'.format(font_family),
        'text-align:{:s}'.format(text_align),
        '">{:s}'.format(text),
        '</h{:d}>'.format(level),
    ])


def p(
    text,
    font_size=100,
    font_family='courier',
    text_align='left',
    line_height=100,
    width=640
):
    return ''.join([
        '<p style="',
        'font-size:{:.1f}%;'.format(float(font_size)),
        'font-family:{:s};'.format(font_family),
        'text-align:{:s};'.format(text_align),
        'width:{:.1f}px;'.format(float(width)),
        'line-height:{:.1f}%"'.format(float(line_height)),
        '>{:s}</p>'.format(text),
    ])


def image(path, width=100):
    return '<img src="{:s}"style="width:{:.1f}px;">'.format(path, width)


def code(
    text,
    font_family='courier',
    font_size=100,
    line_height=100,
    margin_px=0,
    padding_px=0,
    text_align='left'
):
    out = ''
    out += '<pre style="'
    out += 'font-size:{:.1f}%;'.format(font_size)
    out += 'font-family:{:s};'.format(font_family)
    out += 'line-height:{:.1f}%;'.format(line_height)
    out += 'text-align:{:s};'.format(text_align)
    out += 'margin:{:.1f}px;padding:{:.1f}px">'.format(
        margin_px,
        padding_px)
    out += '<code>'
    out += text
    out += '</code>'
    out += '</pre>'
    return out


def page(title, text):
    out = ''
    out += '<!DOCTYPE html>\n'
    out += '<html>\n'
    out += '<head>\n'
    out += '<title>{:s}</title>\n'.format(title)
    out += '</head>\n'
    out += '<body>\n'
    out += text
    out += '</body>\n'
    out += '</html>\n'
    return out


def make_site_table(
    sites,
    energy_bin_edges,
    wild_card='{site_key:s}_key_{energy_bin_index:06d}.jpg',
    site_width=FIGURE_WIDTH_PIXEL*4,
    header=True,
):
    table_width = len(sites)*site_width
    matrix = []

    if header:
        side_head_row = []
        for site_key in sites:
            side_head_row.append(h(site_key, level=5, text_align='center'))
        matrix.append(side_head_row)

    for energy_bin_index in range(len(energy_bin_edges) - 1):
        row = []
        for site_key in sites:
            sub_row = []
            path = wild_card.format(
                site_key=site_key,
                energy_bin_index=energy_bin_index
            )
            img = image(path=path, width=site_width)
            sub_row.append(img)
            row.append(table([sub_row], width=site_width))
        matrix.append(row)

    return table(matrix, width=table_width)


def make_site_particle_index_table(
    sites,
    particles,
    energy_bin_edges,
    wild_card='{site_key:s}_{particle_key:s}_key_{energy_bin_index:06d}.jpg',
    particle_width=FIGURE_WIDTH_PIXEL,
    header=True,
):
    site_width = len(particles)*particle_width
    table_width = len(sites)*site_width

    matrix = []
    if header:
        side_head_row = []
        for site_key in sites:
            side_head_row.append(h(site_key, level=5, text_align='center'))
        matrix.append(side_head_row)

        row = []
        for site_key in sites:
            sub_row = []
            for particle_key in particles:
                sub_row.append(h(particle_key, level=6, text_align='center'))
            row.append(table([sub_row], width=site_width))
        matrix.append(row)

    for energy_bin_index in range(len(energy_bin_edges) - 1):
        row = []
        for site_key in sites:
            sub_row = []
            for particle_key in particles:
                path = wild_card.format(
                    site_key=site_key,
                    particle_key=particle_key,
                    energy_bin_index=energy_bin_index)
                img = image(path=path, width=particle_width)
                sub_row.append(img)
            row.append(table([sub_row], width=site_width))
        matrix.append(row)

    return table(matrix, width=table_width)


doc = ''
doc += h('Cherenkov-plenoscope', level=1)
doc += p(
    "Summarizing the Cherenkov-plenoscope's instrument-response-functions.",
    text_align='justify',
    font_family='calibri',
)


site_matrix = []
_row = []
for site_key in irf_config['config']['sites']:
    _row.append(
        h(site_key, level=5, text_align='center')
    )
site_matrix.append(_row)
_row = []
for site_key in irf_config['config']['sites']:
    _row.append(
        code(
            json.dumps(irf_config['config']['sites'][site_key], indent=4),
            font_size=50,
            line_height=100,
        ),
    )
site_matrix.append(_row)

doc += table(
    site_matrix,
    width=FIGURE_WIDTH_PIXEL*8
)

doc += h('Light-field-trigger', level=2)
_trigger_config = sum_config['trigger'].copy()
_trigger_config.pop('ratescan_thresholds_pe')
doc += code(
    json.dumps(_trigger_config, indent=4),
    font_size=50,
    line_height=100,
)

doc += h('Effective area, ponit source, trigger-level', level=2)
doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0101_trigger_acceptance_for_cosmic_particles_plot',
        '{site_key:s}_point.jpg'
    )
)
doc += h('Effective acceptance, diffuse source, trigger-level', level=2)
doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0101_trigger_acceptance_for_cosmic_particles_plot',
        '{site_key:s}_diffuse.jpg'
    )
)

doc += h('Flux of airshowers', level=2)
doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0050_flux_of_airshowers_plot',
        '{site_key:s}_airshower_differential_flux.jpg'
    )
)

doc += h('Trigger-ratescan', level=2)
doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0130_trigger_ratescan_plot',
        '{site_key:s}_ratescan.jpg'
    )
)

doc += h('Differential trigger-rates, entire field-of-view', level=2)
doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0106_trigger_rates_for_cosmic_particles_plot',
        '{site_key:s}_differential_trigger_rate.jpg'
    )
)

doc += h('Directions of primaries, past trigger', level=2)
doc += p(
    "Primary particle's incidend direction color-coded "
    "with their probability to trigger the plenoscope. "
    "Hatched solid angles are unknown. ",
    text_align='justify',
    font_family='calibri')
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        '0810_grid_direction_of_primaries_plot',
        '{site_key:s}_{particle_key:s}_'
        'grid_direction_pasttrigger_{energy_bin_index:06d}.jpg'
    )
)
doc += h('Cherenkov-intensity on observation-level, past trigger', level=2)
doc += p(
    "Areal intensity of Cherenkov-photons on the observation-level. "
    "Only showing airshowers which passed the plenoscope's trigger. "
    "Color-coding shows the average intensity of a single airshower. ",
    text_align='justify',
    font_family='calibri'
)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        '0815_grid_illumination_plot',
        '{site_key:s}_{particle_key:s}_'
        'grid_area_pasttrigger_{energy_bin_index:06d}.jpg'
    )
)
doc += h('Trigger-probability vs. true Cherenkov-size / p.e.', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '0070_trigger_probability_vs_cherenkov_size',
        '{site_key:s}_{particle_key:s}_'
        'trigger_probability_vs_cherenkov_size.jpg'
    )
)
doc += h('Trigger-probability vs. offaxis-angle', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'trigger_probability_vs_offaxis',
        '{site_key:s}_{particle_key:s}_trigger_probability_vs_offaxis.jpg'
    )
)
doc += h('Trigger-probability vs. offaxis-angle vs. energy', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        'trigger_probability_vs_offaxis',
        '{site_key:s}_{particle_key:s}_'
        'trigger_probability_vs_offaxis_{energy_bin_index:06d}.jpg'
    )
)
doc += h('Cherenkov and night-sky-background classification', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'cherenkov_photon_classification',
        '{site_key:s}_{particle_key:s}_'
        'cherenkovclassification_size_confusion.jpg'
    )
)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'cherenkov_photon_classification',
        '{site_key:s}_{particle_key:s}_'
        'cherenkovclassification_sensitivity_vs_true_energy.jpg'
    )
)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'cherenkov_photon_classification',
        '{site_key:s}_{particle_key:s}_'
        'cherenkovclassification_'
        'true_size_over_extracted_size_vs_true_energy.jpg'
    )
)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'cherenkov_photon_classification',
        '{site_key:s}_{particle_key:s}_'
        'cherenkovclassification_'
        'true_size_over_extracted_size_vs_true_size.jpg'
    )
)

doc += h('Direction-reconstruction', level=2)
doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'gamma_direction_reconstruction',
        '{site_key:s}_gamma_psf_radial.jpg'
    )
)

acceptance_trigger_in_onregion = irf.json_numpy.read_tree(opj(
    pa['summary_dir'],
    "0300_onregion_trigger_acceptance"
))

_row = []
for site_key in irf_config['config']['sites']:
    _onregion = acceptance_trigger_in_onregion[site_key]
    _row.append(
            code(
                json.dumps(_onregion, indent=4),
                font_size=50,
                line_height=100
            )
    )
doc += table(
    matrix=[_row],
    width=FIGURE_WIDTH_PIXEL*8
)

doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=energy_bin_edges_coarse,
    wild_card=opj(
        'gamma_direction_reconstruction',
        '{site_key:s}_gamma_{energy_bin_index:06d}_psf.jpg'
    )
)

doc += h('Effective area, trigger, reconstructed in on-region', level=2)
doc += p(
    "Fade lines show acceptance on trigger-level "
    "without direction-cut in on-region.",
    text_align='justify',
    font_family='calibri')
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'acceptance_trigger_in_onregion_plot',
        '{site_key:s}_{particle_key:s}_point_onregion.jpg'
    )
)

doc += h('Effective acceptance, trigger, reconstructed in on-region', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'acceptance_trigger_in_onregion_plot',
        '{site_key:s}_{particle_key:s}_diffuse_onregion.jpg'
    )
)

doc += h('Braodband-sensitivity, trigger', level=2)
doc += p(
    "A.k.a integral spectral exclusion zone. Only on trigger-level.",
    text_align='justify',
    font_family='calibri')

doc += code(
    json.dumps(sum_config['on_off_measuremnent'], indent=4),
    font_size=50,
    line_height=100,
)

doc += make_site_table(
    sites=irf_config['config']['sites'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'acceptance_trigger_in_onregion_rates',
        '{site_key:s}_integral_spectral_exclusion_zone.jpg'
    )
)

doc += h('Magnetic deflection in atmosphere', level=2)
doc += h('primary azimuth', level=4)
doc += make_site_particle_index_table(
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
doc += h('primary zenith', level=4)
doc += make_site_particle_index_table(
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
doc += h('cherenkov-pool x', level=4)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        '..',
        'magnetic_deflection',
        'control_figures',
        '{site_key:s}_{particle_key:s}_cherenkov_pool_x_m.jpg'
    )
)
doc += h('cherenkov-pool y', level=4)
doc += make_site_particle_index_table(
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


doc += h('Runtime', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'runtime',
        '{site_key:s}_{particle_key:s}_relative_runtime.jpg'
    )
)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card=opj(
        'runtime',
        '{site_key:s}_{particle_key:s}_speed_runtime.jpg'
    )
)

doc += h('Configurations', level=2)
doc += h('Plenoscope-scenery', level=3)
doc += code(
    json.dumps(irf_config['plenoscope_scenery'], indent=4),
    font_size=50,
    line_height=100
)
doc += h('Plenoscope read-out, and night-sky-background', level=3)
doc += code(
    json.dumps(irf_config['merlict_propagation_config'], indent=4),
    font_size=50,
    line_height=100
)
doc += h('Sites and particles', level=3)
doc += code(
    json.dumps(irf_config['config'], indent=4),
    font_size=50,
    line_height=100
)

html = page('summary', doc)

with open(opj(pa['summary_dir'], 'index.html'), 'wt') as fout:
    fout.write(html)

production_name = pa['run_dir']
if production_name[-1] == '/':
    production_name = os.path.dirname(production_name)

weasyprint.HTML(opj(pa['summary_dir'], 'index.html')).write_pdf(
    opj(pa['summary_dir'], '{:s}.pdf'.format(production_name)))
