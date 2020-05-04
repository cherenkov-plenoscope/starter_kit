#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import json
import weasyprint

argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
run_dir = argv[1]
summary_dir = os.path.join(run_dir, 'summary')

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)


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
    padding_px=0
):
    out = ''
    out += '<pre style="'
    out += 'font-size:{:.1f}%;font-family:{:s};line-height:{:.1f}%;'.format(
        font_size,
        font_family,
        line_height)
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


def make_site_particle_index_table(
    sites,
    particles,
    energy_bin_edges,
    wild_card='{site_key:s}_{particle_key:s}_key_{energy_bin_index:06d}.jpg',
    particle_width=100,
    header=True,
):
    site_width = len(particles)*particle_width
    table_width = len(sites)*site_width

    matrix = []
    if header:
        side_head_row = []
        for site_key in sites:
            side_head_row.append(h(site_key, level=3, text_align='center'))
        matrix.append(side_head_row)

        row = []
        for site_key in sites:
            sub_row = []
            for particle_key in particles:
                sub_row.append(h(particle_key, level=4, text_align='center'))
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
    "A summary of the Cherenkov-plenoscope's instrument-response-functions. "
    "This is created automatically. ",
    text_align='justify',
    font_family='calibri')

doc += h('Effective acceptance, diffuse source, past trigger', level=2)
doc += p(
    "Only events where the incident-direction is reconstructed to be within "
    "a suitable field-of-view for on-regions, i.e. max. 2.5deg off axis.",
    text_align='justify',
    font_family='calibri')
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='{site_key:s}_{particle_key:s}_diffuse_trigger.jpg')


doc += h('Effective area, ponit source, past trigger', level=2)
doc += p(
    "Only events where both the reconstructed incident-direction < 2.5deg"
    " off axis, "
    "AND the primary's incident-direction < 2.5deg off axis."
    "<br>Although technically applicable, this might be misleading for charged"
    " cosmic-rays. "
    "For low energetic cosmic-rays there is almost no intersection of the "
    "two directional criteria due to deflection of the airshower in earth's "
    "magnetig field. "
    "This leads to an underestimation of the effective area at low energies. "
    "However, this is not relevant for the estimation of the plenoscope's "
    "performance to detect point-sources of gamma-rays.",
    text_align='justify',
    font_family='calibri')
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='{site_key:s}_{particle_key:s}_point_trigger.jpg')


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
    energy_bin_edges=sum_config['energy_bin_edges_GeV_coarse'],
    wild_card='grid_direction/{site_key:s}_{particle_key:s}_'
    'grid_direction_pasttrigger_{energy_bin_index:06d}.jpg')

doc += h('Cherenkov-intensity on observation-level, past trigger', level=2)
doc += p(
    "Areal intensity of Cherenkov-photons on the observation-level. "
    "Only showing airshowers which passed the plenoscope's trigger. "
    "Color-coding shows the average intensity of a single airshower. ",
    text_align='justify',
    font_family='calibri')

doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=sum_config['energy_bin_edges_GeV_coarse'],
    wild_card='grid_area/'
    '{site_key:s}_{particle_key:s}_'
    'grid_area_pasttrigger_{energy_bin_index:06d}.jpg')


doc += h('Trigger-probability vs. Cherenkov-photon-equivalent (p.e.)', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='trigger_probability_vs_cherenkov_size/'
    '{site_key:s}_{particle_key:s}_'
    'trigger_probability_vs_cherenkov_size.jpg')


doc += h('Trigger-probability vs. offaxis-angle', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='trigger_probability_vs_offaxis/'
    '{site_key:s}_{particle_key:s}_'
    'trigger_probability_vs_offaxis.jpg')


doc += h('Trigger-probability vs. offaxis-angle vs. energy', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1, 2, 3, 4],
    wild_card='trigger_probability_vs_offaxis/'
    '{site_key:s}_{particle_key:s}_'
    'trigger_probability_vs_offaxis_{energy_bin_index:06d}.jpg')


doc += h('Cherenkov and night-sky-background classification', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='cherenkov_photon_classification/'
    '{site_key:s}_{particle_key:s}_'
    'cherenkovclassification_size_confusion.jpg')

doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='cherenkov_photon_classification/'
    '{site_key:s}_{particle_key:s}_'
    'cherenkovclassification_sensitivity_vs_true_energy.jpg')

doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='cherenkov_photon_classification/'
    '{site_key:s}_{particle_key:s}_'
    'cherenkovclassification_'
    'true_size_over_extracted_size_vs_true_energy.jpg')

doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='cherenkov_photon_classification/'
    '{site_key:s}_{particle_key:s}_'
    'cherenkovclassification_'
    'true_size_over_extracted_size_vs_true_size.jpg')


doc += h('Configurations', level=2)

doc += h('Plenoscope-scenery', level=3)
doc += code(
    json.dumps(irf_config['plenoscope_scenery'], indent=4),
    font_size=50,
    line_height=100)

doc += h('Plenoscope read-out, and night-sky-background', level=3)
doc += code(
    json.dumps(irf_config['merlict_propagation_config'], indent=4),
    font_size=50,
    line_height=100)

doc += h('Sites and particles', level=3)
doc += code(
    json.dumps(irf_config['config'], indent=4),
    font_size=50,
    line_height=100)

doc += h('Relative runtime', level=2)
doc += make_site_particle_index_table(
    sites=irf_config['config']['sites'],
    particles=irf_config['config']['particles'],
    energy_bin_edges=[0, 1],
    wild_card='{site_key:s}_{particle_key:s}_relative_runtime.jpg')

html = page('summary', doc)

with open(opj(summary_dir, 'index.html'), 'wt') as fout:
    fout.write(html)

weasyprint.HTML(opj(summary_dir, 'index.html')).write_pdf(
    opj(summary_dir, 'index.pdf'))
