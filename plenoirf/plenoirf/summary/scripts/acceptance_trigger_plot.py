#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_table as spt
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

acceptance_trigger_path = os.path.join(
    pa['summary_dir'],
    "acceptance_trigger",
    "acceptance_trigger.json"
)
with open(acceptance_trigger_path, 'rt') as f:
    A = json.loads(f.read())

energy_bin_edges = np.array(A['energy_bin_edges']['value'])
assert A['energy_bin_edges']['unit'] == "GeV"

trigger_thresholds = np.array(A['trigger_thresholds']['value'])
assert A['trigger_thresholds']['unit'] == "p.e."

trigger_threshold = sum_config['trigger']['threshold_pe']
idx_trigger_threshold = np.where(
    np.array(trigger_thresholds)==trigger_threshold,
)[0][0]
assert trigger_threshold in trigger_thresholds

sources = {
    'diffuse': {
        'label': 'area $\\times$ solid angle',
        'unit': "m$^{2}$ sr",
        'limits': [1e-1, 1e5],
    },
    'point': {
        'label': 'area',
        'unit': "m$^{2}$",
        'limits': [1e1, 1e6],
    }
}

cr = A['cosmic_response']

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

for site_key in irf_config['config']['sites']:
    for source_key in sources:
        for tt in range(len(trigger_thresholds)):

            fig = irf.summary.figure.figure(fig_16_by_9)
            ax = fig.add_axes((.1, .1, .8, .8))

            text_y = 0
            for particle_key in irf_config['config']['particles']:

                Q = np.array(
                    cr[
                        site_key][
                        particle_key][
                        source_key][
                        'value'][
                        tt]
                )
                delta_Q = np.array(
                    cr[
                        site_key][
                        particle_key][
                        source_key][
                        'relative_uncertainty'][
                        tt]
                )
                Q_lower = (1 - delta_Q)*Q
                Q_upper = (1 + delta_Q)*Q

                irf.summary.figure.ax_add_hist(
                    ax=ax,
                    bin_edges=energy_bin_edges,
                    bincounts=Q,
                    linestyle=particle_colors[particle_key],
                    bincounts_upper=Q_upper,
                    bincounts_lower=Q_lower,
                    face_color=particle_colors[particle_key],
                    face_alpha=0.25,
                )

                ax.text(
                    0.9,
                    0.1 + text_y,
                    particle_key,
                    color=particle_colors[particle_key],
                    transform=ax.transAxes
                )
                text_y += 0.06


            ax.set_xlabel('energy / GeV')
            ax.set_ylabel('{:s} / {:s}'.format(
                    sources[source_key]['label'],
                    sources[source_key]['unit']
                )
            )
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_ylim(sources[source_key]['limits'])
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            ax.loglog()
            ax.set_xlim([energy_bin_edges[0], energy_bin_edges[-1]])

            if tt == idx_trigger_threshold:
                fig.savefig(
                    os.path.join(
                        pa['out_dir'],
                        '{:s}_{:s}.jpg'.format(
                            site_key,
                            source_key,
                        )
                    )
                )
            ax.set_title(
                'threshold: {:d}p.e.'.format(trigger_thresholds[tt])
            )
            fig.savefig(
                os.path.join(
                    pa['out_dir'],
                    '{:s}_{:s}_{:06d}.jpg'.format(
                        site_key,
                        source_key,
                        tt,
                    )
                )
            )
            plt.close(fig)
