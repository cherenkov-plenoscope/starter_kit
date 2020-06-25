#!/usr/bin/python
import sys
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

trigger_thresholds = np.array(sum_config['trigger_thresholds_pe'])
nominal_trigger_threshold_idx = sum_config['nominal_trigger_threshold_idx']
nominal_trigger_threshold = trigger_thresholds[nominal_trigger_threshold_idx]

sources = {
    'diffuse': {
        'label': 'acceptance',
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
tt = nominal_trigger_threshold_idx

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        for source_key in sources:

            fig = irf.summary.figure.figure(sum_config['figure_16_9'])
            ax = fig.add_axes((.1, .1, .8, .8))

            Q = np.array(
                cr[site_key][particle_key][source_key]['value'][tt]
            )
            delta_Q = np.array(
                cr[site_key][particle_key][source_key]['relative_uncertainty'][tt]
            )
            Q_lower = (1 - delta_Q)*Q
            Q_upper = (1 + delta_Q)*Q

            if (
                site_key == 'namibia' and
                particle_key == 'gamma' and
                source_key == 'point'
            ):
                hess_ct5_area_path = os.path.join(
                    pa['summary_dir'],
                    'hess_ct5_gamma_area.json'
                )
                if os.path.exists(hess_ct5_area_path):
                    with open(hess_ct5_area_path, 'rt') as f:
                        hess_hess_ct5_area = json.loads(f.read())
                    ax.plot(
                        hess_hess_ct5_area["energy_GeV"],
                        hess_hess_ct5_area["area_m2"],
                        'r-',
                    )

            irf.summary.figure.ax_add_hist(
                ax=ax,
                bin_edges=energy_bin_edges,
                bincounts=Q,
                linestyle='k-',
                bincounts_upper=Q_upper,
                bincounts_lower=Q_lower,
                face_color='k',
                face_alpha=0.25
            )

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
            fig.savefig(
                os.path.join(
                    pa['out_dir'],
                    '{:s}_{:s}_{:s}.png'.format(
                        site_key,
                        particle_key,
                        source_key)))
            plt.close(fig)
