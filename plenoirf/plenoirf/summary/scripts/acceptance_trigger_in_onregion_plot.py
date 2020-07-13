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

trigger_thresholds = sum_config['trigger']['ratescan_thresholds_pe']
trigger_threshold = sum_config['trigger']['threshold_pe']
idx_trigger_threshold = np.where(
    np.array(trigger_thresholds) == trigger_threshold,
)[0][0]
assert trigger_threshold in trigger_thresholds

# trigger
# -------
acceptance_trigger_path = os.path.join(
    pa['summary_dir'],
    "acceptance_trigger",
    "acceptance_trigger.json"
)
with open(acceptance_trigger_path, 'rt') as f:
    A = json.loads(f.read())
A_energy_bin_edges = np.array(A['energy_bin_edges']['value'])
assert A['energy_bin_edges']['unit'] == "GeV"

# trigger detection
# -----------------
acceptance_trigger_in_onregion_path = os.path.join(
    pa['summary_dir'],
    "acceptance_trigger_in_onregion",
    "acceptance_trigger_in_onregion.json"
)
with open(acceptance_trigger_in_onregion_path, 'rt') as f:
    G = json.loads(f.read())
G_energy_bin_edges =  np.array(G['energy_bin_edges']['value'])


ylim = [1e1, 1e6]

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

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

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        for source_key in sources:

            acceptance_trigger = np.array(A[
                'cosmic_response'][
                site_key][
                particle_key][
                source_key][
                'value'][
                idx_trigger_threshold])
            acceptance_trigger_unc = np.array(A[
                'cosmic_response'][
                site_key][
                particle_key][
                source_key][
                'relative_uncertainty'][
                idx_trigger_threshold])

            acceptance_trigger_onregion = np.array(G[
                'cosmic_response'][
                site_key][
                particle_key][
                source_key][
                'value'])
            acceptance_trigger_onregion_unc = np.array(G[
                'cosmic_response'][
                site_key][
                particle_key][
                source_key][
                'relative_uncertainty'])

            fig = irf.summary.figure.figure(fig_16_by_9)
            ax = fig.add_axes((.1, .1, .8, .8))

            irf.summary.figure.ax_add_hist(
                ax=ax,
                bin_edges=A_energy_bin_edges,
                bincounts=acceptance_trigger,
                linestyle='gray',
                bincounts_upper=acceptance_trigger*(1 + acceptance_trigger_unc),
                bincounts_lower=acceptance_trigger*(1 - acceptance_trigger_unc),
                face_color=particle_colors[particle_key],
                face_alpha=0.05,
            )
            irf.summary.figure.ax_add_hist(
                ax=ax,
                bin_edges=G_energy_bin_edges,
                bincounts=acceptance_trigger_onregion,
                linestyle=particle_colors[particle_key],
                bincounts_upper=acceptance_trigger_onregion*(1 + acceptance_trigger_onregion_unc),
                bincounts_lower=acceptance_trigger_onregion*(1 - acceptance_trigger_onregion_unc),
                face_color=particle_colors[particle_key],
                face_alpha=0.25,
            )

            ax.set_xlabel('energy / GeV')
            ax.set_ylabel(
                sources[source_key]['label']
                + ' / ' +
                sources[source_key]['unit']
            )
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_ylim(sources[source_key]['limits'])
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            ax.loglog()
            ax.set_xlim([A_energy_bin_edges[0], A_energy_bin_edges[-1]])
            fig.savefig(
                os.path.join(
                    pa['out_dir'],
                    '{:s}_{:s}_{:s}_onregion.jpg'.format(
                        site_key,
                        particle_key,
                        source_key,
                    )
                )
            )
            plt.close(fig)
