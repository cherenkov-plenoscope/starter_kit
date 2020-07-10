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
energy_bin_edges = np.array(A['energy_bin_edges']['value'])
assert A['energy_bin_edges']['unit'] == "GeV"

# trigger detection
# -----------------
gamma_detection_on_trigger_level_path = os.path.join(
    pa['summary_dir'],
    "gamma_detection_on_trigger_level",
    "gamma_detection_on_trigger_level.json"
)
with open(gamma_detection_on_trigger_level_path, 'rt') as f:
    G = json.loads(f.read())


ylim = [1e1, 1e6]

cr = A['cosmic_response']

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

for site_key in irf_config['config']['sites']:

    area_trigger = np.array(A[
        'cosmic_response'][
        site_key][
        'gamma'][
        'point'][
        'value'][
        idx_trigger_threshold])
    area_trigger_unc = np.array(A[
        'cosmic_response'][
        site_key][
        'gamma'][
        'point'][
        'relative_uncertainty'][
        idx_trigger_threshold])

    area_trigger_detection = np.array(G[
        site_key][
        'point'][
        'value'])
    area_trigger_detection_unc = np.array(G[
        site_key][
        'point'][
        'relative_uncertainty'])


    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((.1, .1, .8, .8))

    irf.summary.figure.ax_add_hist(
        ax=ax,
        bin_edges=energy_bin_edges,
        bincounts=area_trigger,
        linestyle='gray',
        bincounts_upper=area_trigger*(1 + area_trigger_unc),
        bincounts_lower=area_trigger*(1 - area_trigger_unc),
        face_color=particle_colors['gamma'],
        face_alpha=0.05,
    )
    irf.summary.figure.ax_add_hist(
        ax=ax,
        bin_edges=energy_bin_edges,
        bincounts=area_trigger_detection,
        linestyle=particle_colors['gamma'],
        bincounts_upper=area_trigger_detection*(1 + area_trigger_detection_unc),
        bincounts_lower=area_trigger_detection*(1 - area_trigger_detection_unc),
        face_color=particle_colors['gamma'],
        face_alpha=0.25,
    )

    ax.set_xlabel('energy / GeV')
    ax.set_ylabel('area / m$^{2}$')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_ylim(ylim)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.loglog()
    ax.set_xlim([energy_bin_edges[0], energy_bin_edges[-1]])
    fig.savefig(
        os.path.join(
            pa['out_dir'],
            '{:s}_{:s}.jpg'.format(
                site_key,
                'gamma',
            )
        )
    )
    plt.close(fig)
