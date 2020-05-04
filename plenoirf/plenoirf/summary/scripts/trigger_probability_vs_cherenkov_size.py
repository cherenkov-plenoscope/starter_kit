#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sparse_table as spt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)

        pasttrigger_mask = spt.make_mask_of_right_in_left(
            left_indices=event_table['trigger'][spt.IDX],
            right_indices=event_table['pasttrigger'][spt.IDX])

        num_bins = 12
        size_bin_edges = np.geomspace(1, 2**num_bins, num_bins+1)

        num_thrown = np.histogram(
            event_table['trigger']['num_cherenkov_pe'],
            bins=size_bin_edges)[0]

        num_pasttrigger = np.histogram(
            event_table['trigger']['num_cherenkov_pe'],
            bins=size_bin_edges,
            weights=pasttrigger_mask)[0]

        trigger_probability = num_pasttrigger/num_thrown
        trigger_probability_relunc = np.nan*np.ones(num_pasttrigger.shape[0])
        _v = num_pasttrigger > 0
        trigger_probability_relunc[_v] = np.sqrt(
            num_pasttrigger[_v])/num_pasttrigger[_v]
        trigger_probability_absunc = (
            trigger_probability_relunc*trigger_probability)

        trgprb = trigger_probability
        trgprb_absunc = trigger_probability_absunc

        fig = irf.summary.figure.figure(sum_config['figure_16_9'])
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=size_bin_edges,
            bincounts=trgprb,
            linestyle='k-',
            bincounts_upper=trgprb + trgprb_absunc,
            bincounts_lower=trgprb - trgprb_absunc,
            face_color='k',
            face_alpha=.3)
        ax.semilogx()
        ax.semilogy()
        ax.set_ylim([1e-6, 1e-0])
        ax.set_xlabel('Cherenkov-size / p.e.')
        ax.set_ylabel('trigger probability / 1')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        fig.savefig(
            opj(
                pa['out_dir'],
                '{:s}_trigger_probability_vs_cherenkov_size.{:s}'.format(
                    prefix_str,
                    sum_config['figure_16_9']['format'])))
        plt.close(fig)
