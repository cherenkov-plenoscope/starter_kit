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

num_size_bins = 12
size_bin_edges = np.geomspace(1, 2**num_size_bins, (3*num_size_bins)+1)

trigger_modus = sum_config["trigger"]["modus"]
trigger_threshold = sum_config['trigger']['threshold_pe']

fig_16_by_9 = sum_config['plot']['16_by_9']

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        site_particle_dir = opj(pa['out_dir'], site_key ,particle_key)
        os.makedirs(site_particle_dir, exist_ok=True)
        site_particle_prefix = '{:s}_{:s}'.format(site_key, particle_key)

        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'
            ),
            structure=irf.table.STRUCTURE
        )

        key = "trigger_probability_vs_cherenkov_size"

        idx_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=event_table['trigger'],
            threshold=trigger_threshold,
            modus=trigger_modus,
        )
        mask_pasttrigger = spt.make_mask_of_right_in_left(
            left_indices=event_table['trigger'][spt.IDX],
            right_indices=idx_pasttrigger
        )

        num_thrown = np.histogram(
            event_table['trigger']['num_cherenkov_pe'],
            bins=size_bin_edges
        )[0]

        num_pasttrigger = np.histogram(
            event_table['trigger']['num_cherenkov_pe'],
            bins=size_bin_edges,
            weights=mask_pasttrigger
        )[0]

        trigger_probability = irf.analysis.effective_quantity._divide_silent(
            numerator=num_pasttrigger,
            denominator=num_thrown,
            default=np.nan
        )

        trigger_probability_unc = \
            irf.analysis.effective_quantity._divide_silent(
                numerator=np.sqrt(num_pasttrigger),
                denominator=num_pasttrigger,
                default=np.nan
            )

        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=size_bin_edges,
            bincounts=trigger_probability,
            linestyle='k-',
            bincounts_upper=trigger_probability*(1+trigger_probability_unc),
            bincounts_lower=trigger_probability*(1-trigger_probability_unc),
            face_color='k',
            face_alpha=.3)
        ax.semilogx()
        ax.semilogy()
        ax.set_ylim([1e-6, 1.5e-0])
        ax.set_xlabel('true Cherenkov-size / p.e.')
        ax.set_ylabel('trigger probability / 1')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        fig.savefig(opj(pa['out_dir'], site_particle_prefix+"_"+key+".jpg"))
        plt.close(fig)

        irf.json_numpy.write(
            os.path.join(site_particle_dir, key+".json"),
            {
                "true_Cherenkov_size_bin_edges_pe": size_bin_edges,
                "unit": "1",
                "mean": trigger_probability,
                "relative_uncertainty": trigger_probability_unc,
            }
        )
