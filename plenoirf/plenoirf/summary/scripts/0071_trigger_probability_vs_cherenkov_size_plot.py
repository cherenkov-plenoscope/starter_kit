#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

key = 'trigger_probability_vs_cherenkov_size'

trigger_vs_size = irf.json_numpy.read_tree(
    os.path.join(pa['summary_dir'], "0070_"+key)
)

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        site_particle_prefix = '{:s}_{:s}'.format(site_key, particle_key)

        # each particle alone
        # -------------------
        size_bin_edges = np.array(trigger_vs_size[
            site_key][
            particle_key][
            'trigger_probability_vs_cherenkov_size'][
            'true_Cherenkov_size_bin_edges_pe'])

        prob = np.array(trigger_vs_size[
            site_key][
            particle_key][
            'trigger_probability_vs_cherenkov_size'][
            'mean'])
        prob_unc = np.array(trigger_vs_size[
            site_key][
            particle_key][
            'trigger_probability_vs_cherenkov_size'][
            'relative_uncertainty'])

        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=size_bin_edges,
            bincounts=prob,
            linestyle='k-',
            bincounts_upper=prob*(1+prob_unc),
            bincounts_lower=prob*(1-prob_unc),
            face_color='k',
            face_alpha=.3)
        ax.semilogx()
        ax.semilogy()
        ax.set_ylim([1e-6, 1.5e-0])
        ax.set_xlabel('true Cherenkov-size / p.e.')
        ax.set_ylabel('trigger-probability / 1')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        fig.savefig(opj(pa['out_dir'], site_particle_prefix+"_"+key+".jpg"))
        plt.close(fig)

    # all particles together
    # ----------------------
    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    text_y = 0
    for particle_key in irf_config['config']['particles']:

        size_bin_edges = np.array(trigger_vs_size[
            site_key][
            particle_key][
            'trigger_probability_vs_cherenkov_size'][
            'true_Cherenkov_size_bin_edges_pe'])

        prob = np.array(trigger_vs_size[
            site_key][
            particle_key][
            'trigger_probability_vs_cherenkov_size'][
            'mean'])
        prob_unc = np.array(trigger_vs_size[
            site_key][
            particle_key][
            'trigger_probability_vs_cherenkov_size'][
            'relative_uncertainty'])

        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=size_bin_edges,
            bincounts=prob,
            linestyle=particle_colors[particle_key],
            bincounts_upper=prob*(1+prob_unc),
            bincounts_lower=prob*(1-prob_unc),
            face_color=particle_colors[particle_key],
            face_alpha=.25
        )
        ax.text(
            0.9,
            0.1 + text_y,
            particle_key,
            color=particle_colors[particle_key],
            transform=ax.transAxes
        )
        text_y += 0.06
    ax.semilogx()
    ax.semilogy()
    ax.set_ylim([1e-6, 1.5e-0])
    ax.set_xlabel('true Cherenkov-size / p.e.')
    ax.set_ylabel('trigger-probability / 1')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    fig.savefig(opj(pa['out_dir'], site_key+"_"+key+".jpg"))
    plt.close(fig)
