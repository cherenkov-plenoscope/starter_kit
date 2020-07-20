#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_table as spt
import plenoirf as irf
import magnetic_deflection as mdfl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

trigger_modus = sum_config["trigger"]["modus"]
trigger_threshold = sum_config['trigger']['threshold_pe']

energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins']['point_spread_function'] + 1,
)
fig_16_by_9 = sum_config['plot']['16_by_9']


def histogram(
    cradial2_bin_edges_deg2,
    offaxis2_deg2,
    energy_mask,
    pasttrigger_mask,
):
    num_cradial_bins = len(cradial2_bin_edges_deg2) - 1
    num_thrown = np.histogram(
        offaxis2_deg2,
        weights=energy_mask,
        bins=cradial2_bin_edges_deg2)[0]

    energy_pasttrigger_mask = np.logical_and(
        energy_mask,
        pasttrigger_mask).astype(np.int)
    num_pasttrigger = np.histogram(
        offaxis2_deg2,
        weights=energy_pasttrigger_mask,
        bins=cradial2_bin_edges_deg2)[0]

    num_pasttrigger_relunc = np.nan*np.ones(num_cradial_bins)
    for bb in range(num_cradial_bins):
        if num_pasttrigger[bb] > 0:
            num_pasttrigger_relunc[bb] = (
                np.sqrt(num_pasttrigger[bb])/num_pasttrigger[bb])

    trgprb = np.nan*np.ones(num_cradial_bins)
    for bb in range(num_cradial_bins):
        if num_thrown[bb] > 0:
            trgprb[bb] = num_pasttrigger[bb]/num_thrown[bb]
    trgprb_absunc = trgprb*num_pasttrigger_relunc

    _up = trgprb+trgprb_absunc
    _up_valid = np.logical_not(np.isnan(_up))
    _lo = trgprb-trgprb_absunc
    _lo_valid = np.logical_not(np.isnan(_lo))
    if np.sum(_lo_valid) and np.sum(_up_valid):
        ymin = np.min(_lo[_lo_valid])
        if ymin < 0.:
            ymin = np.min(trgprb[_lo_valid])
        ylim = [ymin, np.max(_up[_up_valid])]
    else:
        ylim = None
    return {
        'num_thrown': num_thrown,
        'num_pasttrigger': num_pasttrigger,
        'num_pasttrigger_relunc': num_pasttrigger_relunc,
        'trgprb': trgprb,
        'trgprb_absunc': trgprb_absunc,
        'ylim': ylim,
        'num_energy_bin_pasttrigger': np.sum(energy_pasttrigger_mask)}


def write_figure(
    path,
    cradial2_bin_edges_deg2,
    trgprb,
    trgprb_absunc,
    ylim,
    figure_config,
    title,
):
    fig = irf.summary.figure.figure(figure_config)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    irf.summary.figure.ax_add_hist(
        ax=ax,
        bin_edges=cradial2_bin_edges_deg2,
        bincounts=trgprb,
        linestyle='k-',
        bincounts_upper=trgprb + trgprb_absunc,
        bincounts_lower=trgprb - trgprb_absunc,
        face_color='k',
        face_alpha=.3)
    ax.set_title(title, family='monospace')
    ax.semilogy()
    ax.set_xlabel('angle between(pointing, primary)$^2$ / deg$^2$')
    ax.set_ylabel('trigger-probability / 1')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_xlim([0, np.max(cradial2_bin_edges_deg2)])
    ax.set_ylim(ylim)
    fig.savefig(path)
    plt.close(fig)


os.makedirs(pa['out_dir'], exist_ok=True)

GLOBAL_YLIM = [1e-4, 1.5e0]

pointing_azimuth_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'azimuth_deg']
pointing_zenith_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'zenith_deg']

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        # read
        # ----
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

        # summarize
        # ---------
        idx_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=event_table['trigger'],
            threshold=trigger_threshold,
            modus=trigger_modus,
        )
        pasttrigger_mask = spt.make_mask_of_right_in_left(
            left_indices=event_table['primary'][spt.IDX],
            right_indices=idx_pasttrigger,
        )

        offaxis_deg = mdfl.discovery._angle_between_az_zd_deg(
            az1_deg=np.rad2deg(event_table['primary']['azimuth_rad']),
            zd1_deg=np.rad2deg(event_table['primary']['zenith_rad']),
            az2_deg=pointing_azimuth_deg,
            zd2_deg=pointing_zenith_deg
        )
        offaxis2_deg2 = offaxis_deg**2

        num_cradial_bins = int(0.1*np.sqrt(offaxis_deg.shape[0]))
        max_cradial_deg = np.max(offaxis_deg)
        max_cradial2_deg2 = np.ceil(max_cradial_deg**2)

        cradial2_bin_edges_deg2 = np.linspace(
            0,
            max_cradial2_deg2,
            num_cradial_bins+1
        )

        # all energy
        # ----------
        res = histogram(
            cradial2_bin_edges_deg2=cradial2_bin_edges_deg2,
            offaxis2_deg2=offaxis2_deg2,
            energy_mask=np.ones(offaxis2_deg2.shape[0]),
            pasttrigger_mask=pasttrigger_mask)

        write_figure(
            path=opj(
                pa['out_dir'],
                '{:s}_trigger_probability_vs_offaxis.{:s}'.format(
                    prefix_str,
                    fig_16_by_9['format'])),
            cradial2_bin_edges_deg2=cradial2_bin_edges_deg2,
            trgprb=res['trgprb'],
            trgprb_absunc=res['trgprb_absunc'],
            ylim=GLOBAL_YLIM,
            figure_config=fig_16_by_9,
            title=''
        )

        # versus energy
        # -------------
        num_energy_bins = len(energy_bin_edges) - 1
        coarse_cradial2_bin_edges_deg2 = np.linspace(
            0,
            max_cradial2_deg2,
            (num_cradial_bins//num_energy_bins)+1)

        energy_hists = []
        for ex in range(num_energy_bins):
            energy_bin_maks = np.logical_and(
                event_table['primary']['energy_GeV'] >= energy_bin_edges[ex],
                event_table['primary']['energy_GeV'] < energy_bin_edges[ex+1])
            energy_bin_maks = energy_bin_maks.astype(np.int)

            res = histogram(
                cradial2_bin_edges_deg2=coarse_cradial2_bin_edges_deg2,
                offaxis2_deg2=offaxis2_deg2,
                energy_mask=energy_bin_maks,
                pasttrigger_mask=pasttrigger_mask)
            energy_hists.append(res)

        for ex in range(num_energy_bins):
            write_figure(
                path=opj(
                    pa['out_dir'],
                    '{:s}_trigger_probability_vs_offaxis_{:06d}.{:s}'.format(
                        prefix_str,
                        ex,
                        fig_16_by_9['format']
                    )
                ),
                cradial2_bin_edges_deg2=coarse_cradial2_bin_edges_deg2,
                trgprb=energy_hists[ex]['trgprb'],
                trgprb_absunc=energy_hists[ex]['trgprb_absunc'],
                ylim=GLOBAL_YLIM,
                figure_config=fig_16_by_9,
                title=(
                    'energy {: 7.1f} - {: 7.1f} GeV, ' +
                    'num. events {: 6d}'
                    ).format(
                        energy_bin_edges[ex],
                        energy_bin_edges[ex+1],
                        energy_hists[ex]['num_energy_bin_pasttrigger']
                    )
            )
