#!/usr/bin/python
import sys

import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json
import sparse_table as spt

import plenoirf as irf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
run_dir = argv[1]
summary_dir = os.path.join(run_dir, 'summary')

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)

cfg = irf_config['config']


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
    fig = irf.summary.figure.figure(sum_config['figure_16_9'])
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
    ax.set_title(title)
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


for site_key in cfg['sites']:
    for particle_key in cfg['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        # read
        # ----
        event_table = spt.read(
            path=os.path.join(
                run_dir,
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)

        # summarize
        # ---------
        energy_bin_edges = np.geomspace(
            np.min(sum_config['energy_bin_edges_GeV']),
            np.max(sum_config['energy_bin_edges_GeV']),
            5)

        pasttrigger_mask = spt.make_mask_of_right_in_left(
                left_indices=event_table['primary'][spt.IDX],
                right_indices=event_table['pasttrigger'][spt.IDX])

        prm = event_table['primary']
        prm_dirs = np.zeros((prm.shape[0], 3))
        prm_dirs[:, 0] = np.sin(prm['zenith_rad'])*np.cos(prm['azimuth_rad'])
        prm_dirs[:, 1] = np.sin(prm['zenith_rad'])*np.sin(prm['azimuth_rad'])
        prm_dirs[:, 2] = np.cos(prm['zenith_rad'])

        ple_az = np.deg2rad(cfg['plenoscope_pointing']['azimuth_deg'])
        ple_zd = np.deg2rad(cfg['plenoscope_pointing']['zenith_deg'])
        plenoscope_dir = np.zeros(3)
        plenoscope_dir[0] = np.sin(ple_zd)*np.cos(ple_az)
        plenoscope_dir[1] = np.sin(ple_zd)*np.sin(ple_az)
        plenoscope_dir[2] = np.cos(ple_zd)


        offaxis = irf.grid._make_angle_between(
            directions=prm_dirs,
            direction=plenoscope_dir)

        offaxis_deg = np.rad2deg(offaxis)
        offaxis2_deg2 = offaxis_deg**2

        num_cradial_bins = int(0.1*np.sqrt(offaxis_deg.shape[0]))
        max_cradial_deg = np.max(offaxis_deg)
        max_cradial2_deg2 = np.ceil(max_cradial_deg**2)

        plenoscope_fov_radius_deg = .5*irf_config[
            'light_field_sensor_geometry']['max_FoV_diameter_deg']

        cradial2_bin_edges_deg2 = np.linspace(
            0,
            max_cradial2_deg2,
            num_cradial_bins+1)

        # all energy
        # ----------
        res = histogram(
            cradial2_bin_edges_deg2=cradial2_bin_edges_deg2,
            offaxis2_deg2=offaxis2_deg2,
            energy_mask=np.ones(offaxis2_deg2.shape[0]),
            pasttrigger_mask=pasttrigger_mask)

        write_figure(
            path=opj(
                summary_dir,
                '{:s}_trigger_probability_vs_offaxis.{:s}'.format(
                    prefix_str,
                    sum_config['figure_16_9']['format'])),
            cradial2_bin_edges_deg2=cradial2_bin_edges_deg2,
            trgprb=res['trgprb'],
            trgprb_absunc=res['trgprb_absunc'],
            ylim=res['ylim'],
            figure_config=sum_config['figure_16_9'],
            title='')

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

        ylims = []
        for ex in range(num_energy_bins):
            if energy_hists[ex]['ylim'] is not None:
                ylims.append(energy_hists[ex]['ylim'])
        if len(ylims) == 0:
            global_ylim = None
        else:
            ylims = np.array(ylims)
            global_ylim = [np.min(ylims[:, 0]), np.max(ylims[:, 1])]

        for ex in range(num_energy_bins):
            write_figure(
                path=opj(
                    summary_dir,
                    '{:s}_trigger_probability_vs_offaxis_{:06d}.{:s}'.format(
                        prefix_str,
                        ex,
                        sum_config['figure_16_9']['format'])),
                cradial2_bin_edges_deg2=coarse_cradial2_bin_edges_deg2,
                trgprb=energy_hists[ex]['trgprb'],
                trgprb_absunc=energy_hists[ex]['trgprb_absunc'],
                ylim=global_ylim,
                figure_config=sum_config['figure_16_9'],
                title='energy {:.1f} - {:.1f} GeV, num. events {:d}'.format(
                    energy_bin_edges[ex],
                    energy_bin_edges[ex+1],
                    energy_hists[ex]['num_energy_bin_pasttrigger']))
