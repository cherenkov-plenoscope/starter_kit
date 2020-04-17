#!/usr/bin/python
import sys
import numpy as np
import os
import plenoirf as irf
import json
import sparse_table as spt

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

fc16by9 = sum_config['figure_16_9']
fc1by1 = fc16by9.copy()
fc1by1['rows'] = fc16by9['rows']*(16/9)

CHCL = 'cherenkovclassification'

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        event_table = spt.read(
            path=os.path.join(
                run_dir,
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)

        mrg_chc_fts = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray({
                spt.IDX: event_table['features'][spt.IDX]}),
            level_keys=[
                'primary',
                'trigger',
                'cherenkovclassification',
                'features'
            ]
        )

        # confusion matrix
        # ----------------
        num_bins_size_confusion_matrix = int(
            0.2*np.sqrt(mrg_chc_fts['features'].shape[0]))
        size_bin_edges = np.geomspace(
            1e1,
            1e5,
            num_bins_size_confusion_matrix+1)
        np_bins = np.histogram2d(
            mrg_chc_fts['trigger']['num_cherenkov_pe'],
            mrg_chc_fts['features']['num_photons'],
            bins=[size_bin_edges, size_bin_edges])[0]
        np_exposure_bins = np.histogram(
            mrg_chc_fts['trigger']['num_cherenkov_pe'],
            bins=size_bin_edges)[0]

        np_bins_normalized = np_bins.copy()
        for true_bin in range(num_bins_size_confusion_matrix):
            if np_exposure_bins[true_bin] > 0:
                np_bins_normalized[true_bin, :] /= np_exposure_bins[true_bin]

        fig = irf.summary.figure.figure(fc1by1)
        ax = fig.add_axes([0.1, 0.23, 0.7, 0.7])
        ax_h = fig.add_axes([0.1, 0.08, 0.7, 0.1])
        ax_cb = fig.add_axes([0.85, 0.23, 0.02, 0.7])
        _pcm_confusion = ax.pcolormesh(
            size_bin_edges,
            size_bin_edges,
            np.transpose(np_bins_normalized),
            cmap='Greys',
            norm=plt_colors.PowerNorm(gamma=0.5))
        plt.colorbar(_pcm_confusion, cax=ax_cb, extend='max')
        ax.set_aspect('equal')
        ax.set_title('normalized for each column')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_ylabel('reconstructed Cherenkov-size/p.e.')
        ax.loglog()
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax_h.loglog()
        ax_h.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax_h.set_xlim([np.min(size_bin_edges), np.max(size_bin_edges)])
        ax_h.set_xlabel('true Cherenkov-size/p.e.')
        ax_h.set_ylabel('num. events')
        ax_h.spines['top'].set_color('none')
        ax_h.spines['right'].set_color('none')
        irf.summary.figure.ax_add_hist(
            ax=ax_h,
            bin_edges=size_bin_edges,
            bincounts=np_exposure_bins,
            linestyle='k-')
        plt.savefig(
            os.path.join(
                summary_dir,
                '{:s}_{:s}_size_confusion.{:s}'.format(
                    prefix_str,
                    CHCL,
                    fc1by1['format'])))
        plt.close('all')

        # sensitivity VS. energy
        # -----------------------
        energy_bin_edges = np.array(sum_config['energy_bin_edges_GeV_coarse'])
        num_energy_bins = len(energy_bin_edges) - 1

        tprs = []
        ppvs = []
        num_events = []
        for i in range(num_energy_bins):
            e_start = energy_bin_edges[i]
            e_stop = energy_bin_edges[i+1]
            e_mask = np.logical_and(
                mrg_chc_fts['primary']['energy_GeV'] >= e_start,
                mrg_chc_fts['primary']['energy_GeV'] < e_stop)
            tp = mrg_chc_fts[CHCL]['num_true_positives'][e_mask]
            fn = mrg_chc_fts[CHCL]['num_false_negatives'][e_mask]
            fp = mrg_chc_fts[CHCL]['num_false_positives'][e_mask]
            tpr = tp / (tp + fn)
            ppv = tp / (tp + fp)
            tprs.append(np.median(tpr))
            ppvs.append(np.median(ppv))
            num_events.append(np.sum(e_mask))
        tprs = np.array(tprs)
        ppvs = np.array(ppvs)
        num_events = np.array(num_events)
        num_events_relunc = np.nan*np.ones(num_events.shape[0])
        _v = num_events > 0
        num_events_relunc[_v] = np.sqrt(num_events[_v])/num_events[_v]

        pp = os.path.join(
            summary_dir,
            '{:s}_{:s}_sensitivity_vs_true_energy'.format(
                prefix_str,
                CHCL))
        fig = irf.summary.figure.figure(fc16by9)
        ax = fig.add_axes([.12, .12, .85, .85])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=energy_bin_edges,
            bincounts=tprs,
            linestyle='k-',
            bincounts_upper=tprs*(1 + num_events_relunc),
            bincounts_lower=tprs*(1 - num_events_relunc),
            face_color='k',
            face_alpha=0.05)
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=energy_bin_edges,
            bincounts=ppvs,
            linestyle='k:',
            bincounts_upper=ppvs*(1 + num_events_relunc),
            bincounts_lower=ppvs*(1 - num_events_relunc),
            face_color='k',
            face_alpha=0.05)
        ax.set_xlabel('energy / GeV')
        ax.set_ylabel('true-positive-rate -\npositive-predictive-value :\n')
        ax.set_xlim([np.min(energy_bin_edges), np.max(energy_bin_edges)])
        ax.set_ylim([0, 1])
        ax.semilogx()
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        plt.savefig(pp+'.'+fc16by9['format'])
        plt.close(fig)
        with open(pp+'.json', 'wt') as f:
            out = {
                'energy_bin_edges_GeV': energy_bin_edges.tolist(),
                'num_events': num_events.tolist(),
                'true_positive_rate': tprs.tolist(),
                'positive_predictive_value': ppvs.tolist()}
            f.write(json.dumps(out, indent=4))

        # p.e. true/extracted VS. energy
        # -------------------------------
        true_over_reco_ratios = []
        num_events = []
        for i in range(num_energy_bins):
            e_start = energy_bin_edges[i]
            e_stop = energy_bin_edges[i+1]
            e_mask = np.logical_and(
                mrg_chc_fts['primary']['energy_GeV'] >= e_start,
                mrg_chc_fts['primary']['energy_GeV'] < e_stop)
            true_num_cherenkov_pe = mrg_chc_fts['trigger']['num_cherenkov_pe'][
                e_mask]
            num_cherenkov_pe = mrg_chc_fts['features']['num_photons'][e_mask]

            true_over_reco_ratio = true_num_cherenkov_pe/num_cherenkov_pe
            true_over_reco_ratios.append(np.median(true_over_reco_ratio))
            num_events.append(np.sum(e_mask))
        true_over_reco_ratios = np.array(true_over_reco_ratios)
        num_events = np.array(num_events)
        num_events_relunc = np.nan*np.ones(num_events.shape[0])
        _v = num_events > 0
        num_events_relunc[_v] = np.sqrt(num_events[_v])/num_events[_v]

        pp = os.path.join(
            summary_dir,
            '{:s}_{:s}_true_size_over_extracted_size_vs_true_energy'.format(
                prefix_str,
                CHCL))
        fig = irf.summary.figure.figure(fc16by9)
        ax = fig.add_axes([.12, .12, .85, .85])
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=energy_bin_edges,
            bincounts=true_over_reco_ratios,
            linestyle='k-',
            bincounts_upper=true_over_reco_ratios*(1 + num_events_relunc),
            bincounts_lower=true_over_reco_ratios*(1 - num_events_relunc),
            face_color='k',
            face_alpha=0.1)
        ax.axhline(y=1, color='k', linestyle=':')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_xlabel('energy / GeV')
        ax.set_ylabel('Cherenkov-size true/extracted / 1')
        ax.set_xlim([np.min(energy_bin_edges), np.max(energy_bin_edges)])
        ax.semilogx()
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        plt.savefig(pp+'.'+fc16by9['format'])
        plt.close('all')
        with open(pp+'.json', 'wt') as f:
            out = {
                'energy_bin_edges_GeV': energy_bin_edges.tolist(),
                'num_events': num_events.tolist(),
                'true_over_reco_ratios': true_over_reco_ratios.tolist()}
            f.write(json.dumps(out, indent=4))

        # p.e. true/extracted VS. true p.e.
        # ---------------------------------
        num_ratios = []
        num_events = []
        for i in range(num_bins_size_confusion_matrix):
            pe_start = size_bin_edges[i]
            pe_stop = size_bin_edges[i+1]
            pe_mask = np.logical_and(
                mrg_chc_fts['trigger']['num_cherenkov_pe'] >= pe_start,
                mrg_chc_fts['trigger']['num_cherenkov_pe'] < pe_stop)
            true_num_cherenkov_pe = mrg_chc_fts['trigger']['num_cherenkov_pe'][
                pe_mask]
            num_cherenkov_pe = mrg_chc_fts['features']['num_photons'][pe_mask]
            num_ratio = true_num_cherenkov_pe/num_cherenkov_pe
            num_ratios.append(np.median(num_ratio))
            num_events.append(np.sum(pe_mask))
        num_ratios = np.array(num_ratios)
        num_events = np.array(num_events)
        num_events_relunc = np.nan*np.ones(num_events.shape[0])
        _v = num_events > 0
        num_events_relunc[_v] = np.sqrt(num_events[_v])/num_events[_v]

        pp = os.path.join(
            summary_dir,
            '{:s}_{:s}_true_size_over_extracted_size_vs_true_size'.format(
                prefix_str,
                CHCL))
        fig = irf.summary.figure.figure(fc16by9)
        ax = fig.add_axes([.12, .12, .85, .85])
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=size_bin_edges,
            bincounts=num_ratios,
            linestyle='k-',
            bincounts_upper=num_ratios*(1 + num_events_relunc),
            bincounts_lower=num_ratios*(1 - num_events_relunc),
            face_color='k',
            face_alpha=0.1)
        ax.axhline(y=1, color='k', linestyle=':')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_xlabel('true Cherenkov-size / p.e.')
        ax.set_ylabel('Cherenkov-size true/extracted / 1')
        ax.set_xlim([np.min(size_bin_edges), np.max(size_bin_edges)])
        ax.semilogx()
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        plt.savefig(pp+'.'+fc16by9['format'])
        plt.close('all')
        with open(pp+'.json', 'wt') as f:
            out = {
                'size_bin_edges_pe': size_bin_edges.tolist(),
                'num_events': num_events.tolist(),
                'true_over_reco_ratios': num_ratios.tolist()}
            f.write(json.dumps(out, indent=4))
