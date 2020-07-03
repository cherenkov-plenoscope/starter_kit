#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_table as spt
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

intensity_bin_edges = irf.grid.PH_BIN_EDGES[1:]
num_intensity_bins = len(intensity_bin_edges) - 1

energy_bin_edges = np.geomspace(
    sum_config['lower_energy_GeV'],
    sum_config['upper_energy_GeV'],
    sum_config['num_energy_bins']//4
)
num_energy_bins = len(energy_bin_edges) - 1

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)
        table = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=event_table['primary'][spt.IDX],
            level_keys=['primary', 'grid'],
        )
        table = spt.sort_table_on_common_indices(
            table=table,
            common_indices=table['primary'][spt.IDX],
        )

        for energy_idx in range(num_energy_bins):

            l_energy = energy_bin_edges[energy_idx]
            u_energy = energy_bin_edges[energy_idx + 1]

            energy_mask = (
                (table['primary']['energy_GeV'] >= l_energy)*
                (table['primary']['energy_GeV'] < u_energy)
            ).astype(np.int64)

            num_airshower = np.sum(energy_mask)
            print(site_key, particle_key, l_energy, u_energy, num_airshower)

            mean_grid_hist = np.zeros(num_intensity_bins)
            mean_grid_hist_rel_unc = np.nan*np.ones(num_intensity_bins)

            if num_airshower > 0:
                grid_hist = np.zeros(shape=(num_airshower, num_intensity_bins))
                for bin_idx in range(num_intensity_bins):
                    grid_hist[energy_mask, bin_idx] = table[
                        'grid'][
                        'hist_{:02d}'.format(bin_idx + 1)]

                mean_grid_hist = np.sum(grid_hist, axis=0)
                mean_grid_hist_rel_unc = irf.analysis.effective_quantity._divide_silent(
                    numerator=np.sqrt(mean_grid_hist),
                    denominator=mean_grid_hist,
                    default=np.nan
                )
                mean_grid_hist /= np.sum(mean_grid_hist)

            fig = irf.summary.figure.figure(sum_config['figure_16_9'])
            ax = fig.add_axes((.1, .1, .8, .8))
            irf.summary.figure.ax_add_hist(
                ax=ax,
                bin_edges=intensity_bin_edges,
                bincounts=mean_grid_hist,
                linestyle='k-',
                bincounts_upper=(1 + mean_grid_hist_rel_unc)*mean_grid_hist,
                bincounts_lower=(1 - mean_grid_hist_rel_unc)*mean_grid_hist,
                face_color='k',
                face_alpha=0.33,
            )
            ax.set_title(
                'energy {: >8.2f} - {: >8.2f}GeV, num. airshower {:>6d}'.format(
                    l_energy,
                    u_energy,
                    num_airshower
                ),
                family='monospace'
            )
            ax.loglog()
            ax.set_ylim([1e-3, 1e0])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('intensity of cherenkov-photons in grib-bin / 1')
            ax.set_ylabel('relative intensity of grid-bins / 1')
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            fig.savefig(
                os.path.join(
                    pa['out_dir'],
                    '{:s}_{:s}_{:06d}_grid_intensity.jpg'.format(
                        site_key,
                        particle_key,
                        energy_idx
                    )
                )
            )
            plt.close(fig)
