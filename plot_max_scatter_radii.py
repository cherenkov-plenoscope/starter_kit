import numpy as np
import matplotlib.pyplot as plt
import acp_instrument_response_function as irf
import json
import os

figsizec = (6, 4)
dpic = 320
ax_sizec = (0.11, 0.11, 0.88, 0.88)

out_dir = os.path.join('examples', 'scatter_range_coverage')
os.makedirs(out_dir, exist_ok=True)

figc = plt.figure(figsize=figsizec, dpi=dpic)
axc = figc.add_axes(ax_sizec)
linestyles = ['-', '--', ':']

for i, particle in enumerate(['gamma', 'electron', 'proton']):

    config_path = os.path.join(
        'resources',
        'acp',
        '71m',
        '{:s}_steering.json'.format(particle))
    with open(config_path, 'rt') as fin:
        steering_card = json.loads(fin.read())

    max_scatter_radius_in_bin, energy_bin_edges = (
        irf.utils.energy_bins_and_max_scatter_radius(
            energy=steering_card['energy'],
            max_scatter_radius=steering_card['max_scatter_radius'],
            number_runs=4096))

    axc.plot(
        energy_bin_edges[: -1],
        max_scatter_radius_in_bin,
        'k{:s}'.format(linestyles[i]),
        label='{:s}'.format(particle))

axc.semilogx()
axc.semilogy()
axc.set_ylim([100, 3000])
axc.set_xlim([0.1, 2000])
axc.legend(loc='best', fontsize=10)
axc.spines['right'].set_visible(False)
axc.spines['top'].set_visible(False)
axc.set_xlabel('energy / GeV')
axc.set_ylabel('max scatter-radius / m')
axc.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
figc.savefig(
    os.path.join(out_dir, 'max_scatter_radius.png'.format(particle)))