import matplotlib.pyplot as plt
import os
import json
import numpy

out_dir = os.path.join('examples', 'scatter_range_coverage')
os.makedirs(out_dir, exist_ok=True)

figure_configs = [
    {
        'figsize': (8, 6),
        'figsize2': (8, 4),
        'dpi': 240,
        'ax_size': (0.08, 0.08, 0.9, 0.9),
        'ax_size2': (0.08, 0.12, 0.9, 0.85),
        'path_ext': '',
    },
    {
        'figsize': (1920/300, 1080/300),
        'figsize2': (1920/300, 1080/300),
        'dpi': 300,
        'ax_size': (0.1, 0.15, 0.85, 0.80),
        'ax_size2': (0.1, 0.15, 0.85, 0.80),
        'path_ext': '_beamer',
    },
]

for fc in figure_configs:
    for particle in ['gamma', 'electron', 'proton']:
        events_path = os.path.join(
            'run',
            'irf',
            particle,
            'results',
            'events.json')
        with open(events_path, 'rt') as fin:
            eo = json.loads(fin.read())
            e = {}
            for key in eo:
                e[key] = np.array(eo[key])

        num_triggers = np.sum(e['trigger_mask'])
        num_bins = np.int(np.sqrt(num_triggers)/2)

        # Scatter-radius
        # --------------

        bin_edges = np.linspace(
            0,
            np.max(e['scatter_radii'][e['trigger_mask']]**2),
            num_bins)

        num_triggered = np.histogram(
            e['scatter_radii'][e['trigger_mask']]**2,
            bin_edges)[0]

        num_thrown = np.histogram(
            e['scatter_radii']**2,
            bin_edges)[0]

        fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
        ax = fig.add_axes(fc['ax_size2'])
        for i in range(num_triggered.shape[0]):
            ax.plot(
                1e-6*np.array([bin_edges[i], bin_edges[i + 1]]),
                [num_thrown[i], num_thrown[i]],
                'k',
                label='thrown' if i == 0 else None)
            ax.plot(
                1e-6*np.array([bin_edges[i], bin_edges[i + 1]]),
                [num_triggered[i], num_triggered[i]],
                'k:',
                label='triggered' if i == 0 else None)
        ax.semilogy()
        ax.set_xlim([0, 1.75])
        ax.set_ylim([.9, 3e4])
        ax.set_xlabel(r'(Scatter-radius)$^2$ / (km)$^2$')
        ax.set_ylabel('{:s}s / 1'.format(particle))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.legend(loc='best', fontsize=10)
        fig.savefig(
            os.path.join(
                out_dir,
                'scatter_radius_thrown_and_triggered_{:s}'.format(particle) +
                fc['path_ext'] +
                '.png'))

        ratio = num_triggered/num_thrown
        ratio_delta = np.sqrt(num_triggered)/num_thrown
        ratio_low = ratio - ratio_delta
        ratio_high = ratio + ratio_delta
        ratio_low[ratio_low < 0] = 0

        fig = plt.figure(figsize=fc['figsize'], dpi=fc['dpi'])
        ax = fig.add_axes(fc['ax_size'])
        for i in range(num_triggered.shape[0]):
            x = 1e-6*np.array([bin_edges[i], bin_edges[i + 1]])
            ax.plot(
                x,
                [1e2*ratio[i], 1e2*ratio[i]],
                'k')
            ax.fill_between(
                x=x,
                y1=1e2*np.array([ratio_low[i], ratio_low[i]]),
                y2=1e2*np.array([ratio_high[i], ratio_high[i]]),
                color='k',
                alpha=0.2,
                linewidth=0)
        ax.set_xlim([0, 1.75])
        ax.set_xlabel(r'(Scatter-radius)$^2$ / (km)$^2$')
        ax.set_ylabel('{:s}s triggered/thrown / %'.format(particle))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.legend(loc='best', fontsize=10)
        fig.savefig(
            os.path.join(
                out_dir,
                'scatter_radius_thrown_and_triggered_{:s}_ratio'.format(
                    particle) +
                fc['path_ext'] +
                '.png'))

        # Scatter-angle
        # -------------
        if 'gamma' not in particle:
            scatter_angle_sq_deg = np.rad2deg(e['zenith_distances'])**2

            bin_edges = np.linspace(
                0,
                np.max(scatter_angle_sq_deg[e['trigger_mask']]),
                num_bins)

            num_triggered = np.histogram(
                scatter_angle_sq_deg[e['trigger_mask']],
                bin_edges)[0]

            num_thrown = np.histogram(
                scatter_angle_sq_deg,
                bin_edges)[0]

            fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
            ax = fig.add_axes(fc['ax_size2'])
            for i in range(num_triggered.shape[0]):
                ax.plot(
                    np.array([bin_edges[i], bin_edges[i + 1]]),
                    [num_thrown[i], num_thrown[i]],
                    'k',
                    label='thrown' if i == 0 else None)
                ax.plot(
                    np.array([bin_edges[i], bin_edges[i + 1]]),
                    [num_triggered[i], num_triggered[i]],
                    'k:',
                    label='triggered' if i == 0 else None)
            ax.semilogy()
            ax.set_xlim([0, 45])
            ax.set_ylim([5, 3e3])
            ax.set_xlabel(r'(Scatter-angle)$^2$ / (deg)$^2$')
            ax.set_ylabel('{:s}s / 1'.format(particle))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            ax.axvline(x=3.25**2, color='k', linestyle='-', alpha=0.25)
            ax.legend(loc='best', fontsize=10)
            fig.savefig(
                os.path.join(
                    out_dir,
                    'scatter_angle_thrown_and_triggered_{:s}'.format(
                        particle) +
                    fc['path_ext'] + 
                    '.png'))

            ratio = num_triggered/num_thrown
            ratio_delta = np.sqrt(num_triggered)/num_thrown
            ratio_low = ratio - ratio_delta
            ratio_high = ratio + ratio_delta
            ratio_low[ratio_low < 0] = 0

            fig = plt.figure(figsize=fc['figsize'], dpi=fc['dpi'])
            ax = fig.add_axes(fc['ax_size'])
            for i in range(num_triggered.shape[0]):
                x = np.array([bin_edges[i], bin_edges[i + 1]])
                ax.plot(
                    x,
                    [1e2*ratio[i], 1e2*ratio[i]],
                    'k')
                ax.fill_between(
                    x=x,
                    y1=1e2*np.array([ratio_low[i], ratio_low[i]]),
                    y2=1e2*np.array([ratio_high[i], ratio_high[i]]),
                    color='k',
                    alpha=0.2,
                    linewidth=0)
            ax.set_xlim([0, 45])
            ax.set_xlabel(r'(Scatter-angle)$^2$ / (deg)$^2$')
            ax.set_ylabel('{:s}s triggered/thrown / %'.format(particle))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            ax.axvline(x=3.25**2, color='k', linestyle='-', alpha=0.25)
            ax.legend(loc='best', fontsize=10)
            fig.savefig(
                os.path.join(
                    out_dir,
                    'scatter_angle_thrown_and_triggered_{:s}_ratio'.format(
                        particle) +
                    fc['path_ext'] +
                    '.png'))

        plt.close('all')
