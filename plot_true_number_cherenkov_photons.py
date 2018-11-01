import matplotlib.pyplot as plt
import os
import json
import numpy
import matplotlib.colors as colors

out_dir = os.path.join('examples', 'true_number_cherenkov_photons')
os.makedirs(out_dir, exist_ok=True)

figsize2 = (7, 3.6)
dpi = 240
ax_size2 = (0.08, 0.12, 0.9, 0.85)


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

    # number true Cherenkov photons
    # -----------------------------

    bin_edges = np.zeros(29)
    bin_edges[1] = 1
    for b in range(bin_edges.shape[0]):
        if b > 1:
            bin_edges[b] = np.sqrt(2)*bin_edges[b-1]

    num_triggered = np.histogram(
        e['num_true_cherenkov_photons'][e['trigger_mask']],
        bin_edges)[0]

    num_thrown = np.histogram(
        e['num_true_cherenkov_photons'],
        bin_edges)[0]

    ratio = np.zeros(num_thrown.shape[0])
    ratio_delta = np.zeros(num_thrown.shape[0])
    for j in range(num_thrown.shape[0]):
        if num_thrown[j] == 0:
            ratio[j] = 0
            ratio_delta[j] = 0
        else:
            ratio[j] = num_triggered[j]/num_thrown[j]
            ratio_delta[j] = np.sqrt(num_triggered[j])/num_thrown[j]
    ratio_l = ratio - ratio_delta
    ratio_h = ratio + ratio_delta

    fig = plt.figure(figsize=figsize2, dpi=dpi)
    ax = fig.add_axes(ax_size2)
    for i in range(num_triggered.shape[0]):
        x = [bin_edges[i], bin_edges[i + 1]]
        ax.plot(
            x,
            [ratio[i], ratio[i]],
            'k')
        ax.fill_between(
            x=x,
            y1=[ratio_l[i], ratio_l[i]],
            y2=[ratio_h[i], ratio_h[i]],
            color='k',
            alpha=0.2,
            linewidth=0)
    ax.semilogx()
    ax.set_ylim([-.01, 1.01])
    ax.set_xlabel('true Cherenkov-photons / 1')
    ax.set_ylabel('{:s} triggered/thrown / 1'.format(particle))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    fig.savefig(
        os.path.join(
            out_dir, (
                'trigger_probability_true_number_cherenkov_photons_' +
                '{:s}.png'
            ).format(particle)))

    num_cer_detected = e['num_true_cherenkov_photons'][e['trigger_mask']]
    energies_detected = e['energies'][e['trigger_mask']]

    cer_start_10power = 1
    cer_stop_10power = 4
    num_cer_bins = 24
    bin_edges_cer = np.logspace(
        cer_start_10power,
        cer_stop_10power,
        num_cer_bins)

    energy_start_10power = -1
    energy_stop_10power = 3
    num_energy_bins = 33
    bin_edges_energie = np.logspace(
        energy_start_10power,
        energy_stop_10power,
        num_energy_bins)

    h = np.histogram2d(
        x=num_cer_detected,
        y=energies_detected,
        bins=[bin_edges_cer, bin_edges_energie])[0]

    exposure = np.histogram(energies_detected, bins=bin_edges_energie)[0]

    @plt.FuncFormatter
    def fake_log(x, pos):
        'The two args are the value and tick position'
        if np.abs(np.modf(x)[0]) < 0.01:
            return r'$10^{%1.f}$' % (x)
        else:
            return r''

    h_exp = h.copy()
    for i in range(exposure.shape[0]):
        if exposure[i] == 0.0:
            h_exp[:, i] = 0.0
        else:
            h_exp[:, i] /= exposure[i]

    fig = plt.figure(figsize=(7, 6), dpi=dpi)
    ax = fig.add_axes((0.1, 0.32, 0.75, 0.66))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('true Cherenkov-photons / 1')
    im = ax.pcolor(
        bin_edges_energie,
        bin_edges_cer,
        h_exp,
        cmap='binary',
        norm=colors.PowerNorm(gamma=1./2.))
    cax = fig.add_axes((0.9, 0.32, 0.03, 0.66))
    cbar = fig.colorbar(im, cax=cax)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlim([10**energy_start_10power, 10**energy_stop_10power])
    ax.set_ylim([10**cer_start_10power, 10**cer_stop_10power])
    ax.loglog()

    # exposure
    # --------
    ax2 = fig.add_axes((0.1, 0.08, 0.75, 0.15))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for i in range(exposure.shape[0]):
        ax2.plot(
            [bin_edges_energie[i], bin_edges_energie[i + 1]],
            [exposure[i], exposure[i]],
            'k')
    ax2.loglog()
    ax2.set_xlim([10**energy_start_10power, 10**energy_stop_10power])
    ax2.set_ylim([1, 500])
    ax2.set_ylabel('events / 1')
    ax2.set_xlabel('true energy {:s} / GeV'.format(particle))
    ax2.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    fig.savefig(
        os.path.join(
            out_dir,
            'cherenkov_photons_vs_energy_{:s}.png'.format(particle)))

    plt.close('all')
