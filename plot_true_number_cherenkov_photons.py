import matplotlib.pyplot as plt
import os
import json
import numpy
import matplotlib.colors as colors

out_dir = os.path.join('examples', 'true_number_cherenkov_photons')
os.makedirs(out_dir, exist_ok=True)

figsize2 = (8, 4)
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

    ratio = num_triggered/num_thrown
    ratio_delta = np.sqrt(num_triggered)/num_thrown
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
    ax.set_xlabel(r'number true Cherenkov-photons / 1')
    ax.set_ylabel('trigger-probability {:s} / 1'.format(particle))
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

    bin_edges_cer = np.logspace(
        1,
        4,
        33)

    bin_edges_energie = np.logspace(
        -1,
        3,
        33)

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
        h_exp[:, i] /= exposure[i]

    h_exp[np.isnan(h_exp)] = 0.0

    fig = plt.figure(figsize=(6, 4.5), dpi=dpi)
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.90))
    ax.imshow(
        h_exp,
        origin='lower',
        extent=[
            -1,
            3,
            1,
            4],
        cmap='binary',
        norm=colors.PowerNorm(gamma=1./2.))
    ax.xaxis.set_major_formatter(fake_log)
    ax.yaxis.set_major_formatter(fake_log)
    ax.set_ylabel(r'number true Cherenkov-photons / 1')
    ax.set_xlabel('true energy / GeV')
    fig.savefig(
        os.path.join(
            out_dir,
            'hist_{:s}.png'.format(particle)))
