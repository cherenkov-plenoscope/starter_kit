import plenopy as pl
import os
import pandas as pd


def add_hist(ax, bin_edges, bincounts, linestyle, color, alpha):
    assert bin_edges.shape[0] == bincounts.shape[0] + 1
    for i, bincount in enumerate(bincounts):
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [bincount, bincount],
            linestyle)
        ax.fill_between(
            x=[bin_edges[i], bin_edges[i + 1]],
            y1=[bincount, bincount],
            color=color,
            alpha=alpha,
            edgecolor='none')


light_field_geometry = pl.LightFieldGeometry('run/light_field_calibration/')

particles = {
    'gamma': 'gamma_diffuse',
    'electron': 'electron',
    'proton': 'proton'
}

out_dir = os.path.join(
    '.',
    'examples',
    'benchmark_cherenkov_photon_classification')
os.makedirs(out_dir, exist_ok=True)


def particle_cx_cy_from_momentum(momentum):
    direction = momentum / np.linalg.norm(momentum)
    return direction[0], direction[1]

MIN_NUM_CHERENKOV_PHOTONS = 30

for particle in particles:
    run_path = os.path.join(
        '.',
        'run',
        'irf',
        particles[particle],
        'past_trigger')
    cache_path = os.path.join(
        out_dir,
        particle+'_cherenkov_photon_classification.msg')
    if os.path.exists(cache_path):
        continue

    run = pl.Run(path=run_path, light_field_geometry=light_field_geometry)

    infos = []
    for event_idx in range(run.number_events):
        event = run[event_idx]
        info = {}
        info['run'] = event.simulation_truth.event. \
            corsika_run_header.number
        info['event'] = event.simulation_truth. \
            event.corsika_event_header.number
        info['particle_id'] = event.simulation_truth.event. \
            corsika_event_header.primary_particle_id
        info['particle_energy'] = event.simulation_truth.event. \
            corsika_event_header.total_energy_GeV
        info['particle_x'] = event.simulation_truth.event. \
            corsika_event_header.core_position_x_meter()
        info['particle_y'] = event.simulation_truth.event. \
            corsika_event_header.core_position_y_meter()
        particle_cx, particle_cy = particle_cx_cy_from_momentum(
            event.simulation_truth.event.corsika_event_header.momentum())
        info['particle_cx'] = particle_cx
        info['particle_cy'] = particle_cy
        info['true_num_cherenkov_pe'] = event. \
            simulation_truth.detector.number_air_shower_pulses()
        info['true_num_night_sky_background_pe'] = event. \
            simulation_truth.detector.number_night_sky_background_pulses()

        photons = pl.classify.RawPhotons.from_event(event)
        try:
            roi = pl.classify.center_for_region_of_interest(event)
            cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
                    roi=roi,
                    photons=photons,
                    min_number_photons=17)
            info['method'] = 0
        except IndexError:
            cherenkov_photons = pl.classify.cherenkov_photons_in_image(
                photons=photons,
                light_field_geometry=light_field_geometry)
            info['method'] = 1

        if cherenkov_photons.x.shape[0] < MIN_NUM_CHERENKOV_PHOTONS:
            continue

        br = pl.classify.benchmark(
            pulse_origins=event.simulation_truth.detector.pulse_origins,
            photon_ids_cherenkov=cherenkov_photons.photon_ids)

        info['num_true_positives'] = br['number_true_positives']
        info['num_false_positives'] = br['number_false_positives']
        info['num_true_negatives'] = br['number_true_negatives']
        info['num_false_negatives'] = br['number_false_negatives']
        info['num_cherenkov_pe'] = cherenkov_photons.x.shape[0]
        print(particle, event_idx, 'of', run.number_events, info)
        infos.append(info)
    df = pd.DataFrame(infos)
    df.to_msgpack(cache_path)


for particle in particles:
    run_path = os.path.join(
        '.',
        'run',
        'irf',
        particles[particle],
        'past_trigger')
    cache_path = os.path.join(
        out_dir,
        particle+'_cherenkov_photon_classification.msg')

    df = pd.read_msgpack(cache_path)
    print(df)


    # confusion matrix
    # ----------------
    np_bin_edges = np.geomspace(1e1, 1e5, 33)
    np_bins = np.histogram2d(
        #1e3*np.ones(df.true_num_cherenkov_pe.shape[0]),
        df.true_num_cherenkov_pe,
        df.num_cherenkov_pe,
        bins=[np_bin_edges, np_bin_edges])[0]

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_axes([.2,.2,.72,.72])
    ax.pcolormesh(
        np_bin_edges,
        np_bin_edges,
        np_bins.T,
        cmap='Greys')
    ax.set_xlabel('true size of Cherenkov-photons / p.e.')
    ax.set_ylabel('reconstructed size of Cherenkov-photons / p.e.')
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    plt.savefig(
        os.path.join(
            out_dir,
            '{:s}_{:s}.png'.format(particle, 'confusion')))
    plt.close('all')


    # sensitivity VS. true energy
    # ---------------------------
    num_bins = 25
    true_energy_bin_edges = np.geomspace(.1, 1e3, num_bins+1)

    tprs = []
    ppvs = []
    num_events_in_bin = []
    for i in range(num_bins):
        e_start = true_energy_bin_edges[i]
        e_stop = true_energy_bin_edges[i+1]
        e_mask = (df.particle_energy >= e_start)&(df.particle_energy < e_stop)
        rec_mask = np.logical_not(np.isnan(df.num_true_positives))
        e_mask = e_mask&rec_mask

        tp = df.num_true_positives[e_mask]
        fn = df.num_false_negatives[e_mask]
        fp = df.num_false_positives[e_mask]
        tpr = tp / (tp + fn)
        ppv = tp / (tp + fp)
        tprs.append(np.median(tpr))
        ppvs.append(np.median(ppv))
        num_events_in_bin.append(np.sum(e_mask))
    tprs = np.array(tprs)
    ppvs = np.array(ppvs)

    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_axes([.15,.2,.82,.72])
    add_hist(
        ax=ax,
        bin_edges=true_energy_bin_edges,
        bincounts=tprs,
        linestyle='k-',
        color='b',
        alpha=0.0)

    add_hist(
        ax=ax,
        bin_edges=true_energy_bin_edges,
        bincounts=ppvs,
        linestyle='k:',
        color='r',
        alpha=0.0)
    ax.set_xlabel('true energy / GeV')
    ax.set_ylabel('true positive rate -\npositive predictive value :\n')
    ax.set_xlim([np.min(true_energy_bin_edges), np.max(true_energy_bin_edges)])
    ax.semilogx()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    plt.savefig(
        os.path.join(
            out_dir,
            '{:s}_{:s}.png'.format(particle, 'sensitivity')))
    plt.close('all')


    # p.e. true/extracted VS. true energy
    # -----------------------------------
    num_ratios = []
    num_events_in_bin = []
    for i in range(num_bins):
        e_start = true_energy_bin_edges[i]
        e_stop = true_energy_bin_edges[i+1]
        e_mask = (df.particle_energy >= e_start)&(df.particle_energy < e_stop)
        rec_mask = np.logical_not(np.isnan(df.num_true_positives))
        e_mask = e_mask&rec_mask
        num_ratio = df.true_num_cherenkov_pe[e_mask]/df.num_cherenkov_pe[e_mask]
        num_ratios.append(np.median(num_ratio))
        num_events_in_bin.append(np.sum(e_mask))
    num_ratios = np.array(num_ratios)

    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_axes([.15,.2,.82,.72])
    add_hist(
        ax=ax,
        bin_edges=true_energy_bin_edges,
        bincounts=num_ratios,
        linestyle='k-',
        color='b',
        alpha=0.0)
    ax.axhline(y=1, color='k', linestyle=':')
    ax.set_xlabel('true energy / GeV')
    ax.set_ylabel('Cherenkov size true/extracted / 1')
    ax.set_xlim([np.min(true_energy_bin_edges), np.max(true_energy_bin_edges)])
    ax.semilogx()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    plt.savefig(
        os.path.join(
            out_dir,
            '{:s}_{:s}.png'.format(particle, 'ratio_extraction')))
    plt.close('all')


    # p.e. true/extracted VS. true p.e.
    # ---------------------------------
    num_bins = 12
    true_pe_bin_edges = np.geomspace(10, 1e4, num_bins+1)
    num_ratios = []
    for i in range(num_bins):
        pe_start = true_pe_bin_edges[i]
        pe_stop = true_pe_bin_edges[i+1]
        pe_mask = (df.true_num_cherenkov_pe >= pe_start)&(df.true_num_cherenkov_pe < pe_stop)
        rec_mask = np.logical_not(np.isnan(df.num_true_positives))
        pe_mask = pe_mask&rec_mask
        num_ratio = df.true_num_cherenkov_pe[pe_mask]/df.num_cherenkov_pe[pe_mask]
        num_ratios.append(np.median(num_ratio))
    num_ratios = np.array(num_ratios)

    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_axes([.15,.2,.82,.72])
    add_hist(
        ax=ax,
        bin_edges=true_pe_bin_edges,
        bincounts=num_ratios,
        linestyle='k-',
        color='b',
        alpha=0.0)
    ax.axhline(y=1, color='k', linestyle=':')
    ax.set_xlabel('true Cherenkov size / p.e.')
    ax.set_ylabel('Cherenkov size true/extracted / 1')
    ax.set_xlim([np.min(true_pe_bin_edges), np.max(true_pe_bin_edges)])
    ax.semilogx()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    plt.savefig(
        os.path.join(
            out_dir,
            '{:s}_{:s}.png'.format(particle, 'ratio_vs_pe')))
    plt.close('all')
