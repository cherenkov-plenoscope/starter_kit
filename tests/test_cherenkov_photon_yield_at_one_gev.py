import os
import subprocess as sp
import tempfile
import numpy as np
import corsika_wrapper as cw
import simpleio
import toy_air_shower as tas


def steering_card(atmospheric_absorbtion, num_shower):
    out = ''
    out += 'RUNNR   1\n'
    out += 'EVTNR   1\n'
    out += 'NSHOW   {:d}\n'.format(num_shower)
    out += 'PRMPAR  1\n'
    out += 'ESLOPE  0.\n'
    out += 'ERANGE  1. 1.\n'
    out += 'THETAP  0. 0.\n'
    out += 'PHIP    0. 360.\n'
    out += 'SEED 1 0 0\n'
    out += 'SEED 2 0 0\n'
    out += 'SEED 3 0 0\n'
    out += 'SEED 4 0 0\n'
    out += 'OBSLEV  1e2\n'
    out += 'FIXCHI  0.\n'
    out += 'MAGNET 20.8 -11.4\n'
    out += 'ELMFLG  T   T\n'
    out += 'MAXPRT  1\n'
    out += 'PAROUT  F F\n'
    out += 'TELESCOPE  0. 0. 0. 5000e2 # instrument radius\n'
    out += 'ATMOSPHERE 26 T # Chile, Paranal, ESO site\n'
    out += 'CWAVLG 250 700\n'
    out += 'CSCAT 1 0 0\n'
    out += 'CERQEF F {:s} F # pde, atmo, mirror\n'.format(
        'T' if atmospheric_absorbtion else 'F')
    out += 'CERSIZ 1\n'
    out += 'CERFIL F\n'
    out += 'TSTART T\n'
    out += 'EXIT\n'
    return out

out_dir = os.path.join("tests", "test_cherenkov_photon_yield_at_one_gev")
os.makedirs(out_dir, exist_ok=True)

bin_edges_emission_altitude = np.linspace(5e3, 55e3, 101)
bin_edges_wavelength = np.linspace(250e-9, 700e-9, 101)
num_shower = 100

# CORSIKA KIT
# -----------

cor_hist_emission_altitude = []
cor_hist_wavelength = []
cor_num_cherenkov_photons = []
cor_first_interaction_altitude = []

with tempfile.TemporaryDirectory (prefix='test_corsika_') as tmp:
    corsika_steering_card_path = os.path.join(tmp, "steering.txt")
    with open(corsika_steering_card_path, "wt") as fout:
        fout.write(
            steering_card(
                atmospheric_absorbtion=False,
                num_shower=num_shower))

    eventio_path = os.path.join(tmp, "shower.eventio")
    cw.corsika(
        steering_card=cw.read_steering_card(corsika_steering_card_path),
        output_path=eventio_path,
        save_stdout=True)

    simpleio_path = os.path.join(tmp, "shower.simpleio")
    sp.call([
        os.path.join('build','merlict', 'merlict-eventio-converter'),
        '-i', eventio_path,
        '-o', simpleio_path])

    run = simpleio.SimpleIoRun(simpleio_path)
    for idx in range(len(run)):
        event = run[idx]
        cpb = event.cherenkov_photon_bunches

        cor_first_interaction_altitude.append(-1.*event.header.raw[7 - 1]*1e-2)

        assert(np.sum(cpb.probability_to_reach_observation_level < 0.) == 0)
        assert(np.sum(cpb.probability_to_reach_observation_level > 1.) == 0)

        cor_num_cherenkov_photons.append(
            np.sum(cpb.probability_to_reach_observation_level))

        cor_hist_wavelength.append(
            np.histogram(
                -1.*cpb.wavelength,
                weights=cpb.probability_to_reach_observation_level,
                bins=bin_edges_wavelength)[0])

        cor_hist_emission_altitude.append(
            np.histogram(
                cpb.emission_height,
                weights=cpb.probability_to_reach_observation_level,
                bins=bin_edges_emission_altitude)[0])

cor_hist_emission_altitude = np.array(cor_hist_emission_altitude)
cor_hist_wavelength = np.array(cor_hist_wavelength)
cor_num_cherenkov_photons = np.array(cor_num_cherenkov_photons)
cor_first_interaction_altitude = np.array(cor_first_interaction_altitude)

# toy simulation
# --------------

for bremsstrahlungs_multiplicity in [1, 6]:

    tas_hist_emission_altitude = []
    tas_hist_wavelength = []
    tas_num_cherenkov_photons = []
    tas_first_interaction_altitude = []


    for idx in range(num_shower):
        print(idx)
        particles, cherenkov_photons = tas.simulate_gamma_ray_air_shower(
            random_seed=idx,
            primary_energy=1e9*tas.UNIT_CHARGE,
            wavelength_start=250e-9,
            wavelength_end=700e-9,
            bremsstrahlung_correction_factor=bremsstrahlungs_multiplicity)

        tas_first_interaction_altitude.append(particles[0]['end_altitude'])

        tas_num_cherenkov_photons.append(cherenkov_photons.shape[0])

        tas_hist_wavelength.append(
            np.histogram(
                cherenkov_photons[:, tas.IDX_WAVELENGTH],
                bins=bin_edges_wavelength)[0])

        tas_hist_emission_altitude.append(
            np.histogram(
                cherenkov_photons[:, tas.IDX_ALTITUDE],
                bins=bin_edges_emission_altitude)[0])

    tas_hist_emission_altitude = np.array(tas_hist_emission_altitude)
    tas_hist_wavelength = np.array(tas_hist_wavelength)
    tas_num_cherenkov_photons = np.array(tas_num_cherenkov_photons)
    tas_first_interaction_altitude = np.array(tas_first_interaction_altitude)


    #---------------------

    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    cor_hist_emission_altitude_normalized = np.sum(
        cor_hist_emission_altitude,
        axis=0) / np.sum(cor_num_cherenkov_photons)

    tas_hist_emission_altitude_normalized = np.sum(
        tas_hist_emission_altitude,
        axis=0) / np.sum(tas_num_cherenkov_photons)

    ax.plot(
        1e-3*bin_edges_emission_altitude[: -1],
        cor_hist_emission_altitude_normalized,
        'k',
        drawstyle='steps',
        label="CORSIKA")
    ax.plot(
        1e-3*bin_edges_emission_altitude[: -1],
        tas_hist_emission_altitude_normalized,
        ':k',
        drawstyle='steps',
        label="Sebastian")
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('emission-altitude (a.s.l.) / km')
    ax.set_ylabel('normalized num. Cherenkov-photons / 1')
    ax.legend(loc='upper right')
    fig.savefig(
        os.path.join(
            out_dir,
            'cherenkov-photon-emission-altitude-histogram_{:d}.png'.format(
                bremsstrahlungs_multiplicity)))

    #---------------------

    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    cor_hist_wavelength_normalized = np.sum(
        cor_hist_wavelength,
        axis=0) / np.sum(cor_num_cherenkov_photons)
    tas_hist_wavelength_normalized = np.sum(
        tas_hist_wavelength,
        axis=0) / np.sum(tas_num_cherenkov_photons)
    ax.plot(
        1e9*bin_edges_wavelength[: -1],
        cor_hist_wavelength_normalized,
        'k',
        drawstyle='steps',
        label="CORSIKA")
    ax.plot(
        1e9*bin_edges_wavelength[: -1],
        tas_hist_wavelength_normalized,
        ':k',
        drawstyle='steps',
        label="Sebastian")
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('wavelength / nm')
    ax.set_ylabel('normalized num. Cherenkov-photons / 1')
    ax.legend(loc='upper right')
    fig.savefig(
        os.path.join(
            out_dir,
            'cherenkov-wavelength-histogram_{:d}.png'.format(
                bremsstrahlungs_multiplicity)))

    #---------------------

    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    bin_edges_num_cherenkov_photons = np.linspace(0., 3e5, 25)
    cor_hist_num_cherenkov_photons = np.histogram(
        cor_num_cherenkov_photons,
        bins=bin_edges_num_cherenkov_photons)[0]
    tas_hist_num_cherenkov_photons = np.histogram(
        tas_num_cherenkov_photons,
        bins=bin_edges_num_cherenkov_photons)[0]

    ax.plot(
        bin_edges_num_cherenkov_photons[: -1],
        cor_hist_num_cherenkov_photons,
        'k',
        drawstyle='steps',
        label="CORSIKA")
    #ax.vlines(np.median(cor_num_cherenkov_photons), y_max=y_max, 'k')
    ax.plot(
        bin_edges_num_cherenkov_photons[: -1],
        tas_hist_num_cherenkov_photons,
        ':k',
        drawstyle='steps',
        label="Sebastian")
    #ax.vlines(np.median(tas_num_cherenkov_photons), y_max=y_max, ':k')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('num. Cherenkov-photons / 1')
    ax.set_ylabel('num. events / 1')
    ax.legend(loc='upper right')
    fig.savefig(
        os.path.join(
            out_dir,
            'cherenkov-yield-histogram_{:d}.png'.format(
                bremsstrahlungs_multiplicity)))

    #---------------------

    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    bin_edges_first_interaction_altitude = np.linspace(
        1e3,
        100e3,
        int(2 * np.sqrt(num_shower)))
    cor_hist_first_interaction_altitude = np.histogram(
        cor_first_interaction_altitude,
        bins=bin_edges_first_interaction_altitude)[0]
    tas_hist_first_interaction_altitude = np.histogram(
        tas_first_interaction_altitude,
        bins=bin_edges_first_interaction_altitude)[0]

    ax.plot(
        1e-3*bin_edges_first_interaction_altitude[: -1],
        cor_hist_first_interaction_altitude,
        'k',
        drawstyle='steps',
        label="CORSIKA")
    ax.plot(
        1e-3*bin_edges_first_interaction_altitude[: -1],
        tas_hist_first_interaction_altitude,
        ':k',
        drawstyle='steps',
        label="Sebastian")
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('first interaction altitude (a.s.l.) / km')
    ax.set_ylabel('num. events / 1')
    ax.legend(loc='upper right')
    fig.savefig(
        os.path.join(
            out_dir,
            'first-interaction-altitude-histogram_{:d}.png'.format(
                bremsstrahlungs_multiplicity)))

    #---------------------
