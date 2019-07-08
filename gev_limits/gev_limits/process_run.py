import numpy as np
import os
import tempfile
import json
import subprocess as sp
import scipy
import simpleio as sio
import corsika_wrapper as cw
from . import light_field
from . import lookup
from . import job_structure
from . import thrown_structure


def make_steering_card(
    random_seed,
    num_events,
    instrument,
    particle,
    site,
):
    steering_card = '' \
    'RUNNR {random_seed:d}\n' \
    'EVTNR 1\n' \
    'NSHOW {num_events:d}\n' \
    'PRMPAR  {prmpar:d}\n' \
    'ESLOPE {E_slope:.3e}\n' \
    'ERANGE {E_start:.3e} {E_stop:.3e}\n' \
    'THETAP 0. {max_zenith_angle_deg:.3f}\n' \
    'PHIP 0. 360.\n' \
    'SEED {seed1:d} 0 0\n' \
    'SEED {seed2:d} 0 0\n' \
    'SEED {seed3:d} 0 0\n' \
    'SEED {seed4:d} 0 0\n' \
    'OBSLEV {obslev:.3e}\n' \
    'FIXCHI 0.\n' \
    'MAGNET {Bx:.3e} {Bz:.3e}\n' \
    'ELMFLG T T\n' \
    'MAXPRT 1\n' \
    'PAROUT F F\n' \
    'TELESCOPE 0. 0. 0. {aperture_radius:.3f}\n' \
    'ATMOSPHERE {atmosphere:d} T\n' \
    'CWAVLG 250 700\n' \
    'CSCAT 1 {XSCAT_cm:.3f} 0.\n' \
    'CERQEF F T F\n' \
    'CERSIZ 1\n' \
    'CERFIL F\n' \
    'TELFIL "."\n' \
    'TSTART T\n' \
    ''.format(
        random_seed=random_seed,
        seed1=random_seed + 0,
        seed2=random_seed + 1,
        seed3=random_seed + 2,
        seed4=random_seed + 3,
        num_events=num_events,
        prmpar=particle["prmpar"],
        E_start=particle["E_start"],
        E_stop=particle["E_stop"],
        E_slope=particle["E_slope"],
        max_zenith_angle_deg=particle["max_zenith_angle_deg"],
        XSCAT_cm=particle["max_scatter_radius"]*1e2,
        Bx=site["earth_magnetic_field_x_muT"],
        Bz=site["earth_magnetic_field_z_muT"],
        atmosphere=site["atmosphere"],
        obslev=site["observation_level_altitude_asl"]*1e2,
        aperture_radius=instrument['aperture_radius']*1e2,)
    return steering_card


def process_run(
    tmp_dir,
    random_seed=1,
    num_events=100,
    eventio_converter_path='./build/merlict/merlict-eventio-converter',
    instrument=job_structure.example_job['instrument'],
    particle=job_structure.example_job['particle'],
    site=job_structure.example_job['site'],
    trigger_threshold=job_structure.example_job['trigger_threshold'],
    nsb_rate_pixel=job_structure.example_job['nsb_rate_pixel'],
):
    np.random.seed(random_seed)
    os.makedirs(tmp_dir, exist_ok=True)

    corsika_card = make_steering_card(
        random_seed=random_seed,
        num_events=num_events,
        instrument=instrument,
        particle=particle,
        site=site)
    corsika_card_path = os.path.join(tmp_dir, 'card.txt')
    with open(corsika_card_path, 'wt') as f:
        f.write(corsika_card)

    evtio_run_path = os.path.join(tmp_dir, 'run.evtio')
    cw.corsika(
        steering_card=cw.read_steering_card(corsika_card_path),
        output_path=evtio_run_path,
        save_stdout=True)

    assert os.stat(evtio_run_path).st_size < 3.7e9 , "Can not trust eventio-files larger 3.7GByte."

    sio_run_path = os.path.join(tmp_dir, 'run.sio')
    sp.call([
        eventio_converter_path,
        '-i', evtio_run_path,
        '-o', sio_run_path])

    plenoscope = light_field.init_Plenoscope(
        aperture_radius=instrument['aperture_radius'],
        num_paxel_on_diagonal=instrument['num_paxel_on_diagonal'],
        field_of_view_radius_deg=instrument['field_of_view_radius_deg'],
        num_pixel_on_diagonal=instrument['num_pixel_on_diagonal'],
        time_radius=instrument['time_radius'],
        num_time_slices=instrument['num_time_slices'],)

    lut_path = os.path.join(tmp_dir, 'run_{:06d}.lut'.format(random_seed))
    lua = lookup.LookUpAppender(
        path=lut_path,
        random_seed=random_seed,
        num_events=num_events,
        eventio_converter_path=eventio_converter_path,
        instrument=instrument,
        particle=particle,
        site=site,
        trigger_threshold=trigger_threshold)

    thrown_path = os.path.join(lut_path, lookup.THROWN_PATH)
    thrown = []

    corsika_run = sio.SimpleIoRun(sio_run_path)
    for event_idx in range(corsika_run.number_events):
        event = corsika_run[event_idx]

        features = extract_simulation_truth(event)
        light_field_sequence = extract_light_field_sequence(
            event.cherenkov_photon_bunches,
            plenoscope,
            instrument,
            relative_arrival_times_std=instrument['relative_arrival_times_std'])

        features['max_scatter_radius'] = float(
            1e-2*corsika_run.header.raw[248 - 1])
        assert corsika_run.header.raw[249 - 1] == 0.

        features['trigger'] = int(0)
        features['num_photons'] = int(light_field_sequence.shape[0])

        # image trigger
        # -------------
        image_photons = light_field.get_image_photons(
            lfs=light_field_sequence,
            plenoscope=plenoscope)
        image = light_field.get_image_from_image_photons(
            image_photons=image_photons,
            object_distance=12.5e3,
            cx_bin_edges=plenoscope.cx_bin_edges,
            cy_bin_edges=plenoscope.cy_bin_edges)
        nsb = np.reshape(
            np.random.normal(
                loc=nsb_rate_pixel,
                scale=np.sqrt(nsb_rate_pixel),
                size=image.size
            ),
            newshape=image.shape
        )
        image += nsb
        trigger_kernel = 7/9*np.array([[1,1,1],[1,1,1],[1,1,1]])  # only 7 pixel
        trigger_image = scipy.signal.convolve2d(
            image,
            trigger_kernel,
            mode='same')

        features['trigger_response'] = np.max(trigger_image)

        if features['trigger_response'] > trigger_threshold:
            features['trigger'] = int(1)



        if features['trigger']:
            features = extract_and_append_indexing_features(
                features=features,
                light_field_sequence=light_field_sequence)
            lua.append_event(
                light_field_sequence_uint8=light_field_sequence,
                features=features)

        thrown.append(thrown_structure.features_to_array_float32(features))

    thrown = np.array(thrown)
    with open(thrown_path, 'wb') as fout:
        fout.write(thrown.tobytes())



def extract_simulation_truth(event):
    truth = {}
    truth['event'] = int(event.header.raw[2 - 1])
    truth['run'] = int(event.header.raw[44 - 1])
    truth['particle_id'] = int(event.header.primary_particle_id)
    truth['particle_energy'] = float(event.header.total_energy_GeV)
    particle_momentum = event.header.momentum()
    particle_direction = particle_momentum/np.linalg.norm(
        particle_momentum)
    truth['particle_cx'] = float(particle_direction[0])
    truth['particle_cy'] = float(particle_direction[1])
    truth['particle_x'] = float(event.header.core_position_x_meter())
    truth['particle_y'] = float(event.header.core_position_y_meter())
    truth['particle_height_first_interaction'] = float(
        np.abs(event.header.raw[7 - 1]*1e-2))
    return truth


def extract_and_append_indexing_features(features, light_field_sequence):
    lfs = light_field_sequence
    features['image_cx_median'] = float(np.median(lfs[:, 0]))
    features['image_cy_median'] = float(np.median(lfs[:, 1]))
    return features


def extract_light_field_sequence(
    cherenkov_photon_bunches,
    plenoscope,
    instrument,
    relative_arrival_times_std=1e-9,
):
    cpb = cherenkov_photon_bunches
    photons = light_field.PhotonObservables(
        x=cpb.x,
        y=cpb.y,
        cx=cpb.cx,
        cy=cpb.cy,
        relative_arrival_times=cpb.arrival_time_since_first_interaction)

    num_photons_emitted = cpb.x.shape[0]
    probs = np.random.uniform(size=num_photons_emitted)
    passed_atmosphere = cpb.probability_to_reach_observation_level > probs
    photons = light_field.cut_PhotonObservables(
        photons=photons,
        mask=passed_atmosphere)

    field_of_view_radius = np.deg2rad(instrument['field_of_view_radius_deg'])
    photons_c = np.hypot(photons.cx, photons.cy)
    in_field_of_view = photons_c < field_of_view_radius
    photons = light_field.cut_PhotonObservables(
        photons=photons,
        mask=in_field_of_view)

    num_photons_ground = len(photons.x)
    dices = np.random.uniform(size=num_photons_ground)
    detected = dices < (
        instrument['mirror_reflectivity']*
        instrument['photo_detection_efficiency'])
    photons = light_field.cut_PhotonObservables(
        photons=photons,
        mask=detected)

    relative_arrival_times = photons.relative_arrival_times
    relative_arrival_times += np.random.normal(
        loc=0.,
        scale=relative_arrival_times_std)
    relative_arrival_times -= np.median(relative_arrival_times)
    photons = light_field.PhotonObservables(
        x=photons.x,
        y=photons.y,
        cx=photons.cx,
        cy=photons.cy,
        relative_arrival_times=relative_arrival_times)

    return light_field.photons_to_light_field_sequence(
        photons=photons,
        plenoscope=plenoscope)
