import os
import glob
import subprocess
import numpy as np
import plenopy as pl
import json


def split_list_into_list_of_lists(events, num_events_in_job):
    """
    Splits a list into a list of sublists.
    When the individual event to be processed is too little workload to justify
    startup-overhead on worker-nodes, these events can be packed into jobs with
    num_events_in_job number of events in each.
    """
    num_events = len(events)
    jobs = []
    event_counter = 0
    while event_counter < num_events:
        job_event_counter = 0
        job = []
        while (
            event_counter < num_events and
            job_event_counter < num_events_in_job
        ):
            job.append(events[event_counter])
            job_event_counter += 1
            event_counter += 1
        jobs.append(job)
    return jobs


def make_jobs_cherenkov_classification(
    light_field_geometry_path,
    run_path,
    score_dir,
    num_events_in_job=100,
    override=False,
):
    light_field_geometry_path = os.path.abspath(light_field_geometry_path)
    paths_in_run = glob.glob(os.path.join(run_path, '*'))

    # Events have digit filenames.
    event_numbers = []
    for path_in_run in paths_in_run:
        filename = os.path.split(path_in_run)[-1]
        if str.isdigit(filename):
            event_numbers.append(int(filename))

    # Be lazy. Only process when override==True or output does not exist yet.
    event_numbers_to_process = []
    for event_number in event_numbers:
        if override:
            event_numbers_to_process.append(event_number)
        else:
            event_path = os.path.join(run_path, "{:012d}".format(event_number))
            dense_photon_ids_path = os.path.join(
                event_path,
                'dense_photon_ids.uint32.gz')
            if not os.path.exists(dense_photon_ids_path):
                event_numbers_to_process.append(event_number)

    # Make chunks for efficiency.
    chunks = split_list_into_list_of_lists(
        events=event_numbers_to_process,
        num_events_in_job=num_events_in_job)

    jobs = []
    for i, chunk in enumerate(chunks):
        job = {
            "light_field_geometry_path": light_field_geometry_path,
            "run_path": run_path,
            "event_numbers": chunk,
            "score_path": os.path.join(score_dir, "{:06d}.jsonl".format(i))}
        jobs.append(job)
    return jobs


def run_job_cherenkov_classification(job):
    light_field_geometry = pl.LightFieldGeometry(
        job['light_field_geometry_path'])
    scores = []
    for event_number in job['event_numbers']:
        event_path = os.path.join(job['run_path'], "{:012d}".format(event_number))
        event = pl.Event(event_path, light_field_geometry)
        roi = pl.classify.center_for_region_of_interest(event)
        photons = pl.classify.RawPhotons.from_event(event)
        cherenkov_photons, s = pl.classify.cherenkov_photons_in_roi_in_image(
            roi=roi,
            photons=photons)
        pl.classify.write_dense_photon_ids_to_event(
            event_path=os.path.abspath(event._path),
            photon_ids=cherenkov_photons.photon_ids,
            settings=s)
        score = pl.classify.benchmark(
            pulse_origins=event.simulation_truth.detector.pulse_origins,
            photon_ids_cherenkov=cherenkov_photons.photon_ids)
        score["true_particle_id"] = event.simulation_truth.event. \
            corsika_event_header.primary_particle_id
        score["run_id"] = event.simulation_truth.event. \
            corsika_run_header.number
        score["event_id"] = event.simulation_truth. \
            event.corsika_event_header.number
        scores.append(score)

    with open(job["score_path"], "wt") as fout:
        for score in scores:
            fout.write(json.dumps(score)+"\n")



def make_jobs_light_field_geometry(
    merlict_map_path,
    scenery_path,
    num_photons_per_block,
    out_dir,
    num_blocks,
    random_seed=0
):
    jobs = []
    for seed in np.arange(random_seed, num_blocks):
        jobs.append({
            "merlict_map_path": merlict_map_path,
            "scenery_path": scenery_path,
            "random_seed": seed,
            "out_dir": out_dir,
            "num_photons_per_block": num_photons_per_block})
    return jobs


def run_job_light_field_geometry(job):
    seed_str = '{:d}'.format(job['random_seed'])
    call = [
        job['merlict_map_path'],
        '-s', job['scenery_path'],
        '-n', '{:d}'.format(job['num_photons_per_block']),
        '-o', os.path.join(job['out_dir'], seed_str),
        '-r', seed_str]
    return subprocess.call(call)


def make_jobs_feature_extraction(
    past_trigger_path,
    true_particle_id,
    light_field_geometry_path,
    feature_map_dir,
    num_events_in_job=100
):
    event_ids = pl.tools.acp_format.all_folders_with_digit_names_in_path(
        past_trigger_path)
    event_paths = []
    for event_id in event_ids:
        event_path = os.path.abspath(
            os.path.join(past_trigger_path, "{:012d}".format(event_id)))
        event_paths.append(event_path)

    lol = split_list_into_list_of_lists(
        events=event_paths,
        num_events_in_job=num_events_in_job)

    jobs = []
    for i, l in enumerate(lol):
        job = {}
        job["event_paths"] = l
        job["true_particle_id"] = int(true_particle_id)
        job["light_field_geometry_path"] = light_field_geometry_path
        job["feature_path"] = os.path.join(
            feature_map_dir,
            "{:06d}.jsonl".format(i))
        jobs.append(job)
    return jobs


def run_job_feature_extraction(job):
    event_paths=job["event_paths"]
    light_field_geometry_path=job["light_field_geometry_path"]
    true_particle_id=job["true_particle_id"]

    lfg = pl.LightFieldGeometry(light_field_geometry_path)

    lfg_addon = {}
    lfg_addon["paxel_radius"] = \
        lfg.sensor_plane2imaging_system.\
            expected_imaging_system_max_aperture_radius/\
        lfg.sensor_plane2imaging_system.number_of_paxel_on_pixel_diagonal
    lfg_addon["nearest_neighbor_paxel_enclosure_radius"] = \
        3*lfg_addon["paxel_radius"]
    lfg_addon["paxel_neighborhood"] = pl.features.estimate_nearest_neighbors(
        x=lfg.paxel_pos_x,
        y=lfg.paxel_pos_y,
        epsilon=lfg_addon["nearest_neighbor_paxel_enclosure_radius"])
    lfg_addon["fov_radius"] = \
        .5*lfg.sensor_plane2imaging_system.max_FoV_diameter
    lfg_addon["fov_radius_leakage"] = 0.9*lfg_addon["fov_radius"]
    lfg_addon["num_pixel_on_diagonal"] = \
        np.floor(2*np.sqrt(lfg.number_pixel/np.pi))

    features = []
    for event_path in event_paths:
        event = pl.Event(event_path, light_field_geometry=lfg)

        run_id = event.simulation_truth.event.corsika_run_header.number
        event_id = np.mod(event.number, 1000000)

        try:
            cp = event.cherenkov_photons
            if cp is None:
                raise RuntimeError("No Cherenkov-photons classified yet.")
            f = pl.features.extract_features(
                cherenkov_photons=cp,
                light_field_geometry=lfg,
                light_field_geometry_addon=lfg_addon)
            f["true_particle_id"] = int(true_particle_id)
            f["run_id"] = int(run_id)
            f["event_id"] = int(event_id)
            features.append(f)
        except Exception as e:
            print("Run {:d}, Event: {:d} :".format(run_id, event_id), e)

    with open(job["feature_path"], "wt") as fout:
        for event_features in features:
            fout.write(json.dumps(event_features) + "\n")

    return 0
