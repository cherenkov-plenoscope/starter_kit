import os
import glob
import subprocess
import numpy as np
import plenopy as pl


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
    run_path,
    light_field_geometry_path,
    num_events_in_job=100,
    override=False,
):
    light_field_geometry_path = os.path.abspath(light_field_geometry_path)
    run_path = os.path.abspath(run_path)
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
            event_path = os.path.join(run_path, "{:d}".format(event_number))
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
    for chunk in chunks:
        job = {
            "light_field_geometry_path": light_field_geometry_path,
            "run_path": run_path,
            "event_numbers": chunk}
        jobs.append(job)
    return jobs


def run_job_cherenkov_classification(job):
    light_field_geometry = pl.LightFieldGeometry(
        job['light_field_geometry_path'])
    for event_number in job['event_numbers']:
        event_path = os.path.join(job['run_path'], str(event_number))
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
    light_field_geometry_path,
    num_events_in_job=100
):
    event_ids = pl.tools.acp_format.all_folders_with_digit_names_in_path(
        past_trigger_path)
    event_paths = []
    for event_id in event_ids:
        event_path = os.path.abspath(
            os.path.join(past_trigger_path, "{:d}".format(event_id)))
        event_paths.append(event_path)

    lol = split_list_into_list_of_lists(
        events=event_paths,
        num_events_in_job=num_events_in_job)

    jobs = []
    for i, l in enumerate(lol):
        job = {}
        job["event_paths"] = l
        job["light_field_geometry_path"] = os.path.abspath(
            light_field_geometry_path)
        jobs.append(job)
    return jobs


def run_job_feature_extraction(job):
    features = pl.features.extract_features_from_events(
        event_paths=job["event_paths"],
        light_field_geometry_path=job["light_field_geometry_path"])
    return features