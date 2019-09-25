#! /usr/bin/env python
"""
Estimate the sensitivity of the Atmospheric Cherenkov Plenoscope (ACP).

Usage: trigger_sensitivity [--out_dir=DIR] [--lfc_Mp=NUMBER]

Options:
    -h --help               Prints this help message.
    -o --out_dir=DIR        Output directory [default: ./run]
    --lfc_Mp=NUMBER         How many mega photons to be used during the
                            light field calibration of the plenoscope.
                            [default: 1337]
"""
import docopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sun_grid_engine_map as sge
import plenoscope_map_reduce as plmr
import os
from os import path as op
import shutil
import subprocess
import acp_instrument_response_function as irf
import random
import plenopy as pl
import json
import multiprocessing


def absjoin(*args):
    return op.abspath(op.join(*args))


USE_CLUSTER = False

if __name__ == '__main__':
    try:
        if USE_CLUSTER:
            pool = sge
        else:
            pool = multiprocessing.Pool(4)

        print("-------start---------")
        trigger_steering_path = absjoin(
            'resources',
            'acp',
            '71m',
            'trigger_steering.json')
        with open(trigger_steering_path, 'rt') as fin:
            trigger = json.loads(fin.read())

        arguments = docopt.docopt(__doc__)
        out_dir = op.abspath(arguments['--out_dir'])

        print("-------light-field-geometry---------")
        os.makedirs(out_dir, exist_ok=True)
        lfg_path = absjoin(out_dir, 'light_field_geometry')
        if not op.exists(lfg_path):
            lfg_tmp_dir = lfg_path+".tmp"
            os.makedirs(lfg_tmp_dir)
            lfg_jobs = plmr.make_jobs_light_field_geometry(
                merlict_map_path=absjoin(
                    'build',
                    'merlict',
                    'merlict-plenoscope-calibration-map'),
                scenery_path=absjoin(
                    'resources',
                    'acp',
                    '71m',
                    'scenery'),
                out_dir=lfg_tmp_dir,
                num_photons_per_block=1000*1000,
                num_blocks=int(arguments['--lfc_Mp']),
                random_seed=0)

            rc = pool.map(plmr.run_job_light_field_geometry, lfg_jobs)
            subprocess.call([
                absjoin(
                    'build',
                    'merlict',
                    'merlict-plenoscope-calibration-reduce'),
                '--input', lfg_tmp_dir,
                '--output', lfg_path])
            shutil.rmtree(lfg_tmp_dir)

        print("-------instrument-response---------")
        particles = ['gamma', 'electron', 'proton']
        jobs = []
        os.makedirs(op.join(out_dir, 'irf'), exist_ok=True)
        for p in particles:
            if not op.isdir(op.join(out_dir, 'irf', p)):
                if p in ['electron', 'proton']:
                    location_config_path = absjoin(
                        'resources',
                        'acp',
                        '71m',
                        'chile_paranal_magnet_off.json')
                else:
                    location_config_path = absjoin(
                        'resources',
                        'acp',
                        '71m',
                        'chile_paranal.json')
                jobs += irf.make_output_directory_and_jobs(
                    output_dir=absjoin(out_dir, 'irf', p),
                    num_energy_bins=10,
                    num_events_in_energy_bin=25,
                    particle_config_path=absjoin(
                        'resources',
                        'acp',
                        '71m',
                        '.low_energy_'+p+'_steering.json'),
                    location_config_path=location_config_path,
                    light_field_geometry_path=lfg_path,
                    merlict_plenoscope_propagator_path=absjoin(
                        'build',
                        'merlict',
                        'merlict-plenoscope-propagation'),
                    merlict_plenoscope_propagator_config_path=absjoin(
                        'resources',
                        'acp',
                        'merlict_propagation_config.json'),
                    corsika_path=absjoin(
                        "build",
                        "corsika",
                        "corsika-75600",
                        "run",
                        "corsika75600Linux_QGSII_urqmd"),
                    trigger_patch_threshold=trigger['patch_threshold'],
                    trigger_integration_time_in_slices=trigger[
                        'integration_time_in_slices'])
        random.shuffle(jobs)
        rc = pool.map(irf.run_job, jobs)

        print("-------reducing thrown/triggered---------")
        for p in particles:
            p_dir = op.join(out_dir, 'irf', p)
            if not op.exists(op.join(p_dir, "thrown.jsonl")):
                irf.concatenate_files(
                    wildcard_path=op.join(p_dir, "__thrown", "*.jsonl"),
                    out_path=op.join(p_dir, "thrown.jsonl"))
            if not op.exists(op.join(p_dir, "triggered.jsonl")):
                irf.concatenate_files(
                    wildcard_path=op.join(p_dir, "__triggered", "*.jsonl"),
                    out_path=op.join(p_dir, "triggered.jsonl"))

        print("-------classifying Cherenkov---------")
        cla_jobs = []
        for p in particles:
            run_path = op.join(out_dir, 'irf', p, 'past_trigger')
            score_dir = op.join(out_dir, 'irf', p, '__cherenkov')
            os.makedirs(score_dir, exist_ok=True)
            p_jobs = plmr.make_jobs_cherenkov_classification(
                light_field_geometry_path=lfg_path,
                run_path=run_path,
                score_dir=score_dir,
                num_events_in_job=100,
                override=False)
            cla_jobs += p_jobs
        random.shuffle(cla_jobs)
        rc = pool.map(plmr.run_job_cherenkov_classification, cla_jobs)

        print("-------reducing Cherenkov---------")
        for p in particles:
            score_dir = op.join(out_dir, 'irf', p, '__cherenkov')
            score_path = op.join(out_dir, 'irf', p, 'cherenkov.jsonl')
            if not op.exists(score_path):
                irf.concatenate_files(
                    wildcard_path=op.join(score_dir, "*.jsonl"),
                    out_path=score_path)

        print("-------extracting features---------")
        feature_jobs = []
        for p in particles:
            past_trigger_path = op.join(out_dir, 'irf', p, 'past_trigger')
            feature_path = op.join(out_dir, 'irf', p, 'features.jsonl')
            feature_map_dir = op.join(out_dir, 'irf', p, '__features')
            true_particle_id = irf.__particle_str_to_corsika_id(p)
            os.makedirs(feature_map_dir, exist_ok=True)
            if not op.exists(feature_path):
                feature_jobs += plmr.make_jobs_feature_extraction(
                    past_trigger_path=past_trigger_path,
                    true_particle_id=true_particle_id,
                    light_field_geometry_path=lfg_path,
                    feature_map_dir=feature_map_dir,
                    num_events_in_job=250)

        rc = pool.map(plmr.run_job_feature_extraction, feature_jobs)

        print("-------reducing features---------")
        for p in particles:
            feature_map_dir = op.join(out_dir, 'irf', p, '__features')
            feature_path = op.join(out_dir, 'irf', p, 'features.jsonl')
            if not op.exists(feature_path):
                irf.concatenate_files(
                    wildcard_path=op.join(feature_map_dir, "*.jsonl"),
                    out_path=feature_path)

    except docopt.DocoptExit as e:
        print(e)
