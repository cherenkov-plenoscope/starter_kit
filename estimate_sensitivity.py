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
import acp_instrument_sensitivity_function as isf
import random
import plenopy as pl
import json
import multiprocessing


def absjoin(*args):
    return op.abspath(op.join(*args))

USE_CLUSTER = True

if __name__ == '__main__':
    try:
        if USE_CLUSTER:
            pool = sge
        else:
            pool = multiprocessing.Pool(8)

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

        # ----------------------------------------
        # Light-field-calibration of Plenoscope
        # ----------------------------------------
        print("-------light-field-geometry---------")
        os.makedirs(out_dir, exist_ok=True)
        if not op.isdir(op.join(out_dir, 'light_field_calibration')):
            lfc_tmp_dir = op.join(out_dir, 'light_field_calibration.tmp')
            os.makedirs(lfc_tmp_dir)
            lfc_jobs = plmr.make_jobs_light_field_geometry(
                merlict_map_path=absjoin(
                    'build',
                    'merlict',
                    'merlict-plenoscope-calibration-map'),
                scenery_path=absjoin(
                    'resources',
                    'acp',
                    '71m',
                    'scenery'),
                out_dir=lfc_tmp_dir,
                num_photons_per_block=1000*1000,
                num_blocks=int(arguments['--lfc_Mp']),
                random_seed=0)

            rc = pool.map(
                plmr.run_job_light_field_geometry,
                lfc_jobs)
            subprocess.call([absjoin(
                    'build',
                    'merlict',
                    'merlict-plenoscope-calibration-reduce'),
                '--input', lfc_tmp_dir,
                '--output', op.join(out_dir, 'light_field_calibration')])
            shutil.rmtree(lfc_tmp_dir)

        # --------------------------------------------------
        # Instrument-response to typical cosmic particles
        # --------------------------------------------------
        print("-------instrument-response---------")
        particles = ['gamma', 'electron', 'proton']
        jobs = []
        os.makedirs(op.join(out_dir, 'irf'), exist_ok=True)
        for p in particles:
            if not op.isdir(op.join(out_dir, 'irf', p)):
                if p in ['electron', 'proton']:
                     location_steering_card_path = absjoin(
                        'resources',
                        'acp',
                        '71m',
                        'chile_paranal_magnet_off.json')
                else:
                     location_steering_card_path = absjoin(
                        'resources',
                        'acp',
                        '71m',
                        'chile_paranal.json')
                jobs += irf.trigger_simulation.make_output_directory_and_jobs(
                    particle_steering_card_path=absjoin(
                        'resources',
                        'acp',
                        '71m',
                        p+'_steering.json'),
                    location_steering_card_path=location_steering_card_path,
                    output_path=op.join(
                        out_dir,
                        'irf',
                        p),
                    acp_detector_path=op.join(
                        out_dir,
                        'light_field_calibration'),
                    mct_acp_config_path=absjoin(
                        'resources',
                        'acp',
                        'merlict_propagation_config.json'),
                    mct_acp_propagator_path=absjoin(
                        'build',
                        'merlict',
                        'merlict-plenoscope-propagation'),
                    trigger_patch_threshold=trigger['patch_threshold'],
                    trigger_integration_time_in_slices=trigger[
                        'integration_time_in_slices'])
        random.shuffle(jobs)
        rc = pool.map(
            irf.trigger_simulation.run_job,
            jobs)

        print("-------trigger-study---------")
        for p in particles:
            if (
                op.isdir(op.join(out_dir, 'irf', p)) and
                not op.isdir(op.join(out_dir, 'irf', p, 'results'))
            ):
                irf.trigger_study_analysis.run_analysis(
                    path=op.join(out_dir, 'irf', p),
                    patch_threshold=trigger['patch_threshold'])

        # -------------------------------
        # Classifying Cherenkov-photons
        # -------------------------------
        print("-------classifying Cherenkov---------")
        cla_jobs = []
        for p in particles:
            run_path = op.join(out_dir, 'irf', p, 'past_trigger')
            p_jobs = plmr.make_jobs_cherenkov_classification(
                light_field_geometry_path=absjoin(
                    out_dir,
                    'light_field_calibration'),
                run_path=run_path,
                num_events_in_job=100,
                override=False)
            cla_jobs += p_jobs
        random.shuffle(cla_jobs)
        rc = pool.map(
            plmr.run_job_cherenkov_classification,
            cla_jobs)


        # ------------------------------------------------
        # Sensitivity and time-to-detections of the ACP
        # ------------------------------------------------
        print("-------calculate Sensitivity---------")
        if not op.isdir(op.join(out_dir, 'isf_beamer')):
            os.makedirs(op.join(out_dir, 'isf_beamer'), exist_ok=True)
            results_2 = isf.analysis(
                gamma_collection_area_path=op.join(
                    out_dir, 'irf', 'gamma', 'results', 'irf.csv'),
                electron_collection_acceptance_path=op.join(
                    out_dir, 'irf', 'electron', 'results', 'irf.csv'),
                proton_collection_acceptance_path=op.join(
                    out_dir, 'irf', 'proton', 'results', 'irf.csv'),
                rigidity_cutoff_in_tev=0.01,
                relative_flux_below_cutoff=0.05,
                fov_in_deg=6.5,
                source_name='3FGL J2254.0+1608',
                out_dir=op.join(out_dir, 'isf_beamer'),
                dpi=300,
                pixel_rows=1080,
                pixel_columns=1920,
                lmar=0.12,
                bmar=0.12,
                tmar=0.02,
                rmar=0.02,)

    except docopt.DocoptExit as e:
        print(e)
