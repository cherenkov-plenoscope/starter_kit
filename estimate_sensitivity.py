#! /usr/bin/env python
"""
Estimate the sensitivity of the Atmospheric Cherenkov Plenoscope (ACP).

Usage: trigger_sensitivity [--out_dir=DIR] [-s=PATH] [--lfc_Mp=NUMBER]

Options:
    -h --help               Prints this help message.
    -o --out_dir=DIR        Output directory [default: ./run]
    -s --scoop_hosts=PATH   Path to the scoop hosts text file.
    --lfc_Mp=NUMBER         How many mega photons to be used during the
                            light field calibration of the plenoscope.
                            [default: 1337]
"""
import docopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sun_grid_engine_map as sge
import os
from os.path import join
from os.path import abspath
import shutil
from subprocess import call
import acp_instrument_response_function as irf
import acp_instrument_sensitivity_function as isf
import random
import plenopy as pl
import json


if __name__ == '__main__':
    try:
        trigger_steering_path = join(
            'resources', 'acp', '71m', 'trigger_steering.json')
        with open(trigger_steering_path, 'rt') as fin:
            trigger = json.loads(fin.read())
        # for zero accidental-rate within the events simulated here.
        # for 5.0ns integration-time, i.e. 10 integration_time_in_slices

        arguments = docopt.docopt(__doc__)
        od = os.path.abspath(arguments['--out_dir'])

        # 1) Light-field-calibration of Plenoscope
        # ----------------------------------------
        os.makedirs(od, exist_ok=True)
        if not os.path.isdir(join(od, 'light_field_calibration')):
            lfc_tmp_dir = join(od, 'light_field_calibration.tmp')
            os.makedirs(lfc_tmp_dir)
            lfc_jobs = pl.light_field_geometry.map_reduce.make_jobs(
                merlict_map_path=abspath(
                    join('build', 'merlict', 'merlict-plenoscope-calibration-map')),
                scenery_path=os.path.abspath(join('resources', 'acp', '71m', 'scenery')),
                out_dir=lfc_tmp_dir,
                num_photons_per_block=1000*1000,
                num_blocks=int(arguments['--lfc_Mp']),
                random_seed=0)

            rc = sge.map(pl.light_field_geometry.map_reduce.worker_node, lfc_jobs)
            call([
                abspath(join('build', 'merlict', 'merlict-plenoscope-calibration-reduce')),
                '--input', lfc_tmp_dir,
                '--output', join(od, 'light_field_calibration')])
            shutil.rmtree(lfc_tmp_dir)

        # 2) Instrument-response-functions for typical cosmic particles
        # -------------------------------------------------------------
        particles = ['gamma', 'electron', 'proton']
        jobs = []
        os.makedirs(join(od, 'irf'), exist_ok=True)
        for p in particles:
            if not os.path.isdir(join(od, 'irf', p)):
                jobs += irf.trigger_simulation.make_output_directory_and_jobs(
                    particle_steering_card_path=abspath(join(
                        'resources', 'acp', '71m', p+'_steering.json')),
                    location_steering_card_path=abspath(join(
                        'resources', 'acp', '71m', 'chile_paranal.json')),
                    output_path=join(od, 'irf', p),
                    acp_detector_path=join(od, 'light_field_calibration'),
                    mct_acp_config_path=abspath(join(
                        'resources', 'acp', 'merlict_propagation_config.json')),
                    mct_acp_propagator_path=abspath(join(
                        'build', 'merlict', 'merlict-plenoscope-propagation')),
                    trigger_patch_threshold=trigger['patch_threshold'],
                    trigger_integration_time_in_slices=trigger[
                        'integration_time_in_slices'])

        random.shuffle(jobs)
        rc = sge.map(irf.trigger_simulation.run_job, jobs)

        for p in particles:
            if (
                os.path.isdir(join(od, 'irf', p)) and
                not os.path.isdir(join(od, 'irf', p, 'results'))
            ):
                irf.trigger_study_analysis.run_analysis(
                    path=join(od, 'irf', p),
                    patch_threshold=trigger['patch_threshold'])

        # 3) Sensitivity and time-to-detections of the ACP
        # ------------------------------------------------
        if not os.path.isdir(join(od, 'isf')):
            os.makedirs(join(od, 'isf'), exist_ok=True)
            results = isf.analysis(
                gamma_collection_area_path=join(
                    od, 'irf', 'gamma', 'results', 'irf.csv'),
                electron_collection_acceptance_path=join(
                    od, 'irf', 'electron', 'results', 'irf.csv'),
                proton_collection_acceptance_path=join(
                    od, 'irf', 'proton', 'results', 'irf.csv'),
                rigidity_cutoff_in_tev=0.01,
                relative_flux_below_cutoff=0.05,
                fov_in_deg=6.5,
                source_name='3FGL J2254.0+1608',
                out_dir=join(od, 'isf'))

        if not os.path.isdir(join(od, 'isf_beamer')):
            os.makedirs(join(od, 'isf_beamer'), exist_ok=True)
            results_2 = isf.analysis(
                gamma_collection_area_path=join(
                    od, 'irf', 'gamma', 'results', 'irf.csv'),
                electron_collection_acceptance_path=join(
                    od, 'irf', 'electron', 'results', 'irf.csv'),
                proton_collection_acceptance_path=join(
                    od, 'irf', 'proton', 'results', 'irf.csv'),
                rigidity_cutoff_in_tev=0.01,
                relative_flux_below_cutoff=0.05,
                fov_in_deg=6.5,
                source_name='3FGL J2254.0+1608',
                out_dir=join(od, 'isf_beamer'),
                dpi=300,
                pixel_rows=1080,
                pixel_columns=1920,
                lmar=0.12,
                bmar=0.12,
                tmar=0.02,
                rmar=0.02,)

    except docopt.DocoptExit as e:
        print(e)
