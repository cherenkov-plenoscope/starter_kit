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
import scoop
import os
from os.path import join
from subprocess import call
import acp_instrument_response_function as irf
import acp_instrument_sensitivity_function as isf
import random


if __name__ == '__main__':
    try:
        patch_threshold = 103
        # for zero accidental-rate within the events simulated here.
        # for 5.0ns integration-time, i.e. 10 integration_time_in_slices
        integration_time_in_slices = 10

        arguments = docopt.docopt(__doc__)
        od = arguments['--out_dir']

        # 1) Light-field-calibration of Plenoscope
        # ----------------------------------------
        os.makedirs(od, exist_ok=True)
        if not os.path.isdir(join(od, 'light_field_calibration')):
            call([
                join('build', 'merlict', 'merlict-plenoscope-calibration'),
                '--scenery', join('resources', 'acp', '71m', 'scenery'),
                '--number_mega_photons', arguments['--lfc_Mp'],
                '--output', join(od, 'light_field_calibration')])

        # 2) Instrument-response-functions for typical cosmic particles
        # -------------------------------------------------------------
        particles = ['gamma', 'electron', 'proton']
        jobs = []
        os.makedirs(join(od, 'irf'), exist_ok=True)
        for p in particles:
            if not os.path.isdir(join(od, 'irf', p)):
                jobs += irf.trigger_simulation.make_output_directory_and_jobs(
                    steering_card_path=join(
                        'resources', 'acp', '71m', p+'_steering.json'),
                    output_path=join(od, 'irf', p),
                    acp_detector_path=join(od, 'light_field_calibration'),
                    mct_acp_config_path=join(
                        'resources', 'acp', 'mct_propagation_config.xml'),
                    mct_acp_propagator_path=join(
                        'build', 'merlict', 'merlict-plenoscope-propagation'),
                    trigger_patch_threshold=patch_threshold,
                    trigger_integration_time_in_slices=(
                        integration_time_in_slices))

        random.shuffle(jobs)
        rc = list(scoop.futures.map(irf.trigger_simulation.run_job, jobs))

        for p in particles:
            if (os.path.isdir(join(od, 'irf', p)) and
                not os.path.isdir(join(od, 'irf', p, 'results'))
            ):
                irf.trigger_study_analysis.run_analysis(
                    path=join(od, 'irf', p),
                    patch_threshold=patch_threshold)

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
