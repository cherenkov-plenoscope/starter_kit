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
import sun_grid_engine_map as sge
import os
from os.path import join
from os.path import abspath
import shutil
from subprocess import call
import acp_instrument_response_function as irf
import random
import plenopy as pl


if __name__ == '__main__':
    try:
        patch_threshold = 103
        # for zero accidental-rate within the events simulated here.
        # for 5.0ns integration-time, i.e. 10 integration_time_in_slices
        integration_time_in_slices = 10

        arguments = docopt.docopt(__doc__)
        od = os.path.abspath(arguments['--out_dir'])

        # 1) Light-field-calibration of Plenoscope
        # ----------------------------------------
        assert os.path.isdir(join(od, 'light_field_calibration')):

        # 2) diffuse gammas
        # -------------------------------------------------------------
        assert os.path.isdir(join(od, 'irf'))
        p = 'gamma_diffuse'
        jobs = []
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
                trigger_patch_threshold=patch_threshold,
                trigger_integration_time_in_slices=(
                    integration_time_in_slices))

        random.shuffle(jobs)
        rc = sge.map(irf.trigger_simulation.run_job, jobs)

    except docopt.DocoptExit as e:
        print(e)
