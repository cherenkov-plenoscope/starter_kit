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
import os
from os.path import join
from subprocess import call
import acp_instrument_response_function as acp_irf


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        od = arguments['--out_dir']

        # 1) Light-field-calibration of Plenoscope
        # ----------------------------------------
        os.makedirs(od, exist_ok=True)
        if not os.path.isdir(join(od, 'light_field_calibration')):
            call([
                join('build', 'mctracer', 'mctPlenoscopeCalibration'),
                '--scenery', join('resources', 'acp', '71m', 'scenery'),
                '--number_mega_photons', arguments['--lfc_Mp'],
                '--output', join(od, 'light_field_calibration')])

        # 2) Instrument-response-functions for typical cosmic particles
        # -------------------------------------------------------------
        particles = ['gamma', 'electron', 'proton']
        os.makedirs(join(od, 'irf'), exist_ok=True)
        for p in particles:
            if not os.path.isdir(join(od, 'irf', p)):
                command = [
                    'acp_trigger_irf',
                    '--steering_card', join(
                        'resources', 'acp', '71m', p+'_steering.json'),
                    '--output_path', join(od, 'irf', p),
                    '--acp_detector', join(od, 'light_field_calibration'),
                    '--mct_acp_config', join(
                        'resources', 'acp', 'mct_propagation_config.xml'),
                    '--mct_acp_propagator', join(
                        'build', 'mctracer', 'mctPlenoscopePropagation'),]
                if arguments['--scoop_hosts']:
                    command.insert(1, '--scoop_hosts')
                    command.insert(2, arguments['--scoop_hosts'])
                call(command)

        for p in particles:
            if os.path.isdir(join(od, 'irf', p)):
                acp_irf.trigger_study_analysis.run_analysis(join(od, 'irf', p))

        # 3) Sensitivity and time-to-detections of the ACP
        # ------------------------------------------------
        os.makedirs(join(od, 'isf'), exist_ok=True)
        call([
            'acp_isez',
            '--gamma_area', join(
                od, 'irf', 'gamma', 'results', 'irf.csv'),
            '--electron_acceptance', join(
                od, 'irf', 'electron', 'results', 'irf.csv'),
            '--proton_acceptance', join(
                od, 'irf', 'proton', 'results', 'irf.csv'),
            '--cutoff', '0.01',
            '--rel_flux', '0.05',
            '--fov', '6.5',
            '--src', '3FGL J2254.0+1608',
            '--out', join(od, 'isf')])

    except docopt.DocoptExit as e:
        print(e)
