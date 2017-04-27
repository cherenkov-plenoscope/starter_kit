#! /usr/bin/env python
"""
Run the example workflow to estimate the sensitivity of the 
Atmospheric Cherenkov Plenoscope (ACP).

You can run the simulation in parallel using a python scoop cluster, but you do 
not have to.

The default number of particle observation runs and light field calibration 
photons is rather large (same as in our ACP introduction paper) and will take 
some time (9h on our 96 core cluster).

Usage: example_workflow [-s=SCOOP_HOSTS] [--number_of_runs=RUNS] [--lfc_Mp=MEGA_PHOTONS]

Options:
    -s --scoop_hosts=SCOOP_HOSTS    Path to the scoop hosts text file.
    --number_of_runs=RUNS           How many runs of to simulate for each 
                                    particle [default: 3840]
    --lfc_Mp=MEGA_PHOTONS           How many mega photons to be used during the 
                                    light field calibration of the plenoscope. 
                                    [default: 1000]
"""
import docopt
import os
from os.path import join
from subprocess import call
import acp_instrument_response_function as acp_irf


def main():
    try:
        arguments = docopt.docopt(__doc__)

        # 1) Light field calibration of Plenoscope
        # ----------------------------------------
        os.makedirs('run', exist_ok=True)
        if not os.path.isdir(join('run','light_field_calibration')):
            call([
                join('build','mctracer','mctPlenoscopeCalibration'),
                '--scenery', join('resources', '71m_acp', 'scenery'),
                '--number_mega_photons', arguments['--lfc_Mp'],
                '--output', join('run','light_field_calibration')
            ])


        # 2) Instrument response functions for typical cosmic particles
        # -------------------------------------------------------------
        particles = ['gamma', 'electron', 'proton']
        os.makedirs(join('run','irf'), exist_ok=True)
        for p in particles:
            if not os.path.isdir(join('run','irf',p)):
                command = [
                    'acp_instrument_response_function',
                    '--corsika_card', join('resources', '71m_acp', p+'_steering_card.txt'),
                    '--output_path', join('run','irf',p),
                    '--number_of_runs', arguments['--number_of_runs'],
                    '--acp_detector', join('run','light_field_calibration'),
                    '--mct_acp_config', join('resources', 'mct_propagation_config.xml'),
                    '--mct_acp_propagator', join('build','mctracer','mctPlenoscopePropagation'),
                ]
                if arguments['--scoop_hosts']:
                    command.insert(1, '--scoop_hosts')
                    command.insert(2, arguments['--scoop_hosts'])
                call(command)

        os.makedirs(join('run','irf','results'), exist_ok=True)
        for p in particles:
            result_path = join('run','irf','results',p+'.csv')
            if not os.path.isfile(result_path):
                acp_irf.gamma_limits_bridge.export_effective_area(
                    input_path=join('run','irf',p),
                    detector_responses_key='raw_lixel_sum',
                    detector_response_threshold=100,
                    output_path=result_path,
                    bins=5)

        # 3) Sensitivity and time-to-detections of the ACP
        # ------------------------------------------------
        if not os.path.isdir(join('run','isf')):
            os.makedirs(join('run','isf'), exist_ok=True)
            call([
                'acp_isez',
                '--gamma_area', join('run','irf','results','gamma.csv'),
                '--electron_acceptance', join('run','irf','results','electron.csv'),
                '--proton_acceptance', join('run','irf','results','proton.csv'),
                '--cutoff', '0.01',
                '--rel_flux', '0.05',
                '--fov', '6.5',
                '--src', '3FGL J2254.0+1608',
                '--out', join('run','isf')])

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
