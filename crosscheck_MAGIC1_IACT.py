#! /usr/bin/env python
"""
Run a crosscheck where the ACP simulation tools are used to reproduce the known 
instrument response functions and sensitivities of the MAGIC 1 mono telescope.

Usage: example_workflow [-s=SCOOP_HOSTS]

Options:
    -s --scoop_hosts=SCOOP_HOSTS    Path to the scoop hosts text file.
"""
import docopt
import os
from os.path import join
from subprocess import call
import acp_instrument_response_function as acp_irf


def main():
    try:
        arguments = docopt.docopt(__doc__)

        m1 = 'crosscheck_MAGIC1_IACT'

        # 1) calibration of the 'MAGIC 1 like' telescope
        # ----------------------------------------------
        os.makedirs(m1, exist_ok=True)
        if not os.path.isdir(join(m1,'light_field_calibration')):
            call([
                join('build','mctracer','mctPlenoscopeCalibration'),
                '--scenery', join('resources', '17m_iact', 'scenery'),
                '--number_mega_photons', 25,
                '--output', join(m1,'light_field_calibration')
            ])


        # 2) Instrument response functions for typical cosmic particles
        # -------------------------------------------------------------
        particles = ['gamma', 'electron', 'proton']
        os.makedirs(join(m1,'irf'), exist_ok=True)
        for p in particles:
            if not os.path.isdir(join(m1,'irf',p)):
                command = [
                    'acp_instrument_response_function',
                    '--corsika_card', join('resources', '17m_iact', p+'_steering_card.txt'),
                    '--output_path', join(m1,'irf',p),
                    '--number_of_runs', 192,
                    '--acp_detector', join(m1,'light_field_calibration'),
                    '--mct_acp_config', join('resources','17m_iact','mct_propagation_config.xml'),
                    '--mct_acp_propagator', join('build','mctracer','mctPlenoscopePropagation'),
                ]
                if arguments['--scoop_hosts']:
                    command.insert(1, '--scoop_hosts')
                    command.insert(2, arguments['--scoop_hosts'])
                call(command)

        os.makedirs(join(m1,'irf','results'), exist_ok=True)
        for p in particles:
            result_path = join(m1,'irf','results',p+'.csv')
            if not os.path.isfile(result_path):
                acp_irf.gamma_limits_bridge.export_effective_area(
                    input_path=join(m1,'irf',p),
                    detector_responses_key='raw_lixel_sum',
                    detector_response_threshold=100,
                    output_path=result_path,
                    bins=15,
                    overlay_magic_one=True)

        # 3) Sensitivity and time-to-detections of the ACP
        # ------------------------------------------------
        if not os.path.isdir(join(m1,'isf')):
            os.makedirs(join(m1,'isf'), exist_ok=True)
            call([
                'acp_isez',
                '--gamma_area', join(m1,'irf','results','gamma.csv'),
                '--electron_acceptance', join(m1,'irf','results','electron.csv'),
                '--proton_acceptance', join(m1,'irf','results','proton.csv'),
                '--cutoff', '0.01',
                '--rel_flux', '0.05',
                '--fov', '3.5',
                '--src', '3FGL J2254.0+1608',
                '--out', join(m1,'isf')])

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()