#! /usr/bin/env python
"""
Estimate the sensitivity of the Atmospheric Cherenkov Plenoscope (ACP).

You can run the simulation in parallel using a python scoop cluster, but you do 
not have to.

The default number of particle observation runs and light field calibration 
photons is rather large (same as in our ACP introduction paper) and will take 
some time (9h on our 96 core cluster).

Usage: estimate_sensitivity [--out_dir=DIR] [-s=SCOOP_HOSTS] [--number_of_runs=RUNS] [--lfc_Mp=MEGA_PHOTONS] [--number_of_bins=NUM_BINS]

Options:
    -o --out_dir=DIR                Output directory [default: ./run]
    -s --scoop_hosts=SCOOP_HOSTS    Path to the scoop hosts text file.
    --number_of_runs=RUNS           How many runs to simulate for each 
                                    particle [default: 3840]
    --lfc_Mp=MEGA_PHOTONS           How many mega photons to be used during the 
                                    light field calibration of the plenoscope. 
                                    [default: 1337]
    --number_of_bins=NUM_BINS       The number of energy bins for the instrument
                                    response function. [default: 30]
"""
import docopt
import os
from os.path import join
from subprocess import call
import acp_instrument_response_function as acp_irf


def main():
    try:
        arguments = docopt.docopt(__doc__)
        out_dir = arguments['--out_dir']

        # 1) Light field calibration of Plenoscope
        # ----------------------------------------
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.isdir(join(out_dir,'light_field_calibration')):
            call([
                join('build','mctracer','mctPlenoscopeCalibration'),
                '--scenery', join('resources','acp','71m','scenery'),
                '--number_mega_photons', arguments['--lfc_Mp'],
                '--output', join(out_dir,'light_field_calibration')
            ])


        # 2) Instrument response functions for typical cosmic particles
        # -------------------------------------------------------------
        particles = ['gamma', 'electron', 'proton']
        os.makedirs(join(out_dir,'irf'), exist_ok=True)
        for p in particles:
            if not os.path.isdir(join(out_dir,'irf',p)):
                command = [
                    'acp_instrument_response_function',
                    '--corsika_card', join('resources','acp','71m',p+'_steering_card.txt'),
                    '--output_path', join(out_dir,'irf',p),
                    '--number_of_runs', arguments['--number_of_runs'],
                    '--acp_detector', join(out_dir,'light_field_calibration'),
                    '--mct_acp_config', join('resources','acp','mct_propagation_config_no_night_sky_background.xml'),
                    '--mct_acp_propagator', join('build','mctracer','mctPlenoscopePropagation'),
                ]
                if arguments['--scoop_hosts']:
                    command.insert(1, '--scoop_hosts')
                    command.insert(2, arguments['--scoop_hosts'])
                call(command)

        os.makedirs(join(out_dir,'irf','results'), exist_ok=True)
        for p in particles:
            result_path = join(out_dir,'irf','results',p+'.csv')
            if not os.path.isfile(result_path):
                acp_irf.gamma_limits_bridge.export_effective_area(
                    input_path=join(out_dir,'irf',p),
                    detector_responses_key='raw_lixel_sum',
                    detector_response_threshold=100,
                    output_path=result_path,
                    bins=int(arguments['--number_of_bins']))

        # 3) Sensitivity and time-to-detections of the ACP
        # ------------------------------------------------
        if not os.path.isdir(join(out_dir,'isf')):
            os.makedirs(join(out_dir,'isf'), exist_ok=True)
            call([
                'acp_isez',
                '--gamma_area', join(out_dir,'irf','results','gamma.csv'),
                '--electron_acceptance', join(out_dir,'irf','results','electron.csv'),
                '--proton_acceptance', join(out_dir,'irf','results','proton.csv'),
                '--cutoff', '0.01',
                '--rel_flux', '0.05',
                '--fov', '6.5',
                '--src', '3FGL J2254.0+1608',
                '--out', join(out_dir,'isf')])

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
