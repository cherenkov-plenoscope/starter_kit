import os
from os.path import join
import pkg_resources
from subprocess import call
import acp_instrument_response_function as acp_irf

os.makedirs('run', exist_ok=True)

# Plenoscope optical light field calibration
if not os.path.isdir(join('run','light_field_calibration')):
    call([
        join('build','mctracer','mctPlenoscopeCalibration'),
        '--scenery', pkg_resources.resource_filename(
            'acp_instrument_response_function', 
            join('resources','71m_acp','scenery')),
        '--number_mega_photons', '10',
        '--output', join('run','light_field_calibration')])

particles = ['gamma', 'electron', 'proton']

os.makedirs(join('run','irf'), exist_ok=True)

for p in particles:
    if not os.path.isdir(join('run','irf',p)):
        call([
            'acp_instrument_response_function',
            '--corsika_card', pkg_resources.resource_filename(
                'acp_instrument_response_function', 
                join('resources','71m_acp',p+'_steering_card.txt')),
            '--output_path', join('run','irf',p),
            '--number_of_runs', '4',
            '--acp_detector', join('run','light_field_calibration'),
            '--mct_acp_config', pkg_resources.resource_filename(
                'acp_instrument_response_function', 
                join('resources','propagation_config.xml')),
            '--mct_acp_propagator', join('build','mctracer','mctPlenoscopePropagation'),])

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