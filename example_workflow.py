import os
import pkg_resources
from subprocess import call

os.makedirs('./run', exist_ok=True)

# Plenoscope optical light field calibration
if not os.path.isdir('./run/light_field_calibration'):
    call([
        './build/mctracer/mctPlenoscopeCalibration',
        '--scenery', pkg_resources.resource_filename(
            'acp_instrument_response_function', 
            'resources/71m_acp/scenery'),
        '--number_mega_photons', '10',
        '--output', './run/light_field_calibration'])


particles = {
    'gamma': {
        'out_dir': './run/gamma_irf', 
        'corsika_steering_card_path': pkg_resources.resource_filename(
            'acp_instrument_response_function', 
            'resources/71m_acp/gamma_steering_card.txt' )
    },
    'electron': {
        'out_dir': './run/electron_irf', 
        'corsika_steering_card_path': pkg_resources.resource_filename(
            'acp_instrument_response_function', 
            'resources/71m_acp/electron_steering_card.txt' )
    },
    'proton': {
        'out_dir': './run/electron_irf', 
        'corsika_steering_card_path': pkg_resources.resource_filename(
            'acp_instrument_response_function', 
            'resources/71m_acp/proton_steering_card.txt' )
    },
}


for particle in particles:
    if not os.path.isdir(particles[particle]['out_dir']):
        call([
            'acp_instrument_response_function',
            '--corsika_card', particles[particle]['corsika_steering_card_path'],
            '--output_path', './run/'+particle+'_irf',
            '--number_of_runs', '8',
            '--acp_detector', './run/light_field_calibration',
            '--mct_acp_config', pkg_resources.resource_filename(
                'acp_instrument_response_function', 
                'resources/propagation_config.xml'),
            '--mct_acp_propagator', './build/mctracer/mctPlenoscopePropagation',])

os.makedirs('./run/irf_results', exist_ok=True)

import acp_instrument_response_function as acp_irf

if not os.path.isfile('./run/irf_results/gamma_aeff.dat'):
    acp_irf.gamma_limits_bridge.export_effective_area(
        input_path='./run/gamma_irf',
        detector_responses_key='raw_lixel_sum',
        detector_response_threshold=100,
        output_path='./run/irf_results/gamma_aeff.dat',
        bins=5)

if not os.path.isfile('./run/irf_results/electron_positron_aeff.dat'):
    acp_irf.gamma_limits_bridge.export_effective_area(
        input_path='./run/electron_irf',
        detector_responses_key='raw_lixel_sum',
        detector_response_threshold=100,
        output_path='./run/irf_results/electron_positron_aeff.dat',
        bins=5)

if not os.path.isfile('./run/irf_results/proton_aeff.dat'):
    acp_irf.gamma_limits_bridge.export_effective_area(
        input_path='./run/proton_irf',
        detector_responses_key='raw_lixel_sum',
        detector_response_threshold=100,
        output_path='./run/irf_results/proton_aeff.dat',
        bins=5)

if not os.path.isdir('./run/isf'):
    os.makedirs('./run/isf', exist_ok=True)
    call([
        'acp_isez',
        '--in', './run/irf_results',
        '--cutoff', '0.01',
        '--rel_flux', '0.05',
        '--fov', '6.5',
        '--src', '3FGL J2254.0+1608',
        '--out', './run/isf'])