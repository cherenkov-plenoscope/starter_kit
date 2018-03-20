"""
Usage: trigger_study.py [-c=PATH] [-o=PATH] [-n=NUMBER_RUNS] [-a=PATH] [-p=PATH] [-m=PATH]

Options:
    -h --help                           Prints this help message.
    -c --corsika_card=PATH              [default: resources/acp/71m/gamma_steering_card.txt]
                                            Path to the corsika steering card template.
    -o --output_path=PATH               [default: examples/trigger_study]
                                            Path to write the output directroy.
    -n --number_of_runs=NUMBER_RUNS     [default: 96]
                                            Number of simulation runs to be
                                            executed. The total number of events
                                            is NUMBER_RUNS times NSHOW of the
                                            corsika steering template card.
    -a --acp_detector=PATH              [default: resources/acp/71m/light_field_calibration]
                                            Path to the light-field-calibration.
    -p --mct_acp_config=PATH            [default: resources/acp/mct_propagation_config.xml]
                                            Path to the mctracer ACP propagation configuration.
    -m --mct_acp_propagator=PATH        [default: build/mctracer/mctPlenoscopePropagation]
                                            Path to the mctracer ACP propagation executable.
"""
import docopt
import scoop
import os
import copy
import shutil
import acp_instrument_response_function as irf
from acp_instrument_response_function.simulation import keep_stdout
import corsika_wrapper as cw
import plenopy as pl
import tempfile
import numpy as np


def trigger_study(
    acp_response_path,
    output_path
):
    trigger_settings = [
        {'min_ph_lix': 2, 'min_pax_in_pix': 5, 'time_slices':5},
        {'min_ph_lix': 2, 'min_pax_in_pix': 6, 'time_slices':5},
        {'min_ph_lix': 2, 'min_pax_in_pix': 7, 'time_slices':5},
        {'min_ph_lix': 3, 'min_pax_in_pix': 5, 'time_slices':5},
        {'min_ph_lix': 3, 'min_pax_in_pix': 6, 'time_slices':5},
        {'min_ph_lix': 3, 'min_pax_in_pix': 7, 'time_slices':5},
    ]

    run = pl.Run(acp_response_path)

    pixel_neighborhood = pl.trigger.neighborhood(
        x=run.light_field_geometry.pixel_pos_cx,
        y=run.light_field_geometry.pixel_pos_cy,
        epsilon=np.deg2rad(0.1))

    event_infos = []
    for event in run:
        info = pl.trigger_study.export_trigger_information(event)
        info['light_field_trigger'] = []
        info['num_air_shower_pulses'] = int(
            event.simulation_truth.detector.number_air_shower_pulses())
        for trigger_setting in trigger_settings:
            trigger_sequence = pl.trigger.trigger_on_light_field_sequence(
                light_field=event.light_field,
                pixel_neighborhood=pixel_neighborhood,
                min_photons_in_lixel=trigger_setting['min_ph_lix'],
                min_paxels_above_threshold_in_pixel=trigger_setting['min_pax_in_pix'],
                trigger_integration_time_window_in_slices=trigger_setting['time_slices'])
            trigger_setting['triggers_in_sequence'] = int(np.sum(trigger_sequence))
            info['light_field_trigger'].append(trigger_setting)

        event_infos.append(info)
    pl.trigger_study.write_dict_to_file(event_infos, output_path)


def run_acp_simulation(cfg):
    with tempfile.TemporaryDirectory(prefix='acp_trigger_') as tmp_dir:
        corsika_run_path = os.path.join(tmp_dir, 'airshower.evtio')
        acp_response_path = os.path.join(tmp_dir, 'acp_response.acp')

        cor_rc = cw.corsika(
            steering_card=cfg['current_run']['corsika_steering_card'],
            output_path=corsika_run_path,
            save_stdout=True)

        keep_stdout(corsika_run_path+'.stdout', 'corsika.stdout', cfg)
        keep_stdout(corsika_run_path+'.stderr', 'corsika.stderr', cfg)

        mct_rc = irf.simulation.acp_response(
            corsika_run_path=corsika_run_path,
            output_path=acp_response_path,
            cfg=cfg,
            photon_origins=True)

        keep_stdout(acp_response_path+'.stdout', 'mctPlenoscopePropagation.stdout', cfg)
        keep_stdout(acp_response_path+'.stderr', 'mctPlenoscopePropagation.stderr', cfg)

        trigger_study(
            acp_response_path=acp_response_path,
            output_path=os.path.join(
                cfg['path']['main']['intermediate_results_of_runs']['dir'],
                'run_'+str(cfg['current_run']['number'])+'.json.gz'))
    return {
        'corsika_return_code': cor_rc,
        'mctracer_return_code': mct_rc}


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)

        # Set up configuration and directory environment
        cfg = {}
        cfg['path'] = irf.working_dir.directory_structure(
            arguments['--output_path'])

        os.mkdir(cfg['path']['main']['dir'])
        os.mkdir(cfg['path']['main']['input']['dir'])
        os.mkdir(cfg['path']['main']['stdout']['dir'])
        os.mkdir(cfg['path']['main']['intermediate_results_of_runs']['dir'])

        shutil.copy(
            arguments['--corsika_card'],
            cfg['path']['main']['input']['corsika_steering_card_template'])
        shutil.copytree(
            arguments['--acp_detector'],
            cfg['path']['main']['input']['acp_detector'])
        shutil.copy(
            arguments['--mct_acp_config'],
            cfg['path']['main']['input']['mct_acp_config'])

        cfg['number_of_runs'] = int(arguments['--number_of_runs'])
        cfg['mct_acp_propagator'] = arguments['--mct_acp_propagator']

        cfg['corsika_steering_card_template'] = cw.read_steering_card(
            cfg['path']['main']['input']['corsika_steering_card_template'])

        # SIMULATION
        simulation_instructions = irf.simulation.make_instructions(cfg)
        return_codes = list(scoop.futures.map(run_acp_simulation,
            simulation_instructions))

    except docopt.DocoptExit as e:
        print(e)