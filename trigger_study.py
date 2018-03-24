"""
Usage: trigger_study.py [-c=PATH] [-o=PATH] [-n=NUMBER_RUNS] [-a=PATH] [-p=PATH] [-m=PATH]

Options:
    -h --help                           Prints this help message.
    -c --corsika_card=PATH              [default: resources/acp/71m/gamma_steering_card.txt]
                                            Path to the corsika steering card template.
    -o --output_path=PATH               [default: examples/trigger_study]
                                            Path to write the output directroy.
    -n --number_of_runs=NUMBER_RUNS     [default: 6]
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
    run = pl.Run(acp_response_path)
    integration_time_in_slices = 5
    patch_threshold = 63

    trigger_preparation = pl.trigger.prepare_refocus_sum_trigger(
        run.light_field_geometry,
        object_distances=[10e3, 15e3, 20e3])

    event_infos = []
    for event in run:
        info = pl.trigger_study.export_trigger_information(event)
        info['num_air_shower_pulses'] = int(
            event.simulation_truth.detector.number_air_shower_pulses())

        info['refocus_sum_trigger'] = pl.trigger.apply_refocus_sum_trigger(
            light_field=event.light_field,
            trigger_preparation=trigger_preparation,
            patch_threshold=patch_threshold,
            integration_time_in_slices=integration_time_in_slices)

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

        keep_stdout(
            acp_response_path+'.stdout',
            'mctPlenoscopePropagation.stdout', cfg)
        keep_stdout(
            acp_response_path+'.stderr',
            'mctPlenoscopePropagation.stderr', cfg)

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
