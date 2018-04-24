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
    -a --acp_detector=PATH              [default: run/light_field_calibration]
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
import corsika_wrapper as cw
import plenopy as pl
import tempfile
import numpy as np


def trigger_study(
    acp_response_path,
    output_path,
    past_trigger_path,
    run_number,
    pathch_treshold=67,
):
    run = pl.Run(acp_response_path)
    integration_time_in_slices = 5
    min_number_neighbors = 3

    trigger_preparation = pl.trigger.prepare_refocus_sum_trigger(
        run.light_field_geometry,
        object_distances=[10e3, 15e3, 20e3])

    event_infos = []
    for event in run:
        info = pl.trigger_study.export_trigger_information(event)
        info['num_air_shower_pulses'] = int(
            event.simulation_truth.detector.number_air_shower_pulses())
        info['refocus_sum_trigger'] = pl.trigger.apply_refocus_sum_trigger(
            event=event,
            trigger_preparation=trigger_preparation,
            min_number_neighbors=min_number_neighbors,
            integration_time_in_slices=integration_time_in_slices)
        event_infos.append(info)

        max_patch_threshold = np.max(
            [p['patch_threshold'] for p in info['refocus_sum_trigger']])

        if max_patch_threshold >= pathch_treshold:
            event_filename = '{run:d}{event:06d}'.format(
                run=run_number,
                event=event.number)
            event_path = os.path.join(past_trigger_path, event_filename)
            shutil.copytree(event._path, event_path)
            pl.trigger_study.write_dict_to_file(
                pl.trigger_study.un_numpyify(info['refocus_sum_trigger']),
                os.path.join(event_path, 'refocus_sum_trigger.jsonl'))

    pl.trigger_study.write_dict_to_file(
        pl.trigger_study.un_numpyify(event_infos),
        output_path)


def run_acp_simulation(cfg):
    with tempfile.TemporaryDirectory(prefix='acp_trigger_') as tmp_dir:
        corsika_run_path = os.path.join(tmp_dir, 'airshower.evtio')
        acp_response_path = os.path.join(tmp_dir, 'acp_response.acp')

        cor_rc = cw.corsika(
            steering_card=cfg['current_run']['corsika_steering_card'],
            output_path=corsika_run_path,
            save_stdout=True)

        irf.simulation.keep_stdout(
            corsika_run_path+'.stdout', 'corsika.stdout', cfg)
        irf.simulation.keep_stdout(
            corsika_run_path+'.stderr', 'corsika.stderr', cfg)

        mct_rc = irf.simulation.acp_response(
            corsika_run_path=corsika_run_path,
            output_path=acp_response_path,
            cfg=cfg,
            photon_origins=True)

        irf.simulation.keep_stdout(
            acp_response_path+'.stdout',
            'mctPlenoscopePropagation.stdout', cfg)
        irf.simulation.keep_stdout(
            acp_response_path+'.stderr',
            'mctPlenoscopePropagation.stderr', cfg)

        trigger_study(
            acp_response_path=acp_response_path,
            output_path=os.path.join(
                cfg['path']['main']['intermediate_results_of_runs']['dir'],
                'run_'+str(cfg['current_run']['number'])+'.json.gz'),
            past_trigger_path=cfg['path']['main']['past_trigger']['dir'],
            run_number=cfg['current_run']['number'])
    return {
        'corsika_return_code': cor_rc,
        'mctracer_return_code': mct_rc}


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)

        # Set up configuration and directory environment
        cfg = {}
        cfg['number_of_runs'] = int(arguments['--number_of_runs'])
        cfg['mct_acp_propagator'] = arguments['--mct_acp_propagator']
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

        cfg['corsika_steering_card_template'] = cw.read_steering_card(
            cfg['path']['main']['input']['corsika_steering_card_template'])

        cfg['path']['main']['past_trigger'] = {}
        cfg['path']['main']['past_trigger']['dir'] = os.path.join(
            cfg['path']['main']['dir'], 'past_trigger')
        os.makedirs(
            os.path.join(cfg['path']['main']['past_trigger']['dir'],
                'input'))
        shutil.copytree(
            cfg['path']['main']['input']['acp_detector'],
            os.path.join(
                cfg['path']['main']['past_trigger']['dir'],
                'input', 'plenoscope'))

        # SIMULATION
        simulation_instructions = irf.simulation.make_instructions(cfg)
        return_codes = list(scoop.futures.map(run_acp_simulation,
            simulation_instructions))

    except docopt.DocoptExit as e:
        print(e)
