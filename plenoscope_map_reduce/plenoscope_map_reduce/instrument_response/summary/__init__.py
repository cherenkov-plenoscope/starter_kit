import os
from os.path import join as opj
import pandas as pd
import json
from . import runtime
from .. import table


def summarize(run_dir, out_dir):
    with open(opj(run_dir, 'input', 'config.json'), 'rt') as f:
        config = json.loads(f.read())
    os.makedirs(out_dir, exist_ok=True)


    for site_key in config['sites']:
        for particle_key in config['particles']:
            prefix = '{:s}_{:s}'.format(site_key, particle_key)

            event_table = table.read(opj(
                run_dir,
                site_key,
                particle_key,
                'event_table.tar'))
            runtime_table = runtime.read(opj(
                run_dir,
                site_key,
                particle_key,
                'runtime.csv'))

            extended_runtime_table = runtime.merge_event_table(
                runtime_table=runtime_table,
                event_table=event_table)

            runtime.write(
                path=opj(out_dir, prefix+'_runtime.csv'),
                table=extended_runtime_table,)

            runtime.write_relative_runtime(
                table=extended_runtime_table,
                out_path=opj(out_dir, prefix+'_relative_runtime'))

            runtime.write_speed(
                table=extended_runtime_table,
                out_path=opj(out_dir, prefix+'_speed_runtime'))

    return event_table
