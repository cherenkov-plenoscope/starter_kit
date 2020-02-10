#!/usr/bin/python
import sys
from os.path import join as opj
import os
from plenoscope_map_reduce import instrument_response as irf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 3
run_dir = argv[1]
summary_dir = argv[2]

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        extended_runtime_path = opj(summary_dir, prefix_str+'_runtime.csv')
        if os.path.exists(extended_runtime_path):
            extended_runtime_table = irf.summary.runtime.read(
                extended_runtime_path)
        else:
            event_table = irf.summary.read_event_table_cache(
                summary_dir=summary_dir,
                run_dir=run_dir,
                site_key=site_key,
                particle_key=particle_key)
            runtime_table = irf.summary.runtime.read(opj(
                run_dir,
                site_key,
                particle_key,
                'runtime.csv'))
            extended_runtime_table = irf.summary.runtime.merge_event_table(
                runtime_table=runtime_table,
                event_table=event_table)
            irf.summary.runtime.write(
                path=opj(summary_dir, prefix_str+'_runtime.csv'),
                table=extended_runtime_table,)

        irf.summary.runtime.write_relative_runtime(
            table=extended_runtime_table,
            out_path=opj(summary_dir, prefix_str+'_relative_runtime'),
            figure_config=sum_config['figure_16_9'])

        irf.summary.runtime.write_speed(
            table=extended_runtime_table,
            out_path=opj(summary_dir, prefix_str+'_speed_runtime'),
            figure_config=sum_config['figure_16_9'])
