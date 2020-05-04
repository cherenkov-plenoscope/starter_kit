#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_table as spt
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)
