import docopt
import os
from os.path import join
from subprocess import call
import plenopy as pl
import corsika_wrapper as cw
import numpy as np
import matplotlib.pyplot as plt

"""
A list of interesting gamma events to be found in the example run to be 
simulated.

Run 1, Event 17, E 0.71GeV, I 109p.e., 
Run 1, Event 25, E 2.88GeV, I 146p.e.,
Run 1, Event 32, E 1.41GeV, I 129p.e.,
Run 1, Event 35, E 2.92GeV, I 139p.e.,
Run 1, Event 40, E 0.67GeV, I 145p.e.,
Run 1, Event 46, E 2.71GeV, I 101p.e.,
Run 1, Event 47, E 1.56GeV, I 104p.e.,
Run 1, Event 49, E 3.71GeV, I 2002p.e. nice to be refocused,
Run 1, Event 76, E 2.24GeV, I 161p.e.,
Run 1, Event 87, E 2.19GeV, I 117p.e.,
Run 1, Event 90, E 2.12GeV, I 228p.e.,
Run 1, Event 96, E 3.59GeV, I 1026p.e.,
"""


def write_event_overview_text(event, path):
    with open(path, 'w') as f:
        f.write(event.__repr__()+'\n')
        f.write(event.raw_sensor_response.__repr__()+'\n')
        f.write(event.light_field.__repr__()+'\n')
        f.write(event.sensor_plane2imaging_system.__repr__()+'\n')
        f.write(event.simulation_truth.__repr__()+'\n')
        f.write(event.simulation_truth.event.__repr__()+'\n')
        f.write(event.simulation_truth.detector.__repr__()+'\n')


exa = 'example_events'
os.makedirs(exa, exist_ok=True)

steering_card = cw.read_steering_card(
    join('resources','acp','71m','low_energy_example_gamma_corsika_steering_card.txt'))


if not os.path.exists(join(exa,'gamma.evtio')):
    cw.corsika(    
        steering_card=steering_card, 
        output_path=join(exa,'gamma.evtio'), 
        save_stdout=True)

if not os.path.exists(join(exa,'gamma.acp')):  
    call([
        join('build','mctracer','mctPlenoscopePropagation'),
        '--lixel', join('resources','acp','71m','light_field_calibration'),
        '--input', join(exa,'gamma.evtio'),
        '--config', join('resources','acp','mct_propagation_config.xml'),
        '--output', join(exa,'gamma.acp'),
        '--random_seed', '0',
        '--all_truth'
    ])


run = pl.Run(join(exa,'gamma.acp'))
image_rays = pl.image.ImageRays(run.light_field_geometry)

os.makedirs(join(exa, 'event49_refocus'), exist_ok=True)
event49 = run[48]

pl.plot.refocus.save_side_by_side(
    event=event49, 
    object_distances=np.logspace(np.log(2.5e3), np.log(11.5e3), 5, base=2.73), 
    output_path=join(exa, 'event49_refocus', 'event49_refocus.png'), 
    tims_slice_range=[37,39],
    cx_limit=[+0.25, +1.75],
    cy_limit=[-0.5, +1.0],)

write_event_overview_text(
    event49, 
    join(exa, 'event49_refocus', 'event49_refocus.txt'))
