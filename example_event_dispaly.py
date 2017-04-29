import docopt
import os
from os.path import join
from subprocess import call
import plenopy as pl
import corsika_wrapper as cw


exa = 'example_events'
os.makedirs(exa, exist_ok=True)

steering_card = cw.read_steering_card(
    join('resources','acp','71m','low_energy_example_gamma_corsika_steering_card.txt'))

cw.corsika(    
    steering_card=steering_card, 
    output_path=join(exa,'gamma.evtio'), 
    save_stdout=True)

call([
    join('build','mctracer','mctPlenoscopePropagation'),
    '--lixel', join('resources','acp','71m','light_field_calibration'),
    '--input', join(exa,'gamma.evtio'),
    '--config', join('resources','acp','mct_propagation_config.xml'),
    '--output', join(exa,'gamma.acp'),
    '--random_seed', '0'
])