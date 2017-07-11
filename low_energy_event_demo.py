#! /usr/bin/env python
import docopt
import os
from os.path import join
from subprocess import call
import plenopy as pl
import corsika_wrapper as cw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


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
def write_image(event, path, image_rays, object_distance=22e3):
    image_sequence = event.light_field.pixel_sequence_refocus(
        image_rays.pixel_ids_of_lixels_in_object_distance(object_distance)
    )
    raw_image = pl.light_field.sequence.integrate_around_arrival_peak(
        sequence=image_sequence, 
        integration_radius=1
    )['integral']
    pixel_image = pl.Image(
       raw_image,
       event.light_field.pixel_pos_cx,
       event.light_field.pixel_pos_cy
    )
    fig_size = pl.tools.FigureSize(
        relative_width=10, 
        relative_hight=8, 
        dpi=200
    )
    fig = plt.figure(figsize=(fig_size.width, fig_size.hight))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 6])
    ax_ruler = plt.subplot(gs[0])
    ax_image = plt.subplot(gs[1])
    pl.image.plot.add_pixel_image_to_ax(pixel_image, ax=ax_image)
    pl.tools.add2ax_object_distance_ruler(
        ax=ax_ruler,
        object_distance=object_distance,
        object_distance_min=2.5e3,
        object_distance_max=25e3,
        #label='',
        #print_value=False,
        color='black'
    )
    plt.savefig(path, dpi=192)


def write_event_overview_text(event, path):
    with open(path, 'w') as f:
        f.write(event.__repr__()+'\n')
        f.write(event.raw_sensor_response.__repr__()+'\n')
        f.write(event.light_field.__repr__()+'\n')
        f.write(event.sensor_plane2imaging_system.__repr__()+'\n')
        f.write(event.simulation_truth.__repr__()+'\n')
        f.write(event.simulation_truth.event.__repr__()+'\n')
        f.write(event.simulation_truth.detector.__repr__()+'\n')


def write_event_example(event_number, run, image_rays, output_dir):
    event_dir = join(output_dir,'event'+str(event_number))
    if not os.path.exists(event_dir):
        os.makedirs(event_dir, exist_ok=True)
        event = run[event_number - 1]
        run_number = event.simulation_truth.event.corsika_run_header.number
        id_text = 'Run'+str(run_number)+'Event'+str(event_number)
        object_distance = 22e3
        write_image(
            event=event,
            path=join(event_dir, 'image_'+id_text+'.png'),
            image_rays=image_rays,
            object_distance=object_distance
        )

        write_event_overview_text(
            event, 
            join(event_dir, 'info_'+id_text+'.txt')
        )


out_dir = join('examples', 'low_energy_event_demo')
os.makedirs(out_dir, exist_ok=True)

steering_card = cw.read_steering_card(
    join(
        'resources',
        'acp',
        '71m',
        'low_energy_example_gamma_corsika_steering_card.txt'
    )
)

if not os.path.exists(join(out_dir,'gamma.evtio')):
    cw.corsika(    
        steering_card=steering_card, 
        output_path=join(out_dir,'gamma.evtio'), 
        save_stdout=True
    )

if not os.path.exists(join(out_dir,'gamma.acp')):  
    call([
        join('build','mctracer','mctPlenoscopePropagation'),
        '--lixel', join('resources','acp','71m','light_field_calibration'),
        '--input', join(out_dir,'gamma.evtio'),
        '--config', join('resources','acp','mct_propagation_config.xml'),
        '--output', join(out_dir,'gamma.acp'),
        '--random_seed', '0',
        '--all_truth'
    ])


run = pl.Run(join(out_dir,'gamma.acp'))
image_rays = pl.image.ImageRays(run.light_field_geometry)

# Refocus example
# ---------------

if not os.path.exists(join(out_dir,'event49_refocus')): 
    os.makedirs(join(out_dir, 'event49_refocus'), exist_ok=True)
    event49 = run[48]

    integral = pl.light_field.sequence.integrate_around_arrival_peak(
        sequence=event49.light_field.sequence, 
        integration_radius=1
    )

    pl.plot.refocus.save_side_by_side(
        event=event49, 
        object_distances=np.logspace(np.log(2.5e3), np.log(11.5e3), 5, base=2.73), 
        output_path=join(out_dir, 'event49_refocus', 'event49_refocus.png'), 
        tims_slice_range=[integral['start_slice'], integral['stop_slice']],
        cx_limit=[+0.25, +1.75],
        cy_limit=[-0.5, +1.0],
    )

    write_event_overview_text(
        event49, 
        join(out_dir, 'event49_refocus', 'info.txt')
    )

# Threshold example
# -----------------
interesting_events = [17, 25, 32, 35, 40, 46, 47, 76, 87, 90, 96]

for event_number in  interesting_events:
    write_event_example(
        event_number=event_number, 
        run=run, 
        image_rays=image_rays, 
        output_dir=out_dir
    )