#!/usr/bin/python
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import os
import numpy as np
import json
import plenoirf as irf
import magnetic_deflection as mdfl


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]

state_number = mdfl._latest_state_number(work_dir=work_dir)
latest_state = mdfl._read_state(
    work_dir=work_dir,
    state_number=state_number)

t = latest_state
info_str = "{:.1f}km a.s.l., Atm.-id. {:d}, Bx {:.1f}uT, Bz {:.1f}uT".format(
    t['input']['site']['observation_level_asl_m']*1e-3,
    t['input']['site']["atmosphere_id"],
    t['input']['site']["earth_magnetic_field_x_muT"],
    t['input']['site']["earth_magnetic_field_z_muT"])




keys = {
    'primary_azimuth_deg': {
        "unit": "deg", "name": "primary-azimuth", "factor": 1, "yscale": "lin"},
    'primary_zenith_deg': {
        "unit": "deg", "name": "primary-zenith", "factor": 1, "yscale": "lin"},
    'cherenkov_pool_x_m': {
        "unit": "km", "name": "Cherenkov-pool-x", "factor": 1e-3, "yscale": "log"},
    'cherenkov_pool_y_m': {
        "unit": "km", "name": "Cherenkov-pool-y", "factor": 1e-3, "yscale": "log"}
}

for key in keys:
    figsize = (6, 4)
    dpi = 320
    ax_size = (0.15, 0.15, 0.80, 0.75)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(ax_size)
    ax.plot(
        t["primary_energy_GeV"],
        np.array(t[key])*keys[key]["factor"],
        'ko')
    ax.set_title(info_str)
    ax.semilogx()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('energy / GeV')
    ax.set_xlim([0.5, np.max(t["primary_energy_GeV"])])
    ax.set_ylabel(
        '{key:s} / {unit:s}'.format(
            key=keys[key]["name"],
            unit=keys[key]["unit"]))
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    if keys[key]["yscale"] == "log":
        ax.semilogy()
    fig.savefig(
        os.path.join(
            work_dir,
            '{:06d}_{:s}.jpg'.format(state_number, keys[key]["name"])))


energies = t["primary_energy_GeV"]
xs = np.array(t["cherenkov_pool_x_m"])*1e-3
ys = np.array(t["cherenkov_pool_y_m"])*1e-3

major_axis_x = t["cherenkov_pool_major_x"]
major_axis_y = t["cherenkov_pool_major_y"]
minor_axis_x = t["cherenkov_pool_minor_x"]
minor_axis_y = t["cherenkov_pool_minor_y"]
major_std = np.array(t["cherenkov_pool_major_std_m"])*1e-3
minor_std = np.array(t["cherenkov_pool_minor_std_m"])*1e-3

xlim = [np.min(xs), np.max(xs)]
ylim = [np.min(ys), np.max(ys)]

y_spread = ylim[1] - ylim[0]
x_spread = xlim[1] - xlim[0]

figsize = (6, 10)
dpi = 360
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes(ax_size)
for i in range(len(xs)):
    ax.plot(xs[i], ys[i], 'ko')
    start_x = xs[i]
    start_y = ys[i]
    major_end_x = start_x + major_axis_x[i]*major_std[i]
    major_end_y = start_y + major_axis_y[i]*major_std[i]
    ax.plot([start_x, major_end_x], [start_y, major_end_y], 'k-')
    major_end_x = start_x - major_axis_x[i]*major_std[i]
    major_end_y = start_y - major_axis_y[i]*major_std[i]
    ax.plot([start_x, major_end_x], [start_y, major_end_y], 'k-')

    minor_end_x = start_x + minor_axis_x[i]*minor_std[i]
    minor_end_y = start_y + minor_axis_y[i]*minor_std[i]
    ax.plot([start_x, minor_end_x], [start_y, minor_end_y], 'k:')
    minor_end_x = start_x - minor_axis_x[i]*minor_std[i]
    minor_end_y = start_y - minor_axis_y[i]*minor_std[i]
    ax.plot([start_x, minor_end_x], [start_y, minor_end_y], 'k:')

    if np.mod(i, 5) == 0 and energies[i] < 6:
        ax.text(
            start_x+.05*x_spread,
            start_y+.01*y_spread,
            "{:.1f}GeV".format(energies[i]))

ax.text(.1*x_spread, -.01*y_spread, "major axis")
ax.text(.1*x_spread, -.05*y_spread, "minor axis")
ax.plot([-.05*x_spread, .05*x_spread], [-.01*y_spread, -.01*y_spread], "k-")
ax.plot([-.05*x_spread, .05*x_spread], [-.05*y_spread, -.05*y_spread], "k:")
ax.set_title(info_str)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('x / km')
ax.set_ylabel('y / km')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.set_aspect("equal")
fig.savefig(
    os.path.join(
        work_dir,
        '{:06d}_cherenkov_pool_spread.jpg').format(state_number))
