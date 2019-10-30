import numpy as np
import matplotlib.pyplot as plt
import os
r2d = np.rad2deg
d2r = np.deg2rad

"""
image_integrated = np.zeros((64, 192))
image_integrated[25, 43] = 3
image_scale = 100.

c_parallel_bin_edges=np.linspace(
    np.deg2rad(-.5),
    np.deg2rad(2.5),
    3*64 + 1)
c_perpendicular_bin_edges=np.linspace(
    np.deg2rad(-.5),
    np.deg2rad(+.5),
    64 + 1)

max_core_radius = 250

altitude_bin = 5
energy_bin = 2

azimuth_bin = 2
radius_bin = 50

azimuth_bin_centers = np.linspace(0, 2*np.pi, 8, endpoint=False)
radius_bin_centers = np.linspace(0., max_core_radius, 250)

azimuth = azimuth_bin_centers[azimuth_bin]
radius = radius_bin_centers[radius_bin]

aperture_x = radius*np.cos(azimuth)
aperture_y = radius*np.sin(azimuth)

aperture_bin_radius = 4.6


energy_bin_centers = np.geomspace(0.25, 25, 8)
altitude_bin_edges = np.geomspace(5e3, 5e3+8e3,  8)
num_shower = [
    [0, 0, 0, 1, 5, 7, 9, 9],
    [0, 0, 0, 3, 4, 9, 9, 8],
    [0, 0, 3, 7, 9, 9, 7, 6],
    [0, 2, 5, 8, 8, 7, 4, 1],
    [3, 7, 9, 9, 7, 2, 0, 0],
    [8, 9, 9, 8, 3, 1, 0, 0],
    [9, 9, 8, 4, 3, 0, 0, 0],
    [9, 9, 5, 2, 0, 0, 0, 0],
]

energy = energy_bin_centers[energy_bin]
altitude = altitude_bin_edges[altitude_bin]

path = os.path.join("lookup_view", "{:06d}.png".format(42))
"""

def sz(x_start, x_stop, y_start, y_stop, figsize, dpi):
    num_cols = figsize[0]*dpi
    num_rows = figsize[1]*dpi
    x_s_rel = x_start/num_cols
    x_e_rel = x_stop/num_cols
    y_s_rel = y_start/num_rows
    y_e_rel = y_stop/num_rows
    out = (x_s_rel, y_s_rel, x_e_rel-x_s_rel, y_e_rel-y_s_rel)
    return out

def add_circle(ax, x, y, r, linestyle):
    phis = np.linspace(0, 2*np.pi, 512)
    xs = r*np.cos(phis) + x
    ys = r*np.sin(phis) + y
    ax.plot(xs, ys, linestyle)

def rm_splines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def add_slider_axes(ax, start, stop, value, label, log=False):
    ax.set_ylim([start, stop])
    if log:
        ax.semilogy()
    rm_splines(ax=ax)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(label)
    ax.plot(
        [0, 1],
        [value, value],
        "k",
        linewidth=5)


def save_view(
    path,
    integrated_lookup,
    energy_bin,
    altitude_bin,
    azimuth_bin,
    radius_bin
):
    il = integrated_lookup

    image_integrated = il.image(
        energy_bin=energy_bin,
        altitude_bin=altitude_bin,
        azimuth_bin=azimuth_bin,
        radius_bin=radius_bin)
    #print(image_integrated.shape)
    num_showers = np.array(il.num_showers)
    num_photons = np.array(il.num_photons)

    lookup_population = 0.5*(
        num_photons > il.filling["max_num_photons_in_bin"])
    lookup_population_pos = lookup_population.copy()
    lookup_population_pos[energy_bin, altitude_bin] += 0.5
    image_integrated_size = np.sum(image_integrated)

    c_parallel_bin_edges = il.integrated["c_parallel_bin_edges"]
    c_perpendicular_bin_edges = il.integrated["c_perpendicular_bin_edges"]
    max_core_radius = il.integrated["radius_stop"]
    aperture_bin_radius = il.integrated["aperture_bin_radius"]

    radius_bin_centers = il.integrated["radius_bin_centers"]
    azimuth_bin_centers = il.integrated["azimuth_bin_centers"]

    azimuth = azimuth_bin_centers[azimuth_bin]
    radius = radius_bin_centers[radius_bin]
    aperture_x = radius*np.cos(azimuth)
    aperture_y = radius*np.sin(azimuth)

    energy_bin_centers = il.energy_bin_centers
    altitude_bin_edges = il.altitude_bin_edges

    energy = energy_bin_centers[energy_bin]
    altitude = altitude_bin_edges[altitude_bin]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    figsize = (16, 9)
    dpi = 120
    img_height = 0.5

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax_img = fig.add_axes(sz(75, 1920-75, 480, 480+((1920-150)/3), figsize, dpi))
    ax_img.pcolor(
        r2d(c_parallel_bin_edges),
        r2d(c_perpendicular_bin_edges),
        image_integrated.T,
        cmap="inferno")
    rm_splines(ax=ax_img)
    ax_img.grid(color='white', linestyle='-', linewidth=0.66, alpha=0.3)
    ax_img.set_xlabel("radial/deg")
    ax_img.set_yticks(np.linspace(-.5, .5, 5))
    ax_img.set_xticks(np.linspace(-.5, 2.5, 13))

    ax_aperture = fig.add_axes(sz(75, 75+330, 75, 75+330, figsize, dpi))
    add_circle(ax=ax_aperture, x=0, y=0, r=max_core_radius, linestyle="k:")
    rm_splines(ax=ax_aperture)
    ax_aperture.set_xlim([-max_core_radius, max_core_radius])
    ax_aperture.set_ylim([-max_core_radius, max_core_radius])
    ax_aperture.set_yticklabels([])

    add_circle(
        ax=ax_aperture,
        x=aperture_x, y=aperture_y, r=aperture_bin_radius, linestyle="k-")
    ax_aperture.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax_aperture.set_xlabel("x/m")
    ax_aperture.set_ylabel("y/m")

    ax_ap_text = fig.add_axes(sz(75+330, 75+330+300, 75, 75+330, figsize, dpi))
    ax_ap_text.set_axis_off()
    ax_ap_text.text(0.1, 1.0, "x {:0.1f}m".format(aperture_x))
    ax_ap_text.text(0.1, 0.9, "y {:0.1f}m".format(aperture_y))
    ax_ap_text.text(0.1, 0.7, "azimuth {:0.1f}deg".format(r2d(azimuth)))
    ax_ap_text.text(0.1, 0.6, "radius {:0.1f}m".format(radius))
    ax_ap_text.text(0.1, 0.5, "apertur-radius {:0.1f}m".format(aperture_bin_radius))
    ax_ap_text.text(0.1, 0.3, "azimuth-bin {:d}".format(azimuth_bin))
    ax_ap_text.text(0.1, 0.2, "radius-bin {:d}".format(radius_bin))
    ax_ap_text.text(0.1, 0.1, "energy-bin {:d}".format(energy_bin))
    ax_ap_text.text(0.1, 0.0, "altitude-bin {:d}".format(altitude_bin))

    ax_population = fig.add_axes(sz(680, 680+330, 75, 75+330, figsize, dpi))
    rm_splines(ax=ax_population)

    ax_population.pcolor(
        lookup_population_pos.T,
        cmap="binary",
        vmax=1.0)
    ax_population.set_xlabel("energy-bins")
    ax_population.set_ylabel("altitude-bins")

    ax_population.set_xticks(np.arange(len(energy_bin_centers)))
    ax_population.set_yticks(np.arange(len(altitude_bin_edges)))
    ax_population.set_xticklabels([])
    ax_population.set_yticklabels([])
    ax_population.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.3)

    ax_size = fig.add_axes(sz(1100, 1150, 75, 75+330, figsize, dpi))
    add_slider_axes(
        ax=ax_size,
        start=1e-1,
        stop=1e3,
        value=image_integrated_size,
        log=True,
        label="num. photons\n{:.1f}".format(image_integrated_size))

    ax_altitude_slider = fig.add_axes(sz(1250, 1300, 75, 75+330, figsize, dpi))
    add_slider_axes(
        ax=ax_altitude_slider,
        start=altitude_bin_edges[0]*1e-3,
        stop=altitude_bin_edges[-1]*1e-3,
        value=altitude*1e-3,
        log=False,
        label="altitude/km\n{:.1f}km".format(altitude*1e-3))

    ax_energy_slider = fig.add_axes(sz(1400, 1920-475, 75, 75+330, figsize, dpi))
    add_slider_axes(
        ax=ax_energy_slider,
        start=energy_bin_centers[0],
        stop=energy_bin_centers[-1],
        value=energy,
        log=True,
        label="energy/GeV\n{:.1f}GeV".format(energy))
    fig.savefig(path)
    plt.close(fig)
