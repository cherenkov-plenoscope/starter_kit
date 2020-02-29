#!/usr/bin/python
import sys
from os.path import join as opj
import os
import pandas as pd
import numpy as np
import json
import magnetic_deflection as mdfl
import plenoirf as irf
import scipy


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 3
deflection_table_path = argv[1]
run_dir = argv[2]

deflection_table = mdfl.map_and_reduce.read_deflection_table(
    path=deflection_table_path)

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)


key_map = {
    'primary_azimuth_deg': {
        "unit": "deg",
        "name": "primary-azimuth",
        "factor": 1,
        "start": 90.0},
    'primary_zenith_deg': {
        "unit": "deg",
        "name": "primary-zenith",
        "factor": 1,
        "start": 0.0},
    'cherenkov_pool_x_m': {
        "unit": "km",
        "name": "Cherenkov-pool-x",
        "factor": 1e-3,
        "start": 0.0},
    'cherenkov_pool_y_m': {
        "unit": "km",
        "name": "Cherenkov-pool-y",
        "factor": 1e-3,
        "start": 0.0}
}

charge_signs = {
    "gamma": 0.,
    "electron": -1.,
    "proton": 1.,
    "helium": 1.,
}

figsize = (16/2, 9/2)
dpi = 240
ax_size = (0.15, 0.12, 0.8, 0.8)

def add_circle(ax, x, y, r, linewidth, color, alpha):
    phis = np.linspace(0, 2*np.pi, 1001)
    xs = r*np.cos(phis)
    ys = r*np.sin(phis)
    ax.plot(xs, ys, linewidth=linewidth, color=color, alpha=alpha)


def add_points_in_half_dome(
    ax,
    azimuths_deg,
    zeniths_deg,
    point_diameter,
    color=None,
    alpha=None,
    rgbas=None,
):
    zeniths = np.deg2rad(zeniths_deg)
    azimuths = np.deg2rad(azimuths_deg)

    proj_radii = np.sin(zeniths)
    proj_x = np.cos(azimuths)*proj_radii
    proj_y = np.sin(azimuths)*proj_radii

    if rgbas is not None:
        _colors = rgbas[:, 0:3]
        _alphas = rgbas[:, 3]
    else:
        assert color is not None
        assert alpha is not None
        _colors = [color for i in range(len(zeniths))]
        _alphas = [alpha for i in range(len(zeniths))]

    for i in range(len(zeniths)):
        e1 = plt_patches.Ellipse(
            (proj_x[i], proj_y[i]),
            width=point_diameter*np.cos(zeniths[i]),
            height=point_diameter,
            angle=np.rad2deg(azimuths[i]),
            linewidth=0,
            fill=True,
            zorder=2,
            facecolor=_colors[i],
            alpha=_alphas[i])
        ax.add_patch(e1)


def add_grid_in_half_dome(
    ax,
    azimuths_deg,
    zeniths_deg,
    linewidth,
    color,
    alpha,
    draw_lower_horizontal_edge_deg=None,
):
    zeniths = np.deg2rad(zeniths_deg)
    proj_radii = np.sin(zeniths)
    for i in range(len(zeniths)):
        add_circle(
            ax=ax,
            x=0,
            y=0,
            r=proj_radii[i],
            linewidth=linewidth*np.cos(zeniths[i]),
            color=color,
            alpha=alpha)

    azimuths = np.deg2rad(azimuths_deg)
    for a in range(len(azimuths)):
        for z in range(len(zeniths)):
            if z == 0:
                continue
            r_start = np.sin(zeniths[z-1])
            r_stop = np.sin(zeniths[z])
            start_x = r_start*np.cos(azimuths[a])
            start_y = r_start*np.sin(azimuths[a])
            stop_x = r_stop*np.cos(azimuths[a])
            stop_y = r_stop*np.sin(azimuths[a])
            ax.plot(
                [start_x, stop_x],
                [start_y, stop_y],
                color=color,
                linewidth=linewidth*np.cos(zeniths[z-1]),
                alpha=alpha)

    if draw_lower_horizontal_edge_deg is not None:
        zd_edge = np.deg2rad(draw_lower_horizontal_edge_deg)
        r = np.sin(zd_edge)
        ax.plot(
            [-1, 0],
            [-r, -r],
            color=color,
            linewidth=linewidth,
            alpha=alpha)
        add_circle(
            ax=ax,
            x=0,
            y=0,
            r=r,
            linewidth=linewidth,
            color=color,
            alpha=alpha)


def add_ticklabels_in_half_dome(
    ax,
    azimuths_deg,
    rfov=1.0,
    fmt="{:1.0f}$^\circ$",
):
    xshift = -0.1*rfov
    yshift = -0.05*rfov

    azimuths = np.deg2rad(azimuths_deg)
    azimuth_deg_strs = [fmt.format(az) for az in azimuths_deg]
    xs = rfov*np.cos(azimuths) + xshift
    ys = rfov*np.sin(azimuths) + yshift
    for a in range(len(azimuths)):
        ax.text(
            x=xs[a],
            y=ys[a],
            s=azimuth_deg_strs[a])



def percentile_indices(values, target_value, percentile=90):
    values = np.array(values)
    factor = percentile/100.
    delta = np.abs(values - target_value)
    argsort_delta = np.argsort(delta)
    num_values = len(values)
    idxs = np.arange(num_values)
    idxs_sorted = idxs[argsort_delta]
    idx_limit = int(np.ceil(num_values*factor))
    return idxs_sorted[0: idx_limit]


def smooth(energies, values):
    suggested_num_energy_bins = int(np.ceil(2*np.sqrt(len(values))))
    suggested_energy_bin_edges = np.geomspace(
        np.min(energies),
        np.max(energies),
        suggested_num_energy_bins+1)
    suggested_energy_supports = 0.5*(
        suggested_energy_bin_edges[0:-1] +
        suggested_energy_bin_edges[1:])

    actual_energy_supports = []
    key_med = []
    key_mean80 = []
    key_std80 = []
    for ibin in range(len(suggested_energy_bin_edges) - 1):
        e_start = suggested_energy_bin_edges[ibin]
        e_stop = suggested_energy_bin_edges[ibin+1]
        mask = np.logical_and(energies >= e_start, energies < e_stop)
        if np.sum(mask) > 3:
            actual_energy_supports.append(suggested_energy_supports[ibin])
            med = np.median(values[mask])
            key_med.append(med)
            indices80 = percentile_indices(
                values=values[mask],
                target_value=med,
                percentile=80)
            key_std80.append(np.std(values[mask][indices80]))
            key_mean80.append(np.mean(values[mask][indices80]))
    return {
        "energy_supports": np.array(actual_energy_supports),
        "key_med": np.array(key_med),
        "key_std80": np.array(key_std80),
        "key_mean80": np.array(key_mean80),
    }



for site_key in deflection_table:
    for particle_key in deflection_table[site_key]:
        site = irf_config['config']['sites'][site_key]
        site_str = "".join([
            "{:s}, {:.1f}$\,$km$\,$a.s.l., ",
            "Atm.-id {:d}, ",
            "Bx {:.1f}$\,$uT, ",
            "Bz {:.1f}$\,$uT"]).format(
                site_key,
                site['observation_level_asl_m']*1e-3,
                site["atmosphere_id"],
                site["earth_magnetic_field_x_muT"],
                site["earth_magnetic_field_z_muT"])

        t_raw = deflection_table[site_key][particle_key]
        defelction_detected = t_raw['primary_azimuth_deg'] != 0.
        t = t_raw[defelction_detected]

        energy_fine = np.geomspace(
            np.min(t["energy_GeV"]),
            10*np.max(t["energy_GeV"]),
            1000)

        for key in key_map:
            print(site_key, particle_key, key)

            sres = smooth(energies=t["energy_GeV"], values=t[key])
            energy_supports = sres["energy_supports"]
            key_med = sres["key_med"]
            key_std80 = sres["key_std80"]
            key_mean80 = sres["key_mean80"]
            unc80_upper = key_mean80 + key_std80
            unc80_lower = key_mean80 - key_std80

            if particle_key == "electron":
                valid_range = energy_supports > 0.75
                energy_supports = energy_supports[valid_range]
                key_med = key_med[valid_range]
                key_std80 = key_std80[valid_range]
                key_mean80 = key_mean80[valid_range]
                unc80_upper = unc80_upper[valid_range]
                unc80_lower = unc80_lower[valid_range]

            key_start = charge_signs[particle_key]*key_map[key]["start"]

            energy_bins_ext = np.array(
                energy_supports.tolist() +
                np.geomspace(200, 600, 20).tolist())
            key_mean80_ext = np.array(
                key_mean80.tolist() +
                (key_start*np.ones(20)).tolist())

            def power_law(energy, scale, index):
                # return a*np.exp(b*np.log(t))
                return scale*energy**(index)

            if np.mean(key_mean80 - key_start) > 0:
                sig = -1
            else:
                sig = 1

            expy, pcov = scipy.optimize.curve_fit(
                power_law,
                energy_bins_ext,
                key_mean80_ext - key_start,
                p0=(
                    sig*charge_signs[particle_key],
                    1.
                ))

            print("{:s} = {:1.1E}*energy**({:1.2f}) + {:1.2E}".format(
                key_map[key]["name"],
                expy[0],
                expy[1],
                key_start
            ))

            info_str = particle_key + ", " + site_str

            rec_key = power_law(
                energy=energy_fine,
                scale=expy[0],
                index=expy[1])
            rec_key += key_start

            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes(ax_size)
            ax.plot(
                t["energy_GeV"],
                np.array(t[key])*key_map[key]["factor"],
                'ko',
                alpha=0.05)
            ax.plot(
                energy_supports,
                key_mean80*key_map[key]["factor"],
                'kx')
            for ibin in range(len(energy_supports)):
                _x = energy_supports[ibin]
                _y_low = unc80_lower[ibin]
                _y_high = unc80_upper[ibin]
                ax.plot(
                    [_x, _x],
                    np.array([_y_low, _y_high])*key_map[key]["factor"],
                    'k-')
            ax.plot(
                energy_bins_ext,
                key_mean80_ext*key_map[key]["factor"],
                'bo',
                alpha=0.3)
            ax.plot(
                energy_fine,
                rec_key*key_map[key]["factor"],
                'r-')
            ax.set_title(info_str, alpha=.5)
            ax.semilogx()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('energy$\,/\,$GeV')
            ax.set_xlim([0.4, 110])

            y_fit_lower = key_map[key]["factor"]*np.min(unc80_lower)
            y_fit_upper = key_map[key]["factor"]*np.max(unc80_upper)
            y_fit_range = y_fit_upper - y_fit_lower
            assert y_fit_range > 0
            _ll = y_fit_lower - 0.2*y_fit_range
            _uu = y_fit_upper + 0.2*y_fit_range
            ax.set_ylim([_ll, _uu])
            ax.set_ylabel(
                '{key:s}$\,/\,${unit:s}'.format(
                    key=key_map[key]["name"],
                    unit=key_map[key]["unit"]))
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            fig.savefig(
                os.path.join(
                    deflection_table_path,
                    '{:s}_{:s}_{:s}.jpg'.format(
                        site_key,
                        particle_key,
                        key)))
            plt.close(fig)

        print("Density of Cherenkov-photons")

        num_cherenkov_photons_per_shower = (
            t['char_total_num_photons']/
            t['char_total_num_airshowers'])

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(ax_size)
        ax.plot(
            t["energy_GeV"],
            num_cherenkov_photons_per_shower,
            'ko',
            alpha=0.3)
        ax.set_title(info_str, alpha=.5)
        ax.loglog()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('energy$\,/\,$GeV')
        ax.set_xlim([0.4, 10*np.max(t["energy_GeV"])])
        ax.set_ylabel('size of Cherenkov-photons$\,/\,$1')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                deflection_table_path,
                '{:s}_{:s}_{:s}.jpg'.format(
                    site_key,
                    particle_key,
                    "num_photons_per_shower")))
        plt.close(fig)

        areal_spread_m2 = (
            np.pi*
            t['char_position_std_major_m']*
            t['char_position_std_minor_m'])

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(ax_size)
        ax.plot(
            t["energy_GeV"],
            areal_spread_m2,
            'ko',
            alpha=0.3)
        ax.set_title(info_str, alpha=.5)
        ax.loglog()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('energy$\,/\,$GeV')
        ax.set_xlim([0.4, 10*np.max(t["energy_GeV"])])
        ax.set_ylabel('spread in area$\,/\,$m$^2$')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                deflection_table_path,
                '{:s}_{:s}_{:s}.jpg'.format(
                    site_key,
                    particle_key,
                    "areal_spread")))
        plt.close(fig)


        directional_spread_deg2 = (
            np.pi*
            np.rad2deg(t['char_direction_std_major_rad'])*
            np.rad2deg(t['char_direction_std_minor_rad']))

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(ax_size)
        ax.plot(
            t["energy_GeV"],
            directional_spread_deg2,
            'ko',
            alpha=0.3)
        ax.set_title(info_str, alpha=.5)
        ax.loglog()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('energy$\,/\,$GeV')
        ax.set_xlim([0.4, 10*np.max(t["energy_GeV"])])
        ax.set_ylabel('spread in solid angle$\,/\,$deg$^2$')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                deflection_table_path,
                '{:s}_{:s}_{:s}.jpg'.format(
                    site_key,
                    particle_key,
                    "directional_spread")))
        plt.close(fig)

        light_field_outer_density = (
            num_cherenkov_photons_per_shower/
            (directional_spread_deg2*areal_spread_m2))

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(ax_size)
        ax.plot(
            t["energy_GeV"],
            light_field_outer_density,
            'ko',
            alpha=0.3)
        ax.set_title(info_str, alpha=.5)
        ax.loglog()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('energy$\,/\,$GeV')
        ax.set_xlim([0.4, 10*np.max(t["energy_GeV"])])
        ax.set_ylabel("density of outer light-field$\,/\,$m$^{-2}\,$deg$^{-2}$")
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                deflection_table_path,
                '{:s}_{:s}_{:s}.jpg'.format(
                    site_key,
                    particle_key,
                    "light_field_outer_density")))
        plt.close(fig)

        azimuths_deg_steps = np.linspace(0, 360, 12, endpoint=False)


        if particle_key == "gamma":
            fov_deg = 1.
        elif particle_key == "proton":
            fov_deg = 10.
        elif particle_key == "helium":
            fov_deg = 10.
        else:
            fov_deg = 90.

        fov = np.deg2rad(fov_deg)
        rfov = np.sin(fov)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes((0.07, 0.07, 0.85, 0.85))
        cmap_ax = fig.add_axes((0.8, 0.07, 0.02, 0.85))
        ax.set_title(info_str, alpha=0.5)

        add_grid_in_half_dome(
            ax=ax,
            azimuths_deg=azimuths_deg_steps,
            zeniths_deg=np.linspace(0, fov_deg, 10),
            linewidth=1.4,
            color='k',
            alpha=0.1,
            draw_lower_horizontal_edge_deg=fov_deg)

        cmap_name = "nipy_spectral"
        cmap_norm = plt_colors.LogNorm(
            vmin=np.min(t['energy_GeV']),
            vmax=np.max(t['energy_GeV']))
        cmap_mappable = matplotlib.cm.ScalarMappable(
            norm=cmap_norm,
            cmap=cmap_name)
        plt.colorbar(cmap_mappable, cax=cmap_ax)
        cmap_ax.set_xlabel('energy$\,/\,$GeV')

        rgbas = cmap_mappable.to_rgba(t['energy_GeV'])
        rgbas[:, 3] = 0.25
        add_points_in_half_dome(
            ax=ax,
            azimuths_deg=t['primary_azimuth_deg'],
            zeniths_deg=t['primary_zenith_deg'],
            point_diameter=0.1*rfov,
            rgbas=rgbas)
        ax.text(
            -1.5*rfov,
            0.8*rfov,
            "sky-dome\nw.r.t. magnetic north")
        ax.text(
            -1.0*rfov,
            -1.0*rfov,
            "{:1.1f}$^\circ$".format(fov_deg))

        ax.set_axis_off()
        ax.set_aspect('equal')

        add_ticklabels_in_half_dome(
            ax=ax,
            azimuths_deg=azimuths_deg_steps,
            rfov=rfov)
        ax.set_xlim([-1.01*rfov, 1.01*rfov])
        ax.set_ylim([-1.01*rfov, 1.01*rfov])

        fig.savefig(
            os.path.join(
                deflection_table_path,
                '{:s}_{:s}_{:s}.jpg'.format(
                    site_key,
                    particle_key,
                    "dome")))
        plt.close(fig)

