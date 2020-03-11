#!/usr/bin/python
import sys
from os.path import join as opj
import os
import pandas as pd
import numpy as np
import json
import magnetic_deflection as mdfl
from magnetic_deflection import plot_sky_dome as mdfl_plot
import plenoirf as irf
import scipy


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
deflection_dir = argv[1]

deflection_table = mdfl.map_and_reduce.read_deflection_table(
    path=deflection_dir)

with open(os.path.join(deflection_dir, "sites.json"), "rt") as f:
    sites = json.loads(f.read())

with open(os.path.join(deflection_dir, "particles.json"), "rt") as f:
    particles = json.loads(f.read())

with open(os.path.join(deflection_dir, "pointing.json"), "rt") as f:
    pointing = json.loads(f.read())

key_map = {
    'primary_azimuth_deg': {
        "unit": "deg",
        "name": "primary-azimuth",
        "factor": 1,
        "start": 90.0,
        "etend_high_energies": True,},
    'primary_zenith_deg': {
        "unit": "deg",
        "name": "primary-zenith",
        "factor": 1,
        "start": 0.0,
        "etend_high_energies": True,},
    'cherenkov_pool_x_m': {
        "unit": "km",
        "name": "Cherenkov-pool-x",
        "factor": 1e-3,
        "start": 0.0,
        "etend_high_energies": True,},
    'cherenkov_pool_y_m': {
        "unit": "km",
        "name": "Cherenkov-pool-y",
        "factor": 1e-3,
        "start": 0.0,
        "etend_high_energies": True,},
}

charge_signs = {}
for particle_key in particles:
    charge_signs[particle_key] = np.sign(
        particles[particle_key]["electric_charge_qe"])

figsize = (16/2, 9/2)
dpi = 240
ax_size = (0.15, 0.12, 0.8, 0.8)


def make_site_str(site_key, site):
    return "".join([
        "{:s}, {:.1f}$\,$km$\,$a.s.l., ",
        "Atm.-id {:d}, ",
        "Bx {:.1f}$\,$uT, ",
        "Bz {:.1f}$\,$uT"]).format(
            site_key,
            site['observation_level_asl_m']*1e-3,
            site["atmosphere_id"],
            site["earth_magnetic_field_x_muT"],
            site["earth_magnetic_field_z_muT"])


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


deflection_table = mdfl.analysis.cut_invalid_from_deflection_table(
    deflection_table=deflection_table,
    but_keep_site="Off")
deflection_table = mdfl.analysis.add_density_fields_to_deflection_table(
    deflection_table=deflection_table)

for site_key in deflection_table:
    for particle_key in deflection_table[site_key]:
        print(site_key, particle_key)
        site_str = make_site_str(site_key, sites[site_key])

        t = deflection_table[site_key][particle_key]
        energy_fine = np.geomspace(
            np.min(t["energy_GeV"]),
            10*np.max(t["energy_GeV"]),
            1000)

        if "Off" in site_key:
            continue

        for key in key_map:
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

            if key_map[key]["etend_high_energies"]:
                energy_bins_ext = np.array(
                    energy_supports.tolist() +
                    np.geomspace(200, 600, 20).tolist())
                key_mean80_ext = np.array(
                    key_mean80.tolist() +
                    (key_start*np.ones(20)).tolist())
            else:
                energy_bins_ext = energy_supports.copy()
                key_mean80_ext = key_mean80.copy()


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
            assert y_fit_range >= 0
            y_fit_range = np.max([y_fit_range, 1.])
            _ll = y_fit_lower - 0.2*y_fit_range
            _uu = y_fit_upper + 0.2*y_fit_range
            ax.set_ylim([_ll, _uu])
            ax.set_ylabel(
                '{key:s}$\,/\,${unit:s}'.format(
                    key=key_map[key]["name"],
                    unit=key_map[key]["unit"]))
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            filename = '{:s}_{:s}_{:s}'.format(site_key, particle_key, key)
            filepath = os.path.join(deflection_dir, filename)
            fig.savefig(filepath+'.jpg')
            plt.close(fig)

            with open(filepath+'.json', 'wt') as fout:
                fout.write(json.dumps(
                    {
                        "name": key,
                        "power_law": {
                            "formula": "f(Energy) = A*Energy**B + C",
                            "A": float(expy[0]),
                            "B": float(expy[1]),
                            "C": float(key_start),
                        },
                        "energy_GeV": sres["energy_supports"].tolist(),
                        "mean": sres["key_mean80"].tolist(),
                        "std": sres["key_std80"].tolist(),
                    },
                    indent=4))

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
        mdfl_plot.add_grid_in_half_dome(
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
        mdfl_plot.add_points_in_half_dome(
            ax=ax,
            azimuths_deg=t['primary_azimuth_deg'],
            zeniths_deg=t['primary_zenith_deg'],
            point_diameter=0.1*rfov,
            rgbas=rgbas)
        ax.text(
            -1.6*rfov,
            0.65*rfov,
            "direction of primary\n\nazimuth w.r.t.\nmagnetic north")
        ax.text(
            -1.5*rfov,
            -1.0*rfov,
            "zenith {:1.0f}$^\circ$".format(fov_deg))
        ax.set_axis_off()
        ax.set_aspect('equal')
        mdfl_plot.add_ticklabels_in_half_dome(
            ax=ax,
            azimuths_deg=azimuths_deg_steps,
            rfov=rfov)
        ax.set_xlim([-1.01*rfov, 1.01*rfov])
        ax.set_ylim([-1.01*rfov, 1.01*rfov])
        fig.savefig(
            os.path.join(
                deflection_dir,
                '{:s}_{:s}_{:s}.jpg'.format(
                    site_key,
                    particle_key,
                    "dome")))
        plt.close(fig)


    density_map = {
        "num_cherenkov_photons_per_shower": {
            "label": "size of Cherenkov-pool$\,/\,$1"
        },
        "spread_area_m2": {
            "label": "Cherenkov-pool's spread in area$\,/\,$m$^{2}$"
        },
        "spread_solid_angle_deg2": {
            "label": "Cherenkov-pool's spread in solid angle$\,/\,$deg$^{2}$"
        },
        "light_field_outer_density": {
            "label": "density of Cherenkov-pool$\,/\,$m$^{-2}\,$deg$^{-2}$"
        },
    }

    parmap = {
        "gamma": "k",
        "electron": "b",
        "proton": "r",
        "helium": "orange"
    }

    for den_key in density_map:
        ts = deflection_table[site_key]
        alpha = 0.2
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(ax_size)
        for particle_key in parmap:
            ax.plot(
                ts[particle_key]["energy_GeV"],
                ts[particle_key][den_key],
                'o',
                color=parmap[particle_key],
                alpha=alpha,
                label=particle_key)
        leg = ax.legend()
        for line in leg.get_lines():
            line.set_alpha(0)
        ax.set_title(site_str, alpha=.5)
        ax.loglog()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('energy$\,/\,$GeV')
        ax.set_xlim([0.4, 200.0])
        ax.set_ylim([1e-6, 1e3])
        ax.set_ylabel(density_map[den_key]["label"])
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                deflection_dir,
                '{:s}_{:s}.jpg'.format(
                    site_key,
                    den_key)))
        plt.close(fig)
