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


for site_key in deflection_table:
    for particle_key in deflection_table[site_key]:
        site = irf_config['config']['sites'][site_key]
        site_str = "".join([
            "{:s}, {:.1f}km a.s.l., ",
            "Atm.-id. {:d}, ",
            "Bx {:.1f}uT, ",
            "Bz {:.1f}uT"]).format(
                site_key,
                site['observation_level_asl_m']*1e-3,
                site["atmosphere_id"],
                site["earth_magnetic_field_x_muT"],
                site["earth_magnetic_field_z_muT"])

        t = deflection_table[site_key][particle_key]
        num_energy_bins = 32
        energy_bin_edges = np.geomspace(
            np.min(t["energy_GeV"]),
            np.max(t["energy_GeV"]),
            num_energy_bins+1)
        energy_bins = 0.5*(energy_bin_edges[0:-1] + energy_bin_edges[1:])

        energy_fine = np.geomspace(
            np.min(t["energy_GeV"]),
            10*np.max(t["energy_GeV"]),
            1000)

        for key in key_map:

            print(site_key, particle_key, key)
            key_med = []
            key_mean80 = []
            key_std80 = []
            for ibin in range(len(energy_bin_edges) - 1):
                e_start = energy_bin_edges[ibin]
                e_stop = energy_bin_edges[ibin+1]
                mask = np.logical_and(
                    t["energy_GeV"] >= e_start,
                    t["energy_GeV"] < e_stop)
                med = np.median(t[key][mask])
                key_med.append(med)
                indices80 = percentile_indices(
                    values=t[key][mask],
                    target_value=med,
                    percentile=80)
                key_std80.append(np.std(t[key][mask][indices80]))
                key_mean80.append(np.mean(t[key][mask][indices80]))
            key_med = np.array(key_med)
            key_std80 = np.array(key_std80)
            key_mean80 = np.array(key_mean80)
            unc80_upper = key_mean80 + key_std80
            unc80_lower = key_mean80 - key_std80

            key_start = charge_signs[particle_key]*key_map[key]["start"]

            energy_bins_ext = np.array(
                energy_bins.tolist() +
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

            rec_key = power_law(
                energy=energy_fine,
                scale=expy[0],
                index=expy[1])
            rec_key += key_start

            figsize = (6, 4)
            dpi = 320
            ax_size = (0.15, 0.12, 0.80, 0.75)
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes(ax_size)
            ax.plot(
                t["energy_GeV"],
                np.array(t[key])*key_map[key]["factor"],
                'ko',
                alpha=0.05)
            ax.plot(
                energy_bins,
                key_mean80*key_map[key]["factor"],
                'kx')
            for ibin in range(len(energy_bins)):
                _x = energy_bins[ibin]
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
            info_str = particle_key +"\n" + site_str
            ax.set_title(info_str)
            ax.semilogx()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('energy / GeV')
            ax.set_xlim([0.4, 10*np.max(t["energy_GeV"])])
            ax.set_ylabel(
                '{key:s} / {unit:s}'.format(
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
