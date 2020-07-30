#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import cosmic_fluxes
import os

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

onregion_acceptance = irf.json_numpy.read_tree(
    os.path.join(
        pa['summary_dir'],
        "0300_onregion_trigger_acceptance"
    )
)

energy_lower = sum_config['energy_binning']['lower_edge_GeV']
energy_upper = sum_config['energy_binning']['upper_edge_GeV']
energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config['energy_binning']['num_bins']['trigger_acceptance_onregion'] + 1
)
energy_bin_centers = irf.summary.bin_centers(energy_bin_edges)
fine_energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config['energy_binning']['num_bins']['interpolation'] + 1
)
fine_energy_bin_centers = irf.summary.bin_centers(fine_energy_bin_edges)
fine_energy_bin_width = irf.summary.bin_width(fine_energy_bin_edges)

trigger_thresholds = np.array(sum_config['trigger']['ratescan_thresholds_pe'])
analysis_trigger_threshold = sum_config['trigger']['threshold_pe']
num_trigger_thresholds = len(trigger_thresholds)

# cosmic-ray-flux
# ----------------
airshower_fluxes = irf.summary.read_airshower_differential_flux_zenith_compensated(
    run_dir=pa['run_dir'],
    summary_dir=pa['summary_dir'],
    energy_bin_centers=fine_energy_bin_centers,
    sites=irf_config['config']['sites'],
    geomagnetic_cutoff_fraction=sum_config[
        'airshower_flux'][
        'fraction_of_flux_below_geomagnetic_cutoff'],
)

# gamma-ray-flux of reference source
# ----------------------------------
(
    gamma_differential_flux_per_m2_per_s_per_GeV,
    gamma_name
) = irf.summary.make_gamma_ray_reference_flux(
    summary_dir=pa['summary_dir'],
    gamma_ray_reference_source=sum_config['gamma_ray_reference_source'],
    energy_supports_GeV=fine_energy_bin_centers,
)

comment_differential = (
    "Differential trigger-rate, reconstructed in fix onregion."
)
comment_integral = (
    "Integral trigger-rate, reconstructed in fix onregion."
)

for site_key in irf_config['config']['sites']:
    site_dir = os.path.join(pa['out_dir'], site_key)
    os.makedirs(site_dir, exist_ok=True)

    # gamma-ray
    # ---------
    site_gamma_dir = os.path.join(site_dir, 'gamma')
    os.makedirs(site_gamma_dir, exist_ok=True)

    _area = np.array(onregion_acceptance[site_key]['gamma']['point']['mean'])

    area_m2 = np.interp(
        x=fine_energy_bin_centers,
        xp=energy_bin_centers,
        fp=_area
    )
    gamma_differential_rate_per_s_per_GeV = (
        gamma_differential_flux_per_m2_per_s_per_GeV *
        area_m2
    )
    gamma_rate_per_s = np.sum(
        gamma_differential_rate_per_s_per_GeV *
        fine_energy_bin_width
    )
    T = gamma_rate_per_s
    dT_dE = gamma_differential_rate_per_s_per_GeV

    irf.json_numpy.write(
        os.path.join(site_gamma_dir, "differential_rate.json"),
        {
            "comment": comment_differential + ", " + gamma_name,
            "unit": "s$^{-1} (GeV)$^{-1}$",
            "mean": dT_dE
        }
    )
    irf.json_numpy.write(
        os.path.join(site_gamma_dir, "integral_rate.json"),
        {
            "comment": comment_integral + ", " + gamma_name,
            "unit": "s$^{-1}$",
            "mean": T
        }
    )

    # cosmic-rays
    # -----------
    for cosmic_key in airshower_fluxes[site_key]:
        site_particle_dir = os.path.join(site_dir, cosmic_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        _acceptance = np.array(
            onregion_acceptance[site_key][cosmic_key]['diffuse']['mean']
        )

        acceptance_m2_sr = np.interp(
            x=fine_energy_bin_centers,
            xp=energy_bin_centers,
            fp=_acceptance
        )
        cosmic_differential_rate_per_s_per_GeV = (
            acceptance_m2_sr *
            airshower_fluxes[site_key][cosmic_key]['differential_flux']
        )
        cosmic_rate_per_s = np.sum(
            cosmic_differential_rate_per_s_per_GeV *
            fine_energy_bin_width
        )
        T = cosmic_rate_per_s
        dT_dE = cosmic_differential_rate_per_s_per_GeV

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "differential_rate.json"),
            {
                "comment": comment_differential,
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dT_dE}
        )
        irf.json_numpy.write(
            os.path.join(site_particle_dir, "integral_rate.json"),
            {
                "comment": comment_integral,
                "unit": "s$^{-1}$",
                "mean": T
            }
        )
