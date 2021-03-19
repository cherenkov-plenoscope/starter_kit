import os
import copy
from os.path import join as opj
import pandas
import numpy as np
import json
import cosmic_fluxes
import pkg_resources
import subprocess
import sparse_numeric_table as spt
import glob
from .. import features
from .. import reconstruction
from .. import analysis
from .. import table
from .. import provenance
from .. import merlict
from .. import grid
from . import figure
from . import samtex
from .. import json_numpy
from .cosmic_flux import read_airshower_differential_flux_zenith_compensated
from .cosmic_flux import make_gamma_ray_reference_flux


def init(run_dir):
    summary_config = _guess_summary_config(run_dir)

    summary_dir = os.path.join(run_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    with open(opj(summary_dir, "summary_config.json"), "wt") as fout:
        fout.write(
            json.dumps(summary_config, indent=4, cls=json_numpy.Encoder)
        )

    proton_flux = cosmic_fluxes.read_cosmic_proton_flux_from_resources()
    with open(opj(summary_dir, "proton_flux.json"), "wt") as fout:
        fout.write(json.dumps(proton_flux, indent=4))

    helium_flux = cosmic_fluxes.read_cosmic_helium_flux_from_resources()
    with open(opj(summary_dir, "helium_flux.json"), "wt") as fout:
        fout.write(json.dumps(helium_flux, indent=4))

    ep_flux = cosmic_fluxes.read_cosmic_electron_positron_flux_from_resources()
    with open(opj(summary_dir, "electron_positron_flux.json"), "wt") as fout:
        fout.write(json.dumps(ep_flux, indent=4))

    fermi_fgl = cosmic_fluxes.read_fermi_3rd_galactic_from_resources()
    with open(opj(summary_dir, "gamma_sources.json"), "wt") as fout:
        fout.write(json.dumps(fermi_fgl, indent=4))


def argv_since_py(sys_argv):
    argv = []
    for arg in sys_argv:
        if len(argv) > 0:
            argv.append(arg)
        if ".py" in arg:
            argv.append(arg)
    return argv


def paths_from_argv(argv):
    assert len(argv) == 2
    run_dir = argv[1]
    summary_dir = os.path.join(run_dir, "summary")
    script_name = str.split(os.path.basename(argv[0]), ".")[0]
    return {
        "run_dir": run_dir,
        "script_name": script_name,
        "summary_dir": summary_dir,
        "out_dir": os.path.join(summary_dir, script_name),
    }


def read_summary_config(summary_dir):
    with open(opj(summary_dir, "summary_config.json"), "rt") as fin:
        config = json.loads(fin.read())
    return config


def read_instrument_response_config(run_dir):
    with open(opj(run_dir, "input", "config.json"), "rt") as f:
        config = json.loads(f.read())
    light_field_sensor_geometry = merlict.read_plenoscope_geometry(
        opj(run_dir, "input", "scenery", "scenery.json")
    )

    grid_geometry = grid.init_geometry(
        instrument_aperture_outer_diameter=(
            2.0
            * light_field_sensor_geometry[
                "expected_imaging_system_aperture_radius"
            ]
        ),
        bin_width_overhead=config["grid"]["bin_width_overhead"],
        instrument_field_of_view_outer_radius_deg=(
            0.5 * light_field_sensor_geometry["max_FoV_diameter_deg"]
        ),
        instrument_pointing_direction=[0, 0, 1],
        field_of_view_overhead=config["grid"]["field_of_view_overhead"],
        num_bins_radius=config["grid"]["num_bins_radius"],
    )

    with open(opj(run_dir, "input", "scenery", "scenery.json"), "rt") as f:
        plenoscope_scenery = json.loads(f.read())
    _prop_cfg_path = opj(run_dir, "input", "merlict_propagation_config.json")
    with open(_prop_cfg_path, "rt") as f:
        merlict_propagation_config = json.loads(f.read())
    bundle = {
        "config": config,
        "light_field_sensor_geometry": light_field_sensor_geometry,
        "plenoscope_scenery": plenoscope_scenery,
        "grid_geometry": grid_geometry,
        "merlict_propagation_config": merlict_propagation_config,
    }
    return bundle


def run(run_dir):

    json_numpy.write(
        path=opj(run_dir, "summary", "provenance.json"),
        out_dict=provenance.make_provenance(),
    )

    summary_dir = opj(run_dir, "summary")
    irf_config = read_instrument_response_config(run_dir=run_dir)
    sum_config = read_summary_config(summary_dir=summary_dir)

    script_abspaths = _make_script_abspaths()

    for script_abspath in script_abspaths:
        script_basename = os.path.basename(script_abspath)
        script_name = str.split(script_basename, ".")[0]
        result_path = os.path.join(run_dir, "summary", script_name)
        if os.path.exists(result_path):
            print("[skip] ", script_name)
        else:
            print("[run ] ", script_name)
            subprocess.call(["python", script_abspath, run_dir])


def _make_script_abspaths():
    script_absdir = pkg_resources.resource_filename(
        "plenoirf", os.path.join("summary", "scripts")
    )
    _paths = glob.glob(os.path.join(script_absdir, "*"))
    out = []
    order = []
    for _path in _paths:
        basename = os.path.basename(_path)
        if str.isdigit(basename[0:4]):
            order.append(int(basename[0:4]))
            out.append(_path)
    order = np.array(order)
    argorder = np.argsort(order)
    out_order = [out[arg] for arg in argorder]
    return out_order


def _estimate_num_events_past_trigger(run_dir, irf_config):
    irf_config = read_instrument_response_config(run_dir=run_dir)

    num_events_past_trigger = 10 * 1000
    for site_key in irf_config["config"]["sites"]:
        for particle_key in irf_config["config"]["particles"]:
            event_table = spt.read(
                path=os.path.join(
                    run_dir,
                    "event_table",
                    site_key,
                    particle_key,
                    "event_table.tar",
                ),
                structure=table.STRUCTURE,
            )
            if event_table["pasttrigger"].shape[0] < num_events_past_trigger:
                num_events_past_trigger = event_table["pasttrigger"].shape[0]
    return num_events_past_trigger


def _guess_energy_bins_lower_upper_number(irf_config, num_events):
    particles = irf_config["config"]["particles"]
    min_energies = []
    max_energies = []
    for particle_key in particles:
        e_bins = particles[particle_key]["energy_bin_edges_GeV"]
        min_energies.append(np.min(e_bins))
        max_energies.append(np.max(e_bins))
    min_energy = np.min(min_energies)
    max_energy = np.max(max_energies)

    num_energy_bins = int(np.sqrt(num_events))
    num_energy_bins = 2 * (num_energy_bins // 2)
    num_energy_bins = np.max([np.min([num_energy_bins, 2 ** 6]), 2 ** 2])
    return min_energy, max_energy, num_energy_bins


def _guess_num_direction_bins(num_events):
    num_bins = int(0.5 * np.sqrt(num_events))
    num_bins = np.max([np.min([num_bins, 2 ** 7]), 2 ** 4])
    return num_bins


def make_ratescan_trigger_thresholds(
    lower_threshold,
    upper_threshold,
    num_thresholds,
    collection_trigger_threshold,
    analysis_trigger_threshold,
):
    assert lower_threshold <= collection_trigger_threshold
    assert upper_threshold >= collection_trigger_threshold

    assert lower_threshold <= analysis_trigger_threshold
    assert upper_threshold >= analysis_trigger_threshold

    tt = np.geomspace(lower_threshold, upper_threshold, num_thresholds,)
    tt = np.round(tt)
    tt = tt.tolist()
    tt = tt + [collection_trigger_threshold]
    tt = tt + [analysis_trigger_threshold]
    tt = np.array(tt, dtype=np.int)
    tt = set(tt)
    tt = list(tt)
    tt = np.sort(tt)
    return tt


FERMI_3FGL_CRAB_NEBULA_NAME = "3FGL J0534.5+2201"
FERMI_3FGL_PHD_THESIS_REFERENCE_SOURCE_NAME = "3FGL J2254.0+1608"


def _guess_summary_config(run_dir):
    irf_config = read_instrument_response_config(run_dir=run_dir)

    num_events_past_collection_trigger = _estimate_num_events_past_trigger(
        run_dir=run_dir, irf_config=irf_config
    )

    lower_E, upper_E, num_E_bins = _guess_energy_bins_lower_upper_number(
        irf_config=irf_config, num_events=num_events_past_collection_trigger
    )

    collection_trigger_threshold_pe = irf_config["config"]["sum_trigger"][
        "threshold_pe"
    ]
    analysis_trigger_threshold_pe = int(
        np.round(1.11 * collection_trigger_threshold_pe)
    )

    fov_radius_deg = (
        0.5 * irf_config["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
    )

    summary_config = {
        "energy_binning": {
            "lower_edge_GeV": lower_E,
            "upper_edge_GeV": upper_E,
            "num_bins": {
                "trigger_acceptance": 48,
                "trigger_acceptance_onregion": 24,
                "interpolation": 1337,
                "point_spread_function": 8,
            },
        },
        "direction_binning": {
            "radial_angle_deg": 35.0,
            "num_bins": _guess_num_direction_bins(
                num_events_past_collection_trigger
            ),
        },
        "trigger": {
            "modus": {
                "accepting_focus": 7,
                "rejecting_focus": -1,
                "intensity_ratio_between_foci": 1.0,
                "use_rejection_focus": False,
            },
            "threshold_pe": analysis_trigger_threshold_pe,
            "ratescan_thresholds_pe": make_ratescan_trigger_thresholds(
                lower_threshold=int(collection_trigger_threshold_pe * 0.8),
                upper_threshold=int(collection_trigger_threshold_pe * 1.5),
                num_thresholds=32,
                collection_trigger_threshold=collection_trigger_threshold_pe,
                analysis_trigger_threshold=analysis_trigger_threshold_pe,
            ),
        },
        "night_sky_background": {"max_num_true_cherenkov_photons": 0,},
        "airshower_flux": {"fraction_of_flux_below_geomagnetic_cutoff": 0.05,},
        "gamma_ray_source_direction": {
            "max_angle_relative_to_pointing_deg": fov_radius_deg - 0.5,
        },
        "train_and_test": {"test_size": 0.5},
        "gamma_hadron_seperation": {"gammaness_threshold": 0.5},
        "random_seed": 1,
        "quality": {
            "max_relative_leakage": 0.1,
            "min_reconstructed_photons": 50,
        },
        "point_spread_function": {
            "theta_square": {"max_angle_deg": 3.25, "num_bins": 256,},
            "core_radius": {"max_radius_m": 640, "num_bins": 4},
            "containment_factor": 0.68,
            "pivot_energy_GeV": 2.0,
        },
        "on_off_measuremnent": {
            "on_over_off_ratio": 1 / 5,
            "detection_threshold_std": 5.0,
        },
        "gamma_ray_reference_source": {
            "type": "3fgl",
            "name_3fgl": FERMI_3FGL_CRAB_NEBULA_NAME,
            "generic_power_law": {
                "flux_density_per_m2_per_s_per_GeV": 1e-3,
                "spectral_index": -2.0,
                "pivot_energy_GeV": 1.0,
            },
        },
        "reconstruction": {
            "trajectory": reconstruction.trajectory.make_example_config_for_71m_plenoscope(
                fov_radius_deg=fov_radius_deg),
        }
    }

    summary_config["plot"] = {
        "16_by_9": figure.CONFIG_16_9,
        "particle_colors": {
            "gamma": "black",
            "electron": "blue",
            "proton": "red",
            "helium": "orange",
        },
    }

    return summary_config


def read_train_test_frame(
    site_key,
    particle_key,
    run_dir,
    transformed_features_dir,
    trigger_config,
    quality_config,
    train_test,
    level_keys,
):
    sk = site_key
    pk = particle_key

    airshower_table = spt.read(
        path=os.path.join(run_dir, "event_table", sk, pk, "event_table.tar",),
        structure=table.STRUCTURE,
    )

    airshower_table["transformed_features"] = spt.read(
        path=os.path.join(
            transformed_features_dir, sk, pk, "transformed_features.tar",
        ),
        structure=features.TRANSFORMED_FEATURE_STRUCTURE,
    )["transformed_features"]

    idxs_triggered = analysis.light_field_trigger_modi.make_indices(
        trigger_table=airshower_table["trigger"],
        threshold=trigger_config["threshold_pe"],
        modus=trigger_config["modus"],
    )
    idxs_quality = analysis.cuts.cut_quality(
        feature_table=airshower_table["features"],
        max_relative_leakage=quality_config["max_relative_leakage"],
        min_reconstructed_photons=quality_config["min_reconstructed_photons"],
    )

    EXT_STRUCTRURE = dict(table.STRUCTURE)
    EXT_STRUCTRURE[
        "transformed_features"
    ] = features.TRANSFORMED_FEATURE_STRUCTURE["transformed_features"]

    out = {}
    for kk in ["test", "train"]:
        idxs_valid_kk = spt.intersection(
            [idxs_triggered, idxs_quality, train_test[sk][pk][kk],]
        )
        table_kk = spt.cut_table_on_indices(
            table=airshower_table,
            structure=EXT_STRUCTRURE,
            common_indices=idxs_valid_kk,
            level_keys=level_keys,
        )
        table_kk = spt.sort_table_on_common_indices(
            table=table_kk, common_indices=idxs_valid_kk
        )
        out[kk] = spt.make_rectangular_DataFrame(table_kk)

    return out
