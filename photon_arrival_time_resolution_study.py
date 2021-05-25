import os
import plenoirf
import json
import argparse

parser = argparse.ArgumentParser(
    description=
        "Explore the need for high resolution arrival-time reconstruction of"
        "Cherenkov-photons in the read-out-electronics."
)
parser.add_argument('--out_dir', metavar='PATH', type=str)
args = parser.parse_args()
out_dir = args.out_dir


study_config = {
    "arrival_times_resolution_s": [
        0.4e-9, 0.8e-9, 1.6e-9, 3.2e-9, 6.4e-9,
    ]
}
NUM_EVENTS = 1000*1000
MULTIPROCESSING_POOL = ["sun_grid_engine", "local"][0]


os.makedirs(out_dir, exist_ok=True)
plenoirf.json_numpy.write(
    path=os.path.join(out_dir, "arrival_time_study_config.json"),
    out_dict=study_config
)

# init
# ====

configs = []
for ii in range(len(study_config["arrival_times_resolution_s"])):
    config = dict(plenoirf.EXAMPLE_CONFIG)

    # only gamma
    config["particles"] = dict()
    config["particles"]["gamma"] = dict(
        plenoirf.EXAMPLE_CONFIG["particles"]["gamma"]
    )

    # only namibia
    config["sites"] = dict()
    config["sites"]["namibia"] = dict(
        plenoirf.EXAMPLE_CONFIG["sites"]["namibia"]
    )

    config["runs"] = {"gamma": {"num": NUM_EVENTS//250, "first_run_id": 1}}
    config["num_airshowers_per_run"] = 250

    run_dir = os.path.join(out_dir, "{:06d}_run".format(ii))
    plenoirf.init(
        out_dir=run_dir,
        config=config
    )

    prop_conf = plenoirf.json_numpy.read(
        path=os.path.join(run_dir, "input", "merlict_propagation_config.json")
    )
    prop_conf["photon_stream"][
        "single_photon_arrival_time_resolution"] = study_config[
        "arrival_times_resolution_s"][ii]
    plenoirf.json_numpy.write(
        path=os.path.join(run_dir, "input", "merlict_propagation_config.json"),
        out_dict=prop_conf
    )

# run
# ===

common_resources = []
common_resources.append("magnetic_deflection")
common_resources.append("light_field_geometry")
common_resources.append("trigger_geometry")

run0_dir = os.path.join(out_dir, "{:06d}_run".format(0))
for ii in range(len(study_config["arrival_times_resolution_s"])):
    run_dir = os.path.join(out_dir, "{:06d}_run".format(ii))

    if ii > 0:
        for common_resource in common_resources:
            if not os.path.exists(os.path.join(run_dir, common_resource)):
                plenoirf.network_file_system.copy(
                    src=os.path.join(run0_dir, common_resource),
                    dst=os.path.join(run_dir, common_resource)
                )

    if not os.path.exists(os.path.join(run_dir, "event_table")):
        plenoirf.run(
            path=run_dir,
            MULTIPROCESSING_POOL=MULTIPROCESSING_POOL
        )
