import sun_grid_engine_map
import os
from os import path as op
import acp_instrument_response_function as irf
import magnetic_deflection
import json
import tempfile

NUM_WORKER_PARALLEL = 100

assert op.basename(os.getcwd()) == "starter_kit"

particles = ["electron", "proton"]
locations = ["chile_paranal", "namibia_gamsberg"]
resource_dir = op.join("resources", "acp", "71m")

for particle in particles:
    for location in locations:
        with open(op.join(resource_dir, location+".json"), "rt") as f:
            location_cfg = json.loads(f.read())

        output_filename = "magnetic_deflection_{:s}_{:s}.json".format(
            particle,
            location)

        particle_id = irf.PARTICLE_STR_TO_CORSIKA_ID[particle]
        atmosphere_id = irf.ATMOSPHERE_STR_TO_CORSIKA_ID[
            location_cfg["atmosphere"]]

        initial_state = magnetic_deflection.example_state.copy()
        initial_state["input"]["corsika_particle_id"] = particle_id
        initial_state["input"]["site"][
            "corsika_atmosphere_model"] = atmosphere_id
        for key in [
            "observation_level_altitude_asl",
            "earth_magnetic_field_x_muT",
            "earth_magnetic_field_x_muT"
        ]:
            initial_state["input"]["site"][key] = location_cfg[key]
        initial_state["input"]["initial"]["energy"] = 32.0
        initial_state["input"]["energy_thrown_per_iteration"] = 3200.0

        work_dir = op.join(resource_dir, output_filename+".part")
        os.makedirs(work_dir, exist_ok=False)

        #with tempfile.TemporaryDirectory(prefix="mag_defl_") as tmp:
        work_dir = op.join(tmp, "iterations")

        magnetic_deflection._init_work_dir(
            work_dir=work_dir,
            initial_state=initial_state)

        while True:
            try:
                magnetic_deflection._one_iteration(
                    work_dir=work_dir,
                    pool=sun_grid_engine_map,
                    num_jobs=NUM_WORKER_PARALLEL)
            except RuntimeError:
                pass

        latest_state = magnetic_deflection._read_state(
            work_dir=work_dir,
            state_number=magnetic_deflection._latest_state_number(
                work_dir))

        with open(op.join(resource_dir, output_filename), "wt") as f:
            f.write(json.dumps(latest_state))
