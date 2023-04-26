import plenoirf
import tempfile
from ... import calibration_source


def make_response_to_source(
    source_config,
    light_field_geometry_path,

):


EXAMPLE_STAR_CONFIG = {
    "cx_deg": 0.0,
    "cy_deg": 1.0,
    "num_photons": 1000*1000,
    "seed": 122,
}

def make_response_to_star(
    star_config,
    light_field_geometry_path,
    merlict_config,
):

    prng = 

    light_path = ""

    calibration_source.write_photon_bunches(
        cx=np.deg2rad(star_config["cx_deg"]),
        cy=np.deg2rad(star_config["cy_deg"]),
        size=star_config["num_photons"],
        path=light_path,
        prng=prng,
        aperture_radius=aperture_radius,
        BUFFER_SIZE=10000,
    )

    tmp_dir = ""

    plenoirf.production.merlict.plenoscope_propagator(
        corsika_run_path=light_path,
        output_path=tmp_dir,
        light_field_geometry_path=light_field_geometry_path,
        merlict_plenoscope_propagator_path=merlict_config[
            "merlict_plenoscope_propagator_path"
        ],
        merlict_plenoscope_propagator_config_path=merlict_config[
            "merlict_plenoscope_propagator_config_path"
        ],
        random_seed=star_config["seed"],
        photon_origins=True,
        stdout_path=tmp_dir + ".o",
        stderr_path=tmp_dir + ".e",
    )

    left_over_input_dir = os.path.join(tmp_dir, "input")
    if os.path.exists(left_over_input_dir):
        shutil.rmtree(left_over_input_dir)

















    light_field_geometry = LightFieldGeometry(
        path=os.path.join(
            job["work_dir"],
            "geometries",
            job["mkey"],
            job["pkey"],
            job["akey"],
            "light_field_geometry",
        ),
        off_axis_angle_deg=job["off_axis_angle_deg"],
    )
    event = plenopy.Event(
        path=os.path.join(
            job["work_dir"],
            "responses",
            job["mkey"],
            job["pkey"],
            job["akey"],
            "1",
        ),
        light_field_geometry=light_field_geometry,
    )
    out = analysis.analyse_response_to_calibration_source(
        off_axis_angle_deg=job["off_axis_angle_deg"],
        event=event,
        light_field_geometry=light_field_geometry,
        object_distance_m=job["object_distance_m"],
        containment_percentile=job["containment_percentile"],
        binning=config["binning"],
        prng=prng,
    )
    nfs.write(json_numpy.dumps(out), summary_path, "wt")
    return 1
