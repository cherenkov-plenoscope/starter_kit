import plenopy as pl

run = pl.Run('past_trigger/')




def classify_based_on_aperture(
    event,
    air_shower_photon_ids_based_on_image,
    cx_guess,
    cy_guess,
    time_scan_radius=15e-9,
    time_inlier_radius=2e-9,
    c0=299792458
):
    cxcyt, lixel_ids = pl.photon_stream.cython_reader.stream2_cx_cy_arrivaltime_point_cloud(
        photon_stream=event.raw_sensor_response.photon_stream,
        time_slice_duration=event.raw_sensor_response.time_slice_duration,
        NEXT_READOUT_CHANNEL_MARKER=event.raw_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
        cx=event.light_field_geometry.cx_mean,
        cy=event.light_field_geometry.cy_mean,
        time_delay=event.light_field_geometry.time_delay_mean)

    arrival_times = cxcyt[:, 2][air_shower_photon_ids_based_on_image]
    x = event.light_field_geometry.x_mean[lixel_ids[air_shower_photon_ids_based_on_image]]
    y = event.light_field_geometry.y_mean[lixel_ids[air_shower_photon_ids_based_on_image]]
    z = c0*arrival_times
    z -= np.mean(z)

    cz_guess = np.sqrt(1 - cx_guess**2 - cy_guess**2)

    z_radius = time_scan_radius*c0
    inlier_radius = time_inlier_radius*c0

    N = 1000
    inliers = np.zeros(
        (N, air_shower_photon_ids_based_on_image.shape[0]),
        dtype=np.bool)

    for i, o in enumerate(np.linspace(-z_radius, z_radius, N)):
        plane_model = np.array([-cx_guess, -cy_guess, cz_guess, o])
        xyz_point_cloud = np.c_[x,y,z]

        dists = pl.tools.ransac_3d_plane.distance_to_plane(
            plane_model=plane_model,
            xyz_point_cloud=xyz_point_cloud)

        inliers[i, :] = (dists <= inlier_radius)

    best_plane_model = np.argmax(np.sum(inliers, axis=1))
    return air_shower_photon_ids_based_on_image[inliers[best_plane_model]]





def benchmark_air_shower_photon_classification(
    event,
    trigger_response,
    deg_over_s,
    epsilon_cx_cy_radius,
    min_number_photons,
    roi_time_radius,
    roi_cx_cy_radius,
    roi_object_distance_radius,
    refocusses_for_classification,
):
    roi = pl.trigger.region_of_interest_from_trigger_response(
        trigger_response=trigger_response,
        time_slice_duration=event.raw_sensor_response.time_slice_duration,
        pixel_pos_cx=event.light_field_geometry.pixel_pos_cx,
        pixel_pos_cy=event.light_field_geometry.pixel_pos_cy)

    (   air_shower_photon_ids,
        lixel_ids_of_photons
    ) = pl.photon_classification.classify_air_shower_photons(
        light_field_geometry=event.light_field_geometry,
        raw_sensor_response=event.raw_sensor_response,
        start_time_roi=roi['time_center_roi'] - roi_time_radius,
        end_time_roi=roi['time_center_roi'] + roi_time_radius,
        cx_center_roi=roi['cx_center_roi'],
        cy_center_roi=roi['cy_center_roi'],
        cx_cy_radius_roi=roi_cx_cy_radius,
        object_distances=np.logspace(
            np.log10(roi['object_distance'] - roi_object_distance_radius),
            np.log10(roi['object_distance'] + roi_object_distance_radius),
            refocusses_for_classification),
        deg_over_s=deg_over_s,
        epsilon_cx_cy_radius=epsilon_cx_cy_radius,
        min_number_photons=min_number_photons)

    air_shower_photon_ids = classify_based_on_aperture(
        event=event,
        air_shower_photon_ids_based_on_image=air_shower_photon_ids,
        cx_guess=roi['cx_center_roi'],
        cy_guess=roi['cy_center_roi'],
        time_inlier_radius=1.25e-9)

    pulse_origins = event.simulation_truth.detector.pulse_origins
    air_shower = np.zeros(pulse_origins.shape[0], dtype=np.bool)

    if air_shower_photon_ids.shape[0] > 0:
        air_shower[air_shower_photon_ids] = True

    nsb = np.invert(air_shower)
    return {
        'true_positives': int((pulse_origins[air_shower] >= 0).sum()),
        'false_positives': int((pulse_origins[air_shower] < 0).sum()),
        'true_negatives': int((pulse_origins[nsb] < 0).sum()),
        'false_negatives': int((pulse_origins[nsb] >= 0).sum()),
    }

roi_time_radius=10e-9
roi_cx_cy_radius=np.deg2rad(0.4)
roi_object_distance_radius=2e3
deg_over_s=0.175e9
refocusses_for_classification=5
epsilon_cx_cy_radius=np.deg2rad(0.06)
min_number_photons=13

R = 0
N = 0
M = 0
rs = []
num_poss = []
event_benchmarks = []
part_of_air_showers = []
for event in run:
    trigger_response = pl.tomography.image_domain.image_domain_tomography.read_trigger_response(event)

    b = benchmark_air_shower_photon_classification(
        event=event,
        trigger_response=trigger_response,
        roi_time_radius=roi_time_radius,
        roi_cx_cy_radius=roi_cx_cy_radius,
        roi_object_distance_radius=roi_object_distance_radius,
        deg_over_s=deg_over_s,
        refocusses_for_classification=refocusses_for_classification,
        epsilon_cx_cy_radius=epsilon_cx_cy_radius,
        min_number_photons=min_number_photons)

    num_pos = b['true_positives'] + b['false_positives']
    if num_pos > 0:
        r = b['true_positives']/(num_pos)
        rs.append(r)
        N += 1
        R += r
        M += num_pos
        num_poss.append(num_pos)

        part_of_air_shower = b['true_positives']/(b['false_negatives'] + b['true_positives'])
        part_of_air_showers.append(part_of_air_shower)

        print(np.round(R/N,2), np.round(M/N,1), np.median(num_poss),
            np.round(np.median(part_of_air_showers),2)
        )


    else:
        print('miss')

    b['particle_id'] = int(event.simulation_truth.event.corsika_event_header.primary_particle_id)
    b['energy'] = float(event.simulation_truth.event.corsika_event_header.total_energy_GeV)
    momentum = event.simulation_truth.event.corsika_event_header.momentum()
    b['px'] = float(momentum[0])
    b['py'] = float(momentum[1])
    b['pz'] = float(momentum[2])
    b['core_x'] = float(event.simulation_truth.event.corsika_event_header.core_position_x_meter())
    b['core_y'] = float(event.simulation_truth.event.corsika_event_header.core_position_y_meter())
    event_benchmarks.append(b)


# R/N=0.82, M/N=40, med(num_pos)=32
"""
roi_time_radius=5e-9
roi_cx_cy_radius=np.deg2rad(0.5)
roi_object_distance_radius=5e3
deg_over_s=0.25e9
refocusses_for_classification=5
epsilon_cx_cy_radius=np.deg2rad(0.04)
min_number_photons=8
"""

# R/N=0.75, M/N=84, med(num_pos)=69
"""
roi_time_radius=5e-9
roi_cx_cy_radius=np.deg2rad(0.5)
roi_object_distance_radius=1e3
deg_over_s=0.23e9
refocusses_for_classification=3
epsilon_cx_cy_radius=np.deg2rad(0.055)
min_number_photons=10
"""

# R/N=0.73, M/N=92, med(num_pos)=77
"""
roi_time_radius=5e-9
roi_cx_cy_radius=np.deg2rad(0.3)
roi_object_distance_radius=2e3
deg_over_s=0.20e9
refocusses_for_classification=5
epsilon_cx_cy_radius=np.deg2rad(0.055)
min_number_photons=11
"""

# R/N=0.71, M/N=107, med(num_pos)=92.5, part_air_shower=0.55
"""
roi_time_radius=5e-9
roi_cx_cy_radius=np.deg2rad(0.3)
roi_object_distance_radius=2e3
deg_over_s=0.20e9
refocusses_for_classification=5
epsilon_cx_cy_radius=np.deg2rad(0.055)
min_number_photons=10
"""

# R/N=0.67, M/N=126, med(num_pos)=113, part_air_shower=0.62
"""
roi_time_radius=5e-9
roi_cx_cy_radius=np.deg2rad(0.3)
roi_object_distance_radius=2e3
deg_over_s=0.20e9
refocusses_for_classification=5
epsilon_cx_cy_radius=np.deg2rad(0.055)
min_number_photons=9
"""

# R/N=0.62, M/N=147, med(num_pos)=131, part_air_shower=0.65
"""
roi_time_radius=5e-9
roi_cx_cy_radius=np.deg2rad(0.3)
roi_object_distance_radius=5e3
deg_over_s=0.20e9
refocusses_for_classification=7
epsilon_cx_cy_radius=np.deg2rad(0.055)
min_number_photons=9
"""

# With aperture_arrival-plane-cut
# ------------------------------------------------------------------------------

# R/N=0.82, M/N=95 med(num_pos)=82, part_air_shower=0.58
"""
roi_time_radius=5e-9
roi_cx_cy_radius=np.deg2rad(0.3)
roi_object_distance_radius=2e3
deg_over_s=0.20e9
refocusses_for_classification=5
epsilon_cx_cy_radius=np.deg2rad(0.055)
min_number_photons=9

time_inlier_radius=1e-9
"""


# R/N=0.7, M/N=142 med(num_pos)=132, part_air_shower=0.73
"""
roi_time_radius=10e-9
roi_cx_cy_radius=np.deg2rad(0.4)
roi_object_distance_radius=2e3
deg_over_s=0.175e9
refocusses_for_classification=5
epsilon_cx_cy_radius=np.deg2rad(0.07)
min_number_photons=13

time_inlier_radius=1.25e-9
"""