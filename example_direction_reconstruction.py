import numpy as np
import plenopy as pl
import os
import pandas as pd

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

out_dir = os.path.join('examples', 'incident_direction')
os.makedirs(out_dir, exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join('run', 'light_field_calibration'))

run = pl.Run(
    os.path.join('run', 'irf', 'electron', 'past_trigger'),
    light_field_geometry=light_field_geometry)

number_events = 0
results = []

cah_path = os.path.join(out_dir, 'incident_direction.msg')
if not os.path.exists(cah_path):
    for event in run:
        r = {}
        primary_momentum = event.simulation_truth.event.corsika_event_header.momentum()
        primary_direction = primary_momentum/np.linalg.norm(primary_momentum)

        r['particle_id'] = event.simulation_truth.event.corsika_event_header.primary_particle_id
        r['core_cx'] = - primary_direction[0]
        r['core_cy'] = - primary_direction[1]
        r['core_x'] = event.simulation_truth.event.corsika_event_header.core_position_x_meter()
        r['core_y'] = event.simulation_truth.event.corsika_event_header.core_position_y_meter()
        r['energy'] = event.simulation_truth.event.corsika_event_header.total_energy_GeV

        number_events += 1
        print(number_events, r['energy'])
        if r['energy'] > 5:
            break

        roi = pl.classify.center_for_region_of_interest(event)
        photons = pl.classify.RawPhotons.from_event(event)

        cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
            roi=roi,
            photons=photons)

        # Incident-direction reconstructed using only the trigger-patch
        # -------------------------------------------------------------
        r['trigger_patch_cx'] = roi['cx_center_roi']
        r['trigger_patch_cy'] = roi['cy_center_roi']

        # Incident-direction reconstructed using mean of photons in image-space
        # ---------------------------------------------------------------------
        r['image_mean_cx'] = np.mean(cherenkov_photons.cx)
        r['image_mean_cy'] = np.mean(cherenkov_photons.cy)

        # Incident-direction reconstructed using median of photons in image-space
        # -----------------------------------------------------------------------
        r['image_median_cx'] = np.median(cherenkov_photons.cx)
        r['image_median_cy'] = np.median(cherenkov_photons.cy)

        # Incident-direction reconstructed using light-front on
        # principal-aperture-plane
        # -----------------------------------------------------
        B, inlier = pl.tools.ransac_3d_plane.fit(
            xyz_point_cloud=np.c_[
                cherenkov_photons.x,
                cherenkov_photons.y,
                cherenkov_photons.t_pap*3e8],
            max_number_itarations=500,
            min_number_points_for_plane_fit=10,
            max_orthogonal_distance_of_inlier=0.025,)
        c_pap_time = np.array([B[0], B[1], B[2]])
        if c_pap_time[2] > 0:
            c_pap_time *= -1
        c_pap_time = c_pap_time/np.linalg.norm(c_pap_time)
        r['light_front_cx'] = c_pap_time[0]
        r['light_front_cy'] = c_pap_time[1]
        results.append(r)

    rs = pd.DataFrame(results)
    rs.to_msgpack(cah_path)

rs = pd.read_msgpack(cah_path)

methods = [
    'trigger_patch',
    'image_mean',
    'image_median',
    'light_front']

for method in methods:
    rs[method + '_offset'] = np.hypot(
        rs[method + '_cx'] - rs.core_cx,
        rs[method + '_cy'] - rs.core_cy,)

e_bin_edges = np.logspace(np.log10(0.5), np.log10(5), 6)

for method in methods:
    median = []
    error = []
    for e in range(e_bin_edges.shape[0] - 1):
        energy_mask = (
            (rs.energy > e_bin_edges[e]) & (rs.energy <= e_bin_edges[e+1]))
        med = np.median(rs[method + '_offset'][energy_mask])
        median.append(med)
        number_occurences = np.sum(energy_mask)
        error.append(med * np.sqrt(number_occurences)/number_occurences)
    median = np.array(median)
    error = np.array(error)
    fig = plt.figure(figsize=(6, 6), dpi=320)
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    ax.errorbar(
        x=e_bin_edges[:-1],
        y=np.rad2deg(median),
        yerr=(
            np.rad2deg(error),
            np.rad2deg(error)
        ),
        fmt='ko',)
    ax.set_xlabel('energy/GeV')
    ax.set_ylabel('residual incident-direction/deg')
    ax.loglog()
    plt.savefig(os.path.join(out_dir, 'incident_directions_'+method+'.png'))

    # 1 GeV regime
    energy_mask = ((rs.energy > 0.75) & (rs.energy <= 1.5))
    fig = plt.figure(figsize=(6, 6), dpi=320)
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    ax.hist(
        np.rad2deg(rs[method + '_offset'][energy_mask]),
        bins=np.linspace(0, 2, 20),
        fc='gray',
        ec='none')
    ax.set_xlabel('residual incident-direction/deg')
    ax.set_ylabel('number events/1')
    plt.savefig(
        os.path.join(out_dir, 'incident_directions_at_750MeV_to_1500MeV_'+method+'.png'))

