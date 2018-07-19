import numpy as np
import plenopy as pl
import os
import pandas as pd
import json


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


def integration_width_for_one_sigma(hist, bin_edges):
    one_sigma = 0.68
    integral = np.cumsum(hist/np.sum(hist))
    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
    x = np.linspace(np.min(bin_centers), np.max(bin_centers), 100*bin_centers.shape[0])
    f = np.interp(x=x, fp=integral, xp=bin_centers)
    return x[np.argmin(np.abs(f - one_sigma))]


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
    # 1 GeV regime
    energy_mask = ((rs.energy > 0.75) & (rs.energy <= 1.5))
    fig = plt.figure(figsize=(6, 6), dpi=320)
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    h = ax.hist(
        np.rad2deg(rs[method + '_offset'][energy_mask]),
        bins=np.linspace(0, 2, 20),
        fc='gray',
        ec='none')
    ax.set_xlabel('residual incident-direction/deg')
    ax.set_ylabel('number events/1')

    incident_direction_resolution_one_sigma = integration_width_for_one_sigma(
        h[0], h[1])

    ax.axvline(incident_direction_resolution_one_sigma, color='k')
    plt.savefig(
        os.path.join(out_dir, 'incident_directions_at_750MeV_to_1500MeV_'+method+'.png'))

    # Resolution Figure
    #------------------
    resolution_path = os.path.join(
        'run', 'isf', 'assumed_angular_resolution.json')
    with open(resolution_path, 'rt') as fin:
        out = json.loads(fin.read())

    fig = plt.figure(figsize=(6, 6), dpi=320)
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    ax.plot(
        out['Fermi_LAT']['energy_GeV'],
        out['Fermi_LAT']['resolution_deg'],
        'k-.',
        label='Fermi-LAT')
    ax.plot(
        out['Aharonian_et_al_5at5']['energy_GeV'],
        out['Aharonian_et_al_5at5']['resolution_deg'],
        linestyle='-',
        color='k',
        label='Aharonian et al., 5@5')
    ax.plot(
        out['Hofman_limits']['energy_GeV'],
        out['Hofman_limits']['resolution_deg'],
        'ko--',
        markerfacecolor='k',
        label='Hofmann, limits')
    ax.plot(
        out['Hofman_limits_with_earth_magnetic_field']['energy_GeV'],
        out['Hofman_limits_with_earth_magnetic_field']['resolution_deg'],
        'ko:',
        markerfacecolor='white',
        label='Hofmann, limits, with earth-magnetic-field')

    one_sigma_resolutions = []
    one_sigma_resolution_errors = []
    energy_bin_edges = 0.75*(np.cumprod(1.4*np.ones(6))/1.4)
    for energy_bin in range(len(energy_bin_edges) - 1):
        energy_mask = (
            (rs.energy > energy_bin_edges[energy_bin]) &
            (rs.energy <= energy_bin_edges[energy_bin + 1]))
        reiduals = np.rad2deg(rs['light_front_offset'][energy_mask])
        direction_bin_edges = np.linspace(0, 2, 20)
        hist = np.histogram(reiduals, bins=direction_bin_edges)[0]
        one_sigma_resolution = integration_width_for_one_sigma(
            hist, direction_bin_edges)
        num = np.sum(energy_mask)
        one_sigma_resolutions.append(one_sigma_resolution)
        one_sigma_resolution_errors.append(
            one_sigma_resolution * np.sqrt(num)/num)
    energy_bin_centers = (energy_bin_edges[0:-1] + energy_bin_edges[1:])/2
    ax.errorbar(
        x=energy_bin_centers,
        y=one_sigma_resolutions,
        yerr=one_sigma_resolution_errors,
        fmt='kx:',
        label='Plenoscope, light-front')
    ax.loglog()
    ax.legend(loc='best', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Energy / GeV')
    ax.set_ylabel('Resolution / deg')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    fig.savefig(
        os.path.join(out_dir, 'assumed_angular_resolution.png'))

