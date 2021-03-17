import numpy as np
import corsika_primary_wrapper as cpw

from . import example
from .. import random_seed


def _assert_deflection(site_particle_deflection):
    for example_key in example.EXAMPLE_SITE_PARTICLE_DEFLECTION:
        assert example_key in site_particle_deflection
    assert len(site_particle_deflection["energy_GeV"]) >= 2
    for _key in site_particle_deflection:
        assert len(site_particle_deflection["energy_GeV"]) == len(
            site_particle_deflection[_key]
        )
    for energy in site_particle_deflection["energy_GeV"]:
        assert energy > 0.0
    for zenith_deg in site_particle_deflection["primary_zenith_deg"]:
        assert zenith_deg >= 0.0
    assert np.all(np.diff(site_particle_deflection["energy_GeV"]) >= 0)


def _assert_site(site):
    for _key in example.EXAMPLE_SITE:
        assert _key in site


def _assert_particle(particle):
    for _key in example.EXAMPLE_PARTICLE:
        assert _key in particle
    assert np.all(np.diff(particle["energy_bin_edges_GeV"]) >= 0)
    assert len(particle["energy_bin_edges_GeV"]) == 2


def draw_corsika_primary_steering(
    run_id,
    site,
    particle,
    site_particle_deflection,
    num_events,
):
    assert run_id > 0
    _assert_site(site)
    _assert_particle(particle)
    _assert_deflection(site_particle_deflection)
    assert num_events <= random_seed.STRUCTURE.NUM_AIRSHOWER_IDS_IN_RUN

    max_scatter_rad = np.deg2rad(particle["max_scatter_angle_deg"])

    min_common_energy = np.max(
        [
            np.min(particle["energy_bin_edges_GeV"]),
            np.min(site_particle_deflection["energy_GeV"]),
        ]
    )

    np.random.seed(run_id)
    energies = cpw.random_distributions.draw_power_law(
        lower_limit=min_common_energy,
        upper_limit=np.max(particle["energy_bin_edges_GeV"]),
        power_slope=particle["energy_power_law_slope"],
        num_samples=num_events,
    )
    steering = {}
    steering["run"] = {"run_id": int(run_id), "event_id_of_first_event": 1}
    for key in site:
        steering["run"][key] = site[key]

    steering["primaries"] = []
    for e in range(energies.shape[0]):
        event_id = e + 1
        primary = {}
        primary["particle_id"] = particle["particle_id"]
        primary["energy_GeV"] = energies[e]

        primary["magnet_azimuth_rad"] = np.deg2rad(
            np.interp(
                x=primary["energy_GeV"],
                xp=site_particle_deflection["energy_GeV"],
                fp=site_particle_deflection["primary_azimuth_deg"],
            )
        )
        primary["magnet_zenith_rad"] = np.deg2rad(
            np.interp(
                x=primary["energy_GeV"],
                xp=site_particle_deflection["energy_GeV"],
                fp=site_particle_deflection["primary_zenith_deg"],
            )
        )
        primary["magnet_cherenkov_pool_x_m"] = np.interp(
            x=primary["energy_GeV"],
            xp=site_particle_deflection["energy_GeV"],
            fp=site_particle_deflection["cherenkov_pool_x_m"],
        )
        primary["magnet_cherenkov_pool_y_m"] = np.interp(
            x=primary["energy_GeV"],
            xp=site_particle_deflection["energy_GeV"],
            fp=site_particle_deflection["cherenkov_pool_y_m"],
        )

        az, zd = cpw.random_distributions.draw_azimuth_zenith_in_viewcone(
            azimuth_rad=primary["magnet_azimuth_rad"],
            zenith_rad=primary["magnet_zenith_rad"],
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=max_scatter_rad,
        )

        primary["max_scatter_rad"] = max_scatter_rad
        primary["zenith_rad"] = zd
        primary["azimuth_rad"] = az
        primary["depth_g_per_cm2"] = 0.0
        primary["random_seed"] = cpw.simple_seed(
            random_seed.STRUCTURE.random_seed_based_on(
                run_id=run_id, airshower_id=event_id
            )
        )

        steering["primaries"].append(primary)
    return steering
