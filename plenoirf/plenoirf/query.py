import numpy as np
from . import grid
from . import table


def primary_incident_vector(primary_table):
    prm = primary_table
    cxs = np.cos(prm['azimuth_rad'])*prm['zenith_rad']
    cys = np.sin(prm['azimuth_rad'])*prm['zenith_rad']
    return grid._make_bunch_direction(cxs, cys)


def query_in_viewcone_cx_xy(
    cx_deg,
    cy_deg,
    cone_opening_angle_deg,
    primary_table,
):
    incidents = primary_incident_vector(primary_table)
    target_incident = grid._make_bunch_direction(
        np.array([np.deg2rad(cx_deg)]),
        np.array([np.deg2rad(cy_deg)]))
    deltas = grid._make_angle_between(
        directions=incidents,
        direction=target_incident.T)[:, 0]
    return deltas <= np.deg2rad(cone_opening_angle_deg)


def query_grid_histograms(
    energy_GeV_start,
    energy_GeV_stop,
    cx_deg,
    cy_deg,
    cone_opening_angle_deg,
    primary_table,
    grid_histograms,
    num_bins_radius
):
    prm = primary_table
    mask_viewcone = query_in_viewcone_cx_xy(
        cx_deg=cx_deg,
        cy_deg=cy_deg,
        cone_opening_angle_deg=cone_opening_angle_deg,
        primary_table=prm)
    mask_energy = np.logical_and(
        prm['energy_GeV'] > energy_GeV_start,
        prm['energy_GeV'] <= energy_GeV_stop)
    mask = np.logical_and(mask_viewcone, mask_energy)
    matches = prm[mask]

    num_bins_edge = 2*num_bins_radius
    hist = np.zeros((num_bins_edge, num_bins_edge))
    num_airshower = 0
    for mat in matches:
        hist += grid.bytes_to_histogram(
            grid_histograms[(mat['run_id'], mat['airshower_id'])])
        num_airshower += 1
    return hist, num_airshower
