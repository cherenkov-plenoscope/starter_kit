import numpy as np
import pandas as pd


def add_density_fields_to_deflection_table(deflection_table):
    out = {}
    for site_key in deflection_table:
        out[site_key] = {}
        for particle_key in deflection_table[site_key]:
            t = deflection_table[site_key][particle_key]
            dicout = pd.DataFrame(t).to_dict(orient="list")

            dicout['num_cherenkov_photons_per_shower'] = (
                t['char_total_num_photons']/
                t['char_total_num_airshowers'])

            dicout['spread_area_m2'] = (
                np.pi*
                t['char_position_std_major_m']*
                t['char_position_std_minor_m'])

            dicout['spread_solid_angle_deg2'] = (
                np.pi*
                np.rad2deg(t['char_direction_std_major_rad'])*
                np.rad2deg(t['char_direction_std_minor_rad']))

            dicout['light_field_outer_density'] = (
                dicout['num_cherenkov_photons_per_shower']/
                (dicout['spread_solid_angle_deg2']*dicout['spread_area_m2']))
            out[site_key][particle_key] = pd.DataFrame(dicout).to_records(
                index=False)
    return out


def cut_invalid_from_deflection_table(deflection_table, but_keep_site):
    out = {}
    for site_key in deflection_table:
        if but_keep_site in site_key:
            out[site_key] = deflection_table[site_key]
        else:
            out[site_key] = {}
            for particle_key in deflection_table[site_key]:
                t_raw = deflection_table[site_key][particle_key]
                defelction_valid = t_raw['primary_azimuth_deg'] != 0.
                out[site_key][particle_key] = t_raw[defelction_valid]
    return out
