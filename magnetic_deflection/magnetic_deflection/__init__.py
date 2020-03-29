from . import discovery
from . import map_and_reduce
from . import examples
from . import analysis
from . import light_field_characterization

import os
import json
import pandas
import numpy as np
import scipy
from scipy.optimize import curve_fit as  scipy_optimize_curve_fit
import shutil
import corsika_primary_wrapper as cpw


def estimate_raw_deflection(
    particles,
    sites,
    plenoscope_pointing,
    out_dir,
    multiprocessing_pool,
    max_energy,
    num_energy_supports,
):
    os.makedirs(out_dir, exist_ok=True)
    raw_deflection_table_path = os.path.join(out_dir, "raw")
    os.makedirs(raw_deflection_table_path, exist_ok=True)

    with open(os.path.join(out_dir, 'sites.json'), 'wt') as f:
        f.write(json.dumps(sites, indent=4))
    with open(os.path.join(out_dir, 'pointing.json'), 'wt') as f:
        f.write(json.dumps(plenoscope_pointing, indent=4))
    with open(os.path.join(out_dir, 'particles.json'), 'wt') as f:
        f.write(json.dumps(particles, indent=4))

    jobs = map_and_reduce.make_jobs(
        sites=sites,
        particles=particles,
        plenoscope_pointing=plenoscope_pointing,
        max_energy=max_energy,
        num_energy_supports=num_energy_supports)
    combined_results = multiprocessing_pool.map(
        map_and_reduce.run_job,
        jobs_sorted_energy)
    raw_deflection_table = map_and_reduce.structure_combined_results(
        combined_results=combined_results,
        sites=sites,
        particles=particles)
    map_and_reduce.write_deflection_table(
        deflection_table=raw_deflection_table,
        path=raw_deflection_table_path)


def summarize_raw_deflection(
    out_dir,
    min_fit_energy=0.65,
):
    with open(os.path.join(out_dir, "sites.json"), "rt") as f:
        sites = json.loads(f.read())
    with open(os.path.join(out_dir, "particles.json"), "rt") as f:
        particles = json.loads(f.read())
    with open(os.path.join(out_dir, "pointing.json"), "rt") as f:
        pointing = json.loads(f.read())
    _cut_invalid(
        in_path=os.path.join(out_dir, 'raw'),
        out_path=os.path.join(out_dir, 'raw_valid'),
        min_energy=min_fit_energy)
    _add_density_fields(
        in_path=os.path.join(out_dir, 'raw_valid'),
        out_path=os.path.join(out_dir, 'raw_valid_add'))
    _smooth_and_reject_outliers(
        in_path=os.path.join(out_dir, 'raw_valid_add'),
        out_path=os.path.join(out_dir, 'raw_valid_add_clean'))
    _set_high_energies(
        particles=particles,
        in_path=os.path.join(out_dir, 'raw_valid_add_clean'),
        out_path=os.path.join(out_dir, 'raw_valid_add_clean_high'))
    sites2 = {}
    for site_key in sites:
        if not 'Off' in site_key:
            sites2[site_key] = sites[site_key]
    _fit_power_law(
        particles=particles,
        sites=sites2,
        in_path=os.path.join(out_dir, 'raw_valid_add_clean_high'),
        out_path=os.path.join(out_dir, 'raw_valid_add_clean_high_power'))
    _export_table(
        particles=particles,
        sites=sites2,
        in_path=os.path.join(out_dir, 'raw_valid_add_clean_high_power'),
        out_path=os.path.join(out_dir, 'result'))


def _cut_invalid(
    in_path,
    out_path,
    min_energy,
):
    os.makedirs(out_path, exist_ok=True)
    raw_deflection_table = map_and_reduce.read_deflection_table(
        path=in_path)
    deflection_table = analysis.cut_invalid_from_deflection_table(
        deflection_table=raw_deflection_table,
        min_energy=min_energy)
    map_and_reduce.write_deflection_table(
        deflection_table=deflection_table,
        path=out_path)


def _add_density_fields(
    in_path,
    out_path,
):
    os.makedirs(out_path, exist_ok=True)
    valid_deflection_table = map_and_reduce.read_deflection_table(
        path=in_path)
    deflection_table = analysis.add_density_fields_to_deflection_table(
        deflection_table=valid_deflection_table)
    map_and_reduce.write_deflection_table(
        deflection_table=deflection_table,
        path=out_path)



FIT_KEYS = {
    'primary_azimuth_deg': {
        "start": 90.0,
    },
    'primary_zenith_deg': {
        "start": 0.0,
    },
    'cherenkov_pool_x_m': {
        "start": 0.0,
    },
    'cherenkov_pool_y_m': {
        "start": 0.0,
    },
}


def _smooth_and_reject_outliers(
    in_path,
    out_path
):
    deflection_table = map_and_reduce.read_deflection_table(
        path=in_path)
    smooth_table = {}
    for site_key in deflection_table:
        smooth_table[site_key] = {}
        for particle_key in deflection_table[site_key]:
            t = deflection_table[site_key][particle_key]
            sm = {}
            for key in FIT_KEYS:
                sres = analysis.smooth(
                    energies=t["energy_GeV"],
                    values=t[key])
                if 'energy_GeV' in sm:
                    np.testing.assert_array_almost_equal(
                        x=sm['energy_GeV'],
                        y=sres["energy_supports"],
                        decimal=3)
                else:
                    sm['energy_GeV'] = sres["energy_supports"]
                sm[key] = sres["key_mean80"]
                sm[key + "_std"] = sres["key_std80"]
                df = pandas.DataFrame(sm)
            smooth_table[site_key][particle_key] = df.to_records(index=False)
    os.makedirs(out_path, exist_ok=True)
    map_and_reduce.write_deflection_table(
        deflection_table=smooth_table,
        path=out_path)


def _set_high_energies(
    particles,
    in_path,
    out_path,
    energy_start=200,
    energy_stop=600,
    num_points=20
):
    deflection_table = map_and_reduce.read_deflection_table(
        path=in_path)

    charge_signs = {}
    for particle_key in particles:
        charge_signs[particle_key] = np.sign(
            particles[particle_key]["electric_charge_qe"])

    out = {}
    for site_key in deflection_table:
        out[site_key] = {}
        for particle_key in deflection_table[site_key]:
            t = deflection_table[site_key][particle_key]
            sm = {}
            for key in FIT_KEYS:
                sm["energy_GeV"] = np.array(
                    t["energy_GeV"].tolist() +
                    np.geomspace(
                        energy_start,
                        energy_stop,
                        num_points).tolist()
                )
                key_start = charge_signs[particle_key]*FIT_KEYS[key]["start"]
                sm[key] = np.array(
                    t[key].tolist() +
                    (key_start*np.ones(num_points)).tolist())
            df = pandas.DataFrame(sm)
            df = df[df['primary_zenith_deg'] <= cpw.MAX_ZENITH_DEG]
            out[site_key][particle_key] = df.to_records(index=False)
    os.makedirs(out_path, exist_ok=True)
    map_and_reduce.write_deflection_table(
        deflection_table=out,
        path=out_path)


def _fit_power_law(
    particles,
    sites,
    in_path,
    out_path,
):
    deflection_table = map_and_reduce.read_deflection_table(path=in_path)
    charge_signs = {}
    for particle_key in particles:
        charge_signs[particle_key] = np.sign(
            particles[particle_key]["electric_charge_qe"])
    os.makedirs(out_path, exist_ok=True)
    for site_key in sites:
        for particle_key in particles:
            t = deflection_table[site_key][particle_key]
            fits = {}
            for key in FIT_KEYS:
                key_start = charge_signs[particle_key]*FIT_KEYS[key]["start"]
                if np.mean(t[key] - key_start) > 0:
                    sig = -1
                else:
                    sig = 1
                expy, _ = scipy_optimize_curve_fit(
                    analysis.power_law,
                    t['energy_GeV'],
                    t[key] - key_start,
                    p0=(sig*charge_signs[particle_key], 1.))
                fits[key] = {
                    "formula": "f(energy) = scale*energy**index + offset",
                    "scale": float(expy[0]),
                    "index": float(expy[1]),
                    "offset": float(key_start),
                }
            filename = "{:s}_{:s}".format(site_key, particle_key)
            filepath = os.path.join(out_path, filename)
            with open(filepath+'.json', 'wt') as fout:
                fout.write(json.dumps(fits, indent=4))


def _export_table(
    particles,
    sites,
    in_path,
    out_path,
):
    os.makedirs(out_path, exist_ok=True)
    for site_key in sites:
        for particle_key in particles:
            out = {}
            out["energy_GeV"] = np.geomspace(
                np.min(particles[particle_key]['energy_bin_edges_GeV']),
                np.max(particles[particle_key]['energy_bin_edges_GeV']),
                1024)
            filename = "{:s}_{:s}".format(site_key, particle_key)
            filepath = os.path.join(in_path, filename)
            with open(filepath+'.json', 'rt') as fin:
                power_law = json.loads(fin.read())
            for key in FIT_KEYS:
                rec_key = analysis.power_law(
                    energy=out["energy_GeV"],
                    scale=power_law[key]["scale"],
                    index=power_law[key]["index"])
                rec_key += power_law[key]["offset"]
                out[key] = rec_key
            df = pandas.DataFrame(out)
            df = df[df['primary_zenith_deg'] <= cpw.MAX_ZENITH_DEG]
            csv = df.to_csv(index=False)
            out_filepath = os.path.join(out_path, filename)
            with open(out_filepath+'.csv.tmp', 'wt') as fout:
                fout.write(csv)
            shutil.move(out_filepath+'.csv.tmp', out_filepath+'.csv')
