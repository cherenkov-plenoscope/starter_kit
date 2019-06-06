import os
import subprocess as sp
import numpy as np
import tempfile
import shutil
import subprocess
import acp_instrument_response_function as irf
import corsika_wrapper as cw
import glob


def run(
    corsika_steering_card,
    out_path,
    merlict_path
):
    with tempfile.TemporaryDirectory(prefix='cp_mag_') as tmp:
        corsika_out_path = os.path.join(tmp, 'airshower.evtio')
        summary_out_path = os.path.join(tmp, 'event_summary.float32')

        cor_rc = cw.corsika(
            steering_card=job['corsika_steering_card'],
            output_path=corsika_out_path,
            save_stdout=True)

        op = summary_out_path
        with open(op+'.stdout', 'w') as out, open(op+'.stderr', 'w') as err:
            call = [merlict_path, corsika_out_path, summary_out_path]
            mct_rc = subprocess.call(call, stdout=out, stderr=err)

        dirname = os.path.dirname(out_path)
        os.makedirs(dirname, exist_ok=True)
        out_path_wo_ext = os.path.splitext(out_path)[0]
        shutil.copyfile(
            corsika_out_path+".stdout", out_path_wo_ext+'_corsika.o')
        shutil.copyfile(
            corsika_out_path+".stderr", out_path_wo_ext+'_corsika.e')
        shutil.copyfile(
            summary_out_path+'.stdout', out_path_wo_ext+'_merlict.o')
        shutil.copyfile(
            summary_out_path+'.stderr', out_path_wo_ext+'_merlict.e')
        shutil.copyfile(
            summary_out_path, out_path)

    return {
        'corsika_return_code': cor_rc,
        'mcerlict_return_code': mct_rc}


def run_job(job):
    return run(
        corsika_steering_card=job["corsika_steering_card"],
        out_path=job["out_path"],
        merlict_path=job["merlict_path"])


def make_jobs(
    particle_type,
    max_zenith_scatter_angle_deg,
    observation_level_altitude_asl,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
    atmosphere_model,
    E_start=0.25,
    E_stop=10.,
    out_dir=os.path.abspath('run/magnetic_field/reduce.tmp'),
    merlict_path=os.path.abspath('build/merlict/merlict-magnetic-field-explorer'),
    num_runs=1,
    num_events_in_run=1000,
):
    jobs = []
    for run_idx in range(num_runs):
        run_number = run_idx + 1
        job = {}
        job['out_path'] = os.path.join(
            out_dir,
            '{:06d}.float32'.format(run_number))
        job['merlict_path'] = merlict_path
        job['corsika_steering_card'] = irf.utils.make_corsika_steering_card(
            random_seed=run_number,
            run_number=run_number,
            number_events_in_run=num_events_in_run,
            primary_particle=irf.utils.primary_particle_to_corsika(
                particle_type),
            E_start=0.25,
            E_stop=10.,
            max_zenith_scatter_angle_deg=max_zenith_scatter_angle_deg,
            max_scatter_radius=0.,
            observation_level_altitude_asl=observation_level_altitude_asl,
            instrument_radius=250.,
            atmosphere_model=irf.utils.atmosphere_model_to_corsika(
                atmosphere_model),
            earth_magnetic_field_x_muT=earth_magnetic_field_x_muT,
            earth_magnetic_field_z_muT=earth_magnetic_field_z_muT)
        jobs.append(job)
    return jobs


def reduce_output(in_dir, out_path):
    with open(out_path, "wb") as fout:
        for path in glob.glob(os.path.join(out_dir, "*.float32")):
            with open(path, "rb") as fin:
                fout.write(fin.read())
