#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import airshower_template_generator as atg
import os
import plenopy as pl
from iminuit import Minuit
import pandas
import multiprocessing
import glob


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

bell_model_lut_path = "2020-09-26_gamma_lut/bell_model/bell_model.json"

def read_energy_altitude_guess(
    site_key,
    particle_key,
    path=os.path.join(
        pa["summary_dir"], "0065_learning_airshower_maximim_and_energy"
    ),
):
    tt = irf.json_numpy.read_tree(path)
    guess_energy = {}
    guess_altitude = {}

    sk_alt = tt[site_key][particle_key]["airshower_maximum"]
    num_alt = len(sk_alt["idx"])
    for i in range(num_alt):
        guess_altitude[sk_alt["idx"][i]] = sk_alt["airshower_maximum"][i]

    sk_ene = tt[site_key][particle_key]["energy"]
    num_ene = len(sk_ene["idx"])
    for i in range(num_ene):
        guess_energy[sk_ene["idx"][i]] = sk_ene["energy"][i]

    return guess_energy, guess_altitude


_passed_trigger_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

passed_trigger_idx_sets = {}
for sk in irf_config["config"]["sites"]:
    passed_trigger_idx_sets[sk] = {}
    for pk in irf_config["config"]["particles"]:
        passed_trigger_idx_sets[sk][pk] = set(
            _passed_trigger_indices[sk][pk]["passed_trigger"][spt.IDX]
        )

lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)


fov_radius_deg = (
    0.5 * irf_config["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
)

fit_limits = {}
fit_limits["max_cxcy_radius"] = 1.5 * fov_radius_deg
fit_limits["max_core_radius"] = 1e3


# FIT MODEL
# =========


def model_distance_to_main_axis(c_para, c_perp, perp_distance_threshold):
    num = len(c_perp)

    l_trans_max = atg.model.lorentz_transversal(
        c_deg=0.0, peak_deg=0.0, width_deg=perp_distance_threshold
    )
    l_trans = atg.model.lorentz_transversal(
        c_deg=c_perp, peak_deg=0.0, width_deg=perp_distance_threshold
    )
    l_trans /= l_trans_max

    perp_weight = np.sum(l_trans) / num

    return perp_weight


def disc_potential_wall(r, r0, r1):
    if r > r0:
        return ((r - r0) / (r1 - r0)) ** 2
    else:
        return 0.0


def _f(
    source_cx, source_cy, core_x, core_y, light_field,
):
    WRT_DOWNWARDS = -1.0
    c_para, c_perp = atg.projection.project_light_field_onto_source_image(
        cer_cx_rad=WRT_DOWNWARDS * light_field["cx"],
        cer_cy_rad=WRT_DOWNWARDS * light_field["cy"],
        cer_x_m=light_field["x"],
        cer_y_m=light_field["y"],
        primary_cx_rad=WRT_DOWNWARDS * source_cx,
        primary_cy_rad=WRT_DOWNWARDS * source_cy,
        primary_core_x_m=core_x,
        primary_core_y_m=core_y,
    )
    w = model_distance_to_main_axis(
        c_para=c_para, c_perp=c_perp, perp_distance_threshold=np.deg2rad(1.0)
    )

    return 1.0 - w


class LightField:
    def __init__(
        self,
        cx,
        cy,
        x,
        y,
        time_slice,
        cxcy_radius_0,
        cxcy_radius_1,
        xy_radius_0,
        xy_radius_1,
    ):
        self.cx = cx
        self.cy = cy
        self.x = x
        self.y = y
        self.time_slice = time_slice

        self.cxcy_radius_0 = cxcy_radius_0
        self.cxcy_radius_1 = cxcy_radius_1
        self.xy_radius_0 = xy_radius_0
        self.xy_radius_1 = xy_radius_1

        assert self.cxcy_radius_0 > 0.0
        assert self.cxcy_radius_1 > self.cxcy_radius_0

        assert self.xy_radius_0 > 0.0
        assert self.xy_radius_1 > self.xy_radius_0

    def model(self, source_cx, source_cy, core_x, core_y):
        reppelling_source = disc_potential_wall(
            r=np.hypot(source_cx, source_cy),
            r0=self.cxcy_radius_0,
            r1=self.cxcy_radius_1,
        )

        reppelling_core = disc_potential_wall(
            r=np.hypot(core_x, core_y),
            r0=self.xy_radius_0,
            r1=self.xy_radius_1,
        )

        w = _f(
            source_cx,
            source_cy,
            core_x,
            core_y,
            light_field={
                "cx": self.cx,
                "cy": self.cy,
                "x": self.x,
                "y": self.y,
            },
        )

        return w + reppelling_source + reppelling_core


def make_jobs(loph_chunk_dir, quality, limits, bell_model_lut_path, site_key, particle_key):
    chunk_paths = glob.glob(os.path.join(loph_chunk_dir, "*.tar"))
    jobs = []
    for chunk_path in chunk_paths:
        job = {}
        job["loph_path"] = str(chunk_path)
        job["bell_model_lut_path"] = str(bell_model_lut_path)
        job["quality"] = dict(quality)
        job["limits"] = dict(limits)
        job["site_key"] = str(site_key)
        job["particle_key"] = str(particle_key)

        jobs.append(job)
    return jobs


def run_job(job):
    run = pl.photon_stream.loph.LopfTarReader(job["loph_path"])
    limits = job["limits"]
    bell_model_lut = atg.model.read_bell_model_lut(job["bell_model_lut_path"])

    guess_energy, guess_altitude = read_energy_altitude_guess(
        site_key=job["site_key"],
        particle_key=job["particle_key"]
    )
    INSTRUMENT_RADIUS = 40.0

    result = []
    for event in run:
        airshower_id, loph_record = event
        print("airshower_id", airshower_id)
        num_reconstructed_photons = loph_record["photons"]["channels"].shape[0]

        if (
            num_reconstructed_photons
            < job["quality"]["min_reconstructed_photons"]
        ):
            continue

        slf = atg.model.SplitLightField(
            loph_record=loph_record, light_field_geometry=lfg
        )

        if slf.number_photons  < 200:
            print("skip ", airshower_id)
            continue


        try:
            print("guess_energy", guess_energy[airshower_id], "GeV")
            print("guess_altitude", guess_altitude[airshower_id]*1e-3, "km")
            print("num cherenkov ", slf.number_photons, "p.e.")

            bell_fit = atg.model.BellLightFieldFitter(
                bell_model_lut=bell_model_lut,
                split_light_field=slf,
                energy_GeV=guess_energy[airshower_id],
                altitude_m=guess_altitude[airshower_id],
            )

            max_core_radius = bell_fit.max_core_radius()
            print("max_core_radius ", max_core_radius, "m")

            error_cxcy = np.deg2rad(0.1)
            error_xy = 5.0
            guess_cx = slf.median_cx
            guess_cy = slf.median_cy
            scan_cr = np.deg2rad(1.1 * limits["max_cxcy_radius"])
            scan_r = 0.7 * max_core_radius - INSTRUMENT_RADIUS
            scan_r = np.max([0.0, scan_r])

            guess_x = 0.5*scan_r
            guess_y = 0.5*scan_r

            print("max scan core radius ", scan_r, "m")

            mm = Minuit(
                fcn=bell_fit.model,
                source_cx=guess_cx,
                error_source_cx=error_cxcy,
                limit_source_cx=(guess_cx - scan_cr, guess_cx + scan_cr),
                source_cy=guess_cy,
                error_source_cy=error_cxcy,
                limit_source_cy=(guess_cy - scan_cr, guess_cy + scan_cr),
                core_x=guess_x,
                error_core_x=error_xy,
                limit_core_x=(-scan_r, scan_r),
                core_y=guess_y,
                error_core_y=error_xy,
                limit_core_y=(-scan_r, scan_r),
                print_level=0,
                errordef=Minuit.LIKELIHOOD,
            )
            mm.migrad()

            reco = {
                spt.IDX: airshower_id,
                "cx": mm.values["source_cx"],
                "cy": mm.values["source_cy"],
                "x": mm.values["core_x"],
                "y": mm.values["core_y"],
                "fval": mm.fval,
            }
            print(reco)
            result.append(reco)

        except AssertionError as e:
            print(e)
            print("failed airshower_id", airshower_id)
        except KeyError as e:
            print(e)
            print("failed airshower_id", airshower_id)

        """
        lf = LightField(
            cx=lfg.cx_mean[lixel_ids],
            cy=lfg.cy_mean[lixel_ids],
            x=lfg.x_mean[lixel_ids],
            y=lfg.y_mean[lixel_ids],
            time_slice=phs["photons"]["arrival_time_slices"],
            cxcy_radius_0=np.deg2rad(limits["max_cxcy_radius"]),
            cxcy_radius_1=np.deg2rad(1.1 * limits["max_cxcy_radius"]),
            xy_radius_0=limits["max_core_radius"],
            xy_radius_1=1.1 * limits["max_core_radius"],
        )

        error_cxcy = np.deg2rad(0.1)
        error_xy = 10.0
        guess_cx = np.median(lf.cx)
        guess_cy = np.median(lf.cy)
        scan_cr = np.deg2rad(1.1 * limits["max_cxcy_radius"])
        guess_x = np.random.normal(loc=0.0, scale=error_xy)
        guess_y = np.random.normal(loc=0.0, scale=error_xy)
        scan_r = 1.1 * limits["max_core_radius"]

        mm = Minuit(
            fcn=lf.model,
            source_cx=guess_cx,
            error_source_cx=error_cxcy,
            limit_source_cx=(guess_cx - scan_cr, guess_cx + scan_cr),
            source_cy=guess_cy,
            error_source_cy=error_cxcy,
            limit_source_cy=(guess_cy - scan_cr, guess_cy + scan_cr),
            core_x=guess_x,
            error_core_x=error_xy,
            limit_core_x=(guess_x - scan_r, guess_x + scan_r),
            core_y=guess_y,
            error_core_y=error_xy,
            limit_core_y=(guess_y - scan_r, guess_y + scan_r),
            print_level=0,
            errordef=Minuit.LEAST_SQUARES,
        )
        mm.migrad()
        """


    return result


for sk in ["namibia"]:  # irf_config["config"]["sites"]:
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        raw_loph_run = os.path.join(
            pa["run_dir"], "event_table", sk, pk, "cherenkov.phs.loph.tar",
        )

        loph_run_passed_trigger = os.path.join(
            pa["out_dir"], sk, pk, "passed_trigger_cherenkov.phs.loph.tar",
        )

        if not os.path.exists(loph_run_passed_trigger):
            pl.photon_stream.loph.read_filter_write(
                in_path=raw_loph_run,
                out_path=loph_run_passed_trigger,
                identity_set=passed_trigger_idx_sets[sk][pk],
            )

        loph_chunk_dir = os.path.join(pa["out_dir"], sk, pk, "loph_chunks")
        if not os.path.exists(loph_chunk_dir):
            pl.photon_stream.loph.split_into_chunks(
                loph_path=loph_run_passed_trigger,
                out_dir=loph_chunk_dir,
                chunk_prefix="chunk_",
                num_events_in_chunk=256,
            )

        result_path = os.path.join(site_particle_dir, "reco" + ".json")
        if not os.path.exists(result_path):
            print("make jobs")
            jobs = make_jobs(
                loph_chunk_dir=loph_chunk_dir,
                bell_model_lut_path=bell_model_lut_path,
                quality=sum_config["quality"],
                limits=fit_limits,
                site_key=sk,
                particle_key=pk,
            )
            # limit jobs
            jobs = jobs[0:1]
            """
            pool = multiprocessing.Pool(8)
            _results = pool.map(run_job, jobs)
            """
            _results = []
            for job in jobs:
                print("job", job)
                _results.append(run_job(job))

            results = []
            for chunk in _results:
                for r in chunk:
                    results.append(r)

            reco_df = pandas.DataFrame(results)
            reco_di = reco_df.to_dict(orient="list")

            irf.json_numpy.write(
                path=result_path, out_dict=reco_di,
            )
        else:
            reco_di = irf.json_numpy.read(result_path)
