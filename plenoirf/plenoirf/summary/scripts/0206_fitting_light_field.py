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
    c_para, c_perp = atg.project_light_field_onto_source_image(
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


def make_jobs(loph_chunk_dir, quality, limits):
    chunk_paths = glob.glob(os.path.join(loph_chunk_dir, "*.tar"))
    jobs = []
    for chunk_path in chunk_paths:
        job = {}
        job["loph_path"] = str(chunk_path)
        job["quality"] = dict(quality)
        job["limits"] = dict(limits)

        jobs.append(job)
    return jobs


def run_job(job):
    run = pl.photon_stream.loph.LopfTarReader(job["loph_path"])
    limits = job["limits"]

    result = []
    for event in run:
        airshower_id, phs = event
        lixel_ids = phs["photons"]["channels"]
        num_reconstructed_photons = lixel_ids.shape[0]

        if (
            num_reconstructed_photons
            < job["quality"]["min_reconstructed_photons"]
        ):
            continue

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

        guess_cx = np.median(lf.cx)
        guess_cy = np.median(lf.cy)
        scan_cr = np.deg2rad(1.1 * limits["max_cxcy_radius"])
        guess_x = 0.0
        guess_y = 0.0
        scan_r = 1.1 * limits["max_core_radius"]

        mm = Minuit(
            fcn=lf.model,
            source_cx=guess_cx,
            limit_source_cx=(guess_cx - scan_cr, guess_cx + scan_cr),
            source_cy=guess_cy,
            limit_source_cy=(guess_cy - scan_cr, guess_cy + scan_cr),
            core_x=guess_x,
            limit_core_x=(guess_x - scan_r, guess_x + scan_r),
            core_y=guess_y,
            limit_core_y=(guess_y - scan_r, guess_y + scan_r),
            print_level=0,
            errordef=Minuit.LEAST_SQUARES,
        )
        mm.migrad()

        result.append(
            {
                spt.IDX: airshower_id,
                "cx": mm.values["source_cx"],
                "cy": mm.values["source_cy"],
                "x": mm.values["core_x"],
                "y": mm.values["core_y"],
            }
        )
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
            jobs = make_jobs(
                loph_chunk_dir=loph_chunk_dir,
                quality=sum_config["quality"],
                limits=fit_limits,
            )

            pool = multiprocessing.Pool(8)
            _results = pool.map(run_job, jobs)
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
