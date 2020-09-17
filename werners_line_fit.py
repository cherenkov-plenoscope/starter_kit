from iminuit import Minuit
import plenopy as pl
import plenoirf
import numpy as np
import os
import glob
import sparse_numeric_table as spt
import matplotlib
import homogeneous_transformation as homot
import airshower_template_generator as atg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PORTAL_CXCY_SIGN = -1.0
PORTAL_XY_SIGN = 1.0
TRUE_CORE_XY_SIGN = -1.0
TRUE_CXCY_SIGN = 1.0

def circle(radius):
    phi = np.linspace(0, 2.0 * np.pi, 1337)
    return radius * np.cos(phi), radius * np.sin(phi)


def gaussian_bell(mu, sigma, x):
    #norm = 1/(sigma * np.sqrt(2 * np.pi))
    norm = 1.0
    return norm * np.exp( - (x - mu)**2 / (2 * sigma**2) )


def model_distance_to_main_axis(
    c_para,
    c_perp,
    perp_distance_threshold
):
    """
    ^ w
    |
   1|-
    | \
    |  \
    |   \
    o----|-------> c_perp
        thr
    """
    num = len(c_perp)
    abs_c_perp = np.abs(c_perp)
    # weights = 1.0 - abs_c_perp/perp_distance_threshold
    # weights[weights > 1.0] = 1.0
    # weights[weights < 0.0] = 0.0

    weights = gaussian_bell(mu=0.0, sigma=perp_distance_threshold, x=c_perp)
    perp_weight = np.sum(weights)/num

    return perp_weight


def _f(
    source_cx,
    source_cy,
    core_x,
    core_y,
    light_field,
):
    c_para, c_perp = atg.project_light_field_onto_source_image(
        cer_cx_rad=light_field["cx"],
        cer_cy_rad=light_field["cy"],
        cer_x_m=light_field["x"],
        cer_y_m=light_field["y"],
        primary_cx_rad=source_cx,
        primary_cy_rad=source_cy,
        primary_core_x_m=core_x,
        primary_core_y_m=core_y
    )
    w = model_distance_to_main_axis(
        c_para=c_para,
        c_perp=c_perp,
        perp_distance_threshold=np.deg2rad(0.25)
    )
    return 1.0 - w

class LightField:
    def __init__(self, cx, cy, x, y, time_slice):
        self.cx = cx
        self.cy = cy
        self.x = x
        self.y = y
        self.time_slice = time_slice

    def model(self, source_cx, source_cy, core_x, core_y):
        return _f(
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

fov_radius_deg = 3.25
run_dir = "2020-08-04_portal"

past_trigger_dir = os.path.join(
    run_dir, "event_table", "namibia", "gamma", "past_trigger.map"
)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join(run_dir, "light_field_geometry")
)

gamma_table = spt.read(
    path=os.path.join(
        run_dir, "event_table", "namibia", "gamma", "event_table.tar"
    ),
    structure=plenoirf.table.STRUCTURE,
)


T_plenoscope = homot.HomTra(
    translation=homot.Vec(0.0, 0.0, 0.0),
    rotation=homot.quaternion.set_tait_bryan(0.0, 0.0, 0.0)
)

paths = glob.glob(os.path.join(past_trigger_dir, "*.tar"))
paths.sort()

for event_path in paths:

    idx = int(os.path.basename(event_path).split(".")[0])

    truth = spt.cut_table_on_indices(
        gamma_table, plenoirf.table.STRUCTURE, common_indices=[idx]
    )

    momentum = np.array(
        [
            truth["primary"]["momentum_x_GeV_per_c"][0],
            truth["primary"]["momentum_y_GeV_per_c"][0],
            truth["primary"]["momentum_z_GeV_per_c"][0],
        ]
    )
    momentum = momentum / np.linalg.norm(momentum)
    true_cx = TRUE_CXCY_SIGN * momentum[0]
    true_cy = TRUE_CXCY_SIGN * momentum[1]

    true_x = TRUE_CORE_XY_SIGN * truth["core"]["core_x_m"][0]
    true_y = TRUE_CORE_XY_SIGN * truth["core"]["core_y_m"][0]

    event = pl.event.read_from_tar(event_path, light_field_geometry)

    (
        arrival_slices,
        lixel_ids,
    ) = pl.photon_stream.cython_reader.arrival_slices_and_lixel_ids(
        event.raw_sensor_response
    )
    cer_ids = event.cherenkov_photons.photon_ids

    if len(cer_ids) < 67:
        continue
    print(event)

    lf = LightField(
        cx=PORTAL_CXCY_SIGN * light_field_geometry.cx_mean[lixel_ids[cer_ids]],
        cy=PORTAL_CXCY_SIGN * light_field_geometry.cy_mean[lixel_ids[cer_ids]],
        x=PORTAL_XY_SIGN * light_field_geometry.x_mean[lixel_ids[cer_ids]],
        y=PORTAL_XY_SIGN * light_field_geometry.y_mean[lixel_ids[cer_ids]],
        time_slice=arrival_slices[cer_ids],
    )

    """
    # image
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(np.rad2deg(cx), np.rad2deg(cy), "xb")
    ax.plot(np.rad2deg(true_cx), np.rad2deg(true_cy), "or")

    line_x = np.array([true_cx, -true_x])
    line_y = np.array([true_cy, -true_y])
    ax.plot(np.rad2deg(line_x), np.rad2deg(line_y), "-g")

    fov_cx_deg, fov_cy_deg = circle(fov_radius_deg)
    ax.plot(fov_cx_deg, fov_cy_deg, "k-")
    ax.set_aspect("equal")
    ax.set_xlim([-fov_radius_deg, fov_radius_deg])
    ax.set_ylim([-fov_radius_deg, fov_radius_deg])
    ax.set_xlabel("cx / deg")
    ax.set_ylabel("cy / deg")
    ax.set_title(
        "x {:.2e}m, y {:.2e}m, E {:.2e}GeV".format(
            true_x, true_y, truth["primary"]["energy_GeV"][0]
        )
    )
    fig.savefig("{:06d}_img.jpg".format(idx))
    plt.close(fig)
    """

    guess_cx = np.median(lf.cx)
    guess_cy = np.median(lf.cy)
    scan_cr = np.deg2rad(4.5)
    guess_x = 0.0
    guess_y = 0.0
    scan_r = 500

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
        print_level=2
    )
    mm.migrad()

    # print(mm.values)

    # image
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(np.rad2deg(lf.cx), np.rad2deg(lf.cy), "xb")
    ax.plot(np.rad2deg(true_cx), np.rad2deg(true_cy), "or")
    ax.plot(np.rad2deg(mm.values["source_cx"]), np.rad2deg(mm.values["source_cy"]), "og")

    line_x = np.array([true_cx, -true_x])
    line_y = np.array([true_cy, -true_y])
    ax.plot(np.rad2deg(line_x), np.rad2deg(line_y), "-r")

    line_x = np.array([mm.values["source_cx"], -mm.values["core_x"]])
    line_y = np.array([mm.values["source_cy"], -mm.values["core_y"]])
    ax.plot(np.rad2deg(line_x), np.rad2deg(line_y), "-g")

    fov_cx_deg, fov_cy_deg = circle(fov_radius_deg)
    ax.plot(fov_cx_deg, fov_cy_deg, "k-")
    ax.set_aspect("equal")
    ax.set_xlim([-fov_radius_deg, fov_radius_deg])
    ax.set_ylim([-fov_radius_deg, fov_radius_deg])
    ax.set_xlabel("cx / deg")
    ax.set_ylabel("cy / deg")
    ax.set_title(
        "x {:.2e}m, y {:.2e}m, E {:.2e}GeV".format(
            true_x, true_y, truth["primary"]["energy_GeV"][0]
        )
    )
    fig.savefig("{:06d}_brute.jpg".format(idx))
    plt.close(fig)



    """
    c_para, c_perp = atg.project_light_field_onto_source_image(
        cer_cx_rad=lf.cx,
        cer_cy_rad=lf.cy,
        cer_x_m=lf.x,
        cer_y_m=lf.y,
        primary_cx_rad=true_cx,
        primary_cy_rad=true_cy,
        primary_core_x_m=true_x,
        primary_core_y_m=true_y
    )

    paxel_images = []
    for pa in range(light_field_geometry.number_paxel):
        paxel_images.append([])

    for ph in range(len(c_para)):
        paxel_id = light_field_geometry.paxel_pos_tree.query([lf.x[ph], lf.y[ph]])[1]
        paxel_images[paxel_id].append([c_para[ph], c_perp[ph]])

    for pa in range(light_field_geometry.number_paxel):
        if paxel_images[pa]:
            paxel_images[pa] = np.array(paxel_images[pa])
        else:
            paxel_images[pa] = np.array([]).reshape((0, 2))

    c_para_bin_edges = np.linspace(-3.5, 3.5, 7*12)
    c_perp_bin_edges = np.linspace(-.5, .5, 1*12)
    for pa in range(light_field_geometry.number_paxel):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        hi = np.histogram2d(
            np.rad2deg(paxel_images[pa][:, 0]),
            np.rad2deg(paxel_images[pa][:, 1]),
            bins=(c_para_bin_edges, c_perp_bin_edges)
        )[0]
        ax.pcolor(
            c_para_bin_edges,
            c_perp_bin_edges,
            hi.T,
            cmap="inferno"
        )
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-.5, .5])
        ax.set_xlabel("c_para / deg")
        ax.set_ylabel("c_perp / deg")
        fig.savefig("{:06d}_{:06d}_para_perp.jpg".format(idx, pa))
        plt.close(fig)
    """

    """
    # time
    paxel_timages = []
    for pa in range(light_field_geometry.number_paxel):
        paxel_timages.append([])

    for ph in range(len(c_para)):
        paxel_id = light_field_geometry.paxel_pos_tree.query([lf.x[ph], lf.y[ph]])[1]
        paxel_timages[paxel_id].append([c_para[ph], lf.time_slice[ph]])

    for pa in range(light_field_geometry.number_paxel):
        if paxel_timages[pa]:
            paxel_timages[pa] = np.array(paxel_timages[pa])
        else:
            paxel_timages[pa] = np.array([]).reshape((0, 2))

    c_para_bin_edges = np.linspace(-.5, 3.5, 4*16)
    arrslc_bin_edges = np.linspace(25.0, 75.0, 26)
    for pa in range(light_field_geometry.number_paxel):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        hi = np.histogram2d(
            np.rad2deg(paxel_timages[pa][:, 0]),
            paxel_timages[pa][:, 1],
            bins=(c_para_bin_edges, arrslc_bin_edges)
        )[0]
        ax.pcolor(
            c_para_bin_edges,
            arrslc_bin_edges,
            hi.T,
            cmap="inferno"
        )
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.3)
        ax.set_xlim([-.5, 3.5])
        ax.set_ylim([25.0, 75.0])
        ax.set_xlabel("c_para / deg")
        ax.set_ylabel("arrival slice / 0.5ns")
        fig.savefig("{:06d}_{:06d}_para_time.jpg".format(idx, pa))
        plt.close(fig)
    """