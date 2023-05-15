import numpy as np
import sebastians_matplotlib_addons as sebplt


def make_grid_ticks(center, num_pixel, pixel_angel, tick_angle):
    extent = pixel_angel * num_pixel
    num_ticks = int(np.ceil(extent / tick_angle))
    tick_extent = num_ticks * tick_angle

    tick_start = center - 0.5 * tick_extent
    tick_stop = center + 0.5 * tick_extent
    ticks = np.linspace(tick_start, tick_stop, num_ticks + 1)
    return ticks


def make_explicit_cx_cy_ticks(image_response, tick_angle):
    imgr = image_response
    ticks_cx_deg = make_grid_ticks(
        center=imgr["image"]["binning"]["image"]["center"]["cx_deg"],
        num_pixel=imgr["image"]["binning"]["image"]["num_pixel_cx"],
        pixel_angel=imgr["image"]["binning"]["image"]["pixel_angle_deg"],
        tick_angle=tick_angle,
    )
    ticks_cy_deg = make_grid_ticks(
        center=imgr["image"]["binning"]["image"]["center"]["cy_deg"],
        num_pixel=imgr["image"]["binning"]["image"]["num_pixel_cy"],
        pixel_angel=imgr["image"]["binning"]["image"]["pixel_angle_deg"],
        tick_angle=tick_angle,
    )
    return ticks_cx_deg, ticks_cy_deg


def ax_psf_add_eye(
    ax,
    image_response,
    bin_edges_cx_deg,
    bin_edges_cy_deg,
    linecolor,
    eye_FoV_flat2flat_deg,
):
    imgr = image_response
    sebplt.ax_add_circle(
        ax=ax,
        x=np.mean(
            [
                imgr["image"]["binning"]["image"]["center"]["cx_deg"],
                bin_edges_cx_deg[0],
            ]
        ),
        y=np.mean(
            [
                imgr["image"]["binning"]["image"]["center"]["cy_deg"],
                bin_edges_cy_deg[0],
            ]
        ),
        r=(eye_FoV_flat2flat_deg * 0.5 * 2 / np.sqrt(3)),
        linewidth=0.75,
        linestyle="-",
        color=linecolor,
        alpha=1,
        num_steps=7,
    )


def ax_psf_set_ticks(ax, image_response, grid_angle_deg, x=True, y=True, n=3):
    imgr = image_response
    if x:
        ax.set_xticks(
            [
                imgr["image"]["binning"]["image"]["center"]["cx_deg"]
                - n * grid_angle_deg,
                imgr["image"]["binning"]["image"]["center"]["cx_deg"],
                imgr["image"]["binning"]["image"]["center"]["cx_deg"]
                + n * grid_angle_deg,
            ]
        )
    else:
        ax.set_xticks([])

    if y:
        ax.set_yticks(
            [
                imgr["image"]["binning"]["image"]["center"]["cy_deg"]
                - n * grid_angle_deg,
                imgr["image"]["binning"]["image"]["center"]["cy_deg"],
                imgr["image"]["binning"]["image"]["center"]["cy_deg"]
                + n * grid_angle_deg,
            ]
        )
    else:
        ax.set_yticks([])


CMAPS = {
    "inferno": {"gamma": 0.5, "linecolor": "white",},
    "hot": {"gamma": 0.5, "linecolor": "white",},
    "Blues": {"gamma": 0.5, "linecolor": "black",},
    "binary": {"gamma": 0.5, "linecolor": "black",},
    "magma_r": {"gamma": 0.5, "linecolor": "black",},
}
