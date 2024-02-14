import numpy as np
import sebastians_matplotlib_addons as sebplt
from . import mesh as Mesh


def ax3d_add_mesh(ax3d, mesh, color="b"):
    for e in mesh["edges"]:
        start = mesh["vertices"][e[0]]
        stop = mesh["vertices"][e[1]]
        ax3d.plot(
            xs=[start[0], stop[0]],
            ys=[start[1], stop[1]],
            zs=[start[2], stop[2]],
            color=color,
        )


def ax_add_mesh(ax, mesh, color="b"):
    for e in mesh["edges"]:
        start = mesh["vertices"][e[0]]
        stop = mesh["vertices"][e[1]]
        ax.plot(
            [start[0], stop[0]], [start[1], stop[1]], color=color,
        )


def ax3d_set_xyzlim(ax3d, xlim, ylim, zlim):
    for x in xlim:
        for y in ylim:
            for z in zlim:
                ax3d.plot(xs=[x], ys=[y], zs=[z])


def ax3d_set_pane_colors(
    ax3d,
    x=(1.0, 1.0, 1.0, 1.0),
    y=(1.0, 1.0, 1.0, 1.0),
    z=(1.0, 1.0, 1.0, 1.0),
):
    ax3d.w_xaxis.set_pane_color(x)
    ax3d.w_yaxis.set_pane_color(y)
    ax3d.w_zaxis.set_pane_color(z)


def save_3d_views_of_meshes(
    meshes, scale, xlim, ylim, zlim, elevations, azimuths, paths
):
    Mscn_km = []
    for mscn in meshes:
        mscn_km = Mesh.scale(mesh=mscn, factor=scale)
        Mscn_km.append(mscn_km)

    fig = sebplt.figure(style={"rows": 2 * 1080, "cols": 1080, "fontsize": 1})
    ax3d = fig.add_subplot(111, projection="3d")
    for mscn_km in Mscn_km:
        ax3d_add_mesh(ax3d=ax3d, mesh=mscn_km, color="k")
    ax3d_set_xyzlim(
        ax3d=ax3d,
        xlim=np.array(xlim) * scale,
        ylim=np.array(ylim) * scale,
        zlim=np.array(zlim) * scale,
    )
    ax3d_set_pane_colors(
        ax3d=ax3d,
        x=(1.0, 1.0, 1.0, 1.0),
        y=(1.0, 1.0, 1.0, 1.0),
        z=(1.0, 1.0, 1.0, 1.0),
    )

    for i in range(len(paths)):
        ax3d.view_init(elev=elevations[i], azim=azimuths[i])
        fig.savefig(paths[i])

    sebplt.close(fig)


def write_2d_view_of_meshes_to_path(meshes, path, mesh_colors=None):
    if mesh_colors == None:
        mesh_colors = ["k" for i in range(len(meshes))]

    fig = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1})
    ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.85, 0.85])
    for i in range(len(meshes)):
        ax_add_mesh(ax, meshes[i], color=mesh_colors[i])
    ax.set_aspect("equal")
    fig.savefig(path)
    sebplt.close(fig)
