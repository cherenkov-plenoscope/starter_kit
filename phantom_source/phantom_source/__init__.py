from . import light_field
from . import merlict
from . import plot
from . import mesh
from . import transform
import numpy as np


def make_meshes_of_default_phantom_source(
    intensity=360, depth_start_m=2.5e3, depth_stop_m=20e3,
):
    RR = 1.0
    intensity = 1e4 * intensity

    keys = ["triangle", "spiral", "sun", "smiley", "cross"]
    _depths = np.geomspace(depth_start_m, depth_stop_m, len(keys))
    depths = {}
    for i, key in enumerate(keys):
        depths[key] = _depths[i]

    Mimg = []
    Mimg.append(
        mesh.triangle(
            pos=[-1.0, +1.3, depths["triangle"]],
            radius=1.8,
            density=intensity * (depths["triangle"] / 1e3 ** RR),
        )
    )
    Mimg.append(
        mesh.spiral(
            pos=[-1.0, -1.3, depths["spiral"]],
            turns=2.5,
            outer_radius=1.7,
            density=intensity * (depths["spiral"] / 1e3 ** RR),
            fn=110,
        )
    )
    Mimg.append(
        mesh.sun(
            pos=[1.7, 0.0, depths["sun"]],
            num_flares=11,
            radius=1.0,
            density=intensity * (depths["sun"] / 1e3 ** RR),
            fn=110,
        )
    )
    Mimg.append(
        mesh.smiley(
            pos=[-1.0, +1.3, depths["smiley"]],
            radius=0.9,
            density=intensity * (depths["smiley"] / 1e3 ** RR),
            fn=50,
        )
    )
    Mimg.append(
        mesh.cross(
            pos=[+1.3, -1.3, depths["cross"]],
            radius=0.7,
            density=intensity * (depths["cross"] / 1e3 ** RR),
        )
    )

    # transform to cartesian scenery
    # ------------------------------
    Mscn = []
    for mimg in Mimg:
        mscn = mesh.transform_image_to_scneney(mesh=mimg)
        Mscn.append(mscn)

    return Mscn, Mimg, depths
