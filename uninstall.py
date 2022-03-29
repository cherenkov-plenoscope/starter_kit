#! /usr/bin/env python
import os
from subprocess import call

tools = [
    "corsika_primary",
    "cosmic_fluxes",
    "plenopy",
    "cable_robo_mount",
    "plenoirf",
    "magnetic_deflection",
    "sparse_numeric_table",
    "spectral_energy_distribution_units",
    "lima1983analysis",
    "sebastians_matplotlib_addons",
    "json_numpy",
    "gamma_ray_reconstruction",
    "merlict_camera_server",
]
for tool in tools:
    call(["pip", "uninstall", tool])

call(["rm", "-rf", "build"])
