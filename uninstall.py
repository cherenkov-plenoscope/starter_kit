#! /usr/bin/env python
import os
from subprocess import call

tools = [
    "corsika_primary_wrapper",
    "cosmic_fluxes",
    "plenopy",
    "corsika_wrapper",
    "cable_robo_mount",
    "simpleio",
    "plenoirf",
    "magnetic_deflection",
    "sparse_numeric_table",
    "spectral_energy_distribution_units",
    "lima1983analysis",
    "sebastians_matplotlib_addons",
]
for tool in tools:
    call(["pip", "uninstall", tool])

call(["rm", "-rf", "build"])
