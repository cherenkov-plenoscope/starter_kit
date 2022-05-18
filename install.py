#! /usr/bin/env python
import argparse
import os
from os.path import join
from subprocess import call


def main():
    parser = argparse.ArgumentParser(
        prog="install",
        description=(
            "Install the simulations for the Cherenkov-plenoscope.\n"
            "You need to have the credentials to the KIT-CORSIKA\n"
            "air-shower-simulation-software.\n"
            "\n"
            "Go visit https://www.ikp.kit.edu/corsika/\n"
            "and kindly ask for the username and password combination.\n"
        ),
    )
    parser.add_argument(
        "--username",
        metavar="STR",
        type=str,
        help="The username to access KIT-CORSIKA's downloads",
    )
    parser.add_argument(
        "--password",
        metavar="STR",
        type=str,
        help="The password to access KIT-CORSIKA's downloads",
    )
    arguments = parser.parse_args()

    os.makedirs("build", exist_ok=True)

    # KIT-CORSIKA
    # -----------
    call(["pip", "install", "-e", join(".", "corsika_install")])
    call(
        [
            join(
                ".",
                "corsika_install",
                "corsika_primary",
                "scripts",
                "install.py",
            ),
            "--install_path",
            join(".", "build", "corsika"),
            "--username",
            arguments.username,
            "--password",
            arguments.password,
            "--resource_path",
            join(".", "corsika_install", "resources"),
        ]
    )

    # Photon-propagator merlict
    # -------------------------
    merlict_build_dir = join(".", "build", "merlict")
    os.makedirs(merlict_build_dir, exist_ok=True)
    call(
        [
            "cmake",
            "../../merlict_development_kit",
            "-DCMAKE_C_COMPILER=gcc",
            "-DCMAKE_CXX_COMPILER=g++",
        ],
        cwd=merlict_build_dir,
    )
    call(["make", "-j", "12"], cwd=merlict_build_dir)
    call(
        [
            "touch",
            join(
                ".", "..", "..", "merlict_development_kit", "CMakeLists.txt",
            ),
        ],
        cwd=merlict_build_dir,
    )
    call(["make", "-j", "12"], cwd=merlict_build_dir)

    # Tools
    # -----
    tools = [
        "json_numpy",
        "binning_utils",
        "propagate_uncertainties",
        "sebastians_matplotlib_addons",
        "sparse_numeric_table",
        "cosmic_fluxes",
        "plenopy",
        "cable_robo_mount",
        "gamma_ray_reconstruction",
        "plenoirf",
        "magnetic_deflection",
        "spectral_energy_distribution_units",
        "lima1983analysis",
        "merlict_development_kit/merlict_visual/apps/merlict_camera_server",
    ]
    for tool in tools:
        call(["pip", "install", "-e", join(".", tool)])


if __name__ == "__main__":
    main()
