#! /usr/bin/env python
import argparse
import os
from os.path import join
from subprocess import call


def is_installed(path):
    rc = subprocess.Popen(["which", path], stdout=subprocess.PIPE).wait()
    return True if rc == 0 else False


def build_corsika(username, password):
    if not is_installed("f77"):
        print("CORSIKA uses f77, but it's not in your path.")
        print("Install gfortran (GNU compiler collection) if you have not.")
        print("Make a f77-link pointing to your gfortran.")

    assert os.path.isdir(join(".", "corsika_install"))

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
            username,
            "--password",
            password,
            "--resource_path",
            join(".", "corsika_install", "resources"),
        ]
    )


def build_merlict_cpp(num_threads):
    if not is_installed("cmake"):
        print("Merlict uses cmake, but it's not in your path.")
        print("I suggest to install build-essentials.")

    if not is_installed("g++"):
        print("Merlict uses g++, but it's not in your path.")
        print("I suggest to install build-essentials.")

    num_threads_str = "{:d}".format(num_threads)

    assert os.path.isdir(join(".", "merlict_development_kit"))

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
    call(["make", "-j", num_threads_str], cwd=merlict_build_dir)
    call(
        [
            "touch",
            join(
                ".", "..", "..", "merlict_development_kit", "CMakeLists.txt",
            ),
        ],
        cwd=merlict_build_dir,
    )
    call(["make", "-j", num_threads_str], cwd=merlict_build_dir)


LOCAL_PYHTHON_PACKAGES = [
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


def main():
    parser = argparse.ArgumentParser(
        prog="install",
        description=(
            "Install or uninstall the Cherenkov-plenoscope. "
            "This is meant for development and production."
        ),
    )
    commands = parser.add_subparsers(help="Commands", dest="command")

    in_parser = commands.add_parser(
        "install",
        help="Build and install.",
        description=(
            "Install the simulations of the Cherenkov-plenoscope. "
            "To download CORSIKA you need credentials. "
            "Go visit https://www.ikp.kit.edu/corsika/ "
            "and kindly ask for the username and password combination. "
            "Builds CORSIKA, and merlict. "
            "Installs the local python-packages in editable mode. "
        ),
    )
    un_parser = commands.add_parser(
        "uninstall",
        help="Remove builds and uninstall.",
        description=(
            "Uninstall all local python-packages and remove the build."
        ),
    )

    in_parser.add_argument(
        "username",
        metavar="username",
        type=str,
        help="to download CORSIKA.",
    )
    in_parser.add_argument(
        "password",
        metavar="password",
        type=str,
        help="to download CORSIKA.",
    )
    in_parser.add_argument(
        "-j",
        metavar="num",
        type=int,
        help="number of threads to use when a build can be parallelized.",
        default=1,
    )

    args = parser.parse_args()

    if args.command == "install":
        os.makedirs("build", exist_ok=True)
        build_corsika(username=args.username, password=args.password)
        build_merlict_cpp(num_threads=args.j)

        for package_path in LOCAL_PYHTHON_PACKAGES:
            call(["pip", "install", "-e", join(".", package_path)])

    elif args.command == "uninstall":
        os.rmdir("build")

        for package_path in LOCAL_PYHTHON_PACKAGES:
            package = os.path.split(package_path)[-1]
            call(["pip", "uninstall", package])


if __name__ == "__main__":
    main()
