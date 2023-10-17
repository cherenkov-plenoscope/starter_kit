#! /usr/bin/env python
import argparse
import os
import subprocess


def pip_list():
    po = subprocess.Popen(["pip", "list"], stdout=subprocess.PIPE)
    po.wait()
    bstdout = po.stdout.read()
    stdout = bytes.decode(bstdout, encoding="utf-8")
    lines = str.splitlines(stdout)
    items = [line.split() for line in lines]
    packages = {}
    for item in items:
        packages[item[0]] = {}
        packages[item[0]]["version"] = item[1]
        if len(item) >= 3:
            packages[item[0]]["path"] = item[2]
    return packages


def pip_name(name):
    return str.replace(name, "_", "-")


def is_installed(path):
    rc = subprocess.Popen(["which", path], stdout=subprocess.PIPE).wait()
    return True if rc == 0 else False


def build_corsika(username, password, corsika_tar):
    assert os.path.isdir(os.path.join(".", "corsika_install"))

    subprocess.call(
        ["pip", "install", "-e", os.path.join(".", "corsika_install")]
    )

    if not is_installed("f77"):
        print("CORSIKA uses f77, but it's not in your path.")
        print("Install gfortran (GNU compiler collection) if you have not.")
        print("Make a f77-link pointing to your gfortran.")

    if corsika_tar:
        subprocess.call(
            [
                os.path.join(
                    ".",
                    "corsika_install",
                    "corsika_primary",
                    "scripts",
                    "install.py",
                ),
                "--install_path",
                os.path.join(".", "build", "corsika"),
                "--corsika_tar",
                corsika_tar,
                "--resource_path",
                os.path.join(".", "corsika_install", "resources"),
            ]
        )
    else:
        assert username, "Expected a Corsika-username for download."
        assert password, "Expected a Corsika-password for download."
        subprocess.call(
            [
                os.path.join(
                    ".",
                    "corsika_install",
                    "corsika_primary",
                    "scripts",
                    "install.py",
                ),
                "--install_path",
                os.path.join(".", "build", "corsika"),
                "--username",
                username,
                "--password",
                password,
                "--resource_path",
                os.path.join(".", "corsika_install", "resources"),
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

    assert os.path.isdir(os.path.join(".", "merlict_development_kit"))

    merlict_build_dir = os.path.join(".", "build", "merlict")
    os.makedirs(merlict_build_dir, exist_ok=True)
    subprocess.call(
        [
            "cmake",
            "../../merlict_development_kit",
            "-DCMAKE_C_COMPILER=gcc",
            "-DCMAKE_CXX_COMPILER=g++",
        ],
        cwd=merlict_build_dir,
    )
    subprocess.call(["make", "-j", num_threads_str], cwd=merlict_build_dir)
    subprocess.call(
        [
            "touch",
            os.path.join(
                ".",
                "..",
                "..",
                "merlict_development_kit",
                "CMakeLists.txt",
            ),
        ],
        cwd=merlict_build_dir,
    )
    subprocess.call(["make", "-j", num_threads_str], cwd=merlict_build_dir)


LOCAL_PYHTHON_PACKAGES = [
    {"path": "rename_after_writing", "name": "rename_after_writing"},
    {"path": "json_numpy", "name": "json_numpy_sebastian-achim-mueller"},
    {"path": "json_utils", "name": "json_utils_sebastian-achim-mueller"},
    {"path": "json_line_logger", "name": "json_line_logger"},
    {"path": "binning_utils", "name": "binning_utils_sebastian-achim-mueller"},
    {"path": "solid_angle_utils", "name": "solid_angle_utils"},
    {"path": "ray_voxel_overlap", "name": "ray_voxel_overlap"},
    {"path": "thin_lens", "name": "thin_lens"},
    {"path": "optic_object_wavefronts", "name": "optic_object_wavefronts"},
    {
        "path": "photon_spectra",
        "name": "photon_spectra_cherenkov-plenoscope-project",
    },
    {
        "path": "homogeneous_transformation",
        "name": "homogeneous_transformation",
    },
    {
        "path": "propagate_uncertainties",
        "name": "propagate_uncertainties_sebastian-achim-mueller",
    },
    {
        "path": "confusion_matrix",
        "name": "confusion_matrix_sebastian-achim-mueller",
    },
    {
        "path": "lima1983analysis",
        "name": "lima1983analysis_sebastian-achim-mueller",
    },
    {
        "path": "sparse_numeric_table",
        "name": "sparse_numeric_table_sebastian-achim-mueller",
    },
    {
        "path": "spectral_energy_distribution_units",
        "name": "spectral_energy_distribution_units_sebastian-achim-mueller",
    },
    {
        "path": "sebastians_matplotlib_addons",
        "name": "sebastians_matplotlib_addons",
    },
    {
        "path": "atmospheric_cherenkov_response",
        "name": "atmospheric_cherenkov_response_cherenkov-plenoscope-project",
    },
    {"path": "cosmic_fluxes", "name": "cosmic_fluxes"},
    {"path": "gamma_ray_reconstruction", "name": "gamma_ray_reconstruction"},
    {"path": "magnetic_deflection", "name": "magnetic_deflection"},
    {
        "path": "flux_sensitivity",
        "name": "flux_sensitivity_sebastian-achim-mueller",
    },
    {"path": "corsika_install", "name": "corsika_primary"},
    {"path": "cable_robo_mount", "name": "cable_robo_mount"},
    {"path": "timing_toy_simulation", "name": "timing_toy_simulation"},
    {"path": "phantom_source", "name": "phantom_source"},
    {
        "path": "airshower_template_generator",
        "name": "airshower_template_generator",
    },
    {"path": "plenopy", "name": "plenopy"},
    {"path": "plenoptics", "name": "plenoptics"},
    {"path": "plenoirf", "name": "plenoirf"},
    {
        "path": "merlict_development_kit/merlict_visual/apps/merlict_camera_server",
        "name": "merlict_camera_server",
    },
]


def main():
    parser = argparse.ArgumentParser(
        prog="install",
        description=("Install or uninstall the Cherenkov-plenoscope. "),
    )
    commands = parser.add_subparsers(help="Commands", dest="command")

    in_parser = commands.add_parser(
        "install",
        help="Build and install.",
        description=(
            "Install the simulations of the Cherenkov-plenoscope. "
            "Either provide the CORSIKA-tar or download it. "
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
        "--corsika_tar",
        metavar="PATH",
        type=str,
        help="file with the CORSIKA-tar you would download from KIT.",
    )
    in_parser.add_argument(
        "--username",
        metavar="STRING",
        type=str,
        help="to download CORSIKA from KIT.",
    )
    in_parser.add_argument(
        "--password",
        metavar="STRING",
        type=str,
        help="to download CORSIKA from KIT.",
    )
    in_parser.add_argument(
        "-j",
        metavar="NUM_THREADS",
        type=int,
        default=1,
        help="number of threads to use when a build can be parallelized.",
    )

    args = parser.parse_args()

    if args.command == "install":
        os.makedirs("build", exist_ok=True)
        if os.path.exists(os.path.join("build", "corsika")):
            print(os.path.join("build", "corsika"), "Already done.")
        else:
            build_corsika(
                username=args.username,
                password=args.password,
                corsika_tar=args.corsika_tar,
            )
        if os.path.exists(os.path.join("build", "merlict")):
            print(os.path.join("build", "merlict"), "Already done.")
        else:
            build_merlict_cpp(num_threads=args.j)

        for pypackage in LOCAL_PYHTHON_PACKAGES:
            installed_packages = pip_list()
            if pip_name(pypackage["name"]) in installed_packages:
                print(pypackage["name"], "Already installed.")
            else:
                rc = subprocess.call(
                    [
                        "pip",
                        "install",
                        "-e",
                        os.path.join(".", pypackage["path"]),
                    ]
                )
                if rc != 0:
                    raise AssertionError(
                        "Failed to install {:s}".format(pypackage["path"])
                    )

    elif args.command == "uninstall":
        subprocess.call(["rm", "-rf", "build"])

        for pypackage in LOCAL_PYHTHON_PACKAGES:
            subprocess.call(
                ["pip", "uninstall", "--yes", pip_name(pypackage["name"])]
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
