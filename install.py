#! /usr/bin/env python
import argparse
import os
import shutil
import subprocess
import importlib
import requests
from packaging.version import Version


def get_highest_package_version_tag_on_pypi(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    data = requests.get(url).json()
    versions = list(data["releases"].keys())
    versions.sort(key=Version, reverse=True)
    return versions[0]


def make_importlib_find_the_newly_installed_packages():
    importlib.invalidate_caches()


def _pip_list():
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


def normalize_pip_name(name):
    return str.replace(name, "_", "-")


def pip_list():
    out = {}
    pip = _pip_list()
    for name in pip:
        out[normalize_pip_name(name)] = pip[name]
    return out


def is_installed(path):
    rc = subprocess.Popen(["which", path], stdout=subprocess.PIPE).wait()
    return True if rc == 0 else False


def build_corsika(username, password, corsika_tar):
    assert os.path.isdir(os.path.join(".", "packages", "corsika_primary"))
    assert (
        normalize_pip_name("corsika_primary") in pip_list()
    ), "corsika_primary must be pip-installed before building corsika."

    if not is_installed("gfortran"):
        print("CORSIKA uses gfortran, but it's not in your path.")
        print("Install gfortran (GNU compiler collection) if you have not.")
        print("Maybe also make a f77 link pointing to your gfortran.")

    if corsika_tar:
        subprocess.call(
            [
                os.path.join(
                    ".",
                    "packages",
                    "corsika_primary",
                    "corsika_primary",
                    "scripts",
                    "install.py",
                ),
                "--install_path",
                os.path.join(".", "packages", "build", "corsika"),
                "--corsika_tar",
                corsika_tar,
                "--resource_path",
                os.path.join(".", "packages", "corsika_primary", "resources"),
            ]
        )
    else:
        assert username, "Expected a Corsika-username for download."
        assert password, "Expected a Corsika-password for download."
        subprocess.call(
            [
                os.path.join(
                    ".",
                    "packages",
                    "corsika_primary",
                    "corsika_primary",
                    "scripts",
                    "install.py",
                ),
                "--install_path",
                os.path.join(".", "packages", "build", "corsika"),
                "--username",
                username,
                "--password",
                password,
                "--resource_path",
                os.path.join(".", "packages", "corsika_primary", "resources"),
            ]
        )


def build_merlict_development_kit(num_threads):
    if not is_installed("cmake"):
        print("Merlict uses cmake, but it's not in your path.")
        print("I suggest to install build-essentials.")

    if not is_installed("g++"):
        print("Merlict uses g++, but it's not in your path.")
        print("I suggest to install build-essentials.")

    num_threads_str = "{:d}".format(num_threads)

    assert os.path.isdir(
        os.path.join(".", "packages", "merlict_development_kit")
    )

    build_dir = os.path.join(".", "packages", "build")
    merlict_build_dir = os.path.join(build_dir, "merlict_development_kit")

    os.makedirs(merlict_build_dir, exist_ok=True)
    subprocess.call(
        [
            "cmake",
            os.path.join("..", "..", "merlict_development_kit"),
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


def build_merlict_c89():
    merlict_c89_dir = os.path.join(
        "packages", "merlict", "merlict", "c89", "merlict_c89"
    )
    subprocess.call(["make"], cwd=merlict_c89_dir)

    merlict_c89_build_dir = os.path.join("packages", "build", "merlict_c89")
    os.makedirs(merlict_c89_build_dir, exist_ok=True)
    shutil.copy(
        src=os.path.join(merlict_c89_dir, "build", "bin", "ground_grid"),
        dst=os.path.join(merlict_c89_build_dir, "ground_grid"),
    )


LOCAL_PYHTHON_PACKAGES = [
    {"path": "black_pack", "name": "black_pack"},
    {"path": "rename_after_writing", "name": "rename_after_writing"},
    {"path": "json_numpy", "name": "json_numpy_sebastian-achim-mueller"},
    {"path": "json_utils", "name": "json_utils_sebastian-achim-mueller"},
    {"path": "json_line_logger", "name": "json_line_logger"},
    {"path": "sequential_tar", "name": "sequential_tar"},
    {"path": "spherical_coordinates", "name": "spherical_coordinates"},
    {"path": "dynamicsizerecarray", "name": "dynamicsizerecarray"},
    {"path": "binning_utils", "name": "binning_utils_sebastian-achim-mueller"},
    {"path": "solid_angle_utils", "name": "solid_angle_utils"},
    {"path": "ray_voxel_overlap", "name": "ray_voxel_overlap"},
    {"path": "thin_lens", "name": "thin_lens"},
    {"path": "un_bound_histogram", "name": "un_bound_histogram"},
    {"path": "triangle_mesh_io", "name": "triangle_mesh_io"},
    {"path": "optic_object_wavefronts", "name": "optic_object_wavefronts"},
    {"path": "merlict", "name": "merlict"},
    {
        "path": "computer_aided_design_for_optical_instruments",
        "name": "computer_aided_design_for_optical_instruments",
    },
    {"path": "svg_cartesian_plot", "name": "svg_cartesian_plot"},
    {
        "path": "photon_spectra",
        "name": "photon_spectra_cherenkov-plenoscope-project",
    },
    {
        "path": "merlict_development_kit_python",
        "name": "merlict_development_kit_python_cherenkov-plenoscope-project",
    },
    {"path": "spherical_histogram", "name": "spherical_histogram"},
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
    {
        "path": "cosmic_fluxes",
        "name": "cosmic_fluxes_cherenkov-plenoscope-project",
    },
    {
        "path": "gamma_ray_reconstruction",
        "name": "gamma_ray_reconstruction_cherenkov-plenoscope-project",
    },
    {"path": "corsika_primary", "name": "corsika_primary"},
    {
        "path": "magnetic_deflection",
        "name": "magnetic_deflection_cherenkov-plenoscope-project",
    },
    {
        "path": "flux_sensitivity",
        "name": "flux_sensitivity_sebastian-achim-mueller",
    },
    {"path": "cable_robo_mount", "name": "cable_robo_mount"},
    {"path": "timing_toy_simulation", "name": "timing_toy_simulation"},
    {
        "path": "phantom_source",
        "name": "phantom_source_cherenkov-plenoscope-project",
    },
    {
        "path": "airshower_template_generator",
        "name": "airshower-template-generator-cherenkov-plenoscope",
    },
    {"path": "plenopy", "name": "plenopy"},
    {"path": "plenoptics", "name": "plenoptics_cherenkov-plenoscope-project"},
    {"path": "plenoirf", "name": "plenoirf_cherenkov-plenoscope-project"},
    {
        "path": "merlict_development_kit/merlict_visual/apps/merlict_camera_server",
        "name": "merlict_camera_server",
    },
]


def read_version_file_local_python_package(path):
    package_name = os.path.basename(path)
    version_file_path = os.path.join(path, package_name, "version.py")
    with open(version_file_path) as f:
        txt = f.read()
        last_line = txt.splitlines()[-1]
        version_string = last_line.split()[-1]
        version = version_string.strip("\"'")
    return version


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

    pypi_parser = commands.add_parser(
        "pypi",
        help="Compares the local packages to what is hosted on PyPi.",
        description=(
            "This is meant to get an overview about what packages need to "
            "be updated on PyPi again. And about what packages are not on "
            "PyPi at all."
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
    build_dir = os.path.join(".", "packages", "build")

    if args.command == "install":
        os.makedirs(build_dir, exist_ok=True)

        # python packages
        must_update_pip_list = True
        for pypackage in LOCAL_PYHTHON_PACKAGES:
            if must_update_pip_list:
                installed_packages = pip_list()
                must_update_pip_list = False
            if normalize_pip_name(pypackage["name"]) in installed_packages:
                print(pypackage["name"], "Already installed.")
            else:
                must_update_pip_list = True
                rc = subprocess.call(
                    [
                        "pip",
                        "install",
                        "-e",
                        os.path.join(".", "packages", pypackage["path"]),
                    ]
                )
                if rc != 0:
                    raise AssertionError(
                        "Failed to install {:s}".format(pypackage["path"])
                    )

        make_importlib_find_the_newly_installed_packages()

        # corsika
        if os.path.exists(os.path.join(build_dir, "corsika")):
            print(os.path.join(build_dir, "corsika"), "Already done.")
        else:
            build_corsika(
                username=args.username,
                password=args.password,
                corsika_tar=args.corsika_tar,
            )

        import corsika_primary

        corsika_primary.configfile.write(
            config=corsika_primary.configfile.default(
                build_dir=os.path.join(build_dir, "corsika")
            )
        )

        # merlict_development_kit
        if os.path.exists(os.path.join(build_dir, "merlict_development_kit")):
            print(
                os.path.join(build_dir, "merlict_development_kit"),
                "Already done.",
            )
        else:
            build_merlict_development_kit(num_threads=args.j)

        import merlict_development_kit_python as mdkpy

        mdkpy.configfile.write(
            config=mdkpy.configfile.default(
                build_dir=os.path.join(build_dir, "merlict_development_kit")
            )
        )

        # merlict_c89
        if os.path.exists(os.path.join(build_dir, "merlict_c89")):
            print(os.path.join(build_dir, "merlict_c89"), "Already done.")
        else:
            build_merlict_c89()
        import plenoirf

        plenoirf.configfile.write(
            config=plenoirf.configfile.default(
                merlict_c89_ground_grid_path=os.path.join(
                    build_dir, "merlict_c89", "ground_grid"
                )
            )
        )

    elif args.command == "uninstall":
        subprocess.call(["rm", "-rf", build_dir])

        for pypackage in LOCAL_PYHTHON_PACKAGES:
            subprocess.call(
                [
                    "pip",
                    "uninstall",
                    "--yes",
                    normalize_pip_name(pypackage["name"]),
                ]
            )

    elif args.command == "pypi":
        local_packages = pip_list()

        header = ""
        header += "{:<60s}   ".format("[package name]")
        header += "{:<16s}".format("[local pip]")
        header += "   "
        header += "{:<16s}".format("[local file]")
        header += "   "
        header += "{:<16s}".format("[remote pypi]")
        header += "   "
        header += "{:<16s}".format("[suggestion]")
        print(header)
        print("=" * len(header))

        for pypackage in LOCAL_PYHTHON_PACKAGES:
            normalized_name = normalize_pip_name(pypackage["name"])

            if normalized_name in local_packages:
                pip_version = local_packages[normalized_name]["version"]
            else:
                pip_version = None

            try:
                file_version = read_version_file_local_python_package(
                    path=os.path.join("packages", pypackage["path"])
                )
            except Exception as err:
                file_version = None

            try:
                pypi_version = get_highest_package_version_tag_on_pypi(
                    package_name=normalized_name
                )
            except KeyError as err:
                pypi_version = None

            ppp = "{:<60s}   ".format(normalized_name)
            if pip_version:
                ppp += "{:<16s}".format(pip_version)
            else:
                ppp += "{:<16s}".format("-")

            if file_version:
                ppp += "{:<16s}".format(file_version)
            else:
                ppp += "{:<16s}".format("-")

            ppp += "   "
            if pypi_version:
                ppp += "{:<16s}".format(pypi_version)
            else:
                ppp += "{:<16s}".format("-")

            ppp += "   "
            if pypi_version and file_version:
                if Version(pypi_version) < Version(file_version):
                    ppp += "{:<16s}".format("Update PyPi,")

            if file_version and pip_version:
                if Version(pip_version) < Version(file_version):
                    ppp += "{:<16s}".format("Re install with pip,")

            print(ppp)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
