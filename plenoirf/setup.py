import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="plenoirf",
    version="0.2.2",
    description="Explore magnetic deflection of cosmic-rays below 10GeV.",
    long_description=long_description,
    url="https://github.com/cherenkov-plenoscope",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    license="GPL v3",
    packages=["plenoirf",],
    install_requires=[
        "cosmic_fluxes",
        "corsika_primary",
        "atmospheric_cherenkov_response_cherenkov-plenoscope-project",
        "json_line_logger>=0.0.3",
        "propagate_uncertainties_sebastian-achim-mueller>=0.2.3",
        "shapely",
        "binning_utils_sebastian-achim-mueller",
        "json_numpy_sebastian-achim-mueller",
        "confusion_matrix_sebastian-achim-mueller>=0.0.4",
        "flux_sensitivity_sebastian-achim-mueller>=0.0.1",
        "rename_after_writing",
    ],
    package_data={"plenoirf": [os.path.join("summary", "scripts", "*")]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
