import setuptools
import os


with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("cosmic_fluxes", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="cosmic_fluxes_cherenkov-plenoscope-project",
    version=version,
    description="Fluxes of cosmic gamma-rays and cosmic-rays relevant "
    "for the atmospheric Cherenkov-method",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    url="https://github.com/cherenkov-plenoscope/",
    packages=[
        "cosmic_fluxes",
        "cosmic_fluxes.pulsars",
    ],
    package_data={
        "cosmic_fluxes": [
            os.path.join("resources", "*"),
            os.path.join("pulsars", "resources", "*"),
        ],
    },
    python_requires=">=3",
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
    ],
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
