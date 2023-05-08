import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="timing_toy_simulation",
    version="0.1.0",
    description="Estimate the absolute timing accuracy of gamma-rays in the atmospheric Cherenkov-method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    url="https://github.com/cherenkov-plenoscope/",
    license="GPL v3",
    packages=["timing_toy_simulation"],
    package_data={},
    python_requires=">=3",
    install_requires=["corsika_primary",],
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
