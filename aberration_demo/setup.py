import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="aberration_demo",
    version="0.0.1",
    description="A simple simulation to show the power of plenoptic perception",
    long_description=long_description,
    url="https://github.com/cherenkov-plenoscope",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    license="MIT",
    packages=["aberration_demo",],
    package_data={
        "aberration_demo": [
            os.path.join("deformations", "scripts", "*"),
            os.path.join("offaxis", "scripts", "*"),
        ],
    },
    install_requires=["perlin_noise"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ],
)
