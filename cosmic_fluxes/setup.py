import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='cosmic_fluxes',
    version='0.0.0',
    description='Fluxes of cosmic gamma-rays and cosmic-rays relevant '
    'for the atmospheric Cherenkov-method',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sebastian Achim Mueller',
    author_email='sebastian-achim.mueller@mpi-hd.mpg.de',
    url='https://github.com/cherenkov-plenoscope/',
    license='GPL v3',
    packages=['cosmic_fluxes'],
    package_data={'cosmic_fluxes': [os.path.join('resources', '*')]},
    python_requires='>=3',
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
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
