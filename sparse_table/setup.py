import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='sparse_table',
    version='0.0.0',
    description='Read, write, and manipulate sparse tables',
    long_description=long_description,
    url='https://github.com/cherenkov-plenoscope',
    author='Sebastian Achim Mueller',
    author_email='sebastian-achim.mueller@mpi-hd.mpg.de',
    license='mit',
    packages=[
        'sparse_table',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ],
)
