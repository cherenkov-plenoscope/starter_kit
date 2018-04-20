![img](readme/show.gif)

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The starter kit with all the packages needed to explore the ACP.

Recommendations:
- Operating System: Linux (we develope on [Ubuntu](https://www.ubuntu.com/download/desktop) LTS)
- Python environment: [Anaconda Python 3.6](https://www.continuum.io/DOWNLOADS)

## Download
```bash
git clone --recursive git@github.com:TheBigLebowSky/starter_kit.git
```

## Install

You need:

* Credentials for the download of KIT-CORSIKA.
* Fortran77-compiler from the gcc.
* The gcc build-essentials
* [libopencv-dev](https://github.com/TheBigLebowSky/mctracer) for the mctracer.

```bash
cd starter_kit/
python install.py
```

## Run tests
```bash
py.test .
```

## Updating

```bash
git pull
git submodule update
```