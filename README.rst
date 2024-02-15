####################
Cherenkov Plenoscope
####################
|CcByLicenseBadge|

Welcome to the Cherenkov plenoscope.
This 'starter_kit' is where the Cherenkov plenoscope and its
astronomical performance are simulated and estimated.

The cherenkov-plenoscope project consists of many rather independent packages
and programs. In this 'starter_kit' all the bits come together. Some packages
are truely standalone, others only make sense inside of this starter_kit.

************
Installation
************

Recommendations
---------------

We develop on:

- GNU-Linux Debian 12

- gnu compiler collection 12.2.0

- python 3.11.5


Requirements
------------

- Credentials to download CORSIKA from KIT

- Fortran77-compiler from the gnu compiler collection (gcc)

- The gcc package build-essentials


Install
-------
First the starter_kit needs to be cloned recursively with all its submodules.

.. code-block:: bash

    git clone --recursive git@github.com:cherenkov-plenoscope/starter_kit.git


For development, the python packages are all installed editable right in place.
Also some some executables need to build from sources such as CORSIKA and merlict.
To do this, there is the ``install.py`` script.
When you got access to the CORSIKA sources, it is easiest when you download the
sources for CORSIKA and provide them directly.

.. code-block:: bash

    python ./install install --corsika_tar path/to/my/corsika-75600.tar.gz


Uninstall
---------
The ``install.py`` can also uninstall all packages and remove the builds.

.. code-block:: bash

    python ./install uninstall 



Updating
--------

.. code-block:: bash

    git pull
    git submodule update


.. |CcByLicenseBadge| image:: https://img.shields.io/badge/license-CC--BY--4.0-lightgrey.svg
    :target: https://creativecommons.org/licenses/by/4.0/deed.en
