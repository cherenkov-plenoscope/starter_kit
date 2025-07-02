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

- Credentials to download CORSIKA_ from KIT_

.. _KIT: https://www.kit.edu/index.php
.. _CORSIKA: https://www.iap.kit.edu/corsika/index.php

- GNU Compiler Collection (build-essentials)

- Fortran77 compiler from the GNU compiler collection

- git

Install
-------
First the starter_kit needs to be cloned recursively with all its submodules.

.. code-block:: bash

    git clone --recursive git@github.com:cherenkov-plenoscope/starter_kit.git


For development, the python packages are all installed ``--editable`` right in place.
Also some some executables need to build from sources such as CORSIKA and merlict.
To do this, there is the ``install.py`` script.
When you got access to the CORSIKA sources, it is easiest when you download the
sources for CORSIKA and provide them directly.

.. code-block:: bash

    python ./install.py install --corsika_tar path/to/my/corsika-75600.tar.gz


Uninstall
---------
The ``install.py`` can also uninstall all packages and remove the builds.

.. code-block:: bash

    python ./install.py uninstall


Configfiles
-----------
The cherenkov plenoscope simulation uses configfiles in the user's home.

* ``.corsika_primary.json`` is used by ``packages/corsika_primary`` to link the
  CORSIKA executables.

* ``.merlict_develompment_kit_python.json`` is used by
  ``packages/merlict_develompment_kit_python`` to link the
  ``merlict_develompment_kit`` executables.

* ``.plenoirf.json`` is used by ``packages/plenoirf`` to link the executable
  for the ``ground_grid``.

* ``.plenoirf.production-run-id-range.json`` is used by ``packages/plenoirf`` to define ranges for ``run_id``s (which serve as random seeds) for the production of the instrument response function.




Updating
--------

.. code-block:: bash

    git pull
    git submodule update


Development
-----------
List the version strings of local and remote packages on ``PyPi``.

.. code-block:: bash

    python ./install.py pypi



.. |CcByLicenseBadge| image:: https://img.shields.io/badge/license-CC--BY--4.0-lightgrey.svg
    :target: https://creativecommons.org/licenses/by/4.0/deed.en
