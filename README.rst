####################
Cherenkov Plenoscope
####################
|CcByLicenseBadge|

Welcome to the Cherenkov plenoscope.
This 'starter_kit' is where the Cherenkov plenoscope and its
astronomical performance are simulated and estimated.

The cherenkov-plenoscope project consists of many rather independent packages
and programs. In this 'starter_kit' all the bits come together.


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

.. code-block::bash

    git clone --recursive git@github.com:cherenkov-plenoscope/starter_kit.git


Updating
--------

.. code-block::bash

    git pull
    git submodule update


.. |CcByLicenseBadge| image:: https://img.shields.io/badge/license-CC--BY--4.0-lightgrey.svg
    :target: https://creativecommons.org/licenses/by/4.0/deed.en
