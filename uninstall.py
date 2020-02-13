#! /usr/bin/env python
import os
from subprocess import call

call(['pip', 'uninstall', 'corsika_primary_wrapper'])
call(['pip', 'uninstall', 'corsika_wrapper'])
call(['pip', 'uninstall', 'plenopy'])
call(['pip', 'uninstall', 'reflector_study'])
call(['pip', 'uninstall', 'plenoscope_map_reduce'])
call(['pip', 'uninstall', 'simpleio'])
call(['pip', 'uninstall', 'cosmic_fluxes'])
call(['rm', '-rf', 'build'])
