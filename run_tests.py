#! /usr/bin/env python
import os
from subprocess import call

call(['py.test', './corsika_install'])
call(['py.test', './corsika_wrapper'])

os.chdir('./mctracer/Tests')
call(['./../../build/mctracer/mctTest'])
os.chdir('../../')

call(['py.test', './plenopy'])
call(['py.test', './gamma_limits_sensitivity'])
call(['py.test', './instrument_response_function'])
call(['py.test', './instrument_sensitivity_function'])

