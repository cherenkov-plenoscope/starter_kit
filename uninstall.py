#! /usr/bin/env python
import os
from subprocess import call

call(['pip','uninstall', 'acp_corsika_install'])
call(['pip','uninstall', 'corsika_wrapper'])
call(['pip','uninstall', 'plenopy'])
call(['pip','uninstall', 'gamma_limits_sensitivity'])
call(['pip','uninstall', 'acp_instrument_response_function'])
call(['pip','uninstall', 'acp_instrument_sensitivity_function'])
call(['pip','uninstall', 'reflector_study'])

call(['rm', '-rf', 'build'])
