import os
from subprocess import call

call(['pip','uninstall', 'acp_corsika_install'])
call(['pip','uninstall', 'corsika_wrapper'])
call(['pip','uninstall', 'plenopy'])
call(['pip','uninstall', 'gamma_limits_sensitivity'])
call(['pip','uninstall', 'instrument_response_function'])
call(['pip','uninstall', 'instrument_sensitivity_function'])

call(['rm', '-rf', 'build'])