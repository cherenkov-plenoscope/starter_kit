import os
from subprocess import call

corsika_username = '' 
corsika_password = ''

call(['pip','install', './corsika_install/'])

call(['mkdir', 'build'])

call([
    'acp_corsika_install',
    '-p', './build/corsika', 
    '--username', corsika_username, 
    '--password', corsika_password])

call(['pip','install', './corsika_wrapper/'])
call([
    'corsika', 
    '-c',
    'build/corsika/corsika-75600/run/corsika75600Linux_QGSII_urqmd'])

call(['mkdir', './build/mctracer'])
os.chdir('./build/mctracer')
call(['cmake', '../../mctracer'])
call(['make', '-j', '12'])
call(['touch', './../../mctracer/CMakeLists.txt'])
call(['make', '-j', '12'])
os.chdir('./../..')

call(['pip','install', './plenopy/'])
call(['pip','install', './gamma_limits_sensitivity/'])
call(['pip','install', './instrument_response_function/'])
call(['pip','install', './instrument_sensitivity_function/'])