#! /usr/bin/env python
"""
Install the Atmospheric Cherenkov Plenoscope (ACP) starter kit.

We apologize for this, but you need to have the credentials to the KIT CORSIKA
air-shower simulation software.

Go visit https://www.ikp.kit.edu/corsika/ and kindly ask for the username and
password combination.

Usage: install --username=USERNAME --password=PASSWORD

Options:
    --username=USERNAME                 Username for the KIT CORSIKA ftp-server
    --password=PASSWORD                 Password fot the KIT CORSIKA ftp-server
"""
import docopt
import os
from subprocess import call


def main():
    try:
        arguments = docopt.docopt(__doc__)

        call(['pip', 'install', './corsika_install/'])
        call(['mkdir', 'build'])
        call([
            'acp_corsika_install',
            '-p', './build/corsika',
            '--username', arguments['--username'],
            '--password', arguments['--password']])
        call(['pip', 'install', './corsika_wrapper/'])
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
        call(['pip', ' install', './plenopy/'])
        call(['pip', 'install', './gamma_limits_sensitivity/'])
        call(['pip', 'install', './instrument_response_function/'])
        call(['pip', 'install', './instrument_sensitivity_function/'])
        call(['pip', 'install', './robo_mount/'])

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
