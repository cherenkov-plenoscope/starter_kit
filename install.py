#! /usr/bin/env python
"""
Install the Cherenkov-plenoscope's starter-kit.

You need to have the credentials to the KIT-CORSIKA
air-shower-simulation-software.

Go visit https://www.ikp.kit.edu/corsika/ and kindly ask for the username and
password combination.

Usage: install --username=USERNAME --password=PASSWORD

Options:
    --username=USERNAME                 Username for the KIT CORSIKA ftp-server
    --password=PASSWORD                 Password fot the KIT CORSIKA ftp-server
"""
import docopt
import os
from os.path import join
from subprocess import call


def main():
    try:
        arguments = docopt.docopt(__doc__)
        os.makedirs('build', exist_ok=True)

        # KIT-CORSIKA
        # -----------
        call([
            join('.', 'corsika_install', 'install.py'),
            '--install_path', join('.', 'build', 'corsika'),
            '--username', arguments['--username'],
            '--password', arguments['--password'],
            '--resource_path', join('.', 'corsika_install', 'resources')])
        call(['pip', 'install', '-e', join('.', 'corsika_wrapper')])
        call([
            'corsika',
            '--corsika_path',
            join(
                'build',
                'corsika',
                'corsika-75600',
                'run',
                'corsika75600Linux_QGSII_urqmd')])

        # Photon-propagator merlict
        # -------------------------
        merlict_build_dir = join('.', 'build', 'merlict')
        os.makedirs(merlict_build_dir, exist_ok=True)
        call(['cmake', '../../merlict_development_kit'], cwd=merlict_build_dir)
        call(['make', '-j', '12'], cwd=merlict_build_dir)
        call(['touch', './../../merlict_development_kit/CMakeLists.txt'], cwd=merlict_build_dir)
        call(['make', '-j', '12'], cwd=merlict_build_dir)

        # Tools
        # -----
        call(['pip', 'install', '-e', join(
            '.', 'plenopy')])
        call(['pip', 'install', '-e', join(
            '.', 'gamma_limits_sensitivity')])
        call(['pip', 'install', '-e', join(
            '.', 'instrument_response_function')])
        call(['pip', 'install', '-e', join(
            '.', 'instrument_sensitivity_function')])
        call(['pip', 'install', '-e', join(
            '.', 'robo_mount')])

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
