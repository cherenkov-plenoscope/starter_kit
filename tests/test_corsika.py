import os
import subprocess as sp
import tempfile
import numpy as np
import pytest
import corsika_wrapper as cw
import shutil


def corsika_steering_card_electron(zenith_angle_deg, azimuth_angle_deg):
    out = ''
    out += 'RUNNR    1\n'
    out += 'EVTNR    1\n'
    out += 'NSHOW    1\n'
    out += 'PRMPAR   3 # e-minus\n'
    out += 'ESLOPE  0.0\n'
    out += 'ERANGE  1 1\n'
    out += 'THETAP  {:f} {:f}\n'.format(zenith_angle_deg, zenith_angle_deg)
    out += 'PHIP    {:f} {:f}\n'.format(azimuth_angle_deg, azimuth_angle_deg)
    out += 'SEED 1 0 0\n'
    out += 'SEED 2 0 0\n'
    out += 'SEED 3 0 0\n'
    out += 'SEED 4 0 0\n'
    out += 'OBSLEV  5000.0e2 # 563 g/cm**2\n'
    out += 'FIXCHI  0.\n'
    out += 'MAGNET 1e-99 1e-99\n'
    out += 'ELMFLG  T   T\n'
    out += 'MAXPRT  1\n'
    out += 'PAROUT  F F\n'
    out += 'TELESCOPE  0. 0. 0. 300.e2 # instrument radius\n'
    out += 'ATMOSPHERE 26 T # Chile, Paranal, ESO site\n'
    out += 'CWAVLG 250 700\n'
    out += 'CSCAT 1 0 0\n'
    out += 'CERQEF F T F # pde, atmo, mirror\n'
    out += 'CERSIZ 1\n'
    out += 'CERFIL F\n'
    out += 'TSTART T\n'
    out += 'EXIT\n'
    return out


out_dir = 'test_coordinate_system'

os.makedirs(out_dir, exist_ok=True)

zenith_angle_range = np.linspace(5., 45., 3)
azimuth_angle_range = np.linspace(0., 360., 3, endpoint=False)

idx = 0
for zenith in zenith_angle_range:
    for azimuth in azimuth_angle_range:
        event_dir = os.path.join(out_dir, "{:06d}".format(idx))
        os.makedirs(event_dir, exist_ok=True)

        corsika_steering_card_path = os.path.join(event_dir, "steering.txt")
        with open(corsika_steering_card_path, "wt") as fout:
            fout.write(
                corsika_steering_card_electron(
                    zenith_angle_deg=zenith,
                    azimuth_angle_deg=azimuth))

        eventio_path = os.path.join(event_dir, "shower.eventio")
        cw.corsika(
            steering_card=cw.read_steering_card(corsika_steering_card_path),
            output_path=eventio_path,
            save_stdout=True)

        saneio_path = os.path.join(event_dir, "shower.saneio")
        sp.call([
            os.path.join('..', 'build','merlict', 'merlict-eventio-converter'),
            '-i', eventio_path,
            '-o', saneio_path])
        os.remove(eventio_path)

        idx += 1
        print(zenith, azimuth)


"""
@pytest.fixture(scope='session')
def tmp(tmpdir_factory):
    fn = tmpdir_factory.mktemp('just_corsika')
    return fn


def test_corsika_simulation(tmp):
    card = cw.read_steering_card(
        join('resources', 'acp', '71m', 'calibration_gamma_event.txt'))
    rc = cw.corsika(
        card,
        join(tmp, 'calibration_gamma.evtio'),
        save_stdout=True)
    assert rc == 0
    assert os.path.exists(join(tmp, 'calibration_gamma.evtio'))


def test_propagation_with_mctracer(tmp):
    assert os.path.exists(join(tmp, 'calibration_gamma.evtio'))
    rc = call([
        join('build', 'merlict', 'merlict-plenoscope-propagation'),
        '--lixel', join(tmp, 'light_field_geometry'),
        '--config', join(
            'resources', 'acp',
            'merlict_propagation_config_no_night_sky_background.json'),
        '--input', join(tmp, 'calibration_gamma.evtio'),
        '--output', join(tmp, 'calibration_gamma.acp'),
        '--all_truth',
        '--random_seed', '1'
    ])
    assert rc == 0
"""

