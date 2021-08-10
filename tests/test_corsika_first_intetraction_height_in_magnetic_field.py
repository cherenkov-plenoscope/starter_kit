import numpy as np
import corsika_wrapper as cw
import os
import subprocess as sp
import tempfile
import corsika_wrapper as cw
import simpleio


def steering_card(Bx, Bz, num_shower):
    c = ''
    c += 'RUNNR   1\n'
    c += 'EVTNR   1\n'
    c += 'NSHOW   {:d}\n'.format(num_shower)
    c += 'PRMPAR  1\n'
    c += 'ESLOPE  0.\n'
    c += 'ERANGE  1. 1.\n'
    c += 'THETAP  0. 0.\n'
    c += 'PHIP    0. 360.\n'
    c += 'SEED 1 0 0\n'
    c += 'SEED 2 0 0\n'
    c += 'SEED 3 0 0\n'
    c += 'SEED 4 0 0\n'
    c += 'OBSLEV  1e2\n'
    c += 'FIXCHI  0.\n'
    c += 'MAGNET {Bx:.3E} {Bz:.3E}\n'.format(Bx=Bx, Bz=Bz)
    c += 'ELMFLG  T T\n'
    c += 'MAXPRT  1\n'
    c += 'PAROUT  F F\n'
    c += 'TELESCOPE  0. 0. 0. 50e2 # instrument radius\n'
    c += 'ATMOSPHERE 26 T # Chile, Paranal, ESO site\n'
    c += 'CWAVLG 250 700\n'
    c += 'CSCAT 1 0 0\n'
    c += 'CERQEF F T F # pde, atmo, mirror\n'
    c += 'CERSIZ 1\n'
    c += 'CERFIL F\n'
    c += 'TSTART T\n'
    c += 'EXIT\n'
    return c


out_dir = os.path.join("tests", "test_corsika_first_interaction")
os.makedirs(out_dir, exist_ok=True)

num_shower = 2560

# CORSIKA
# -------
Bxs = np.linspace(1e-9, 50, 2)
Bzs = np.linspace(1e-9, 50, 2)

first_interaction_heights = np.zeros(shape=(len(Bxs), len(Bzs), num_shower))

for iBx in range(len(Bxs)):
    for iBz in range(len(Bzs)):
        with tempfile.TemporaryDirectory (prefix='test_corsika_') as tmp:
            corsika_steering_card_path = os.path.join(tmp, "steering.txt")
            with open(corsika_steering_card_path, "wt") as fout:
                fout.write(
                    steering_card(
                        Bx=Bxs[iBx],
                        Bz=Bzs[iBz],
                        num_shower=num_shower))

            eventio_path = os.path.join(tmp, "shower.eventio")
            cw.corsika(
                steering_card=cw.read_steering_card(corsika_steering_card_path),
                output_path=eventio_path,
                save_stdout=True)

            simpleio_path = os.path.join(tmp, "shower.simpleio")
            sp.call([
                os.path.join('build','merlict', 'merlict-eventio-converter'),
                '-i', eventio_path,
                '-o', simpleio_path])

            run = simpleio.SimpleIoRun(simpleio_path)
            heights = []
            for idx in range(len(run)):
                event = run[idx]
                first_interaction_height = -1.*event.header.raw[7 - 1]*1e-2
                heights.append(first_interaction_height)
            heights = np.array(heights)

            first_interaction_heights[iBx, iBz, :] = heights

relative_uncertainty = np.sqrt(num_shower)/num_shower
mean_first_intersection_height = np.mean(first_interaction_heights, axis=2)

