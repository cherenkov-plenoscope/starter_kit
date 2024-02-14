import os
import numpy as np
import glob
from .. import utils


def init(pulsar_name, resources_dir=None):
    if resources_dir is None:
        resources_dir = utils.get_resources_dir()
    with open(os.path.join(resources_dir, "SEDpulsars.dat"), "rt") as f:
        seds = loads_spectral_energy_distribution(dat_str=f.read())
    with open(os.path.join(resources_dir, pulsar_name + ".txt")) as f:
        phaseogram = loads_phaseogram(txt_str=f.read())
    return {"phaseogram": phaseogram, "spectrum": seds[pulsar_name]}


def list_pulsar_names(resources_dir=None):
    if resources_dir is None:
        resources_dir = utils.get_resources_dir()
    paths = glob.glob(os.path.join(resources_dir, "J*.txt"))
    basenames = [os.path.basename(p) for p in paths]
    pulsar_names = [str.split(p, ".")[0] for p in basenames]
    return pulsar_names


def loads_spectral_energy_distribution(dat_str):
    # 0 |name
    # 1 |ra
    # 2 |dec
    # 3 | period
    # 4 |pdot
    # 5 |photon_index
    # 6 |cutoff
    # 7 |plec1_pref
    # 8 |plec1_scaling_energy
    # 9 |plec1_cutoff
    lines = str.splitlines(dat_str)
    out = {}

    first_line = lines[0]

    numeric_lines = lines[1:]

    for line in numeric_lines:
        tokens = str.split(line, "|")
        tokens = [str.strip(token) for token in tokens]
        tokens = tokens[1:-1]

        pulsar_name_PSR = tokens[0]
        pulsar_name = str.split(pulsar_name_PSR, " ")[-1]

        try:
            plec1_pref_per_cm2_per_MeV_per_s = float(tokens[7])
            plec1_pref_per_m2_per_MeV_per_s = (
                1e4 * plec1_pref_per_cm2_per_MeV_per_s
            )
            plec1_pref_per_m2_per_GeV_per_s = (
                1e3 * plec1_pref_per_m2_per_MeV_per_s
            )

            plec1_scaling_energy_MeV = float(tokens[8])
            plec1_scaling_energy_GeV = 1e-3 * plec1_scaling_energy_MeV

            plec1_cutoff_MeV = float(tokens[9])
            plec1_cutoff_GeV = 1e-3 * plec1_cutoff_MeV

            photon_index = -float(tokens[5])

            out[pulsar_name] = {
                "spectrum_type": "PLExpCutoff",
                "spectral_index": photon_index,
                "exp_index": 1.0,
                "pivot_energy_GeV": plec1_scaling_energy_GeV,
                "cutoff_energy_GeV": plec1_cutoff_GeV,
                "flux_density_per_m2_per_GeV_per_s": plec1_pref_per_m2_per_GeV_per_s,
            }
        except:
            pass
    return out


def loads_phaseogram(txt_str):
    lines = str.splitlines(txt_str)

    phase = []
    amplitude = []
    for line in lines:
        tokens = str.split(line, " ")
        phase.append(float(tokens[0]))
        amplitude.append(float(tokens[1]))

    phase = np.array(phase)
    amplitude = np.array(amplitude)

    phase = phase * 2 * np.pi

    return {
        "phase_rad": phase,
        "amplitude": amplitude,
    }
