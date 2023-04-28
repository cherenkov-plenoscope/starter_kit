import numpy as np
import cosmic_fluxes
import os
from .. import utils


def ppog_init():
    return {
        "phase_bin_edges": [],
        "phase_bin_edges_unit": "rad",
        "energy_bin_edges": [],
        "energy_bin_edges_unit": "GeV",
        "relative_amplitude_vs_phase": [],
        "relative_amplitude_vs_phase_cdf": [],
        "differential_flux_vs_energy": [],
        "differential_flux_vs_energy_unit": "m$^{-2}$ s^{-1} (GeV)$^{-1}$",
    }


EXAMPLE_PULSAR_SPECTRUM = {
    "spectrum_type": "PLExpCutoff",
    "spectral_index": -1.5,
    "exp_index": 1.0,
    "pivot_energy_GeV": 0.5,
    "cutoff_energy_GeV": 2.5,
    "flux_density_per_m2_per_GeV_per_s": 3e-4,
}


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def ppog_init_from_profiles(
    profiles_dir, pulsar_name, energy_bin_edges,
):
    pulsar = read_pulsar_from_profiles(
        profiles_dir=profiles_dir, pulsar_name=pulsar_name,
    )

    pulsar_phase_bin_edges = np.array(
        pulsar["phaseogram"]["phase_rad"].tolist() + [2 * np.pi]
    )

    return ppog_init_from_phaseogram(
        pulsar_phase_bin_edges=pulsar_phase_bin_edges,
        pulsar_amplitude_vs_phase=pulsar["phaseogram"]["amplitude"],
        pulsar_spectrum=pulsar["spectrum"],
        energy_bin_edges=energy_bin_edges,
    )


def ppog_init_from_phaseogram(
    pulsar_phase_bin_edges,
    pulsar_amplitude_vs_phase,
    pulsar_spectrum,
    energy_bin_edges,
):
    ppog = ppog_init()
    num_phase_bins = len(pulsar_phase_bin_edges) - 1
    assert num_phase_bins >= 1
    num_energy_bins = len(energy_bin_edges) - 1
    assert num_energy_bins >= 1

    ppog["phase_bin_edges"] = np.array(pulsar_phase_bin_edges)
    ppog["relative_amplitude_vs_phase"] = np.array(
        pulsar_amplitude_vs_phase
    ) / np.sum(pulsar_amplitude_vs_phase)
    cdf = np.array(
        [0] + np.cumsum(ppog["relative_amplitude_vs_phase"]).tolist()
    )
    ppog["relative_amplitude_vs_phase_cdf"] = cdf

    ppog["energy_bin_edges"] = np.array(energy_bin_edges)
    ppog["differential_flux_vs_energy"] = np.zeros(num_energy_bins)

    for e in range(num_energy_bins):
        E_start = ppog["energy_bin_edges"][e]
        E_stop = ppog["energy_bin_edges"][e + 1]
        E = np.mean([E_start, E_stop])
        dKdE = cosmic_fluxes.flux_of_fermi_source(
            fermi_source=pulsar_spectrum, energy=E,
        )
        ppog["differential_flux_vs_energy"][e] = dKdE

    return ppog


def ppog_init_dummy(
    num_phase_bins=2048,
    energy_bin_edges=np.geomspace(1e-1, 1e2, 15),
    pulsar_spectrum=EXAMPLE_PULSAR_SPECTRUM,
    peak_std_rad=0.05,
    base_fraction_of_peak=0.1,
):
    assert num_phase_bins >= 1
    pulsar_phase_bin_edges = np.linspace(0, 2 * np.pi, num_phase_bins + 1)

    phase_normalized = np.zeros(num_phase_bins)
    for p in range(num_phase_bins):
        phi_start = pulsar_phase_bin_edges[p]
        phi_stop = pulsar_phase_bin_edges[p + 1]
        phi = np.mean([phi_start, phi_stop])

        peak_1_amplitude = gaussian(
            x=phi, mu=(2 * np.pi * 1 / 3), sigma=2 * peak_std_rad
        )
        peak_2_amplitude = gaussian(
            x=phi, mu=(2 * np.pi * 2 / 3), sigma=peak_std_rad
        )
        peak_amplitude = peak_1_amplitude + peak_2_amplitude

        base_1 = base_fraction_of_peak * (0.5 + 0.4 * np.sin(phi))
        base_2 = (
            base_fraction_of_peak * (1 / 2) * (0.5 + 0.4 * np.cos(2 * phi))
        )
        base_3 = (
            base_fraction_of_peak * (1 / 3) * (0.5 + 0.4 * np.cos(3 * phi))
        )
        base_amplitude = base_1 + base_2 + base_3
        phase_normalized[p] = base_amplitude + peak_amplitude

    pulsar_amplitude_vs_phase = phase_normalized / np.sum(phase_normalized)

    return ppog_init_from_phaseogram(
        pulsar_phase_bin_edges=pulsar_phase_bin_edges,
        pulsar_amplitude_vs_phase=pulsar_amplitude_vs_phase,
        energy_bin_edges=energy_bin_edges,
        pulsar_spectrum=pulsar_spectrum,
    )


def ppog_draw_phase(ppog, prng, num=1):
    r = prng.uniform(low=0.0, high=1.0, size=num)
    phases = np.interp(
        x=r,
        xp=ppog["relative_amplitude_vs_phase_cdf"],
        fp=ppog["phase_bin_edges"],
    )
    phases = np.mod(phases, (2 * np.pi))
    return phases


def read_pulsar_from_profiles(profiles_dir, pulsar_name):
    with open(os.path.join(profiles_dir, "SEDpulsars.dat"), "rt") as f:
        seds = read_sed_pulsars(dat_str=f.read())

    with open(os.path.join(profiles_dir, pulsar_name + ".txt")) as f:
        phaseogram = read_pulsar_phaseogram(txt_str=f.read())

    return {"phaseogram": phaseogram, "spectrum": seds[pulsar_name]}


def read_sed_pulsars(dat_str):
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


def read_pulsar_phaseogram(txt_str):
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
