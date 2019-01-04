import os
import json
import numpy
import pkg_resources
import acp_instrument_sensitivity_function as isf
import matplotlib.pyplot as plt

out_dir = os.path.join('examples', 'trigger_ratescan')
os.makedirs(out_dir, exist_ok=True)

nsb_cherenkov_photon_limit = 25

air_showers = {}
nsb = {}
nsb['threshold'] = np.array([])


for particle in ['gamma', 'electron', 'proton']:
    response_path = os.path.join(
        'run',
        'irf',
        particle,
        'results',
        'events.json')
    with open(response_path, 'rt') as fin:
        eo = json.loads(fin.read())

        num_exposure_time_in_slices = eo['num_exposure_time_slices'][0]
        time_slice_durations = eo['time_slice_durations'][0]

        num_cerenkov_photons = np.array(
            eo['num_true_cherenkov_photons'])
        nsb_mask = num_cerenkov_photons < nsb_cherenkov_photon_limit
        cos_mask = np.invert(nsb_mask)

        max_scatter_radii = np.array(eo['max_scatter_radii'])
        energies = np.array(eo['energies'])
        threshold = np.max(np.array(eo['trigger_responses']), axis=1)
        p = {
            'energies': energies,
            'threshold': threshold,
            'area_thrown': np.pi*max_scatter_radii**2}
        nsb['threshold'] = np.hstack([
            nsb['threshold'],
            threshold[nsb_mask]])
        air_showers[particle] = p

num_energy_bins = 35
energy_bin_edges = np.logspace(
    np.log10(0.5),
    np.log10(1000),
    num_energy_bins + 1)

# Load proton_flux
path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function',
    os.path.join('resources', 'proton_spec.dat'))
proton_flux = np.genfromtxt(path)
proton_flux[:, 0] *= 1  # in GeV
proton_flux[:, 1] /= proton_flux[:, 0]**2.7
below_cutoff = proton_flux[:, 0] < 10
proton_flux[below_cutoff, 1] = 0.05*proton_flux[below_cutoff, 1]
proton_diff_flux = np.array(
    [
        energy_bin_edges,
        np.interp(
            x=energy_bin_edges,
            xp=proton_flux[:, 0],
            fp=proton_flux[:, 1])
    ]
).T

# Load electron_positron_flux
path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function',
    os.path.join('resources', 'e_plus_e_minus_spec.dat'))
electron_flux = np.genfromtxt(path)
electron_flux[:, 0] *= 1  # in GeV
electron_flux[:, 1] /= electron_flux[:, 0]**3.0
below_cutoff = electron_flux[:, 0] < 10
electron_flux[below_cutoff, 1] = 0.05*electron_flux[below_cutoff, 1]
electron_diff_flux = np.array(
    [
        energy_bin_edges,
        np.interp(
            x=energy_bin_edges,
            xp=electron_flux[:, 0],
            fp=electron_flux[:, 1])
    ]
).T

exposure_time_per_event = num_exposure_time_in_slices*time_slice_durations

thresholds = np.arange(80, 160)

# NSB rate
# --------
nsb_exposure_time = nsb['threshold'].shape[0]*exposure_time_per_event

# Proton rate
# -----------
num_proton_thrown = np.histogram(
    air_showers['proton']['energies'],
    bins=energy_bin_edges)[0]
sarg_energy = np.argsort(air_showers['proton']['energies'])
proton_area_thrown = np.interp(
    x=energy_bin_edges[:-1],
    xp=air_showers['proton']['energies'][sarg_energy],
    fp=air_showers['proton']['area_thrown'][sarg_energy])
proton_solid_angle_thrown = (
    np.ones(num_energy_bins) *
    isf.utils.solid_angle_of_cone(6.5))

# Electron and Positron rate
# -----------
num_electron_thrown = np.histogram(
    air_showers['electron']['energies'],
    bins=energy_bin_edges)[0]
sarg_energy = np.argsort(air_showers['electron']['energies'])
electron_area_thrown = np.interp(
    x=energy_bin_edges[:-1],
    xp=air_showers['electron']['energies'][sarg_energy],
    fp=air_showers['electron']['area_thrown'][sarg_energy])
electron_solid_angle_thrown = (
    np.ones(num_energy_bins) *
    isf.utils.solid_angle_of_cone(6.5))

num_nsb_triggers = []
electron_acceptances = []
electron_diff_trigger_rates = []
proton_acceptances = []
proton_diff_trigger_rates = []

for threshold in thresholds:
    num_nsb_triggers.append(
        np.sum(nsb['threshold'] > threshold))

    electron_triggers = air_showers['electron']['threshold'] > threshold
    electron_energies = air_showers['electron']['energies'][electron_triggers]
    num_electron_detected = np.histogram(
        electron_energies,
        bins=energy_bin_edges)[0]

    electron_acceptance = (
        electron_area_thrown *
        electron_solid_angle_thrown *
        (num_electron_detected/num_electron_thrown))

    electron_acceptances.append(electron_acceptance)

    electron_diff_trigger_rate = electron_acceptance*electron_diff_flux[:-1, 1]
    electron_diff_trigger_rates.append(electron_diff_trigger_rate)

    proton_triggers = air_showers['proton']['threshold'] > threshold
    proton_energies = air_showers['proton']['energies'][proton_triggers]
    num_proton_detected = np.histogram(
        proton_energies,
        bins=energy_bin_edges)[0]

    proton_acceptance = (
        proton_area_thrown *
        proton_solid_angle_thrown *
        (num_proton_detected/num_proton_thrown))

    proton_acceptances.append(proton_acceptance)

    proton_diff_trigger_rate = proton_acceptance*proton_diff_flux[:-1, 1]
    proton_diff_trigger_rates.append(proton_diff_trigger_rate)


num_nsb_triggers = np.array(num_nsb_triggers)

proton_diff_trigger_rates = np.array(proton_diff_trigger_rates)
inan = np.isnan(proton_diff_trigger_rates)
proton_diff_trigger_rates[inan] = 0.0

electron_diff_trigger_rates = np.array(electron_diff_trigger_rates)
inan = np.isnan(electron_diff_trigger_rates)
electron_diff_trigger_rates[inan] = 0.0


energy_bin_width = np.gradient(energy_bin_edges)
integrated_proton_rate = np.sum(
    proton_diff_trigger_rates*energy_bin_width[:-1],
    axis=1)
integrated_electron_rate = np.sum(
    electron_diff_trigger_rates*energy_bin_width[:-1],
    axis=1)
nsb_rate = num_nsb_triggers/nsb_exposure_time


plt.plot(
    energy_bin_edges[:-1],
    electron_acceptances[23])
plt.xlabel('energy / GeV')
plt.ylabel('acceptance / m^2 sr')
plt.grid()
plt.loglog()
plt.savefig(os.path.join(out_dir, 'electron_acceptance.png'))
plt.close('all')


plt.plot(
    energy_bin_edges[:-1],
    proton_acceptances[23])
plt.xlabel('energy / GeV')
plt.ylabel('acceptance / m^2 sr')
plt.grid()
plt.loglog()
plt.savefig(os.path.join(out_dir, 'proton_acceptance.png'))
plt.close('all')


plt.plot(
    energy_bin_edges,
    proton_diff_flux[:, 1], 'r')
plt.plot(
    energy_bin_edges,
    electron_diff_flux[:, 1], 'b')
plt.xlabel('energy / GeV')
plt.ylabel('diff flux / m^{-2} s^{-s} sr^{-1} GeV^{-1}')
plt.grid()
plt.loglog()
plt.savefig(os.path.join(out_dir, 'proton_r_electron_b_diff_flux.png'))
plt.close('all')

cfg = [
    {
        'style': 'print',
        'figsize': (8, 8),
        'dpi': 240,
        'ax_size': (0.08, 0.08, 0.88, 0.88)
    },
    {
        'style': 'beamer',
        'figsize': (6.4, 3.6),
        'dpi': 300,
        'ax_size': (0.1, 0.13, 0.85, 0.83)
    }
]

for c in cfg:
    fig = plt.figure(figsize=c['figsize'], dpi=c['dpi'])
    ax = fig.add_axes(c['ax_size'])
    ax.plot(
        thresholds,
        nsb_rate +
        integrated_proton_rate +
        integrated_electron_rate,
        'k',
        label='night-sky + cosmic-rays')
    ax.plot(
        thresholds,
        nsb_rate,
        'k:',
        label='night-sky')
    ax.semilogy()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('trigger-threshold / photo-electrons')
    ax.set_ylabel(r'trigger-rate / s$^{-1}$')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.legend(loc='best', fontsize=10)
    ax.axvline(x=103, color='k', linestyle='-', alpha=0.25)
    fig.savefig(
        os.path.join(
            out_dir,
            'ratescan_{:s}.png'.format(c['style'])))
    plt.close('all')
