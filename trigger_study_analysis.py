import glob
import json
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import acp_instrument_response_function.utils as irfutils

os.makedirs('results', exist_ok=True)

events = []
for p in glob.glob(os.path.join('intermediate_results_of_runs', '*.gz')):
    with gzip.open(p, 'rt') as fin:
        d = json.loads(fin.read())
        for item in d:
            events.append(item)

num_air_shower_photons = []
exposure_times = []
thresholds = []
energies = []
max_scatter_radii = []
for e in events:
    exposure_times.append(
        e['refocus_sum_trigger'][0]['exposure_time_in_slices'])
    energies.append(e['simulation_truth']['energy'])
    num_air_shower_photons.append(e['num_air_shower_pulses'])
    max_scatter_radii.append(e['simulation_truth']['scatter_radius'])
    t0 = e['refocus_sum_trigger'][0]['patch_threshold']
    t1 = e['refocus_sum_trigger'][1]['patch_threshold']
    t2 = e['refocus_sum_trigger'][2]['patch_threshold']
    thresholds.append(np.array([t0, t1, t2]))
thresholds = np.array(thresholds)
num_air_shower_photons = np.array(num_air_shower_photons)
exposure_times = np.array(exposure_times)
energies = np.array(energies)
max_scatter_radii = np.array(max_scatter_radii)

# Trigger
# -------
trigger_mask = ((thresholds >= 67).sum(axis=1)) >= 1
# 1 out of 3, thresh 67, 0Hz accidental, 72.2
# 2 out of 3, thresh 65, 0Hz accidental, 68.4
# 3 out of 3, thresh 60, 0Hz accidental, 70.4

bins = np.zeros(22)
bins[1] = 1
for b in range(bins.shape[0]):
    if b > 1:
        bins[b] = np.sqrt(2)*bins[b-1]

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.75, 0.8, 0.2], xticklabels=[])
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.6], ylim=(-0.05, 1.05))

wt = np.histogram(
    num_air_shower_photons,
    weights=trigger_mask.astype(np.int),
    bins=bins)[0]

wwot = np.histogram(
    num_air_shower_photons,
    bins=bins)[0]

relative_uncertainties = np.sqrt(wwot)/wwot

ax2.step(bins[:-1], wt/wwot, color='C0')

w = wt/wwot
w_low = w - w*relative_uncertainties
w_upp = w + w*relative_uncertainties

EXPOSURE_TIME = 50e-9
accidental_rate = 1/(EXPOSURE_TIME/w[0])
accidental_rate_uncertainty = accidental_rate*(np.sqrt(wwot[0])/wwot[0])

for i, b in enumerate(bins):
    if i > 0 and i < len(bins) - 1:
        ax2.fill_between(
            [bins[i-1], bins[i]],
            w_low[i],
            w_upp[i],
            alpha=0.2,
            color='C0')

ax1.set_title(
    'Accidental rate: ' + str(np.round(accidental_rate, 0)) +
    ' +- ' + str(np.round(accidental_rate_uncertainty, 0)) + ' s**(-1)')
ax1.step(bins[:-1], wwot, 'k')
ax1.semilogx()
ax1.semilogy()
ax1.set_ylabel('# events/1')
ax2.semilogx()
ax2.set_xlabel('# detected air-shower-photons/1')
ax2.set_ylabel('probability to trigger/1')
plt.savefig(os.path.join('results', 'trigger.png'))

max_scatter_area = np.pi*max_scatter_radii**2
num_energy_bins = int(np.sqrt(energies.shape[0])/6)

energy_bin_edges = np.logspace(
    np.log10(np.min(energies)),
    np.log10(np.max(energies)),
    50)

num_thrown = np.histogram(
    energies,
    bins=energy_bin_edges)[0]

num_detected = np.histogram(
    energies,
    weights=trigger_mask.astype(np.int),
    bins=energy_bin_edges)[0]

area_thrown = np.histogram(
    energies,
    weights=max_scatter_area,
    bins=energy_bin_edges)[0]

area_detected_trigger = np.histogram(
    energies,
    weights=max_scatter_area*trigger_mask,
    bins=energy_bin_edges)[0]

area_detected_analysis = np.histogram(
    energies,
    weights=max_scatter_area*trigger_mask*(num_air_shower_photons >= 100),
    bins=energy_bin_edges)[0]

area_detected_100pe = np.histogram(
    energies,
    weights=max_scatter_area*(num_air_shower_photons >= 100),
    bins=energy_bin_edges)[0]

effective_area_trigger = area_detected_trigger/area_thrown*(
    area_thrown/num_thrown)
effective_area_analysis = area_detected_analysis/area_thrown*(
    area_thrown/num_thrown)
effective_area_100pe = area_detected_100pe/area_thrown*(
    area_thrown/num_thrown)

plt.figure()
l0, = plt.step(
    energy_bin_edges[:-1],
    effective_area_trigger,
    'k',
    label='trigger')
l1, = plt.step(
    energy_bin_edges[:-1],
    effective_area_analysis,
    'r',
    label='trigger && >= 100pe')
l2, = plt.step(
    energy_bin_edges[:-1],
    effective_area_100pe,
    'b',
    label='>= 100pe')
plt.legend(handles=[l0, l1, l2])
plt.semilogx()
plt.semilogy()
plt.ylabel('effective area/m**2')
plt.xlabel('energy/GeV')
plt.savefig(os.path.join('results', 'effective_area.png'))


steering_card = irfutils.read_json(join('input', 'steering.json'))
acp_geometry = irfutils.read_acp_design_geometry(join(
    'input',
    'acp_detector',
    'input',
    'scenery',
    'scenery.xml'))

max_zenith_scatter = np.deg2rad(irfutils.max_zenith_scatter_angle_deg(
    steering_card['source_geometry'],
    acp_geometry['max_FoV_diameter_deg']))

scatter_solid_angle = irfutils.scatter_solid_angle(max_zenith_scatter)

log10_E_TeV = np.log10(energy_bin_edges[0: -1]*1e-3)

acceptence_cm2 = effective_area_trigger*1e2*1e2

out =  '# Atmospheric-Cherenkov-Plenoscope\n'
out += '# --------------------------------\n'
out += '#\n'
out += '# Sebastian A. Mueller\n'
out += '# Max Ludwig Ahnen\n'
out += '# Dominik Neise\n'
out += '# Adrian Biland\n'
out += '#\n'
out += '# steering card\n'
out += '# -------------\n'
card_json = json.dumps(steering_card, indent=2).split('\n')
for line in card_json:
    out += '# ' + line + '\n'
out += '#\n'
if scatter_solid_angle > 0.0:
    out += '# log10(Primary Particle Energy) [log10(TeV)], '
    out += 'Effective Acceptance [sr*cm^2], '
    acceptence_cm2 *= scatter_solid_angle
else:
    out += '# log10(Primary Particle Energy) [log10(TeV)], '
    out += 'Effective Area [cm^2], '
out += 'number thrown [#], number detected [#]\n'

for i in range(len(log10_E_TeV)):
    if num_thrown[i] > 0:
        out += '{e:f}, {a:f}, {nt:d}, {nd:d}\n'.format(
            e=log10_E_TeV[i],
            a=acceptence_cm2[i],
            nt=num_thrown[i],
            nd=num_detected[i])

with open(os.path.join('results', 'irf.csv'), 'wt') as fout:
    fout.write(out)