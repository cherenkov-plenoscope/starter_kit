import glob
import json
import gzip
import os

events = []
for p in glob.glob('*.gz'):
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

colors = ['C0', 'C1', 'C2']
o = 0
label_handles = []
# for o, obj in enumerate(e['light_field_trigger']['object_distances']):

median_pe = 61
std = np.sqrt(median_pe)

wt = np.histogram(
    num_air_shower_photons,
    weights=trigger_mask.astype(np.int),
    bins=bins)[0]

wwot = np.histogram(
    num_air_shower_photons,
    bins=bins)[0]

relative_uncertainties = np.sqrt(wwot)/wwot

label_handles.append(
    ax2.step(
        bins[:-1],
        wt/wwot,
        # label='refocused to '+str(np.round(obj/1e3, 1))+'km',
        color=colors[o]
    )[0]
)

w = wt/wwot
w_low = w - w*relative_uncertainties
w_upp = w + w*relative_uncertainties

EXPOSURE_TIME = 40e-9
accidental_rate = 1/(EXPOSURE_TIME/w[0])
accidental_rate_uncertainty = accidental_rate*(np.sqrt(wwot[0])/wwot[0])

print(
    'Accidental rate: ' + str(np.round(accidental_rate, 0)) +
    ' +- ' + str(np.round(accidental_rate_uncertainty, 0)) + 's^-1')

for i, b in enumerate(bins):
    if i > 0 and i < len(bins) - 1:
        ax2.fill_between(
            [bins[i-1], bins[i]],
            w_low[i],
            w_upp[i],
            alpha=0.2,
            color=colors[o])

ax1.step(bins[:-1], wwot, 'k')
ax1.semilogx()
ax1.semilogy()
ax1.set_ylabel('# events/1')

ax2.semilogx()
ax2.set_xlabel('# detected air-shower-photons/1')
ax2.set_ylabel('probability to trigger/1')
# ax2.legend(handles=label_handles)

plt.show()

max_scatter_area = np.pi*max_scatter_radii**2

energy_bin_edges = np.logspace(
    np.log10(0.25),
    np.log10(25),
    50)

num_thrown = np.histogram(
    energies,
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
plt.ylabel('effective area/m^2')
plt.xlabel('energy gamma-ray/GeV')
plt.show()
