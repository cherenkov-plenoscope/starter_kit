"""
Toy simulation to understand spectra and bin problem

Problem
-------
When estimating effective areas, a particle's energy is thrown with spectral index A.
However, the natural spectral index is B != A.

Now we bin event-energies into a histogram.
Here the resulting bin-counts are misleading as the energy-distribution
inside a bin is different between thrown and natural.
"""

import numpy as np
import corsika_primary_wrapper
import sebastians_matplotlib_addons as seb

prng = np.random.Generator(np.random.MT19937(seed=1337))

E_start = 1e0
E_stop = 1e1
E_thresohd = 5e0
E_thresohd_rel_std = 0.5
num_bins = 5

spectral_index_thrown = -2.0
spectral_index_nature = -2.8

num_thrown = 1 * 1000 * 1000

over_sampling = 2
bin_edges = np.geomspace(E_start, E_stop, num_bins + 1)
fbin_edges = np.geomspace(E_start, E_stop, (over_sampling * num_bins) + 1)

over_sampling_weights = np.linspace(1, 0.5, over_sampling)
# over_sampling_weights = np.ones(over_sampling)
over_sampling_weights /= np.sum(over_sampling_weights)


def geombincenters(bin_edges):
    return np.exp(0.5 * (np.log(bin_edges[0:-1]) + np.log(bin_edges[1:])))


def trigger_decision(energy, energy_threshold, std):
    noise = prng.normal(loc=1.0, scale=std, size=len(energy))
    cherenkov_density = noise * energy
    return (cherenkov_density > energy_threshold).astype(np.int)


def estimate_effective_area(energies, trigger_mask, bin_edges):
    count_thrown = np.histogram(energies, bins=bin_edges)[0]
    count_detected = np.histogram(
        energies, bins=bin_edges, weights=trigger_mask
    )[0]
    area = count_detected / count_thrown
    area_unc = np.sqrt(count_detected) / count_detected
    return area, area_unc


def downsample_histogram(fine_conts, over_sampling, over_sampling_weights):
    num_fine_bins = len(fine_conts)
    num_bins = num_fine_bins // over_sampling
    counts = np.zeros(num_bins)
    for b in range(num_bins):
        fb = over_sampling * b
        for o in range(over_sampling):
            counts[b] += over_sampling_weights[o] * fine_conts[fb + o]
    return counts


def downsample_histogram_center_log(fine_conts, over_sampling):
    num_fine_bins = len(fine_conts)
    num_bins = num_fine_bins // over_sampling
    counts = np.zeros(num_bins)
    for b in range(num_bins):
        fb = over_sampling * b
        for o in range(over_sampling):
            counts[b] += np.log10(fine_conts[fb + o])
        counts[b] /= over_sampling
        counts[b] = 10 ** counts[b]
    return counts


bin_centers = geombincenters(bin_edges)
fbin_centers = geombincenters(fbin_edges)

E_thrown = corsika_primary_wrapper.random_distributions.draw_power_law(
    prng=prng,
    lower_limit=E_start,
    upper_limit=E_stop,
    power_slope=spectral_index_thrown,
    num_samples=num_thrown,
)

E_nature = corsika_primary_wrapper.random_distributions.draw_power_law(
    prng=prng,
    lower_limit=E_start,
    upper_limit=E_stop,
    power_slope=spectral_index_nature,
    num_samples=num_thrown,
)

mask_thrown = trigger_decision(
    energy=E_thrown, energy_threshold=E_thresohd, std=E_thresohd_rel_std
)
A_thrown, A_thrown_unc = estimate_effective_area(
    energies=E_thrown, bin_edges=bin_edges, trigger_mask=mask_thrown
)
fA_thrown, fA_thrown_unc = estimate_effective_area(
    energies=E_thrown, bin_edges=fbin_edges, trigger_mask=mask_thrown
)

mask_nature = trigger_decision(
    energy=E_nature, energy_threshold=E_thresohd, std=E_thresohd_rel_std
)
A_nature, A_nature_unc = estimate_effective_area(
    energies=E_nature,
    bin_edges=bin_edges,
    trigger_mask=mask_nature,
)

A_rel = A_thrown / A_nature

rA_thrown = downsample_histogram_center_log(
    fine_conts=fA_thrown,
    over_sampling=over_sampling,
)

rA_rel = rA_thrown / A_nature


fig = seb.figure(seb.FIGURE_16_9)
ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
seb.ax_add_histogram(
    ax=ax,
    bin_edges=bin_edges,
    bincounts=A_nature,
    linestyle="-",
    linecolor="k",
    bincounts_upper=A_nature * (1 + A_nature_unc),
    bincounts_lower=A_nature * (1 - A_nature_unc),
    face_color="k",
    face_alpha=0.25,
)
ax.plot(bin_centers, A_nature, color="k", linestyle="-")

seb.ax_add_histogram(
    ax=ax,
    bin_edges=bin_edges,
    bincounts=A_thrown,
    linestyle="-",
    linecolor="b",
    bincounts_upper=A_thrown * (1 + A_thrown_unc),
    bincounts_lower=A_thrown * (1 - A_thrown_unc),
    face_color="b",
    face_alpha=0.25,
)
ax.plot(bin_centers, A_thrown, color="b", linestyle="-")


seb.ax_add_histogram(
    ax=ax,
    bin_edges=fbin_edges,
    bincounts=fA_thrown,
    linestyle="-",
    linecolor="g",
    bincounts_upper=fA_thrown * (1 + fA_thrown_unc),
    bincounts_lower=fA_thrown * (1 - fA_thrown_unc),
    face_color="g",
    face_alpha=0.25,
)
ax.plot(fbin_centers, fA_thrown, color="g", linestyle="-")

seb.ax_add_histogram(
    ax=ax,
    bin_edges=bin_edges,
    bincounts=rA_thrown,
    linestyle="-",
    linecolor="orange",
)
ax.plot(bin_centers, rA_thrown, color="orange", linestyle="-")
ax.set_xlim([0.9 * bin_edges[0], 1.1 * bin_edges[-1]])
ax.set_ylim([1e-4, 2])
ax.loglog()
fig.savefig("Aeff.jpg")
seb.close(fig)
