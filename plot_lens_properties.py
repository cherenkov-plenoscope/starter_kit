import numpy as np
import matplotlib.pyplot as plt
import photon_spectra.silica_glass_suprasil_311_312_313 as si
import os

fig = plt.figure(figsize=(6, 3), dpi=320)
ax = fig.add_axes((0.1, 0.17, 0.87, 0.75))
ax.plot(
    si.heraeus_silica_glass_suprasil_311_312_313_transmission[:, 0]*1e9,
    si.heraeus_silica_glass_suprasil_311_312_313_transmission[:, 1],
    'k:',
    label='silica-glass, Suprasil 311/312/313,\nincluding Fresnel-reflection')
ax.plot(
    si.fresnell_reflection_losses[:, 0]*1e9,
    si.fresnell_reflection_losses[:, 1],
    'k-',
    label='Fresnel-reflection')
ax.set_ylim([0, 1])
ax.set_xlim([100, 5000])
ax.semilogx()
ax.legend(loc='best', fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Wavelength/nm')
ax.set_ylabel('Transmission/1')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
fig.savefig(os.path.join('examples', 'lens_transmission.png'))



fig = plt.figure(figsize=(6, 3), dpi=320)
ax = fig.add_axes((0.1, 0.17, 0.87, 0.75))
ax.plot(
    si.suprasil_refractive_index[:, 0]*1e9,
    si.suprasil_refractive_index[:, 1],
    'k')
ax.set_ylim([1.4, 1.6])
ax.set_xlim([100, 1000])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Wavelength/nm')
ax.set_ylabel('refractive index/1')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
fig.savefig(os.path.join('examples', 'lens_refraction.png'))
