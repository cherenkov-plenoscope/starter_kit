import numpy as np
import matplotlib.pyplot as plt
import photon_spectra.silica_glass_suprasil_311_312_313 as si
import photon_spectra.cta_mirrors as mi
import photon_spectra.hamamatsu_r11920_100_05 as pmt
import os

figsize = (6, 4)
dpi = 320
ax_size = (0.11, 0.13, 0.86, 0.80)


fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes(ax_size)

wavelength_100to800 = np.linspace(100e-9, 800e-9, 500)
suprasil_incl_Fresnel_100to800 = np.interp(
    x=wavelength_100to800,
    xp=si.heraeus_silica_glass_suprasil_311_312_313_transmission[:, 0],
    fp=si.heraeus_silica_glass_suprasil_311_312_313_transmission[:, 1],)
suprasil_only_Fresnel_100to800 = np.interp(
    x=wavelength_100to800,
    xp=si.fresnell_reflection_losses[:, 0],
    fp=si.fresnell_reflection_losses[:, 1],)
ax.plot(
    wavelength_100to800*1e9,
    ((1 - suprasil_only_Fresnel_100to800) + suprasil_incl_Fresnel_100to800),
    'k',
    label='Suprasil silica-glass')
ax.plot(
    si.fresnell_reflection_losses[:, 0]*1e9,
    si.fresnell_reflection_losses[:, 1],
    'k:',
    label='Fresnel-reflection')
ax.set_ylim([0.0, 1.02])
ax.set_xlim([100, 800])
#ax.semilogx()
ax.legend(loc='best', fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Wavelength/nm')
ax.set_ylabel(r'Transmission through 1cm/1')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
fig.savefig(
    os.path.join(
        'examples',
        'silica_glass_suprasil_311_312_313_transmission.png'))


fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes(ax_size)
ax.plot(
    si.suprasil_refractive_index[:, 0]*1e9,
    si.suprasil_refractive_index[:, 1],
    'k')
ax.set_ylim([1.425, 1.575])
ax.set_xlim([100, 800])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Wavelength/nm')
ax.set_ylabel('Refraction/1')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
fig.savefig(
    os.path.join(
        'examples',
        'silica_glass_suprasil_311_312_313_refraction.png'))


fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes(ax_size)
ax.plot(
    mi.mst_dielectric[:, 0]*1e9,
    mi.mst_dielectric[:, 1],
    'k')
ax.set_ylim([0, 1])
ax.set_xlim([100, 800])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Wavelength/nm')
ax.set_ylabel('Reflection/1')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
fig.savefig(
    os.path.join('examples', 'mirror_reflectivity_cta_mst_dielectric.png'))


fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes(ax_size)
ax.plot(
    pmt.hamamatsu_r11920_100_05[:, 0]*1e9,
    pmt.hamamatsu_r11920_100_05[:, 1],
    'k')
ax.set_ylim([0, 0.5])
ax.set_xlim([100, 800])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Wavelength/nm')
ax.set_ylabel('Photon-detection-efficiency/1')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
fig.savefig(
    os.path.join('examples', 'hamatsu_r11930_100_05_pmt_pde.png'))
