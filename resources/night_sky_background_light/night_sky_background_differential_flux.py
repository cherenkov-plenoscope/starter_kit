"""
La Palma night sky background light
Taken from:

@article{gaug2013night,
    title={Night Sky Background Analysis for the Cherenkov Telescope Array using the Atmoscope instrument},
    author={Gaug, Markus and others},
    journal={arXiv preprint arXiv:1307.3053},
    year={2013}
}

Figure 4. The CTA icrc2013 proceeding claims the data (black solid line) of
Figure 4 to be from 'Benn Ch.R., Ellison S.R., La Palma tech. note, 2007'
but actual I did not find ths data in the specified technical note 115.
There is sth. similar but not down to this low wavelengths.
"""

import numpy as np
import matplotlib.pyplot as plt

h = 6.626e-34 # [J s]
c = 299792458 

# 1 Jy = 10^-26 J/(s m^2 Hz)
# wavelength [nm], flux [Jy / (deg^2)] // [10^-26 J/(s m^2 Hz deg^2)]
raw = np.array([
    [240, 8], # extrapolated
    [270, 11], # extrapolated
    [300, 14.7], # extrapolated
    [330, 15.7],
    [340, 20],
    [394, 30],
    [395, 145],
    [396, 30],
    [433, 40],
    [435, 81],
    [437, 40],
    [462, 50],
    [485, 60],
    [525, 80],
    [555, 97],
    [557.5, 948],
    [561, 93],
    [585, 105],
    [591, 424],
    [595, 146],
    [600, 130],
    [605, 106],
    [627, 161],
    [630, 851],
    [632, 145],
    [638, 371],
    [640, 105],
    [645, 105],
    [647, 140],
    [660, 140],
    [670, 105],
    [678, 105],
    [685, 217],
    [701, 140],
    ])

raw[:,0] *= 1e-9 # from nm to m -> wavelength [m]

raw[:,1] *= 1e-26 # from Jy to J -> flux [J / (s m^2 Hz deg^2)]

square_deg_2_steradian = (2.0*np.pi/360)**2.0
raw[:,1] /= square_deg_2_steradian # from deg^2 to sr -> flux [J / (s m^2 Hz sr)]

f = c/raw[:,0]
# remove frequency
raw[:,1] *= f # from Hz to 1 -> flux [J / (s m^2 1 sr)]

lamb = raw[:,0]
raw[:,1] /= lamb # from Hz to 1 -> flux [J / (s m^2 m sr)]

# differential flux
raw[:,1] /= ((h*c)/raw[:,0]) # flux [1 / (s m^2 m sr)]

"""
Taken from:
@article{preuss2002study,
    title={Study of the photon flux from the night sky at La Palma and Namibia, in the wavelength region relevant for imaging atmospheric Cherenkov telescopes},
    author={Preuss, S and Hermann, G and Hofmann, W and Kohnle, A},
    journal={Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment},
    volume={481},
    number={1},
    pages={229--240},
    year={2002},
    publisher={Elsevier}
}
Figure 8
"""
# wavelength [m], nsb flux [1/(s m^2 sr m)]
hoffmann = np.array([
    [362.5e-9, 3.8e18],
    [410.0e-9, 4.3e18],
    [440.0e-9, 5.2e18],
    [450.0e-9, 5.5e18],
    [500.0e-9, 6.5e18],
    [550.0e-9, 7.3e18],
])

plt.plot(raw[:,0], raw[:,1], label='benn')
plt.plot(hoffmann[:,0], hoffmann[:,1], 'r', label='hoff')
plt.semilogy()
plt.legend(('Benn and Ellision (inter and extrapolated)', 'W. Hoffmann and friends'), loc='lower right')
plt.xlabel('wavelength/m')
plt.ylabel('nsb flux [1/(s m^2 sr m)]')
plt.savefig('nsb_CTA_and_Benn_and_Ellison_and_Hoffmann.png')

xml = ''
xml+= '<function name="nsb_flux_vs_wavelength" comment="Night Sky Background Analysis for the Cherenkov Telescope Array using the Atmoscope instrument, Markus Gaug, arXiv preprint arXiv:1307.3053, based on Benn Ch.R., Ellison S.R., La Palma tech. note, 2007">\n'
xml+= '    <linear_interpolation>\n'

for lambda_flux in raw:
    xml+= '        <xy x="{wavelength:.2e}" y="{flux:.3e}"/>\n'.format(
        wavelength=lambda_flux[0],
        flux=lambda_flux[1])

xml+= '    </linear_interpolation>\n'
xml+= '</linear_interpolation>\n'

print(xml)