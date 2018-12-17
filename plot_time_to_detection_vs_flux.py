import matplotlib.pyplot as plt
import os
import json
import numpy

out_dir = os.path.join('examples', 'time_to_detection_vs_flux')
os.makedirs(out_dir, exist_ok=True)

inpath =  os.path.join('run', 'isf_beamer', 'time_to_detections.csv')

heads = []
sources = []
with open(inpath, 'rt') as fin:
	for line in fin:
		if '#' not in line:
			sources.append(line.replace('\n', ''))
		else:
			heads.append(line.replace('\n', ''))

head = []
for h in heads[1].split(','):
	h = h.replace('#', '')
	h = h.replace('\n', '')
	h = h.strip()
	head.append(h)

r = []
for source in sources:
	s = {}
	source_split = source.split(',')
	for idx, key in enumerate(head):
		s[key] = source_split[idx]
	r.append(s)


time_to_detection = []
flux_1GeV_to_100GeV = [] # cm^{-2} s^{-1}

for s in r:
	time_to_detection.append(float(s['time_est']))
	flux_1GeV_to_100GeV.append(float(s['flux1000']))

time_to_detection = np.array(time_to_detection)
flux_1GeV_to_100GeV = np.array(flux_1GeV_to_100GeV)

rows = 1080
cols = 1920
dpi = 300

fig = plt.figure(figsize=(cols/dpi, rows/dpi), dpi=dpi)
ax = fig.add_axes([.13, .15, .84, .81])
ax.plot(
	time_to_detection,
	flux_1GeV_to_100GeV,
	'ok',
	alpha=0.15)
ax.axvline(3600*50, color='k', alpha=0.5)
ax.loglog()
ax.set_xlabel('time-to-detection / s')
ax.set_ylabel('Fermi-LAT 3FGL Flux1000 / cm$^{-2}$ s$^{-1}$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
fig.savefig(
    os.path.join(
        out_dir,
        'time_to_detection_vs_fermilat_3fgl_flux1000.png'))