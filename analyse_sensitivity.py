import pandas as pd
import numpy as np
import os
from os import path as op
import sklearn
from sklearn import neural_network
import json
import acp_instrument_response_function as irf


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


run_dir = "run2019-08-31_0051"

particles = ["gamma", "electron", "proton"]
stages = ["thrown", "features"]

event_ids = ["run", "event"]

events = {}

for stage in stages:
    events[stage] = {}
    for particle in particles:
        path = op.join(run_dir, 'irf', particle, stage+'.jsonl')
        events[stage][particle] = pd.read_json(path, lines=True)

gammas = events["thrown"]["gamma"].merge(
    events["features"]["gamma"],
    on=event_ids)

electrons = events["thrown"]["electron"].merge(
    events["features"]["electron"],
    on=event_ids)

protons = events["thrown"]["proton"].merge(
    events["features"]["proton"],
    on=event_ids)


num_gammas = gammas.shape[0]
num_electrons = electrons.shape[0]
num_protons = protons.shape[0]

num_bins = int(np.sqrt(gammas.shape[0]))//2

figure_dir = op.join(run_dir, 'results', 'features')
os.makedirs(figure_dir, exist_ok=True)

# =============================================================================
# re-normalizing
gamma_spectral_index = -2.5

energy_bin_edges = np.linspace(0, 1e3, 137)
energy_upper_bin_edge = energy_bin_edges[1:]
gamma_thrown_bins = np.histogram(
    events['thrown']['gamma'].true_particle_energy,
    bins=energy_bin_edges)[0]

gamma_weights_index0 = 1/gamma_thrown_bins
gamma_weights_index0 = gamma_weights_index0/np.sum(gamma_weights_index0)
gamma_weights = energy_upper_bin_edge**gamma_spectral_index
gamma_weights *= gamma_weights_index0
gamma_weights /= np.sum(gamma_weights)

gammas['spectral_weight'] = np.interp(
    x=gammas.true_particle_energy,
    xp=energy_upper_bin_edge,
    fp=gamma_weights)

# =============================================================================
# re-normalizing
proton_spectral_index = -2.7

# energy_bin_edges = np.linspace(0, 1e3, 137)
# energy_upper_bin_edge = energy_bin_edges[1:]
proton_thrown_bins = np.histogram(
    events['thrown']['proton'].true_particle_energy,
    bins=energy_bin_edges)[0]

proton_weights_index0 = 1/proton_thrown_bins
proton_weights_index0 = proton_weights_index0/np.sum(proton_weights_index0)
proton_weights = energy_upper_bin_edge**proton_spectral_index
proton_weights *= proton_weights_index0
proton_weights /= np.sum(proton_weights)

protons['spectral_weight'] = np.interp(
    x=protons.true_particle_energy,
    xp=energy_upper_bin_edge,
    fp=proton_weights)

# =============================================================================


def add_hist(ax, bin_edges, bincounts, linestyle, color, alpha):
    assert bin_edges.shape[0] == bincounts.shape[0] + 1
    for i, bincount in enumerate(bincounts):
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [bincount, bincount],
            linestyle)
        ax.fill_between(
            x=[bin_edges[i], bin_edges[i + 1]],
            y1=[bincount, bincount],
            color=color,
            alpha=alpha,
            edgecolor='none')


gamma_mask = gammas.true_particle_energy <= 10

cfgs = [
    {
        "key": "num_photons",
        "unit": "1",
        "x_start": 10,
        "x_end": 1e4,
        "x_scale": "geomspace",
        "num_bins": num_bins,
    },
    {
        "key": "image_smallest_ellipse_solid_angle",
        "unit": "sr",
        "x_start": 1e-7,
        "x_end": 1e-3,
        "x_scale": "geomspace",
        "num_bins": num_bins,
    },
    {
        "key": "image_smallest_ellipse_object_distance",
        "unit": "m",
        "x_start": 1e3,
        "x_end": 50e3,
        "x_scale": "geomspace",
        "num_bins": num_bins
    },
    {
        "key": "paxel_intensity_peakness_max_over_mean",
        "unit": "1",
        "x_start": 1,
        "x_end": 100,
        "x_scale": "geomspace",
        "num_bins": num_bins
    },
    {
        "key": "paxel_intensity_peakness_std_over_mean",
        "unit": "1",
        "x_start": .1,
        "x_end": 10,
        "x_scale": "geomspace",
        "num_bins": num_bins
    },
    {
        "key": "image_smallest_ellipse_half_depth",
        "unit": "m",
        "x_start": 2e3,
        "x_end": 200e3,
        "x_scale": "geomspace",
        "num_bins": num_bins
    },
    {
        "key": "image_num_islands",
        "unit": "1",
        "x_start": 1,
        "x_end": 20,
        "x_scale": "linspace",
        "num_bins": 20
    },

]

for cfg in cfgs:
    if cfg["x_scale"] == "geomspace":
        f_bin_edges = np.geomspace(
            cfg["x_start"], cfg["x_end"], cfg["num_bins"])
    elif cfg["x_scale"] == "linspace":
        f_bin_edges = np.linspace(
            cfg["x_start"], cfg["x_end"], cfg["num_bins"])

    f_gamma = np.histogram(
        gammas[cfg["key"]][gamma_mask],
        bins=f_bin_edges,
        weights=gammas['spectral_weight'][gamma_mask])[0]
    f_proton = np.histogram(
        protons[cfg["key"]],
        bins=f_bin_edges,
        weights=protons['spectral_weight'])[0]
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([.1, .15, .85, .8])
    add_hist(
        ax=ax,
        bin_edges=f_bin_edges,
        bincounts=f_gamma/np.sum(f_gamma),
        linestyle='k-',
        color='blue',
        alpha=0.5)
    add_hist(
        ax=ax,
        bin_edges=f_bin_edges,
        bincounts=f_proton/np.sum(f_proton),
        linestyle='k-',
        color='red',
        alpha=0.5)
    ax.text(
        0.05, 0.95,
        'gamma@{:.1f}'.format(gamma_spectral_index),
        color='blue',
        transform=ax.transAxes)
    ax.text(
        0.05, 0.9,
        'proton@{:.1f}'.format(proton_spectral_index),
        color='red',
        transform=ax.transAxes)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('{:s}/{:s}'.format(cfg["key"], cfg["unit"]))
    ax.set_ylabel(r'number events/Sum=1')
    plt.savefig(op.join(
        figure_dir,
        '{:s}.png'.format(cfg["key"])))
    plt.close('all')

# learn
# -------------------
event_id_keys = ["true_particle_id", "run", "event"]


def norm_num_photons(events):
    return np.log10(events.num_photons)


def norm_image_smallest_ellipse_object_distance(events):
    obj = events.image_smallest_ellipse_object_distance
    return np.log10(obj) - 3


def norm_image_smallest_ellipse_solid_angle(events):
    sa = events.image_smallest_ellipse_solid_angle
    return np.log10(sa) + 7


def norm_paxel_intensity_offset(events):
    slope = np.hypot(
        events.paxel_intensity_median_x,
        events.paxel_intensity_median_y)
    return np.sqrt(slope)/np.sqrt(71./2.)

# gammas
# -------


gamma_E_cut = gammas.true_particle_energy <= 12.
gamma_leakage_cut = \
    gammas.image_smallest_ellipse_num_photons_on_edge_field_of_view <= \
    0.5*gammas.num_photons

gamma_cut = (gamma_E_cut) & (gamma_leakage_cut)


X_gamma = np.array([
    norm_num_photons(gammas[gamma_cut]),
    norm_image_smallest_ellipse_object_distance(gammas[gamma_cut]),
    norm_image_smallest_ellipse_solid_angle(gammas[gamma_cut]),
    norm_paxel_intensity_offset(gammas[gamma_cut])
]).T
y_gamma = np.vstack([
    1*np.ones(gammas.shape[0])[gamma_cut],
    gammas.true_particle_energy[gamma_cut]
]).T
id_gamma = gammas[gamma_cut][event_id_keys].values


# protons
# -------
proton_leakage_cut = \
    protons.image_smallest_ellipse_num_photons_on_edge_field_of_view <= \
    0.5*protons.num_photons
proton_cut = proton_leakage_cut

X_proton = np.array([
    norm_num_photons(protons[proton_cut]),
    norm_image_smallest_ellipse_object_distance(protons[proton_cut]),
    norm_image_smallest_ellipse_solid_angle(protons[proton_cut]),
    norm_paxel_intensity_offset(protons[proton_cut])
]).T
y_proton = np.vstack([
    0*np.ones(protons.shape[0])[proton_cut],
    protons.true_particle_energy[proton_cut]
]).T
id_proton = protons[proton_cut][event_id_keys].values


(
    x_train, x_test,
    y_train, y_test,
    id_train, id_test
) = sklearn.model_selection.train_test_split(
    np.concatenate([X_gamma, X_proton]),
    np.concatenate([y_gamma, y_proton]),
    np.concatenate([id_gamma, id_proton]),
    test_size=0.25,
    random_state=27)

id_test = pd.DataFrame({
    "true_particle_id": id_test[:, 0],
    "run": id_test[:, 1],
    "event": id_test[:, 2],})

id_train = pd.DataFrame({
    "true_particle_id": id_train[:, 0],
    "run": id_train[:, 1],
    "event": id_train[:, 2],})

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf = sklearn.neural_network.MLPRegressor(
    solver='lbfgs',
    alpha=1e-3,
    hidden_layer_sizes=(5, 5, 5),
    random_state=1,
    verbose=True,
    max_iter=1000)

clf.fit(x_train, y_train[:, 0])

fpr, tpr, thresholds = sklearn.metrics.roc_curve(
    y_true=y_test[:, 0],
    y_score=clf.predict(x_test))

auc = sklearn.metrics.roc_auc_score(
    y_true=y_test[:, 0],
    y_score=clf.predict(x_test))

fig = plt.figure(figsize=(4, 4), dpi=100)
ax = fig.add_axes([.2, .2, .72, .72])
ax.plot(fpr, tpr, 'k')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.set_title('area under curve {:.2f}'.format(auc))
ax.set_xlabel('false positive rate / 1\nproton acceptance')
ax.set_ylabel('true positive rate / 1\ngamma-ray acceptance')
# ax.semilogx()
plt.savefig(op.join(
    figure_dir,
    '{:s}.png'.format('roc')))
plt.close('all')

# intstrument-response after all cuts:


def make_detection_mask(
    events_thrown,
    events_detected,
    event_id_keys
):
    events_detected_ids = pd.DataFrame(events_detected[event_id_keys])
    events_detected_ids["detected"] = True
    detection_mask = events_thrown.merge(
        events_detected_ids,
        on=event_id_keys,
        how='left')["detected"] == True
    return detection_mask.astype(np.int).values


def estimate_effective_area(
    energy_bin_edges,
    energies,
    max_scatter_areas,
    detection_mask
):
    num_thrown = np.histogram(
        energies,
        bins=energy_bin_edges)[0]

    num_detected = np.histogram(
        energies,
        weights=detection_mask,
        bins=energy_bin_edges)[0]

    area_thrown = np.histogram(
        energies,
        weights=max_scatter_areas,
        bins=energy_bin_edges)[0]

    area_detected = np.histogram(
        energies,
        weights=max_scatter_areas*detection_mask,
        bins=energy_bin_edges)[0]

    num_bins = energy_bin_edges.shape[0] - 1
    effective_area = np.nan*np.ones(num_bins)
    for i in range(num_bins):
        if num_thrown[i] > 0 and area_thrown[i] > 0.:
            effective_area[i] = \
                area_detected[i]/area_thrown[i]*(area_thrown[i]/num_thrown[i])

    return {
        "energy_bin_edges": energy_bin_edges,
        "num_thrown": num_thrown,
        "num_detected": num_detected,
        "area_thrown": area_thrown,
        "area_detected": area_detected,
        "effective_area": effective_area,
    }


def save_effective_area_figure(
    response,
    path,
    y_start=1e1,
    y_stop=1e6,
):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([.1, .15, .85, .8])
    add_hist(
        ax=ax,
        bin_edges=energy_bin_edges,
        bincounts=response["effective_area"],
        linestyle='k-',
        color='blue',
        alpha=0.0)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('energy / GeV')
    ax.set_ylabel(r'effective area / m$^2$')
    ax.set_ylim([y_start, y_stop])
    ax.set_xlim([
        np.min(response["energy_bin_edges"]),
        np.max(response["energy_bin_edges"]),
    ])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.savefig(path)
    plt.close('all')


number_energy_bins = 32
energy_start = 2.5e-1
energy_stop = 1e3
energy_bin_edges = np.geomspace(energy_start, energy_stop, number_energy_bins)

point_like_scatter_angle = np.deg2rad(1.5)

effective_areas = {}
source_shapes = ["diffuse", "point"]

for source_shape in source_shapes:
    effective_areas[source_shape] = {}
    for particle in particles:
        if source_shape == "diffuse":
            events_thrown = events['thrown'][particle]
            events_detected = events['features'][particle]
        elif source_shape == "point":
            from_point_source_mask = \
                events['thrown'][particle].true_particle_zenith < \
                point_like_scatter_angle
            events_thrown = events['thrown'][particle][from_point_source_mask]
            events_detected = events['features'][particle]

        max_scatter_radii = \
            events_thrown.true_particle_max_core_scatter_radius.values

        detection_mask = make_detection_mask(
            events_thrown=events_thrown,
            events_detected=events_detected,
            event_id_keys=event_ids)

        effective_areas[source_shape][particle] = estimate_effective_area(
            energy_bin_edges=energy_bin_edges,
            energies=events_thrown.true_particle_energy,
            max_scatter_areas=np.pi*max_scatter_radii**2.,
            detection_mask=detection_mask)

        path = '{:s}_effective_area_{:s}'.format(particle, source_shape)

        with open(op.join(figure_dir, path+'.json'), 'wt') as f:
            f.write(
                json.dumps(
                    effective_areas[source_shape][particle],
                    cls=NumpyEncoder,
                    indent=4))

        save_effective_area_figure(
            response=effective_areas[source_shape][particle],
            path=op.join(figure_dir, path+'.png'))


# after all cuts
# --------------

gammaness_threshold = 0.5
gammaness = clf.predict(x_test)
reconstructed_gamma_mask = gammaness >= gammaness_threshold
test_reconstructed_ids = id_test[reconstructed_gamma_mask]
test_thrown_ids = id_test.copy()


all_events_thrown = events['thrown']['gamma']
all_events_thrown = all_events_thrown.append(events['thrown']['proton'])
all_events_thrown = all_events_thrown.append(events['thrown']['electron'])


test_events_thrown = all_events_thrown.merge(test_thrown_ids, on=event_id_keys)
test_events_detected = all_events_thrown.merge(test_reconstructed_ids, on=event_id_keys)

source_shape = "diffuse"
for particle in ['gamma']:
    if particle == "gamma":
        true_particle_id = 1
    elif particle == "proton":
        true_particle_id = 14
    events_thrown = test_events_thrown[
        test_events_thrown['true_particle_id'] == true_particle_id]
    events_detected = test_events_detected[
        test_events_detected['true_particle_id'] == true_particle_id]

    max_scatter_radii = \
        events_thrown.true_particle_max_core_scatter_radius.values

    detection_mask = make_detection_mask(
        events_thrown=events_thrown,
        events_detected=events_detected,
        event_id_keys=event_ids)

    effective_areas[source_shape][particle] = estimate_effective_area(
        energy_bin_edges=energy_bin_edges,
        energies=events_thrown.true_particle_energy,
        max_scatter_areas=np.pi*max_scatter_radii**2.,
        detection_mask=detection_mask)

    path = 'final_{:s}_effective_area_{:s}'.format(particle, source_shape)

    with open(op.join(figure_dir, path+'.json'), 'wt') as f:
        f.write(
            json.dumps(
                effective_areas[source_shape][particle],
                cls=NumpyEncoder,
                indent=4))

    save_effective_area_figure(
        response=effective_areas[source_shape][particle],
        path=op.join(figure_dir, path+'.png'))