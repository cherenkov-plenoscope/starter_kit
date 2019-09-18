import pandas as pd
import numpy as np
import os
from os import path as op
import sklearn
from sklearn import neural_network
import json
import acp_instrument_response_function as irf
import acp_instrument_sensitivity_function as isf
import pkg_resources
import pickle
import astropy

"""
thrown-table                        All events ever thrown
------------
id
true_particle_type
true_particle_energy
true_particle_first_interaction_height
true_particle_core_x
true_particle_core_y
true_particle_azimuth
true_particle_zenith

true_particle_momentum_x
true_particle_momentum_y
true_particle_momentum_z

true_particle_max_core_scatter_radius
true_particle_max_scatter_angle

observation_level_altitude_asl


trigger-threshold-table             For all thrown events
-----------------------
id
trigger_threshold


past_trigger                        Passed trigger (at specific threshold)
------------
id


feature-table                       Events where features could be extracted
-------------
id
hillas_1
hillas_2
...



"""


figure_configs = [
    {
        'figsize': (8, 6),
        'figsize2': (8, 4),
        'dpi': 240,
        'ax_size': (0.08, 0.08, 0.9, 0.9),
        'ax_size2': (0.08, 0.12, 0.9, 0.85),
        'path_ext': '',
    },
    {
        'figsize': (1920/300, 1080/300),
        'figsize2': (1920/300, 1080/300),
        'dpi': 300,
        'ax_size': (0.1, 0.15, 0.85, 0.80),
        'ax_size2': (0.1, 0.15, 0.85, 0.80),
        'path_ext': '_beamer',
    },
]

fc = figure_configs[1]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def cone_solid_angle(cone_radial_opening_angle):
    cap_hight = (1.0 - np.cos(cone_radial_opening_angle))
    return 2.0*np.pi*cap_hight


def convert_ams02_cosmic_ray_fluxes(out_dir):
    path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        os.path.join('resources', 'proton_spec.dat'))
    proton_flux = np.genfromtxt(path)
    proton_flux[:, 0] *= 1  # in GeV
    proton_flux[:, 1] /= proton_flux[:, 0]**2.7
    opath = os.path.join(out_dir, "cosmic_proton_flux.json")
    with open(opath, "wt") as fout:
        fout.write(
            json.dumps(
                {
                    "energy": {
                        "values": proton_flux[:, 0].tolist(),
                        "unit": "GeV"
                    },
                    "differential_flux": {
                        "values": proton_flux[:, 1].tolist(),
                        "unit": "m^{-2} s^{-1} sr^{-1} GeV^{-1}"
                    },
                },
                indent=4
            )
        )

    path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        os.path.join('resources', 'e_plus_e_minus_spec.dat'))
    electron_flux = np.genfromtxt(path)
    electron_flux[:, 0] *= 1  # in GeV
    electron_flux[:, 1] /= electron_flux[:, 0]**3.0
    opath = os.path.join(out_dir, "cosmic_e_plus_e_minus_flux.json")
    with open(opath, "wt") as fout:
        fout.write(
            json.dumps(
                {
                    "energy": {
                        "values": electron_flux[:, 0].tolist(),
                        "unit": "GeV"
                    },
                    "differential_flux": {
                        "values": electron_flux[:, 1].tolist(),
                        "unit": "m^{-2} s^{-1} sr^{-1} GeV^{-1}"
                    },
                },
                indent=4
            )
        )



def convert_20190831(
    path_run_20190831,
    out_dir,
    cone_radial_opening_angle_deg=6.5,
    trigger_threshold=103,
):
    os.makedirs(out_dir, exist_ok=True)
    particles = {"gamma": 1, "electron": 3, "proton": 14}

    cone_radial_opening_angle = np.deg2rad(cone_radial_opening_angle_deg)
    solid_angle_thrown_sr = cone_solid_angle(cone_radial_opening_angle)

    print("features")
    #---------
    path_features = os.path.join(out_dir, "events_features.jsonl")
    with open(path_features, "wt") as fout:
        for particle in particles:
            path_features_20190831_particle = op.join(
                path_run_20190831,
                'irf',
                particle,
                'features.jsonl')
            with open(path_features_20190831_particle, "rt") as fin:
                for line in fin:
                    ein = json.loads(line)
                    ein['particle'] = particles[particle]
                    fout.write(json.dumps(ein) + '\n')

    print("thrown")
    #-------
    path_thrown = os.path.join(out_dir, "events_thrown.jsonl")
    with open(path_thrown, "wt") as fout:
        for particle in particles:
            path_thrown_20190831_particle = op.join(
                path_run_20190831,
                'irf',
                particle,
                'thrown.jsonl')
            with open(path_thrown_20190831_particle, "rt") as fin:
                for line in fin:
                    ein = json.loads(line)
                    eout = {
                        'particle': ein['true_particle_id'],
                        'run': ein['run'],
                        'event': ein['event'],
                        'energy': ein['true_particle_energy'],
                        'core_x': ein['true_particle_core_x'],
                        'core_y': ein['true_particle_core_y'],
                        'azimuth_phi': ein['true_particle_azimuth'],
                        'zenith_theta': ein['true_particle_zenith'],
                        'first_interaction_height': \
                            ein['true_particle_first_interaction_height'],
                        'area_thrown': \
                            np.pi*ein[
                                'true_particle_max_core_scatter_radius']**2.,
                        'solid_angle_thrown': solid_angle_thrown_sr,
                        }
                    fout.write(json.dumps(eout) + '\n')

    print("trigger_threshold")
    #------------------
    path_trigger_threshold = os.path.join(
        out_dir,
        "events_trigger_threshold.jsonl")
    with open(path_trigger_threshold, "wt") as fout:
        for particle in particles:
            path_thrown_20190831_particle = op.join(
                path_run_20190831,
                'irf',
                particle,
                'thrown.jsonl')
            with open(path_thrown_20190831_particle, "rt") as fin:
                for line in fin:
                    ein = json.loads(line)
                    eout = {
                        'particle': ein['true_particle_id'],
                        'run': ein['run'],
                        'event': ein['event'],
                        'trigger_patch_threshold_0': \
                            ein['trigger_patch_threshold_0'],
                        'trigger_patch_threshold_1': \
                            ein['trigger_patch_threshold_1'],
                        'trigger_patch_threshold_2': \
                            ein['trigger_patch_threshold_2'],
                        }
                    fout.write(json.dumps(eout) + '\n')


    print("past_trigger")
    #--------------
    path_past_trigger = os.path.join(out_dir, "events_past_trigger.jsonl")
    with open(path_past_trigger, "wt") as fout:
        for particle in particles:
            path_thrown_20190831_particle = op.join(
                path_run_20190831,
                'irf',
                particle,
                'thrown.jsonl')
            with open(path_thrown_20190831_particle, "rt") as fin:
                for line in fin:
                    ein = json.loads(line)
                    trigger_response = np.max([
                        ein['trigger_patch_threshold_0'],
                        ein['trigger_patch_threshold_1'],
                        ein['trigger_patch_threshold_2'],])
                    if trigger_response >= trigger_threshold:
                        eout = {
                            'particle': ein['true_particle_id'],
                            'run': ein['run'],
                            'event': ein['event']}
                        fout.write(json.dumps(eout) + '\n')


def estimate_instrument_response(
    energy_bin_edges,
    events_energie,
    events_scatter_area,
    events_scatter_solid_angle,
    events_detection_mask
):
    num_thrown = np.histogram(
        events_energie,
        bins=energy_bin_edges)[0]

    num_detected = np.histogram(
        events_energie,
        weights=events_detection_mask,
        bins=energy_bin_edges)[0]

    area_thrown = np.histogram(
        events_energie,
        weights=events_scatter_area,
        bins=energy_bin_edges)[0]

    area_detected = np.histogram(
        events_energie,
        weights=events_scatter_area*events_detection_mask,
        bins=energy_bin_edges)[0]

    solid_angle_thrown = np.histogram(
        events_energie,
        weights=events_scatter_solid_angle,
        bins=energy_bin_edges)[0]

    solid_angle_detected = np.histogram(
        events_energie,
        weights=events_scatter_solid_angle*events_detection_mask,
        bins=energy_bin_edges)[0]

    num_bins = energy_bin_edges.shape[0] - 1
    effective_area = np.zeros(num_bins)
    effective_solid_angle = np.zeros(num_bins)
    for i in range(num_bins):
        if num_thrown[i] > 0 and area_thrown[i] > 0.:
            effective_area[i] = \
                area_detected[i]/area_thrown[i]*\
                    (area_thrown[i]/num_thrown[i])
            print(effective_area[i])
        if num_thrown[i] > 0 and solid_angle_thrown[i] > 0.:
            effective_solid_angle[i] = \
                solid_angle_detected[i]/solid_angle_thrown[i]*\
                    (solid_angle_thrown[i]/num_thrown[i])

    return {
        "energy_bin_edges": energy_bin_edges,
        "num_thrown": num_thrown,
        "num_detected": num_detected,
        "area_thrown": area_thrown,
        "area_detected": area_detected,
        "solid_angle_thrown": solid_angle_thrown,
        "solid_angle_detected": solid_angle_detected,
        "effective_area": effective_area,
        "effective_solid_angle": effective_solid_angle
    }


def add_hist(ax, bin_edges, bincounts, linestyle, color, alpha, alpha_line):
    assert bin_edges.shape[0] == bincounts.shape[0] + 1
    for i, bincount in enumerate(bincounts):
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [bincount, bincount],
            linestyle,
            alpha=alpha_line)
        ax.fill_between(
            x=[bin_edges[i], bin_edges[i + 1]],
            y1=[bincount, bincount],
            color=color,
            alpha=alpha,
            edgecolor='none')


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


work_dir = 'wow'
out_dir = os.path.join('analysis_results')
os.makedirs(out_dir, exist_ok=True)

# read tables
# -----------

event_id_keys = ["particle", "run", "event"]
particles = {"gamma": 1, "electron": 3, "proton": 14}

events_thrown = pd.read_json(
    os.path.join(work_dir, "events_thrown.jsonl"),
    lines=True)

events_past_trigger = pd.read_json(
    os.path.join(work_dir, "events_past_trigger.jsonl"),
    lines=True)

events_features = pd.read_json(
    os.path.join(work_dir, "events_features.jsonl"),
    lines=True)

events_trigger_threshold = pd.read_json(
    os.path.join(work_dir, "events_trigger_threshold.jsonl"),
    lines=True)

trigger_mask = pd.merge(
    events_thrown,
    pd.DataFrame({
        "particle": events_past_trigger.particle,
        "run": events_past_trigger.run,
        "event": events_past_trigger.event,
        "past_trigger": np.ones(
            events_past_trigger.event.shape[0],
            dtype=np.int64),
        }),
    on=event_id_keys,
    how="left")
trigger_mask = (trigger_mask["past_trigger"] == 1).values

# All rec arrays
# --------------
events_thrown = events_thrown.to_records(index=False)
events_past_trigger = events_past_trigger.to_records(index=False)
events_features = events_features.to_records(index=False)
events_trigger_threshold = events_trigger_threshold.to_records(index=False)

print("SCATTER AREA")
print("============")

scatter_radii = np.hypot(
    events_thrown.core_x,
    events_thrown.core_y)

for particle in particles:
    print(particle)
    particle_mask = events_thrown.particle == particles[particle]
    particle_and_trigger_mask = np.logical_and(
        particle_mask,
        trigger_mask)

    num_triggers = np.sum(particle_and_trigger_mask)
    num_bins = np.int(np.sqrt(num_triggers)/2)

    bin_edges = np.linspace(
        0,
        np.max(scatter_radii[particle_mask])**2,
        num_bins)

    num_triggered = np.histogram(
        scatter_radii[particle_and_trigger_mask]**2,
        bin_edges)[0]

    num_thrown = np.histogram(
        scatter_radii[particle_mask]**2,
        bin_edges)[0]

    ratio = num_triggered/num_thrown
    ratio_delta = np.sqrt(num_triggered)/num_thrown
    ratio_low = ratio - ratio_delta
    ratio_high = ratio + ratio_delta
    ratio_low[ratio_low < 0] = 0

    outd = {
        "bin_edges_radius_square": bin_edges.tolist(),
        "num_thrown": num_thrown.tolist(),
        "num_triggered": num_triggered.tolist(),
        "ratio": ratio.tolist(),
        "ratio_delta": ratio_delta.tolist(),
        "ratio_low": ratio_low.tolist(),
        "ratio_high": ratio_high.tolist(),
    }
    path = os.path.join(
        out_dir,
        "scatter_radius_thrown_and_triggered_{:s}".format(particle))
    with open(path+".json", 'wt') as fout:
        fout.write(json.dumps(outd, indent=4))


    fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
    ax = fig.add_axes(fc['ax_size2'])
    for i in range(num_triggered.shape[0]):
        ax.plot(
            1e-6*np.array([bin_edges[i], bin_edges[i + 1]]),
            [num_thrown[i], num_thrown[i]],
            'k',
            label='thrown' if i == 0 else None)
        ax.plot(
            1e-6*np.array([bin_edges[i], bin_edges[i + 1]]),
            [num_triggered[i], num_triggered[i]],
            'k:',
            label='triggered' if i == 0 else None)
    ax.semilogy()
    ax.set_xlim([0, 1.75])
    ax.set_ylim([.9, 3e4])
    ax.set_xlabel(r'(Scatter-radius)$^2$ / (km)$^2$')
    ax.set_ylabel('{:s}s / 1'.format(particle))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.legend(loc='best', fontsize=10)
    fig.savefig(os.path.join(path+".png"))


    fig = plt.figure(figsize=fc['figsize'], dpi=fc['dpi'])
    ax = fig.add_axes(fc['ax_size'])
    for i in range(num_triggered.shape[0]):
        x = 1e-6*np.array([bin_edges[i], bin_edges[i + 1]])
        ax.plot(
            x,
            [1e2*ratio[i], 1e2*ratio[i]],
            'k')
        ax.fill_between(
            x=x,
            y1=1e2*np.array([ratio_low[i], ratio_low[i]]),
            y2=1e2*np.array([ratio_high[i], ratio_high[i]]),
            color='k',
            alpha=0.2,
            linewidth=0)
    ax.set_xlim([0, 1.75])
    ax.set_xlabel(r'(Scatter-radius)$^2$ / (km)$^2$')
    ax.set_ylabel('{:s}s triggered/thrown / %'.format(particle))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    fig.savefig(os.path.join(path+"_ratio.png"))

print("SCATTER ANGLE")
print("=============")

scatter_angle_deg = np.rad2deg(events_thrown.zenith_theta)

for particle in particles:
    print(particle)
    particle_mask = events_thrown.particle == particles[particle]
    particle_and_trigger_mask = np.logical_and(
        particle_mask,
        trigger_mask)

    bin_edges = np.linspace(
        0,
        np.max(scatter_angle_deg[particle_mask])**2,
        num_bins)

    num_triggered = np.histogram(
        scatter_angle_deg[particle_and_trigger_mask]**2,
        bin_edges)[0]

    num_thrown = np.histogram(
        scatter_angle_deg[particle_mask]**2,
        bin_edges)[0]

    ratio = num_triggered/num_thrown
    ratio_delta = np.sqrt(num_triggered)/num_thrown
    ratio_low = ratio - ratio_delta
    ratio_high = ratio + ratio_delta
    ratio_low[ratio_low < 0] = 0

    outd = {
        "bin_edges_angle_square_deg_square": bin_edges.tolist(),
        "num_thrown": num_thrown.tolist(),
        "num_triggered": num_triggered.tolist(),
        "ratio": ratio.tolist(),
        "ratio_delta": ratio_delta.tolist(),
        "ratio_low": ratio_low.tolist(),
        "ratio_high": ratio_high.tolist(),
    }
    path = os.path.join(
        out_dir,
        "scatter_angle_thrown_and_triggered_{:s}".format(particle))
    with open(path+".json", 'wt') as fout:
        fout.write(json.dumps(outd, indent=4))

    fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
    ax = fig.add_axes(fc['ax_size2'])
    for i in range(num_triggered.shape[0]):
        ax.plot(
            np.array([bin_edges[i], bin_edges[i + 1]]),
            [num_thrown[i], num_thrown[i]],
            'k',
            label='thrown' if i == 0 else None)
        ax.plot(
            np.array([bin_edges[i], bin_edges[i + 1]]),
            [num_triggered[i], num_triggered[i]],
            'k:',
            label='triggered' if i == 0 else None)
    ax.semilogy()
    ax.set_xlim([0, 45])
    ax.set_ylim([5, 3e4])
    ax.set_xlabel(r'(Scatter-angle)$^2$ / (deg)$^2$')
    ax.set_ylabel('{:s}s / 1'.format(particle))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.axvline(x=3.25**2, color='k', linestyle='-', alpha=0.25)
    ax.legend(loc='best', fontsize=10)
    fig.savefig(path+'.png')

    fig = plt.figure(figsize=fc['figsize'], dpi=fc['dpi'])
    ax = fig.add_axes(fc['ax_size'])
    for i in range(num_triggered.shape[0]):
        x = np.array([bin_edges[i], bin_edges[i + 1]])
        ax.plot(
            x,
            [1e2*ratio[i], 1e2*ratio[i]],
            'k')
        ax.fill_between(
            x=x,
            y1=1e2*np.array([ratio_low[i], ratio_low[i]]),
            y2=1e2*np.array([ratio_high[i], ratio_high[i]]),
            color='k',
            alpha=0.2,
            linewidth=0)
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'(Scatter-angle)$^2$ / (deg)$^2$')
    ax.set_ylabel('{:s}s triggered/thrown / %'.format(particle))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.axvline(x=3.25**2, color='k', linestyle='-', alpha=0.25)
    #ax.legend(loc='best', fontsize=10)
    fig.savefig(path+'_ratio.png')


print("GAMMA-HADRON-SEPARATION")
print("=======================")
print("cut on leakage.")

is_not_leaking_features = events_features.\
    image_smallest_ellipse_num_photons_on_edge_field_of_view <= \
    0.5*events_features.num_photons
is_gammma_or_proton_features = np.logical_or(
    events_features.particle == particles['gamma'],
    events_features.particle == particles['proton'])
cut_mask = np.logical_and(is_not_leaking_features, is_gammma_or_proton_features)
events_features_gh = events_features[cut_mask]

is_gammma_or_proton_thrown = np.logical_or(
    events_thrown.particle == particles['gamma'],
    events_thrown.particle == particles['proton'])
events_thrown_gh = events_thrown[is_gammma_or_proton_thrown]

test_size = 0.25
print("split test and training. test_size = {:.2f}".format(test_size))

(
    events_thrown_gh_train,
    events_thrown_gh_test
) = sklearn.model_selection.train_test_split(
    events_thrown_gh,
    test_size=test_size,
    random_state=27)

events_features_gh_train = pd.merge(
    right=pd.DataFrame(events_thrown_gh_train),
    left=pd.DataFrame(events_features_gh),
    on=event_id_keys).to_records(index=False)

events_features_gh_test = pd.merge(
    right=pd.DataFrame(events_thrown_gh_test),
    left=pd.DataFrame(events_features_gh),
    on=event_id_keys).to_records(index=False)

print("Num. events features extracted:")
print("test: {:d}".format(events_features_gh_test.shape[0]))
print("train: {:d}".format(events_features_gh_train.shape[0]))


# prepare learning
# ----------------

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


# learn gamma / hadron seperation
# -------------------------------

gamma_E_max = 10.
print("Learn gammas below E = {:.2f}GeV".format(gamma_E_max))

gamma_mask_train = np.logical_and(
    events_features_gh_train.particle == particles['gamma'],
    events_features_gh_train.energy <= gamma_E_max)
x_train_unscaled = np.array([
    norm_num_photons(events_features_gh_train),
    norm_image_smallest_ellipse_object_distance(events_features_gh_train),
    norm_image_smallest_ellipse_solid_angle(events_features_gh_train),
    norm_paxel_intensity_offset(events_features_gh_train)
]).T
y_train = np.array([
    gamma_mask_train,
    events_features_gh_train.energy
]).T

gamma_mask_test = np.logical_and(
    events_features_gh_test.particle == particles['gamma'],
    events_features_gh_test.energy <= gamma_E_max)
x_test_unscaled = np.array([
    norm_num_photons(events_features_gh_test),
    norm_image_smallest_ellipse_object_distance(events_features_gh_test),
    norm_image_smallest_ellipse_solid_angle(events_features_gh_test),
    norm_paxel_intensity_offset(events_features_gh_test)
]).T
y_test = np.array([
    gamma_mask_test,
    events_features_gh_test.energy
]).T

print("scale feature space")

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train_unscaled)
x_train = scaler.transform(x_train_unscaled)
x_test = scaler.transform(x_test_unscaled)

print("train model")

clf = sklearn.neural_network.MLPRegressor(
    solver='lbfgs',
    alpha=1e-3,
    hidden_layer_sizes=(5, 5, 5),
    random_state=1,
    verbose=True,
    max_iter=1000)
clf.fit(x_train, y_train[:, 0])

model_gh_path = os.path.join(out_dir, "gamma_hadron_model")
with open(model_gh_path+".pkl", "wb") as fout:
    fout.write(pickle.dumps(clf))

print("receiver operating characteristic")

fpr_gh, tpr_gh, thresholds_gh = sklearn.metrics.roc_curve(
    y_true=y_test[:, 0],
    y_score=clf.predict(x_test))

auc_gh = sklearn.metrics.roc_auc_score(
    y_true=y_test[:, 0],
    y_score=clf.predict(x_test))

roc_gh = {
    "false_positive_rate": fpr_gh.tolist(),
    "true_positive_rate": tpr_gh.tolist(),
    "gamma_hadron_threshold": thresholds_gh.tolist(),
    "area_under_curve": float(auc_gh),
    "num_events_for_training": int(x_train.shape[0]),
    "max_gamma_ray_energy": float(gamma_E_max),
}
roc_gh_path = os.path.join(
    out_dir,
    "receiver_operating_characteristic_gamma_hadron_separation")
with open(roc_gh_path+".json", "wt") as fout:
    fout.write(json.dumps(roc_gh, indent=4))

fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
ax = fig.add_axes([.2, .2, .72, .72])
ax.plot(fpr_gh, tpr_gh, 'k')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.set_title('area under curve {:.2f}'.format(auc_gh))
ax.set_xlabel('false positive rate / 1\nproton acceptance')
ax.set_ylabel('true positive rate / 1\ngamma-ray acceptance')
ax.semilogx()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(roc_gh_path+".png")
plt.close('all')

print("instrument response")

def make_detection_mask(
    events_thrown,
    events_detected,
    event_id_keys
):
    df_events_thrown = pd.DataFrame(events_thrown)
    df_events_detected = pd.DataFrame(events_detected)
    df_events_detected_ids = pd.DataFrame(df_events_detected[event_id_keys])
    df_events_detected_ids["detected"] = True
    detection_mask = pd.merge(
        left=df_events_thrown,
        right=df_events_detected_ids,
        on=event_id_keys,
        how='left')["detected"] == True
    return detection_mask.astype(np.int).values

source_geometries = {"onaxis": 2.5, "diffuse": 90.}

number_energy_bins = 30
energy_start = 2.5e-1
energy_stop = 1e3
energy_bin_edges = np.geomspace(energy_start, energy_stop, number_energy_bins + 1)
energy_bin_start = energy_bin_edges[:-1]

gammaness_cut = 0.33

irf = {}
gh = {}

for sg in source_geometries:
    gh[sg] = {}
    irf[sg] = {}
    max_scatter_angle = np.deg2rad(source_geometries[sg])

    _ga_thr_msk = np.logical_and(
        events_thrown_gh_test.particle == particles["gamma"],
        events_thrown_gh_test.zenith_theta <= max_scatter_angle)
    _gammas_events_thrown_gh_test = events_thrown_gh_test[_ga_thr_msk]
    _gammas_events_features_gh_test = pd.merge(
        right=pd.DataFrame(_gammas_events_thrown_gh_test),
        left=pd.DataFrame(events_features_gh_test),
        on=event_id_keys).to_records(index=False)
    gh[sg]["gamma"] = {
        "events_thrown": _gammas_events_thrown_gh_test,
        "events_features": _gammas_events_features_gh_test,}

    _pr_thr_msk = np.logical_and(
        events_thrown_gh_test.particle == particles["proton"],
        events_thrown_gh_test.zenith_theta <= max_scatter_angle)
    _protons_events_thrown_gh_test = events_thrown_gh_test[_pr_thr_msk]
    _protons_events_features_gh_test = pd.merge(
        right=pd.DataFrame(_protons_events_thrown_gh_test),
        left=pd.DataFrame(events_features_gh_test),
        on=event_id_keys).to_records(index=False)
    gh[sg]["proton"] = {
        "events_thrown": _protons_events_thrown_gh_test,
        "events_features": _protons_events_features_gh_test,}

    _el_thr_msk = np.logical_and(
        events_thrown.particle == particles["electron"],
        events_thrown.zenith_theta <= max_scatter_angle)
    _electrons_events_thrown_test = events_thrown[_el_thr_msk]
    _electrons_events_features_gh_test = pd.merge(
        right=pd.DataFrame(_electrons_events_thrown_test),
        left=pd.DataFrame(events_features),
        on=event_id_keys).to_records(index=False)
    gh[sg]["electron"] = {
        "events_thrown": _electrons_events_thrown_test,
        "events_features": _electrons_events_features_gh_test,}

    for p in particles:
        x_unscaled = np.array([
            norm_num_photons(
                gh[sg][p]["events_features"]),
            norm_image_smallest_ellipse_object_distance(
                gh[sg][p]["events_features"]),
            norm_image_smallest_ellipse_solid_angle(
                gh[sg][p]["events_features"]),
            norm_paxel_intensity_offset(
                gh[sg][p]["events_features"])
        ]).T
        gh[sg][p]["x"] = scaler.transform(x_unscaled)
        gammaness = clf.predict(gh[sg][p]["x"])
        detection_feature_mask = gammaness >= gammaness_cut
        gh[sg][p]["detection_mask"] = make_detection_mask(
            events_thrown=gh[sg][p]["events_thrown"],
            events_detected=gh[sg][p]["events_features"][
                detection_feature_mask],
            event_id_keys=event_id_keys)

        irf[sg][p] = {}
        irf[sg][p]["past_cuts"] = estimate_instrument_response(
            energy_bin_edges=energy_bin_edges,
            events_energie=gh[sg][p]["events_thrown"].energy,
            events_scatter_area=gh[sg][p]["events_thrown"].area_thrown,
            events_scatter_solid_angle=gh[sg][p]["events_thrown"].solid_angle_thrown,
            events_detection_mask=gh[sg][p]["detection_mask"])

    print("past trigger")

    for p in particles:
        _mask_thrown = np.logical_and(
            events_thrown.particle == particles[p],
            events_thrown.zenith_theta <= max_scatter_angle)
        evts_thrown = events_thrown[_mask_thrown]
        evts_past_trigger = pd.merge(
            right=pd.DataFrame(evts_thrown[event_id_keys]),
            left=pd.DataFrame(events_past_trigger),
            on=event_id_keys).to_records(index=False)
        past_trigger_mask = make_detection_mask(
            events_thrown=evts_thrown,
            events_detected=evts_past_trigger,
            event_id_keys=event_id_keys)

        irf[sg][p]["past_trigger"] = estimate_instrument_response(
            energy_bin_edges=energy_bin_edges,
            events_energie=evts_thrown.energy,
            events_scatter_area=evts_thrown.area_thrown,
            events_scatter_solid_angle=evts_thrown.solid_angle_thrown,
            events_detection_mask=past_trigger_mask)

    print("past features")

    for p in particles:
        print(p, particles[p])
        _mask_thrown = np.logical_and(
            events_thrown.particle == particles[p],
            events_thrown.zenith_theta <= max_scatter_angle)
        evts_thrown = events_thrown[_mask_thrown]
        evts_past_features = pd.merge(
            right=pd.DataFrame(evts_thrown[event_id_keys]),
            left=pd.DataFrame(events_features),
            on=event_id_keys).to_records(index=False)
        past_feature_mask = make_detection_mask(
            events_thrown=evts_thrown,
            events_detected=evts_past_features,
            event_id_keys=event_id_keys)

        irf[sg][p]["past_features"] = estimate_instrument_response(
            energy_bin_edges=energy_bin_edges,
            events_energie=evts_thrown.energy,
            events_scatter_area=evts_thrown.area_thrown,
            events_scatter_solid_angle=evts_thrown.solid_angle_thrown,
            events_detection_mask=past_feature_mask)

    y_start = 1e0
    y_stop = 1e6

    for p in particles:
        irf_path = os.path.join(out_dir, "irf_{:s}_{:s}".format(sg, p))

        with open(irf_path+".json", "wt") as fout:
            fout.write(json.dumps(
                {
                    "energy_bin_edges": irf[sg][p]["past_trigger"][
                        "energy_bin_edges"].tolist(),
                    "area_past_trigger": irf[sg][p]["past_trigger"][
                        "effective_area"].tolist(),
                    "area_past_features": irf[sg][p]["past_features"][
                        "effective_area"].tolist(),
                    "area_past_cuts": irf[sg][p]["past_cuts"][
                        "effective_area"].tolist(),
                },
                indent=4))

        fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
        ax = fig.add_axes([.1, .15, .85, .8])

        add_hist(
            ax=ax,
            bin_edges=irf[sg][p]["past_trigger"]["energy_bin_edges"],
            bincounts=irf[sg][p]["past_trigger"]["effective_area"],
            linestyle='k:',
            color='blue',
            alpha=0.0,
            alpha_line=0.3)

        add_hist(
            ax=ax,
            bin_edges=irf[sg][p]["past_features"]["energy_bin_edges"],
            bincounts=irf[sg][p]["past_features"]["effective_area"],
            linestyle='k--',
            color='blue',
            alpha=0.0,
            alpha_line=0.3)

        add_hist(
            ax=ax,
            bin_edges=irf[sg][p]["past_cuts"]["energy_bin_edges"],
            bincounts=irf[sg][p]["past_cuts"]["effective_area"],
            linestyle='k-',
            color='blue',
            alpha=0.0,
            alpha_line=1.)

        ax.loglog()
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.set_xlabel('energy / GeV')
        ax.set_ylabel(r'effective area / m$^2$')
        ax.set_ylim([y_start, y_stop])
        ax.set_xlim([
            np.min(irf[sg][p]["past_trigger"]["energy_bin_edges"]),
            np.max(irf[sg][p]["past_trigger"]["energy_bin_edges"]),])
        ax.text(x=.75, y=.1, s="trigger", transform=ax.transAxes)
        ax.plot([.9, .95], [.12, .12], "k:", transform=ax.transAxes, alpha=.3)

        ax.text(x=.75, y=.2, s="features", transform=ax.transAxes)
        ax.plot([.9, .95], [.22, .22], "k--", transform=ax.transAxes, alpha=.3)

        ax.text(x=.75, y=.3, s="all cuts", transform=ax.transAxes)
        ax.plot([.9, .95], [.32, .32], "k-", transform=ax.transAxes)

        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.savefig(irf_path+".png")
        plt.close('all')

print("Fluxes")
print("======")

solid_angle_thrown = events_thrown.solid_angle_thrown[0]
assert np.alltrue(events_thrown.solid_angle_thrown == solid_angle_thrown)


cutoff_energy = 10.
cutoff_reduction = 0.05

print("electrons and positrons")

electron_flux_path = os.path.join(work_dir, "cosmic_e_plus_e_minus_flux.json")
with open(electron_flux_path, 'rt') as fin:
    c = json.loads(fin.read())
    c["energy"] = np.array(c["energy"]["values"])
    c["differential_flux"] = np.array(c["differential_flux"]["values"])
electron_flux = c

energy_below_cutoff = electron_flux["energy"] <= cutoff_energy
electron_flux["differential_flux"][energy_below_cutoff] *= cutoff_reduction

print("protons")

proton_flux_path = os.path.join(work_dir, "cosmic_proton_flux.json")
with open(proton_flux_path, 'rt') as fin:
    c = json.loads(fin.read())
    c["energy"] = np.array(c["energy"]["values"])
    c["differential_flux"] = np.array(c["differential_flux"]["values"])
proton_flux = c

energy_below_cutoff = proton_flux["energy"] <= cutoff_energy
proton_flux["differential_flux"][energy_below_cutoff] *= cutoff_reduction


print("export proton air-shower flux")
fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
ax = fig.add_axes([.15, .15, .8, .8])
ax.plot(
    proton_flux["energy"],
    proton_flux["differential_flux"],
    "k")
ax.loglog()
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.set_xlabel('energy / GeV')
ax.set_ylabel(r'differential flux / m$^{-2}$ sr$^{-1}$ s$^{-1}$ GeV$^{-1}$')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.savefig(os.path.join(out_dir, "proton_air_shower_flux.png"))
plt.close('all')

print("export electron air-shower flux")
fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
ax = fig.add_axes([.15, .15, .8, .8])
ax.plot(
    electron_flux["energy"],
    electron_flux["differential_flux"],
    "k")
ax.loglog()
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.set_xlabel('energy / GeV')
ax.set_ylabel(r'differential flux / m$^{-2}$ sr$^{-1}$ s$^{-1}$ GeV$^{-1}$')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.savefig(os.path.join(out_dir, "electron_air_shower_flux.png"))
plt.close('all')

print("gamma-rays")
# We need differential flux for point-source in units of:
# m^{-2} s^{-1} GeV^{-1}

fermi_3fgl_path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function',
    resource_name=os.path.join(
        'resources',
        'FermiLAT_3FGL_gll_psc_v16.fit'))
gamma_sources = []
fermi_keys = [
    "Source_Name", #str
    'RAJ2000', # deg
    'DEJ2000', # deg
    'GLON', # deg
    'GLAT', # deg
    'SpectrumType', # str
    'Pivot_Energy', # MeV
    'Spectral_Index', # 1
    'Flux_Density', # photons cm^{-2} MeV^{-1} s^{-1}
    'beta', # 1
    'Cutoff', # MeV
    'Exp_Index', # 1
    'Flux1000', # photons cm^{-2} s^{-1}
]
with astropy.io.fits.open(fermi_3fgl_path) as fits:
    num_sources = fits[1].header["NAXIS2"]
    for source_idx in range(num_sources):
        s = {}
        for fermi_key in fermi_keys:
            s[fermi_key] = fits[1].data[source_idx][fermi_key]
        gamma_sources.append(s)

source_name = "3FGL J2254.0+1608"
for i in range(len(gamma_sources)):
    __gamma_source = gamma_sources[i]
    if __gamma_source["Source_Name"] == source_name:
        break

print(__gamma_source)
gamma_source = __gamma_source.copy()
gamma_source["Pivot_Energy"] *= 1e-3 # MeV to GeV
gamma_source["Cutoff"] *= 1e-3 # MeV to GeV
gamma_source["Flux1000"] *= 1e4 # cm^2 to m^2
gamma_source["Flux_Density"] *= 1e4 # cm^2 to m^2
gamma_source["Flux_Density"] *= 1e3 # MeV^{-1} to GeV^{-1}

def power_law_super_exp_cutoff_according_to_3fgl(
    energy,
    flux_density,
    spectral_index,
    pivot_energy,
    cutoff_energy,
    exp_index
):
    '''
    pl super exponential cutoff as defined in 3FGL cat,
    but with already negative spectral_index
    '''
    return (flux_density*(energy/pivot_energy)**(spectral_index))*np.exp(
        (pivot_energy/cutoff_energy)**exp_index -
        (energy/cutoff_energy)**exp_index
    )

hr_energy_bin_edges = np.geomspace(
    energy_start,
    energy_stop,
    number_energy_bins*10 + 1)
hr_energy_bin_start = hr_energy_bin_edges[:-1]
hr_energy_bin_range = (
    hr_energy_bin_edges[1:] -
    hr_energy_bin_edges[:-1])

hr_g_point_dFdE = power_law_super_exp_cutoff_according_to_3fgl(
    energy=hr_energy_bin_start,
    flux_density=gamma_source["Flux_Density"],
    spectral_index=-gamma_source["Spectral_Index"],
    pivot_energy=gamma_source["Pivot_Energy"],
    cutoff_energy=gamma_source["Cutoff"],
    exp_index=gamma_source["Exp_Index"]) # m^{-2} s^{-1} GeV^{-1}

fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
ax = fig.add_axes([.15, .15, .8, .8])
ax.plot(
    hr_energy_bin_start,
    hr_g_point_dFdE,
    "k")
ax.loglog()
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.set_xlabel('energy / GeV')
ax.set_ylabel(r'differential flux / m$^{-2}$ s$^{-1}$ GeV$^{-1}$')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.text(x=0.5, y=0.9, s=gamma_source["Source_Name"], transform=ax.transAxes)
plt.savefig(os.path.join(out_dir, "gamma_ray_air_shower_flux.png"))
plt.close('all')

hr_p_diffuse_dFdE = np.interp(
    x=hr_energy_bin_start,
    xp=proton_flux['energy'],
    fp=proton_flux['differential_flux'])

hr_e_diffuse_dFdE = np.interp(
    x=hr_energy_bin_start,
    xp=electron_flux['energy'],
    fp=electron_flux['differential_flux'])

onaxis_solid_angle = cone_solid_angle(
    cone_radial_opening_angle=np.deg2rad(source_geometries["onaxis"]))

onregion_solid_angle = cone_solid_angle(
    cone_radial_opening_angle=np.deg2rad(.6))

solid_angle_ratio_onregion = onregion_solid_angle/onaxis_solid_angle

hr_p_onaxis_area_pt = np.interp(
    x=hr_energy_bin_start,
    xp=energy_bin_start,
    fp=irf["onaxis"]['proton']['past_trigger']['effective_area'])
hr_p_onaxis_acce_pt = hr_p_onaxis_area_pt*onaxis_solid_angle

hr_e_onaxis_area_pt = np.interp(
    x=hr_energy_bin_start,
    xp=energy_bin_start,
    fp=irf["onaxis"]['electron']['past_trigger']['effective_area'])
hr_e_onaxis_acce_pt = hr_e_onaxis_area_pt*onaxis_solid_angle

hr_p_onaxis_area_ac = np.interp(
    x=hr_energy_bin_start,
    xp=energy_bin_start,
    fp=irf["onaxis"]['proton']['past_cuts']['effective_area'])
hr_p_onaxis_acce_ac = hr_p_onaxis_area_ac*onaxis_solid_angle

hr_e_onaxis_area_ac = np.interp(
    x=hr_energy_bin_start,
    xp=energy_bin_start,
    fp=irf["onaxis"]['electron']['past_cuts']['effective_area'])
hr_e_onaxis_acce_ac = hr_e_onaxis_area_ac*onaxis_solid_angle

# gamma
hr_g_onaxis_area_pt = np.interp(
    x=hr_energy_bin_start,
    xp=energy_bin_start,
    fp=irf["onaxis"]['gamma']['past_trigger']['effective_area'])
hr_g_onaxis_area_ac = np.interp(
    x=hr_energy_bin_start,
    xp=energy_bin_start,
    fp=irf["onaxis"]['gamma']['past_cuts']['effective_area'])

# differential trigger-rates onaxis
onregion_containmant = 0.68
hr_g_dTdE_pt = hr_g_point_dFdE * hr_g_onaxis_area_pt * onregion_containmant
hr_g_dTdE_ac = hr_g_point_dFdE * hr_g_onaxis_area_ac * onregion_containmant

hr_p_dTdE_pt = hr_p_diffuse_dFdE * hr_p_onaxis_acce_pt * solid_angle_ratio_onregion
hr_p_dTdE_ac = hr_p_diffuse_dFdE * hr_p_onaxis_acce_ac * solid_angle_ratio_onregion

hr_e_dTdE_pt = hr_e_diffuse_dFdE * hr_e_onaxis_acce_pt * solid_angle_ratio_onregion
hr_e_dTdE_ac = hr_e_diffuse_dFdE * hr_e_onaxis_acce_ac * solid_angle_ratio_onregion

rate_onregion_g_pt = np.sum(hr_g_dTdE_pt * hr_energy_bin_range)
rate_onregion_g_ac = np.sum(hr_g_dTdE_ac * hr_energy_bin_range)

rate_onregion_p_pt = np.sum(hr_p_dTdE_pt * hr_energy_bin_range)
rate_onregion_p_ac = np.sum(hr_p_dTdE_ac * hr_energy_bin_range)

rate_onregion_e_pt = np.sum(hr_e_dTdE_pt * hr_energy_bin_range)
rate_onregion_e_ac = np.sum(hr_e_dTdE_ac * hr_energy_bin_range)

print("trigger: g {:.1f}, e {:.1f}, p {:.1f}".format(
    rate_onregion_g_pt,
    rate_onregion_e_pt,
    rate_onregion_p_pt,))

print("after cuts: g {:.1f}, e {:.1f}, p {:.1f}".format(
    rate_onregion_g_ac,
    rate_onregion_e_ac,
    rate_onregion_p_ac,))


trigger_rate_path = os.path.join(
    out_dir,
    "differential_trigger_rate_onregion")

with open(trigger_rate_path+".json", "wt") as fout:
    fout.write(json.dumps(
        {
            "energy_bin_edges": hr_energy_bin_edges.tolist(),
            "energy_bin_start": hr_energy_bin_start.tolist(),
            "dTdE_onregion_gamma_trigger": hr_g_dTdE_pt.tolist(),
            "dTdE_onregion_gamma_cuts": hr_g_dTdE_ac.tolist(),

            "dTdE_onregion_proton_trigger": hr_p_dTdE_pt.tolist(),
            "dTdE_onregion_proton_cuts": hr_p_dTdE_ac.tolist(),

            "dTdE_onregion_electron_trigger": hr_e_dTdE_pt.tolist(),
            "dTdE_onregion_electron_cuts": hr_e_dTdE_ac.tolist(),
        },
        indent=4))

ylow = 1e-6
fig = plt.figure(figsize=fc['figsize2'], dpi=fc['dpi'])
ax = fig.add_axes([.15, .15, .8, .8])
# gamma
ax.plot(
    hr_energy_bin_start[hr_g_dTdE_pt > ylow],
    hr_g_dTdE_pt[hr_g_dTdE_pt > ylow],
    "k",
    alpha=0.15)
ax.plot(
    hr_energy_bin_start[hr_g_dTdE_ac > ylow],
    hr_g_dTdE_ac[hr_g_dTdE_ac > ylow],
    "k",)

# proton
ax.plot(
    hr_energy_bin_start[hr_p_dTdE_pt > ylow],
    hr_p_dTdE_pt[hr_p_dTdE_pt > ylow],
    "k:",
    alpha=0.15)
ax.plot(
    hr_energy_bin_start[hr_p_dTdE_ac > ylow],
    hr_p_dTdE_ac[hr_p_dTdE_ac > ylow],
    "k:",)

# electron
ax.plot(
    hr_energy_bin_start[hr_e_dTdE_pt > ylow],
    hr_e_dTdE_pt[hr_e_dTdE_pt > ylow],
    "k--",
    alpha=0.15)
ax.plot(
    hr_energy_bin_start[hr_e_dTdE_ac > ylow],
    hr_e_dTdE_ac[hr_e_dTdE_ac > ylow],
    "k--",)

ax.loglog()
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.set_xlabel('energy / GeV')
ax.set_ylabel(r'differential rate / s$^{-1}$ GeV$^{-1}$')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_ylim([1e-6, 1e2])
ax.text(x=0.5, y=0.9, s=gamma_source["Source_Name"], transform=ax.transAxes)
plt.savefig(trigger_rate_path+".png")
plt.close('all')

