import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import figure


def estimate_effective_quantity(
    energy_bin_edges,
    energies,
    max_scatter_quantities,
    thrown_mask,
    thrown_weights,
    detection_mask,
    detection_weights,
):
    num_thrown = np.histogram(
        energies,
        weights=thrown_mask*thrown_weights,
        bins=energy_bin_edges)[0]
    num_detected = np.histogram(
        energies,
        weights=detection_mask*detection_weights,
        bins=energy_bin_edges)[0]

    num_detected_no_weights = np.histogram(
        energies,
        weights=detection_mask,
        bins=energy_bin_edges)[0]

    quantity_thrown = np.histogram(
        energies,
        weights=max_scatter_quantities,
        bins=energy_bin_edges)[0]
    quantity_detected = np.histogram(
        energies,
        weights=max_scatter_quantities*detection_mask*detection_weights,
        bins=energy_bin_edges)[0]
    num_bins = energy_bin_edges.shape[0] - 1
    effective_quantity = np.nan*np.ones(num_bins)
    for i in range(num_bins):
        if num_thrown[i] > 0 and quantity_thrown[i] > 0.:
            effective_quantity[i] = (
                (quantity_detected[i]/quantity_thrown[i])*
                (quantity_thrown[i]/num_thrown[i]))

    effective_quantity_relunc = np.nan*np.ones(num_bins)
    effective_quantity_absunc = np.nan*np.ones(num_bins)
    for i in range(num_bins):
        if num_detected_no_weights[i] > 0:
            effective_quantity_relunc[i] = (
                np.sqrt(num_detected_no_weights[i])/num_detected_no_weights[i])
            effective_quantity_absunc[i] = (
                effective_quantity_relunc[i]*effective_quantity[i])

    return {
        "energy_bin_edges": energy_bin_edges,
        "num_thrown": num_thrown,
        "num_detected": num_detected,
        "quantity_thrown": quantity_thrown,
        "quantity_detected": quantity_detected,
        "effective_quantity": effective_quantity,
        "effective_quantity_abs_uncertainty": effective_quantity_absunc}


def write_effective_quantity_figure(
    effective_quantity,
    quantity_label,
    path,
    figure_config=figure.CONFIG_16_9,
    y_start=1e1,
    y_stop=1e6,
    linestyle='k-',
):
    uu = (
        effective_quantity["effective_quantity"] +
        effective_quantity["effective_quantity_abs_uncertainty"])
    ll = (
        effective_quantity["effective_quantity"] -
        effective_quantity["effective_quantity_abs_uncertainty"])
    fig = figure.figure(figure_config)
    ax = fig.add_axes([.1, .15, .85, .8])
    figure.ax_add_hist(
        ax=ax,
        bin_edges=effective_quantity["energy_bin_edges"],
        bincounts=effective_quantity["effective_quantity"],
        linestyle=linestyle,
        bincounts_upper=uu,
        bincounts_lower=ll,
        face_color='k',
        face_alpha=0.33,)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('energy / GeV')
    ax.set_ylabel(quantity_label)
    ax.set_ylim([y_start, y_stop])
    ax.set_xlim([
        np.min(effective_quantity["energy_bin_edges"]),
        np.max(effective_quantity["energy_bin_edges"]),])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.savefig(path+'.'+figure_config['format'])
    plt.close(fig)


def write_effective_quantity_table(
    path,
    effective_quantity,
    quantity_key,
):
    key = quantity_key
    eq = effective_quantity
    out = {}
    out["energy_bin_edges_GeV"] = eq['energy_bin_edges'].tolist()
    out["num_thrown"] = eq['num_thrown'].tolist()
    out["num_detected"] = eq['num_detected'].tolist()
    out[key+"_thrown"] = eq['quantity_thrown'].tolist()
    out[key+"_detected"] = eq['quantity_detected'].tolist()
    out["effective_"+key] = eq['effective_quantity'].tolist()
    out["effective_"+key+"_abs_uncertainty"] = eq[
        'effective_quantity_abs_uncertainty'].tolist()
    with open(path, 'wt') as fout:
        fout.write(json.dumps(out, indent=4))