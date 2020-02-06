import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pandas as pd
import numpy as np
import os
import json
import shutil
from .. import logging


def read(path):
    return pd.read_csv(path).to_records(index=False)


def write(path, table):
    df = pd.DataFrame(table)
    with open(path+'.tmp', 'wt') as f:
        f.write(df.to_csv(index=False))
    os.rename(path+'.tmp', path)


def _sum_energy_in_runs(event_table, run_ids):
    _out = []
    for run_id in run_ids:
        mask = event_table["primary"]["run_id"] == run_id
        sum_energy = np.sum(event_table["primary"]["energy_GeV"][mask])
        _out.append({
            "run_id": int(run_id),
            "energy_sum_GeV": float(sum_energy)})
    out = pd.DataFrame(_out).to_records(index=False)
    return out


def _num_events_in_runs(event_table, level_key, run_ids, key):
    _out = []
    for run_id in run_ids:
        mask = event_table[level_key]["run_id"] == run_id
        _out.append({
            "run_id": int(run_id),
            key: int(np.sum(mask))})
    out = pd.DataFrame(_out).to_records(index=False)
    return out


def merge_event_table(runtime_table, event_table):
    runtime = runtime_table
    sum_energies = _sum_energy_in_runs(
        event_table=event_table,
        run_ids=runtime["run_id"])
    num_events_corsika = _num_events_in_runs(
        event_table=event_table,
        level_key="primary",
        run_ids=runtime["run_id"],
        key="num_events_corsika")
    num_events_merlict = _num_events_in_runs(
        event_table=event_table,
        level_key="trigger",
        run_ids=runtime["run_id"],
        key="num_events_merlict")
    num_events_past_trigger = _num_events_in_runs(
        event_table=event_table,
        level_key="pasttrigger",
        run_ids=runtime["run_id"],
        key="num_events_pasttrigger")
    rta = pd.DataFrame(runtime)
    rta = pd.merge(rta, pd.DataFrame(sum_energies), on=["run_id"])
    rta = pd.merge(rta, pd.DataFrame(num_events_corsika), on=["run_id"])
    rta = pd.merge(rta, pd.DataFrame(num_events_merlict), on=["run_id"])
    rta = pd.merge(rta, pd.DataFrame(num_events_past_trigger), on=["run_id"])
    return rta.to_records(index=False)


def write_relative_runtime(table, out_path):
    ert = table
    total_times = {}
    total_time = 0
    for key in logging.KEYS:
        total_times[key] = np.sum(ert[key])
        total_time += total_times[key]

    relative_times = {}
    for key in logging.KEYS:
        relative_times[key] = float(total_times[key]/total_time)

    sc = 0.4
    fig = plt.figure(figsize=(sc*16, sc*9), dpi=120/sc)
    ax = fig.add_axes([0.3, 0.15, 0.5, 0.8])
    labels = []
    sizes = []
    _y = np.arange(len(logging.KEYS))
    for ikey, key in enumerate(relative_times):
        labels.append(key)
        sizes.append(relative_times[key])
        x = relative_times[key]
        ax.plot(
            [0, x, x, 0],
            [_y[ikey]-.5, _y[ikey]-.5, _y[ikey]+.5, _y[ikey]+.5],
            "k")
    ax.set_xlabel('rel. runtime')
    ax.set_yticks(_y)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlim([0, 1])
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    out_path_jpg = out_path+'.jpg'
    fig.savefig(out_path_jpg+".tmp.jpg")
    os.rename(out_path_jpg+".tmp.jpg", out_path_jpg)
    plt.close(fig)
    out_path_json = out_path+'.json'
    with open(out_path_json+'.tmp', "wt") as fout:
        fout.write(json.dumps(relative_times))
    os.rename(out_path_json+'.tmp', out_path_json)


def write_speed(table, out_path):
    ert = table
    speed_keys = {
        "corsika": "num_events_corsika",
        "grid": "num_events_corsika",
        "merlict": "num_events_merlict",
        "trigger": "num_events_merlict",
        "feature_extraction": "num_events_pasttrigger",
        "cherenkov_classification": "num_events_pasttrigger",
    }
    speeds = {}
    for key in speed_keys:
        num_events = ert[speed_keys[key]]
        mask = num_events > 0
        if np.sum(mask) == 0:
            speeds[key] = 0.
        else:
            speeds[key] = float(np.mean(ert[key][mask]/num_events[mask]))

    sc = 0.4
    fig = plt.figure(figsize=(sc*16, sc*9), dpi=120/sc)
    ax = fig.add_axes([0.3, 0.15, 0.5, 0.8])
    labels = []
    sizes = []
    _y = np.arange(len(speeds))
    for ikey, key in enumerate(speeds):
        labels.append(key)
        sizes.append(speeds[key])
        x = speeds[key]
        ax.plot(
            [0, x, x, 0],
            [_y[ikey]-.5, _y[ikey]-.5, _y[ikey]+.5, _y[ikey]+.5],
            "k")
    sizes = np.array(sizes)
    valid = np.logical_not(np.logical_or(np.isinf(sizes), np.isnan(sizes)))
    ax.set_xlabel('speed / events s$^{-1}$')
    ax.set_yticks(_y)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlim([0, np.max(sizes[valid])*1.1])
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    fig.savefig(out_path+'.tmp'+'.jpg')
    os.rename(out_path+'.tmp'+'.jpg', out_path+'.jpg')
    plt.close(fig)
    with open(out_path+'.json'+'.tmp', "wt") as fout:
        fout.write(json.dumps(speeds))
    os.rename(out_path+'.json'+'.tmp', out_path+'.json')
