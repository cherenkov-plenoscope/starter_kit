#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

trigger_threshold = sum_config["trigger"]["threshold_pe"]
trigger_modus = sum_config["trigger"]["modus"]

max_relative_leakage = sum_config["quality"]["max_relative_leakage"]
min_reconstructed_photons = sum_config["quality"]["min_reconstructed_photons"]

num_energy_bins = sum_config["energy_binning"]["num_bins"][
    "trigger_acceptance"
]
energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    num_energy_bins + 1,
)
fine_energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    sum_config["energy_binning"]["num_bins"]["interpolation"] + 1,
)
fine_energy_bin_centers = irf.summary.bin_centers(fine_energy_bin_edges)


fig_16_by_9 = sum_config["plot"]["16_by_9"]
particle_colors = sum_config["plot"]["particle_colors"]

# cosmic-ray-flux
# ----------------
airshower_fluxes = irf.summary.read_airshower_differential_flux_zenith_compensated(
    run_dir=pa["run_dir"],
    summary_dir=pa["summary_dir"],
    energy_bin_centers=fine_energy_bin_centers,
    sites=irf_config["config"]["sites"],
    geomagnetic_cutoff_fraction=sum_config["airshower_flux"][
        "fraction_of_flux_below_geomagnetic_cutoff"
    ],
)

# gamma-ray-flux of reference source
# ----------------------------------
(
    gamma_differential_flux_per_m2_per_s_per_GeV,
    gamma_name,
) = irf.summary.make_gamma_ray_reference_flux(
    summary_dir=pa["summary_dir"],
    gamma_ray_reference_source=sum_config["gamma_ray_reference_source"],
    energy_supports_GeV=fine_energy_bin_centers,
)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

for sk in SITES:
    airshower_fluxes[sk]["gamma"] = {
        "differential_flux": gamma_differential_flux_per_m2_per_s_per_GeV,
    }

tables = {}
reweight_spectrum = {}
for sk in SITES:
    tables[sk] = {}
    reweight_spectrum[sk] = {}
    for pk in PARTICLES:
        reweight_spectrum[sk][pk] = {}
        _table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        reweight_spectrum[sk][pk]["energy_bin_edges"] = energy_bin_edges
        reweight_spectrum[sk][pk]["energy_min"] = np.min(
            _table["primary"]["energy_GeV"]
        )
        reweight_spectrum[sk][pk]["energy_max"] = np.max(
            _table["primary"]["energy_GeV"]
        )
        reweight_spectrum[sk][pk]["histogram_energy_thrown"] = np.histogram(
            _table["primary"]["energy_GeV"], bins=energy_bin_edges,
        )[0]

        idx_triggered = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=_table["trigger"],
            threshold=trigger_threshold,
            modus=trigger_modus,
        )
        idx_quality = irf.analysis.cuts.cut_quality(
            feature_table=_table["features"],
            max_relative_leakage=max_relative_leakage,
            min_reconstructed_photons=min_reconstructed_photons,
        )
        idx_features = _table["features"][spt.IDX]

        idx_common = spt.intersection([idx_triggered, idx_features,])

        table = spt.cut_table_on_indices(
            table=_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_common,
            level_keys=["primary", "features"],
        )
        tables[sk][pk] = spt.sort_table_on_common_indices(
            table=table, common_indices=idx_common
        )

# guess bin edges
lims = {}
Sfeatures = irf.table.STRUCTURE["features"]

for fk in Sfeatures:
    lims[fk] = {}
    for sk in SITES:
        lims[fk][sk] = {}
        for pk in PARTICLES:
            lims[fk][sk][pk] = {}
            features = tables[sk][pk]["features"]
            num_bins = int(np.sqrt(features.shape[0]))
            num_bin_edges = num_bins + 1
            lims[fk][sk][pk]["bin_edges"] = {}
            lims[fk][sk][pk]["bin_edges"]["num"] = num_bin_edges

            if Sfeatures[fk]["histogram"] == "geomspace":
                start = 10 ** np.floor(np.log10(np.min(features[fk])))
                stop = 10 ** np.ceil(np.log10(np.max(features[fk])))
                if start == 0.0:
                    start = 1e-3 * stop
            elif Sfeatures[fk]["histogram"] == "linspace":
                start = np.min(features[fk])
                stop = np.max(features[fk])

            lims[fk][sk][pk]["bin_edges"]["start"] = start
            lims[fk][sk][pk]["bin_edges"]["stop"] = stop

# find same bin-edges for all particles
for fk in Sfeatures:
    for sk in SITES:
        starts = [lims[fk][sk][pk]["bin_edges"]["start"] for pk in PARTICLES]
        stops = [lims[fk][sk][pk]["bin_edges"]["stop"] for pk in PARTICLES]
        nums = [lims[fk][sk][pk]["bin_edges"]["num"] for pk in PARTICLES]
        start = np.min(starts)
        stop = np.max(stops)
        num = np.max(nums)
        for pk in PARTICLES:
            lims[fk][sk][pk]["bin_edges"]["stop"] = stop
            lims[fk][sk][pk]["bin_edges"]["start"] = start
            lims[fk][sk][pk]["bin_edges"]["num"] = num


for sk in SITES:
    for pk in PARTICLES:
        reweight_spectrum[sk][pk][
            "weights"
        ] = irf.analysis.reweight_energy_spectrum(
            initial_energy_bin_centers=irf.summary.bin_centers(
                reweight_spectrum[sk][pk]["energy_bin_edges"]
            ),
            initial_rate=reweight_spectrum[sk][pk]["histogram_energy_thrown"],
            target_energy_bin_centers=fine_energy_bin_centers,
            target_rate=airshower_fluxes[sk][pk]["differential_flux"],
            energies_of_events=tables[sk][pk]["primary"]["energy_GeV"],
        )

    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    for pk in PARTICLES:
        w_energies = np.geomspace(
            reweight_spectrum[sk][pk]["energy_min"],
            reweight_spectrum[sk][pk]["energy_max"],
            1024,
        )
        w_weights = irf.analysis.reweight_energy_spectrum(
            initial_energy_bin_centers=irf.summary.bin_centers(
                reweight_spectrum[sk][pk]["energy_bin_edges"]
            ),
            initial_rate=reweight_spectrum[sk][pk]["histogram_energy_thrown"],
            target_energy_bin_centers=fine_energy_bin_centers,
            target_rate=airshower_fluxes[sk][pk]["differential_flux"],
            energies_of_events=w_energies,
        )

        ax.plot(
            w_energies, w_weights, color=particle_colors[pk],
        )
    ax.loglog()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel("relative re-weights/ 1")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([1e-6, 1.0])
    fig.savefig(
        os.path.join(pa["out_dir"], "{:s}_reweight.jpg".format(sk, fk))
    )
    plt.close(fig)


for fk in Sfeatures:
    for sk in SITES:

        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

        for pk in PARTICLES:
            if Sfeatures[fk]["histogram"] == "geomspace":
                myspace = np.geomspace
            elif Sfeatures[fk]["histogram"] == "linspace":
                myspace = np.linspace

            bin_edges_fk = myspace(
                lims[fk][sk][pk]["bin_edges"]["start"],
                lims[fk][sk][pk]["bin_edges"]["stop"],
                lims[fk][sk][pk]["bin_edges"]["num"],
            )
            bin_counts_fk = np.histogram(
                tables[sk][pk]["features"][fk], bins=bin_edges_fk
            )[0]
            bin_counts_weight_fk = np.histogram(
                tables[sk][pk]["features"][fk],
                weights=reweight_spectrum[sk][pk]["weights"],
                bins=bin_edges_fk,
            )[0]
            with np.errstate(divide="warn"):
                bin_counts_unc_fk = np.sqrt(bin_counts_fk) / bin_counts_fk
                bin_counts_weight_norm_fk = bin_counts_weight_fk / np.sum(
                    bin_counts_weight_fk
                )

            irf.summary.figure.ax_add_hist(
                ax=ax,
                bin_edges=bin_edges_fk,
                bincounts=bin_counts_weight_norm_fk,
                linestyle="-",
                linecolor=particle_colors[pk],
                linealpha=1.0,
                bincounts_upper=bin_counts_weight_norm_fk
                * (1 + bin_counts_unc_fk),
                bincounts_lower=bin_counts_weight_norm_fk
                * (1 - bin_counts_unc_fk),
                face_color=particle_colors[pk],
                face_alpha=0.3,
            )

        if Sfeatures[fk]["histogram"] == "geomspace":
            ax.loglog()
        elif Sfeatures[fk]["histogram"] == "linspace":
            ax.semilogy()

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(
            "Reweighted to energy-spectrum of airshowers. "
            "Gamma-ray-source: {:s}".format(gamma_name)
        )
        ax.set_xlabel("{:s} / {:s}".format(fk, Sfeatures[fk]["unit"]))
        ax.set_ylabel("relative intensity / 1")
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        ax.set_xlim(
            [
                lims[fk][sk][pk]["bin_edges"]["start"],
                lims[fk][sk][pk]["bin_edges"]["stop"],
            ]
        )
        fig.savefig(
            os.path.join(pa["out_dir"], "{:s}_{:s}.jpg".format(sk, fk))
        )
        plt.close(fig)
