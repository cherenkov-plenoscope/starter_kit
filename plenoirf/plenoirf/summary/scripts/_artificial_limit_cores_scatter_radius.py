import plenoirf
import sparse_numeric_table as snt
import sys
import os
import shutil
import glob

assert len(sys.argv) == 3

input_run_path = sys.argv[1]
output_run_path = sys.argv[2]

# the maximum scatter-radii from Sebastian's phd-thesis VS. energy.
phd_scatters = {
    "gamma": {
        "energy":             [0.23, 0.8, 3.0, 35,   81,   432,  1000],
        "max_scatter_radius": [150,  150, 460, 1100, 1235, 1410, 1660],
    },
    "electron": {
        "energy":             [0.23, 1.0,  10,  100,  1000],
        "max_scatter_radius": [150,  150,  500, 1100, 2600],
    },
    "proton": {
        "energy":             [5.0, 25, 250, 1000],
        "max_scatter_radius": [200, 350, 700, 1250],
    }
}

# Helium was not used in the phd-thesis
phd_scatters['helium'] = phd_scatters['proton'].copy()

assert not os.path.exists(output_run_path)
shutil.copytree(
    src=input_run_path,
    dst=output_run_path
)

irf_config = plenoirf.summary.read_instrument_response_config(
    run_dir=input_run_path
)

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:

        _tab = snt.read(
            path=os.path.join(
                input_run_path,
                "event_table",
                site_key,
                particle_key,
                "event_table.tar"
            ),
            structure=plenoirf.table.STRUCTURE
        )

        idx_primary_core = snt.intersection([
            _tab['primary'][snt.IDX],
            _tab['core'][snt.IDX],
        ])
        tab = snt.cut_table_on_indices(
            table=_tab,
            structure=plenoirf.table.STRUCTURE,
            common_indices=idx_primary_core,
            level_keys=['primary', 'core']
        )
        tab = snt.sort_table_on_common_indices(
            table=tab,
            common_indices=idx_primary_core
        )

        max_allowed_scatter_radius = np.interp(
            x=tab['primary']['energy_GeV'],
            xp=phd_scatters[particle_key]['energy'],
            fp=phd_scatters[particle_key]['max_scatter_radius'],
        )

        actual_scatter_radius = np.hypot(
            tab['core']["core_x_m"],
            tab['core']["core_y_m"],
        )

        mask_valid_in_phd_thesis = (
            actual_scatter_radius <= max_allowed_scatter_radius
        )

        idx_valid_in_phd_thesis = tab[
            'primary'][
            snt.IDX][
            mask_valid_in_phd_thesis]

        outtab = snt.cut_table_on_indices(
            table=_tab,
            structure=plenoirf.table.STRUCTURE,
            common_indices=idx_valid_in_phd_thesis,
        )

        snt.write(
            path=os.path.join(
                output_run_path,
                "event_table",
                site_key,
                particle_key,
                "event_table.tar"
            ),
            table=outtab,
            structure=plenoirf.table.STRUCTURE
        )
