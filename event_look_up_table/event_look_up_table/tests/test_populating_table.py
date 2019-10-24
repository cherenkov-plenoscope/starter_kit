import event_look_up_table as elut
import numpy as np
import tempfile
import os


def test_populating():
    opj = os.path.join
    with tempfile.TemporaryDirectory(prefix='plenoscope_lookup_') as tmp:
        lookup_path = os.path.join(tmp, "my_table")
        elut.init(
            lookup_path=lookup_path,
            max_num_photons_in_bin=1000)

        elut._add_energy_to_lookup(
            lookup_path=lookup_path,
            energy_bin_center=0,
            energy_per_iteration=5.)

        ope = os.path.exists
        assert ope(opj(lookup_path, "000000_energy"))
        assert ope(opj(lookup_path, "000000_energy", "fill.json"))
        assert ope(opj(lookup_path, "000000_energy", "000000_altitude"))
