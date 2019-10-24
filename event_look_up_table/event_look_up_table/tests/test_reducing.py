import event_look_up_table as elut
import numpy as np
import tempfile
import os


def test_reducing():
    opj = os.path.join
    NUM_APERTURE_BINS = 8
    NUM_JOBS = 8
    with tempfile.TemporaryDirectory(prefix='plenoscope_lookup_') as tmp:
        for job in range(NUM_JOBS):
            job_path = opj(
                tmp,
                "{:06d}".format(job))
            os.makedirs(job_path)
            for aperture_bin in range(NUM_APERTURE_BINS):
                aperture_bin_path = opj(
                    job_path,
                    elut.APERTURE_BIN_FILENAME.format(aperture_bin))
                with open(aperture_bin_path, "wb") as f:
                    f.write(np.int64(job*aperture_bin).tobytes())

        final_path = opj(tmp, "final")
        os.makedirs(final_path)

        for job in range(NUM_JOBS):
            elut._append_apperture_bin(
                input_altitude_bin_path=opj(tmp, "{:06d}".format(job)),
                output_altitude_bin_path=final_path,
                num_aperture_bins=NUM_APERTURE_BINS)

        for aperture_bin in range(NUM_APERTURE_BINS):
            final_aperture_bin_path = opj(
                final_path,
                elut.APERTURE_BIN_FILENAME.format(aperture_bin))
            assert os.path.exists(final_aperture_bin_path)
            with open(final_aperture_bin_path, "rb") as f:
                content = np.frombuffer(f.read(), dtype=np.int64)

            assert content.shape[0] == NUM_JOBS
            print(content)
            for job in range(NUM_JOBS):
                assert content[job] == job*aperture_bin
