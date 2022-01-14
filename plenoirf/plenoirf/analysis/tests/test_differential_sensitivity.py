import plenoirf
import numpy as np

def test_bell_spectrum_mask_shape():

    for n in range(3,19):
        Ptgr = np.eye(n)

        mask = plenoirf.analysis.differential_sensitivity.make_mask_for_energy_confusion_matrix_for_bell_spectrum(
            probability_true_given_reco=Ptgr,
            containment=0.68,
        )

        assert mask.shape == Ptgr.shape

