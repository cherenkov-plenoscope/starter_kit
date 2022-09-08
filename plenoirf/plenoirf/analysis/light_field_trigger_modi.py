import numpy as np
import sparse_numeric_table as spt


def make_mask(
    trigger_table, threshold, modus,
):
    """
    The plenoscope's trigger has written its response into the trigger-table.
    The response includes the max. photon-equivalent seen in each
    trigger-image. The trigger-images are focused to different
    object-distances.
    Based on this response, different modi for the final trigger are possible.
    """
    KEY = "focus_{:02d}_response_pe"

    assert threshold >= 0
    assert modus["accepting_focus"] >= 0
    assert modus["rejecting_focus"] >= 0

    accepting_response_pe = trigger_table[KEY.format(modus["accepting_focus"])]
    rejecting_response_pe = trigger_table[KEY.format(modus["rejecting_focus"])]

    threshold_accepting_over_rejecting = np.interp(
        x=accepting_response_pe,
        xp=modus["accepting"]["response_pe"],
        fp=modus["accepting"]["threshold_accepting_over_rejecting"],
        left=None,
        right=None,
        period=None,
    )

    accepting_over_rejecting = (
        accepting_response_pe / rejecting_response_pe
    )

    size_over_threshold = accepting_response_pe >= threshold
    ratio_over_threshold = (
        accepting_over_rejecting >= threshold_accepting_over_rejecting
    )

    return np.logical_and(size_over_threshold, ratio_over_threshold)


def make_indices(
    trigger_table, threshold, modus,
):
    mask = make_mask(trigger_table, threshold, modus,)
    return trigger_table[spt.IDX][mask]
