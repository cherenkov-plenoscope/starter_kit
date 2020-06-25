import numpy as np
import sparse_table as spt


def make_mask(
    trigger_table,
    threshold,
    modus,
):
    """
    The plenoscope's trigger has written its response into the trigger-table.
    The response includes the max. photon-equivalent seen in each
    trigger-image. The trigger-images are focused to different
    object-distances.
    Based on this response, different modi for the final trigger are possible.
    """
    accepting_focus = modus['accepting_focus']
    rejecting_focus = modus['rejecting_focus']
    intensity_ratio_between_foci = modus['intensity_ratio_between_foci']
    use_rejection_focus = modus['use_rejection_focus']

    assert threshold >= 0
    assert accepting_focus >= 0
    if use_rejection_focus:
        assert rejecting_focus >= 0
    else:
        assert rejecting_focus == -1

    tt = trigger_table
    KEY = 'focus_{:02d}_response_pe'
    accepting_key = KEY.format(accepting_focus)
    accepting_mask = tt[accepting_key] >= threshold

    if use_rejection_focus:
        rejecting_key = KEY.format(rejecting_focus)

        rejecting_mask = tt[rejecting_key] < (
            tt[accepting_key]/intensity_ratio_between_foci)

        trigger_mask = accepting_mask*rejecting_mask
    else:
        trigger_mask = accepting_mask

    return trigger_mask


def make_indices(
    trigger_table,
    threshold,
    modus,
):
    mask = make_mask(
        trigger_table,
        threshold,
        modus,
    )
    return trigger_table[spt.IDX][mask]


def make_trigger_modi(
    intensity_ratios_between_foci=np.linspace(1.0, 1.1, 2),
    add_telescope_mode=True,
    add_light_field_mode=True,
    lowest_accepting_focus=3,
    lowest_rejecting_focus=0,
):
    trigger_modi = []

    if add_telescope_mode:
        for accepting_focus in range(num_trigger_foci_simulated()):
            m = {
                "accepting_focus": int(accepting_focus),
                "rejecting_focus": -1,
                "use_rejection_focus": 0,
                "intensity_ratio_between_foci": 0,
            }
            trigger_modi.append(m)

    if add_light_field_mode:
        focus_combinations = list_trigger_focus_combinations(
            lowest_accepting_focus=lowest_accepting_focus,
            lowest_rejecting_focus=lowest_rejecting_focus
        )
        for focus_combination in focus_combinations:
            for intensity_ratio in intensity_ratios_between_foci:
                m = {
                    "accepting_focus": int(focus_combination[0]),
                    "rejecting_focus": int(focus_combination[1]),
                    "use_rejection_focus": 1,
                    "intensity_ratio_between_foci": float(intensity_ratio),
                }
                trigger_modi.append(m)

    return trigger_modi
