"""
Take control what intermediate results, or truth of an even is exported.
By default, we do not export all results and we do not export all truth as this
takes too much time and space.
"""

EXAMPLE_CONDITIONS = {
    "skip_num_events": 25,
    "cherry_pick": {
        "energy_GeV": {"start": 100, "stop": 9e99,},
        "distance_to_core_m": {"start": 0, "stop": 250,},
        "off_axis_angle_deg": {"start": 0, "stop": 5,},
        "reconstructed_cherenkov_pe": {"start": 250, "stop": 9e99,},
    },
}


def _assert_range_gt(range_dict):
    assert range_dict["start"] >= 0.0
    assert range_dict["start"] <= range_dict["stop"]


def assert_conditions_valid(conditions):
    assert conditions["skip_num_events"] > 0

    CP = conditions["cherry_pick"]
    _assert_range_gt(CP["energy_GeV"])
    _assert_range_gt(CP["distance_to_core_m"])
    _assert_range_gt(CP["off_axis_angle_deg"])
    _assert_range_gt(CP["reconstructed_cherenkov_pe"])


def _in_range(val, range_dict):
    return range_dict["start"] <= val <= range_dict["stop"]


def conditions_met(
    conditions,
    event_uid,
    event_energy_GeV,
    event_distance_to_core_m,
    event_off_axis_angle_deg,
    event_reconstructed_cherenkov_pe,
):
    # export after certain number of events no matter what
    # ----------------------------------------------------
    if event_uid % conditions["skip_num_events"] == 0:
        return True

    # export when certain conditions are met
    # --------------------------------------
    CP = conditions["cherry_pick"]
    if (
        _in_range(event_energy_GeV, CP["energy_GeV"])
        and _in_range(event_distance_to_core_m, CP["distance_to_core_m"])
        and _in_range(event_off_axis_angle_deg, CP["off_axis_angle_deg"])
        and _in_range(
            event_reconstructed_cherenkov_pe, CP["reconstructed_cherenkov_pe"],
        )
    ):
        return True

    return False
