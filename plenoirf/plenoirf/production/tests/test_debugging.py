import plenoirf


def test_example_conditions_valid():
    plenoirf.production.debugging.assert_conditions_valid(
        conditions=plenoirf.production.debugging.EXAMPLE_CONDITIONS
    )


def test_pass_example_conditions():
    assert plenoirf.production.debugging.conditions_met(
        conditions=plenoirf.production.debugging.EXAMPLE_CONDITIONS,
        event_uid=1,
        event_energy_GeV=101,
        event_distance_to_core_m=249,
        event_off_axis_angle_deg=4.9,
        event_reconstructed_cherenkov_pe=251,
    )

    assert not plenoirf.production.debugging.conditions_met(
        conditions=plenoirf.production.debugging.EXAMPLE_CONDITIONS,
        event_uid=1,
        event_energy_GeV=99,
        event_distance_to_core_m=249,
        event_off_axis_angle_deg=4.9,
        event_reconstructed_cherenkov_pe=251,
    )
