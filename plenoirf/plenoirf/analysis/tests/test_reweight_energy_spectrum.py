import plenoirf
import numpy as np
import corsika_primary_wrapper as cpw


def test_random_spectrum():
    prng = np.random.Generator(np.random.MT19937(seed=0))
    rnd = prng.uniform
    N = 10

    for case in range(10):
        energy = np.linspace(0, 1, N)
        flat_spectrum = np.linspace(1, 1, N)
        poly = [rnd(), rnd(), 1]
        some_spectrum = np.polyval(poly, energy)

        weights = plenoirf.analysis.reweight.reweight(
            initial_energies=energy,
            initial_rates=flat_spectrum,
            target_energies=energy,
            target_rates=some_spectrum,
            event_energies=np.linspace(0, 1, N),
        )
        some_spectrum_norm = some_spectrum / np.max(some_spectrum)

        assert np.all(np.abs(weights - some_spectrum_norm) < 1e-2)


def test_power_law_spectrum():
    N = 10
    energy = np.geomspace(1, 10, N)
    thrown_spectrum = energy ** (-2.0)
    target_spectrum = energy ** (-2.7)

    weights = plenoirf.analysis.reweight.reweight(
        initial_energies=energy,
        initial_rates=thrown_spectrum,
        target_energies=energy,
        target_rates=target_spectrum,
        event_energies=np.linspace(1, 10, 10),
    )


def test_estimate_binning():
    s, n = plenoirf.analysis.reweight.estimate_binning(
        energies=np.linspace(1, 10, 100), start_energy=1, stop_energy=10
    )
    assert s == "linspace"
    assert 50 ** (1 / 3) <= n <= 200 ** (1 / 3)

    s, n = plenoirf.analysis.reweight.estimate_binning(
        energies=np.geomspace(1, 10, 1000), start_energy=1, stop_energy=10
    )

    assert s == "geomspace"
    assert 500 ** (1 / 3) <= n <= 2000 ** (1 / 3)


def test_estimate_relative_rates_with_power_law_minus_one():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    power = cpw.random_distributions.draw_power_law(
        prng=prng,
        lower_limit=1.0,
        upper_limit=10.0,
        power_slope=-1.0,
        num_samples=10000,
    )

    energy, rate = plenoirf.analysis.reweight.estimate_relative_rates(
        energies=power, start_energy=1, stop_energy=10
    )
    assert np.std(rate) < 0.1 * np.mean(rate)


def test_estimate_relative_rates_with_power():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    power = cpw.random_distributions.draw_power_law(
        prng=prng,
        lower_limit=1.0,
        upper_limit=10.0,
        power_slope=-2.7,
        num_samples=10000,
    )

    energy, rate = plenoirf.analysis.reweight.estimate_relative_rates(
        energies=power, start_energy=1, stop_energy=10
    )
    assert np.std(rate) > 0.1 * np.mean(rate)


def test_histogram_weights_equal_one():
    prng = np.random.Generator(np.random.MT19937(seed=0))
    thrown_slope = -2.0
    cosmic_slope = -2.7

    N = 1000
    energy = cpw.random_distributions.draw_power_law(
        prng=prng,
        lower_limit=1.0,
        upper_limit=10.0,
        power_slope=thrown_slope,
        num_samples=N,
    )
    weights = np.ones(N)
    bin_edges = np.geomspace(1, 10, 6)

    np_bin_counts, _ = np.histogram(a=energy, bins=bin_edges, weights=weights)

    (
        rw_bin_counts,
        _,
    ) = plenoirf.analysis.reweight.histogram_with_bin_wise_power_law_reweighting(
        a=energy,
        bins=bin_edges,
        weights=weights,
        target_power_law_slope=cosmic_slope,
        max_power_law_weight_factor=5,
    )

    # as long as weights are 1, power-law reweighting must not matter.
    np.testing.assert_array_almost_equal(
        x=np_bin_counts, y=rw_bin_counts, decimal=6
    )


def test_histogram_weights_equal_one():
    prng = np.random.Generator(np.random.MT19937(seed=0))
    thrown_slope = -2.0
    cosmic_slope = -2.7

    N = 10 * 1000
    energy = cpw.random_distributions.draw_power_law(
        prng=prng,
        lower_limit=1.0,
        upper_limit=10.0,
        power_slope=thrown_slope,
        num_samples=N,
    )
    weights = 1.0 * energy / 10.0
    bin_edges = np.geomspace(1, 10, 6)

    np_bin_counts, _ = np.histogram(a=energy, bins=bin_edges, weights=weights)

    (
        rw_bin_counts,
        _,
    ) = plenoirf.analysis.reweight.histogram_with_bin_wise_power_law_reweighting(
        a=energy,
        bins=bin_edges,
        weights=weights,
        target_power_law_slope=cosmic_slope,
        min_event_weight=0.3,
        max_event_weight=3,
    )

    # When weights ~ energy, then the non reweighted histogram should
    # overestimate the effective-area as the thrown spectrum is harder then
    # the cosmic spectrum
    print(np_bin_counts / rw_bin_counts)
    assert np.all(np_bin_counts / rw_bin_counts > 1.02)
    assert np.all(np_bin_counts / rw_bin_counts < 1.04)
