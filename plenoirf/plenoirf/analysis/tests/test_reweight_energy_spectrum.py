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
        some_spectrum_norm = some_spectrum/np.max(some_spectrum)

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
        energies=np.linspace(1, 10, 100),
        start_energy=1,
        stop_energy=10
    )
    assert s == "linspace"
    assert 50 ** (1/3) <= n <= 200 ** (1/3)

    s, n = plenoirf.analysis.reweight.estimate_binning(
        energies=np.geomspace(1, 10, 1000),
        start_energy=1,
        stop_energy=10
    )

    assert s == "geomspace"
    assert 500 ** (1/3) <= n <= 2000 ** (1/3)



def test_estimate_relative_rates_with_power_law_minus_one():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    power = cpw.random_distributions.draw_power_law(
        prng=prng,
        lower_limit=1.0,
        upper_limit=10.0,
        power_slope=-1.0,
        num_samples=10000
    )

    energy, rate = plenoirf.analysis.reweight.estimate_relative_rates(
        energies=power,
        start_energy=1,
        stop_energy=10
    )
    assert np.std(rate) < 0.1 * np.mean(rate)


def test_estimate_relative_rates_with_power():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    power = cpw.random_distributions.draw_power_law(
        prng=prng,
        lower_limit=1.0,
        upper_limit=10.0,
        power_slope=-2.7,
        num_samples=10000
    )

    energy, rate = plenoirf.analysis.reweight.estimate_relative_rates(
        energies=power,
        start_energy=1,
        stop_energy=10
    )
    assert np.std(rate) > 0.1 * np.mean(rate)


def test_histogram():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    N = 100000
    thrown = cpw.random_distributions.draw_power_law(
        prng=prng,
        lower_limit=1.0,
        upper_limit=10.0,
        power_slope=-2.0,
        num_samples=N
    )
    bin_edges = np.geomspace(1, 10, 6)

    np_bin_counts = np.histogram(
        a=thrown,
        bins=bin_edges,
        weights=np.ones(N)
    )

    nw_bin_counts = plenoirf.analysis.reweight.histogram_power_law_weights(
        a=thrown,
        bins=bin_edges,
        weights=np.ones(N),
        target_power_law_slope=-2.7,
        max_power_law_weight_factor=5,
    )

