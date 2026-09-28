"""

Automated sanity checks for memory.py (linear memory capacity) and
experiments.py (gain/size sweeps that combine memory capacity with the
criticality estimators).
"""

import numpy as np
import pytest

from eoc.network import RandomRNN
from eoc.memory import (
    compute_memory_capacity_spectrum,
    total_memory_capacity,
    generate_input,
    _squared_correlation,
)
from eoc.experiments import SweepConfig, run_gain_sweep


# Tests for memory.py


def test_squared_correlation_perfect_relationship():
    """A perfectly linear relationship must give a squared correlation of 1."""
    y_true = np.linspace(-1, 1, 100)
    y_pred = 3.0 * y_true + 0.5  # any nonzero linear rescaling + offset
    assert _squared_correlation(y_true, y_pred) == pytest.approx(1.0, abs=1e-8)


def test_squared_correlation_no_relationship():
    """
    Completely unrelated (independent) sequences should give a squared
    correlation close to 0 - used a moderately large N and a generous
    tolerance since this is a statistical, not exact, statement.
    """
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=5000)
    y_pred = rng.normal(size=5000)
    assert _squared_correlation(y_true, y_pred) < 0.01


def test_squared_correlation_handles_constant_input():
    """
    If the prediction is a constant (zero variance), the correlation is
    mathematically undefined -returning 0.0
    """
    y_true = np.linspace(-1, 1, 50)
    y_pred = np.full(50, 0.3)  # constant
    result = _squared_correlation(y_true, y_pred)
    assert result == 0.0
    assert not np.isnan(result)


def test_generate_input_statistics():
    """Sanity-checking the basic statistics of both supported input distributions."""
    rng = np.random.default_rng(0)
    normal_input = generate_input(200_000, "normal", rng)
    assert normal_input.mean() == pytest.approx(0.0, abs=0.02)
    assert normal_input.std() == pytest.approx(1.0, abs=0.02)

    rng2 = np.random.default_rng(0)
    uniform_input = generate_input(200_000, "uniform", rng2)
    assert uniform_input.min() >= -1.0
    assert uniform_input.max() <= 1.0
    assert uniform_input.mean() == pytest.approx(0.0, abs=0.02)


def test_generate_input_rejects_unknown_distribution():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_input(100, "not_a_real_distribution", rng)


def test_mc_spectrum_values_are_bounded():
    """
    Every MC_k must lie in [0, 1] (up to a small numerical tolerance),
    since it is defined as a squared correlation coefficient, which is
    mathematically bounded in [0, 1].
    """
    net = RandomRNN(n_units=100, gain=1.2, seed=0)
    rng = np.random.default_rng(1)
    spectrum = compute_memory_capacity_spectrum(
        net, k_max=20, n_train=500, n_test=250, n_transient=150, rng=rng
    )
    assert np.all(spectrum >= -1e-6)
    assert np.all(spectrum <= 1.0 + 1e-6)


def test_mc1_is_high_for_a_reasonable_network():
    """
    MC_1 (reconstructing the input from just 1 step ago) should be
    close to 1 for any network that isn't extremely degenerate -- the
    previous input value has a large, direct, linear effect on the
    current state via w_in, so a linear readout should recover it almost
    perfectly.
    """
    net = RandomRNN(n_units=100, gain=1.0, seed=0)
    rng = np.random.default_rng(1)
    spectrum = compute_memory_capacity_spectrum(
        net, k_max=10, n_train=500, n_test=250, n_transient=150, rng=rng
    )
    assert spectrum[0] > 0.8  # MC_1


def test_total_memory_capacity_matches_manual_sum():
    spectrum = np.array([0.9, 0.5, 0.2, 0.05])
    assert total_memory_capacity(spectrum) == pytest.approx(1.65)


def test_jaeger_bound_total_mc_does_not_greatly_exceed_n():
    """
    Jaeger's (2002) theoretical result again: for a linear reservoir readout
    driven by an scalar input, the total memory capacity is
    bounded by the number of reservoir units, MC <= N 

    Using here a generous tolerance (1.2x) since (a) reservoir is
    nonlinear rather than exactly linear, and (b) any finite-sample
    correlation estimate carries some upward statistical bias
    """
    n_units = 100
    net = RandomRNN(n_units=n_units, gain=1.0, seed=2)
    rng = np.random.default_rng(3)
    spectrum = compute_memory_capacity_spectrum(
        net, k_max=40, n_train=800, n_test=400, n_transient=150, rng=rng
    )
    mc_total = total_memory_capacity(spectrum)
    assert mc_total <= 1.2 * n_units, (
        f"Total MC={mc_total:.1f} greatly exceeds the number of units "
        f"({n_units}) -- this would violate Jaeger's theoretical bound "
        "and likely indicates overfitting (e.g. ridge_alpha too small, "
        "or n_train too small relative to n_units)."
    )


def test_mc_requires_transient_at_least_k_max():
    """`compute_memory_capacity_spectrum` should refuse an impossible
    configuration where k_max exceeds the available warm-up history."""
    net = RandomRNN(n_units=20, gain=1.0, seed=0)
    with pytest.raises(ValueError):
        compute_memory_capacity_spectrum(
            net, k_max=50, n_train=100, n_test=50, n_transient=10
        )


def test_mc_rejects_wrong_length_driving_input():
    net = RandomRNN(n_units=20, gain=1.0, seed=0)
    wrong_length_input = np.zeros(5)  # far too short
    with pytest.raises(ValueError):
        compute_memory_capacity_spectrum(
            net, k_max=10, n_train=100, n_test=50, n_transient=20,
            driving_input=wrong_length_input,
        )


def test_mc_reproducible_with_same_seed_and_rng():
    """Same network + same seeded RNG must give bit-for-bit identical MC."""
    def run_once():
        net = RandomRNN(n_units=80, gain=1.1, seed=5)
        rng = np.random.default_rng(42)
        return compute_memory_capacity_spectrum(
            net, k_max=15, n_train=400, n_test=200, n_transient=150, rng=rng
        )

    spectrum_a = run_once()
    spectrum_b = run_once()
    np.testing.assert_array_equal(spectrum_a, spectrum_b)


def test_mc_spectrum_roughly_decreasing_on_average():
    """
    I do not expect MC_k to be completely monotonically decreasing , but on average -- comparing the mean of
    the first few delays to the mean of the last few delays -- short
    delays should be much easier to reconstruct than long delays. (H3 relation)
    """
    net = RandomRNN(n_units=150, gain=1.0, seed=0)
    rng = np.random.default_rng(1)
    spectrum = compute_memory_capacity_spectrum(
        net, k_max=30, n_train=800, n_test=400, n_transient=150, rng=rng
    )
    early_mean = spectrum[:3].mean()   # MC_1, MC_2, MC_3
    late_mean = spectrum[-3:].mean()   # MC_28, MC_29, MC_30
    assert early_mean > late_mean


def test_ridge_alpha_zero_vs_larger_alpha_both_run():
    """
    Sanity check; both a (near)unregularised fit and a more
    strongly regularised fit run without numerical errors, and that
    heavier regularisation does not increase MC 
    """
    net_a = RandomRNN(n_units=80, gain=1.0, seed=0)
    net_b = RandomRNN(n_units=80, gain=1.0, seed=0)
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)

    spectrum_small_alpha = compute_memory_capacity_spectrum(
        net_a, k_max=10, n_train=400, n_test=200, n_transient=150,
        ridge_alpha=1e-8, rng=rng_a,
    )
    spectrum_large_alpha = compute_memory_capacity_spectrum(
        net_b, k_max=10, n_train=400, n_test=200, n_transient=150,
        ridge_alpha=10.0, rng=rng_b,
    )
    total_small = total_memory_capacity(spectrum_small_alpha)
    total_large = total_memory_capacity(spectrum_large_alpha)
    assert total_large <= total_small + 0.5  # generous tolerance


# Tests for experiments.py


def test_sweep_config_total_input_length():
    config = SweepConfig(
        gains=(1.0,), seeds=(0,), n_transient=100, n_train=300, n_test=150,
    )
    assert config.total_input_length == 100 + 300 + 150


def test_sweep_config_roundtrips_through_dict():
    config = SweepConfig(gains=(0.5, 1.0), seeds=(0, 1), n_units=50)
    d = config.to_dict()
    assert d["gains"] == [0.5, 1.0]
    assert d["seeds"] == [0, 1]
    assert d["n_units"] == 50


def test_run_gain_sweep_output_shapes():
    config = SweepConfig(
        gains=(0.5, 1.0, 1.5),
        seeds=(0, 1),
        n_units=50,
        k_max=10,
        n_train=200,
        n_test=100,
        n_transient=100,
    )
    results = run_gain_sweep(config, verbose=False)

    n_seeds, n_gains, k_max = 2, 3, 10
    assert results.mc_total.shape == (n_seeds, n_gains)
    assert results.mc_spectrum.shape == (n_seeds, n_gains, k_max)
    assert results.lambda_driven.shape == (n_seeds, n_gains)
    assert results.perturbation_driven.shape == (n_seeds, n_gains)


def test_run_gain_sweep_reproducible():
    """Running the exact same sweep config twice must give identical results."""
    config = SweepConfig(
        gains=(0.5, 1.5), seeds=(0,), n_units=50, k_max=8,
        n_train=200, n_test=100, n_transient=100,
    )
    results_a = run_gain_sweep(config, verbose=False)
    results_b = run_gain_sweep(config, verbose=False)
    np.testing.assert_array_equal(results_a.mc_total, results_b.mc_total)
    np.testing.assert_array_equal(results_a.lambda_driven, results_b.lambda_driven)
    np.testing.assert_array_equal(
        results_a.perturbation_driven, results_b.perturbation_driven
    )


def test_extreme_gain_low_end_has_low_memory_and_negative_lambda():
    """
    Extreme-parameter sanity check: a very small gain should give both a
    clearly negative driven Lyapunov exponent (ordered regime) AND
    noticeably reduced total memory capacity compared to a near-critical
    gain (an extremely contractive network forgets its input history
    almost immediately)
    """
    config = SweepConfig(
        gains=(0.05, 1.2),
        seeds=(0, 1, 2),
        n_units=150,
        k_max=20,
        n_train=500,
        n_test=250,
        n_transient=150,
    )
    results = run_gain_sweep(config, verbose=False)
    mc_mean = results.mc_total.mean(axis=0)
    lam_mean = results.lambda_driven.mean(axis=0)

    assert lam_mean[0] < 0  # gain=0.05: clearly ordered
    assert mc_mean[0] < mc_mean[1]  # much less memory than the near-critical gain


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
