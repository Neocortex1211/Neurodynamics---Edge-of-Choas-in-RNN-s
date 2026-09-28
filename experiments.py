"""

Coordinates the actual research-question experiments- sweeping the gain g
while measuring memory
capacity and the criticality estimators from criticality.py, so that they
can be directly compared against each other

It's not new scientific measurement of its own
-- everything it computes is a call to a function already defined and
already tested in network.py, criticality.py, or memory.py

-----------
Why the same driving input matters;
Both the driven Lyapunov exponent (criticality.py) and memory capacity
(memory.py) are measured by driving a network with a long random input
sequence. If it generated a novel random input sequence for every single
measurement, then any difference between, MC(g=1.0) and
MC(g=1.2) could in principle, be partly due to the two measurements
having used two different random input sequences
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

from .network import RandomRNN
from .criticality import estimate_lyapunov_exponent, perturbation_spreading_factor
from .memory import compute_memory_capacity_spectrum, total_memory_capacity, generate_input


# Configuration


@dataclass
class SweepConfig:
    """
    All parameters of a gain sweep, bundled into one object
    """

    gains: tuple 
    seeds: tuple
    n_units: int = 300
    k_max: int = 30
    n_train: int = 800
    n_test: int = 400
    n_transient: int = 300
    ridge_alpha: float = 1e-6
    input_distribution: str = "normal"

    @property
    def total_input_length(self) -> int:
        """Length of the single shared input sequence generated per seed."""
        return self.n_transient + self.n_train + self.n_test

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gains"] = list(self.gains)
        d["seeds"] = list(self.seeds)
        return d


# Main sweep: gain vs. (memory capacity, criticality measures)


@dataclass
class SweepResults:
    """
    Container for the results of `run_gain_sweep`.

    All arrays are indexed as [seed_index, gain_index], except
    `mc_spectrum` which additionally has a trailing delay-index axis
    [seed_index, gain_index, k_index].
    """

    config: SweepConfig
    mc_total: np.ndarray            # shape (n_seeds, n_gains)
    mc_spectrum: np.ndarray         # shape (n_seeds, n_gains, k_max)
    lambda_driven: np.ndarray       # shape (n_seeds, n_gains)
    perturbation_driven: np.ndarray  # shape (n_seeds, n_gains)
    runtime_seconds: float = 0.0


def run_gain_sweep(config: SweepConfig, verbose: bool = True) -> SweepResults:
    """
    Runs here the full gain sweep described by `config`.

    For every seed in config.seeds:
        1. Generates 1 shared input sequence of length
           `config.total_input_length`
        2. For every gain in `config.gains`:
            a. Build a fresh network with that (seed, gain)
            b. Measures memory capacity (full spectrum + total).
            c. Measures the driven Lyapunov exponent, using the same
               shared input.
            d. Measures the driven perturbation-spreading factor, again
               using the same shared input.

    Parameters
    ----------
    config : SweepConfig
    verbose : bool, default True
        If True, it prints progress (one line per seed) and a final timing
        summary.

    Returns
    -------
    SweepResults
        All measured arrays
    """
    t_start = time.time()

    gains = np.asarray(config.gains, dtype=float)
    seeds = list(config.seeds)
    n_seeds, n_gains = len(seeds), len(gains)

    mc_total = np.empty((n_seeds, n_gains))
    mc_spectrum = np.empty((n_seeds, n_gains, config.k_max))
    lambda_driven = np.empty((n_seeds, n_gains))
    perturbation_driven = np.empty((n_seeds, n_gains))

    n_lambda_steps = config.n_train + config.n_test  # matches MC's post-transient length
    n_psf_trials = config.n_train + config.n_test

    for si, seed in enumerate(seeds):
        # One shared input sequence for this whole repetition (all gains)
        input_rng = np.random.default_rng(seed)
        shared_input = generate_input(
            config.total_input_length, config.input_distribution, input_rng
        )
        lambda_rng = np.random.default_rng(seed + 1_000_003)
        psf_rng = np.random.default_rng(seed + 2_000_003)

        for gi, gain in enumerate(gains):
            # Memory capacity 
            net_mc = RandomRNN(n_units=config.n_units, gain=float(gain), seed=seed)
            spectrum = compute_memory_capacity_spectrum(
                net_mc,
                k_max=config.k_max,
                n_train=config.n_train,
                n_test=config.n_test,
                n_transient=config.n_transient,
                ridge_alpha=config.ridge_alpha,
                driving_input=shared_input,
            )
            mc_spectrum[si, gi] = spectrum
            mc_total[si, gi] = total_memory_capacity(spectrum)

            # Driven Lyapunov exponent, same shared input
            net_lambda = RandomRNN(n_units=config.n_units, gain=float(gain), seed=seed)
            lambda_driven[si, gi] = estimate_lyapunov_exponent(
                net_lambda,
                n_steps=n_lambda_steps,
                n_transient=config.n_transient,
                driving_input=shared_input,
                rng=lambda_rng,
            )

            #  Driven perturbation-spreading factor, same shared input
            net_psf = RandomRNN(n_units=config.n_units, gain=float(gain), seed=seed)
            perturbation_driven[si, gi] = perturbation_spreading_factor(
                net_psf,
                n_trials=n_psf_trials,
                n_transient=config.n_transient,
                driving_input=shared_input,
                rng=psf_rng,
            )

        if verbose:
            elapsed = time.time() - t_start
            print(
                f"  seed {seed} ({si + 1}/{n_seeds}) done "
                f"[{elapsed:6.1f}s elapsed]"
            )

    runtime = time.time() - t_start
    if verbose:
        print(f"Sweep finished in {runtime:.1f}s "
              f"({n_seeds} seeds x {n_gains} gains = "
              f"{n_seeds * n_gains} network evaluations).")

    return SweepResults(
        config=config,
        mc_total=mc_total,
        mc_spectrum=mc_spectrum,
        lambda_driven=lambda_driven,
        perturbation_driven=perturbation_driven,
        runtime_seconds=runtime,
    )


# 2nd sweep: robustness across network size N (for H4)


@dataclass
class SizeSweepResults:
    """Container for the results of `run_network_size_sweep`."""

    config: SweepConfig  # `n_units` in this config is a placeholder/unused
    n_units_list: tuple
    mc_total: np.ndarray  # shape (len(n_units_list), n_seeds, n_gains)


def run_network_size_sweep(
    n_units_list: Sequence[int],
    gains: Sequence[float],
    seeds: Sequence[int],
    k_max: int = 30,
    n_train: int = 800,
    n_test: int = 400,
    n_transient: int = 300,
    ridge_alpha: float = 1e-6,
    input_distribution: str = "normal",
    verbose: bool = True,
) -> SizeSweepResults:
    """
    Repeat sthe gain sweep 

    Parameters mirror `SweepConfig` / `run_gain_sweep`, except `n_units`
    is replaced by `n_units_list`, the set of sizes to compare

    Returns
    SizeSweepResults
    """
    t_start = time.time()
    gains = np.asarray(gains, dtype=float)
    seeds = list(seeds)
    mc_total = np.empty((len(n_units_list), len(seeds), len(gains)))

    for ni, n_units in enumerate(n_units_list):
        for si, seed in enumerate(seeds):
            input_rng = np.random.default_rng(seed)
            total_len = n_transient + n_train + n_test
            shared_input = generate_input(total_len, input_distribution, input_rng)

            for gi, gain in enumerate(gains):
                net = RandomRNN(n_units=n_units, gain=float(gain), seed=seed)
                spectrum = compute_memory_capacity_spectrum(
                    net,
                    k_max=k_max,
                    n_train=n_train,
                    n_test=n_test,
                    n_transient=n_transient,
                    ridge_alpha=ridge_alpha,
                    driving_input=shared_input,
                )
                mc_total[ni, si, gi] = total_memory_capacity(spectrum)

        if verbose:
            elapsed = time.time() - t_start
            print(f"  N={n_units} done [{elapsed:6.1f}s elapsed]")

    if verbose:
        print(f"Network-size sweep finished in {time.time() - t_start:.1f}s.")

    placeholder_config = SweepConfig(
        gains=tuple(gains.tolist()), seeds=tuple(seeds),
        n_units=-1, k_max=k_max, n_train=n_train, n_test=n_test,
        n_transient=n_transient, ridge_alpha=ridge_alpha,
        input_distribution=input_distribution,
    )
    return SizeSweepResults(
        config=placeholder_config,
        n_units_list=tuple(n_units_list),
        mc_total=mc_total,
    )


# Saving / loading results


def save_sweep_results(results: SweepResults, path: str) -> None:
    np.savez(
        path,
        mc_total=results.mc_total,
        mc_spectrum=results.mc_spectrum,
        lambda_driven=results.lambda_driven,
        perturbation_driven=results.perturbation_driven,
        runtime_seconds=results.runtime_seconds,
        config_json=json.dumps(results.config.to_dict()),
    )


def load_sweep_results(path: str) -> SweepResults:
    data = np.load(path, allow_pickle=False)
    config_dict = json.loads(str(data["config_json"]))
    config_dict["gains"] = tuple(config_dict["gains"])
    config_dict["seeds"] = tuple(config_dict["seeds"])
    config = SweepConfig(**config_dict)
    return SweepResults(
        config=config,
        mc_total=data["mc_total"],
        mc_spectrum=data["mc_spectrum"],
        lambda_driven=data["lambda_driven"],
        perturbation_driven=data["perturbation_driven"],
        runtime_seconds=float(data["runtime_seconds"]),
    )

