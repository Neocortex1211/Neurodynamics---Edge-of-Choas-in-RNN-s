"""
criticality.py
===============

Three independent ways of asking the same question: for this network,
with this gain g, is the dynamics ORDERED, CHAOTIC, or right at the
CRITICAL boundary between the two?
---------------------------------------------------------

  1. spectral_radius_estimate()
        An ANALYTICAL, purely LINEAR measure. It asks: if I nudge the
        network only infinitesimally away from its resting state at the
        origin, does that nudge grow or shrink, according to the
        network's LINEARISED dynamics?

  2. estimate_lyapunov_exponent()
        A NUMERICAL, fully NONLINEAR measure. It actually simulates the
        network (potentially while being driven by real input, exactly
        as it will be used in the rest of the project) and measures how
        fast two almost-identical trajectories drift apart over time.
        It's is the gold standard measure used throughout the
        literature you read (Sompolinsky et al. 1988; Bertschinger &
        Natschläger 2004; Legenstein & Maass 2007; Boedecker et al. 2012;
        Matzner 2017). it captures the REAL, nonlinear dynamics and it is measured along the network's actual, typical
        trajectory

  3. perturbation_spreading_factor()
        A NUMERICAL, model-free "damage spreading" measure
        inspired by the Derrida parameter used for Random Boolean
        Networks in the Roli et al. review (Section 2, and the
        discussion of Ribeiro et al., Krawitz & Shmulevich, etc.). It
        asks a intuitive question than the Lyapunov
        exponent: if I perturb the network's state a little bit
        right now, how much bigger or smaller is the resulting
        difference after exactly 1 step?

All three measures should roughly AGREE on where the critical gain g_c
lies
"""

from __future__ import annotations

import numpy as np

from .network import RandomRNN


# ==========================================================================
# 1. Spectral radius estimate (analytical, linear, instantaneous)
# ==========================================================================


def spectral_radius_estimate(network: RandomRNN) -> float:
    """
    Compute the spectral radius of the network's Jacobian at the origin.

    The "spectral radius" of a matrix is simply the largest absolute
    value among all of its eigenvalues.

    Interpretation of the returned value ``rho``:
        rho < 1  ->  the origin is a LOCALLY STABLE fixed point.
                     Small perturbations near x=0 shrink over time.
                     ("ordered", at least locally)
        rho > 1  ->  the origin is a LOCALLY UNSTABLE fixed point.
                     Small perturbations near x=0 grow over time.
                     ("unstable" -- a necessary, though not on its own
                     sufficient, condition for chaotic behaviour once
                     the nonlinearity of tanh is taken into account)
        rho == 1 ->  the critical point, right at the boundary.
    """
    J = network.jacobian_at_origin()
    eigenvalues = np.linalg.eigvals(J)
    spectral_radius = np.max(np.abs(eigenvalues))
    return float(spectral_radius)


# ==========================================================================
# 2. Numerical Lyapunov exponent estimate (nonlinear, trajectory-based)
# ==========================================================================


def estimate_lyapunov_exponent(
    network: RandomRNN,
    n_steps: int = 2000,
    n_transient: int = 200,
    perturbation_size: float = 1e-8,
    driving_input: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Numerically estimates the largest Lyapunov exponent of the network's
    dynamics, using the classic two trajectories + periodic
    renormalisation method

    --------------------------------------------------------------------
    The idea, step by step
    --------------------------------------------------------------------
    1. Runs the network for `n_transient` steps first, and throws this
       part of the trajectory away. This is the "warm-up" / transient
       period

    2. After the warm-up, makes an exact COPY of the network's current
       state, and nudges the copy by a TINY random vector of length
       `perturbation_size`. Now we have two trajectories (referenced & perturbed)

    3. Advances both trajectories by exactly one time step, using the
       exact same input u(t) for both 

    4. Measures the new distance between the two trajectories. If the
       dynamics are "chaotic" in this region, this distance will have
       grown compared to `perturbation_size`; if "ordered", it must have
       shrunk.

    5. Records how much the distance grew or shrank (as a log-ratio --
       see below for why), then RESCALEs the perturbed trajectory back
       down so the two trajectories are again exactly `perturbation_size`
       apart

    6. Repeats steps 3-5 for `n_steps` iterations, and average the
       log-ratios from step 4. This average, " the (largest)
       Lyapunov exponent", should tell the typical EXPONENTIAL rate at which
       two nearby trajectories separate, per time step

           lambda_max > 0  ->  nearby trajectories separate exponentially
                               on average -> CHAOTIC dynamics.
           lambda_max < 0  ->  nearby trajectories converge exponentially
                               on average -> ORDERED dynamics.
           lambda_max = 0  ->  the CRITICAL point.

    Parameters
    ----------
    network : RandomRNN
    n_steps : int, default 2000
        the cost of more computation.
    n_transient : int, default 200
        Number of warm-up steps to discard before starting measurement
    perturbation_size : float, default 1e-8
        The (tiny) initial distance between the two trajectories. Must
        be small enough that the LINEARISED dynamics are a good
        approximation over a single time step 
    driving_input : np.ndarray of shape (n_transient + n_steps,), optional
        The external input sequence u(t) used to drive the network
        during BOTH the transient and the measurement.
    rng : np.random.Generator

    Returns
    -------
    float
        The estimated largest Lyapunov exponent, lambda_max, in units of
        "nats per time step" (since we use the natural logarithm).
    """
    if rng is None:
        rng = np.random.default_rng()

    total_steps = n_transient + n_steps
    if driving_input is None:
        driving_input = rng.normal(size=total_steps)
    else:
        driving_input = np.asarray(driving_input, dtype=float)
        if len(driving_input) != total_steps:
            raise ValueError(
                "driving_input must have length n_transient + n_steps "
                f"= {total_steps}, got {len(driving_input)}."
            )

    # --- Step 1: warm-up / transient period 
    for t in range(n_transient):
        network.step(driving_input[t])

    # --- Step 2: the perturbed "shadow" trajectory 
    reference_state = network.state.copy()

    # A random direction in n_units-dimensional space, normalised to
    # length 1, then scaled to the desired (tiny) perturbation size.
    direction = rng.normal(size=network.n_units)
    direction /= np.linalg.norm(direction)
    perturbed_state = reference_state + perturbation_size * direction

    log_growth_factors = np.empty(n_steps)

    W, g, w_in = network.W, network.gain, network.w_in

    for i in range(n_steps):
        u_t = driving_input[n_transient + i]

        # --- Step 3: advances both trajectories by one step, same input
        reference_state = np.tanh(g * (W @ reference_state) + w_in * u_t)
        perturbed_state = np.tanh(g * (W @ perturbed_state) + w_in * u_t)

        # --- Step 4: measures how far apart they now are -------------------
        separation = np.linalg.norm(perturbed_state - reference_state)
        separation = max(separation, 1e-300)

        log_growth_factors[i] = np.log(separation / perturbation_size)

        # --- Step 5: renormalises the perturbed trajectory
        diff = perturbed_state - reference_state
        diff = diff / np.linalg.norm(diff) * perturbation_size
        perturbed_state = reference_state + diff

    network.state = reference_state

    # --- Step 6: averages the log growth factors
    lambda_max = float(np.mean(log_growth_factors))
    return lambda_max


# ==========================================================================
# 3. Perturbation-spreading factor (numerical, "Derrida-style", one-step)
# ==========================================================================


def perturbation_spreading_factor(
    network: RandomRNN,
    n_trials: int = 500,
    n_transient: int = 200,
    perturbation_size: float = 1e-6,
    driving_input: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Estimates the average ONE-STEP perturbation expansion factor

    This is probably the simplest possible dynamic measure of
    order/chaos: there is no iterative
    renormalisation --  just perturbs, takes 1 step, measures, and
    repeats with a fresh perturbation many times.

    Interpretation of the returned value ``factor``:
        factor < 1  ->  perturbations shrink after one step, ON AVERAGE,
                        across many different points along the
                        trajectory -> ORDERED dynamics.
        factor > 1  ->  perturbations grow after one step, ON AVERAGE
                        -> CHAOTIC dynamics.
        factor = 1  ->  the CRITICAL point.


    Parameters
    ----------
    network : RandomRNN
    n_trials : int, default 500
    n_transient : int, default 200
        Number of warm-up steps to run before starting to collect
        trials
    perturbation_size : float, default 1e-6
        The size of each individual random perturbation.
    driving_input : np.ndarray of shape (n_transient + n_trials,)
        Input sequence driving the network while trials are collected
    rng : np.random.Generator

    """
    if rng is None:
        rng = np.random.default_rng()

    total_steps = n_transient + n_trials
    if driving_input is None:
        driving_input = rng.normal(size=total_steps)
    else:
        driving_input = np.asarray(driving_input, dtype=float)
        if len(driving_input) != total_steps:
            raise ValueError(
                "driving_input must have length n_transient + n_trials "
                f"= {total_steps}, got {len(driving_input)}."
            )

    W, g, w_in = network.W, network.gain, network.w_in

    # --- Warm-up ----------------------------------------------------------
    for t in range(n_transient):
        network.step(driving_input[t])

    expansion_factors = np.empty(n_trials)

    for i in range(n_trials):
        u_t = driving_input[n_transient + i]
        reference_state = network.state

        # Fresh random perturbation direction for every trial.
        direction = rng.normal(size=network.n_units)
        direction /= np.linalg.norm(direction)
        perturbed_state = reference_state + perturbation_size * direction

        reference_next = np.tanh(g * (W @ reference_state) + w_in * u_t)
        perturbed_next = np.tanh(g * (W @ perturbed_state) + w_in * u_t)

        separation_after = np.linalg.norm(perturbed_next - reference_next)
        expansion_factors[i] = separation_after / perturbation_size

        # Advancse the actual network state by one step (only using the unperturbed/reference update)
        network.state = reference_next

    return float(np.mean(expansion_factors))

