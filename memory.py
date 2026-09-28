"""

Measures here a network's linear short-term MEMORY CAPACITY (MC): how well can a
simple linear readout, looking only at the network's current state x(t),
reconstruct the external input u(t-k) that was fed in k time steps ago

-------------------------------------------------------------------------
CONTEXT 

For each delay k = 1, 2, 3, ..., :
  1. Fit a linear readout  that tries to predict u(n-k) from x(n), using
     a training portion of the data.
  2. Checking how well that readout reconstructs u(n-k) on a
     separate test portion of the data 
  3. Summary of how well as MC_k, the squared correlation between the
     readout's predictions and the true delayed input, on the test data.
     MC_k = 1 means perfect linear reconstruction; MC_k = 0 means the
     readout does no better than guessing the mean

Summing MC_k over many delays k gives the total memory capacity, MC,
which is the main quantity of interest

A train/test split, just fitting and checking on the
same data, should be necessary.
A linear readout has N free weights (one per neuron) plus one bias term.
If it evaluates it on the same data it was fit on, and N is not much
smaller than the number of time points it have, the readout can partly
just memorise the particular training sequence -- inflating MC estimate with something that
would not generalise to new data. 

Ridge Regression here, because:
The network states can be strongly correlated with each other, which makes the readout-fitting
problem numerically ill-conditioned- A tiny
amount of L2 ("ridge") regularisation stabilises the fit without
 changing the result when the problem is already
well-conditioned
"""

from __future__ import annotations

import numpy as np

from .network import RandomRNN


def generate_input(
    length: int, distribution: str, rng: np.random.Generator
) -> np.ndarray:
    
    if distribution == "normal":
        return rng.normal(loc=0.0, scale=1.0, size=length)
    elif distribution == "uniform":
        return rng.uniform(low=-1.0, high=1.0, size=length)
    else:
        raise ValueError(
            f"Unknown input distribution '{distribution}'. "
            "Use 'normal' or 'uniform'."
        )


def compute_memory_capacity_spectrum(
    network: RandomRNN,
    k_max: int,
    n_train: int = 800,
    n_test: int = 400,
    n_transient: int = 300,
    ridge_alpha: float = 1e-6,
    input_distribution: str = "normal",
    driving_input: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Computes the memory-capacity SPECTRUM MC_k, for delays k = 1..k_max, for
    a single network.

    
    Parameters,
    network : RandomRNN
    k_max : int
        Largest delay to evaluate. Must satisfy k_max <= n_transient (we
        need at least k_max steps of "history" available before the
        first evaluated state 
    n_train : int, 800
        Number of (state, target) pairs used to fit each delay's readout
    n_test : int,  400
        Number of (state, target) pairs used to evaluate each delay's
        readout
    n_transient : int, default 300
        Warm-up steps discarded before collecting any pairs
    ridge_alpha : float, default 1e-6
        L2 regularisation strength for the ridge regression
    input_distribution : {"normal", "uniform"}, default "normal"
        Statistics of the driving input
    driving_input : np.ndarray of shape (n_transient + n_train + n_test,)
        Explicit input sequence to drive the network with. 
    rng : np.random.Generator (generates driving input)

    Returns
    -------
    np.ndarray of shape (k_max,)
        MC_k for k = 1, 2, ..., k_max (mc_spectrum[0] is MC_1, etc.).
        Each entry lies (up to small finite-sample noise) in [0, 1]
    """
    if k_max > n_transient:
        raise ValueError(
            f"k_max ({k_max}) must not exceed n_transient ({n_transient}): "
            "need at least k_max steps of input history available "
            "before the first state we evaluate, so that every target "
            "u(n-k) refers to an input value that actually exists."
        )

    total_len = n_transient + n_train + n_test
    if driving_input is None:
        if rng is None:
            rng = np.random.default_rng()
        driving_input = generate_input(total_len, input_distribution, rng)
    else:
        driving_input = np.asarray(driving_input, dtype=float)
        if len(driving_input) != total_len:
            raise ValueError(
                "driving_input must have length "
                f"n_transient + n_train + n_test = {total_len}, "
                f"got {len(driving_input)}."
            )

    network.reset_state()
    states = network.run(driving_input)  # shape 

    #evaluation window
    eval_start = n_transient
    eval_end = n_transient + n_train + n_test
    X = states[eval_start:eval_end]  # shape

    ones_column = np.ones((X.shape[0], 1))
    X_aug = np.hstack([X, ones_column])  # shape (n_train + n_test, n_units + 1)

    X_train, X_test = X_aug[:n_train], X_aug[n_train:]

    #  Precomputing the ridge solve matrix, and reuse it for every
    # delay k. 
    n_features = X_train.shape[1]
    gram = X_train.T @ X_train
    # Regulariseing all but the bias term
    reg_matrix = ridge_alpha * np.eye(n_features)
    reg_matrix[-1, -1] = 0.0  
    solve_matrix = gram + reg_matrix  # shape

    mc_spectrum = np.empty(k_max)

    for k in range(1, k_max + 1):
        # Target: u(n - k), for every n in the evaluation window
        target_start = eval_start - k
        target_end = eval_end - k
        y = driving_input[target_start:target_end]
        y_train, y_test = y[:n_train], y[n_train:]

        # Solvingg the ridge-regularised normal equations for this delay's
        # readout weights.
        rhs = X_train.T @ y_train
        weights = np.linalg.solve(solve_matrix, rhs)

        y_pred_test = X_test @ weights

        mc_spectrum[k - 1] = _squared_correlation(y_test, y_pred_test)

    return mc_spectrum


def _squared_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Squared Pearson correlation coefficient between y_true and y_pred,
    i.e. Jaeger's (2002) definition of MC_k:

        MC_k = Cov(y_true, y_pred)^2 / ( Var(y_true) * Var(y_pred) )
    """
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true < 1e-12 or std_pred < 1e-12:
        return 0.0
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation**2)


def total_memory_capacity(mc_spectrum: np.ndarray) -> float:
    """
    Sum the memory-capacity spectrum into a single total-memory-capacity
    number: MC = sum_k MC_k.
    """
    return float(np.sum(mc_spectrum))
