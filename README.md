# EDGE OF CHOAS IN RANDOM RECURRENT NETWORKS

**Research question:** Does the short-term memory capacity of a random recurrent neural network peak at (or near) the critical gain separating ordered from chaotic dynamics?
This project is a small, self-contained replication and extension of a well-established
finding in the reservoir-computing / dynamical-systems literature (Bertschinger &
Natschläger, 2004; Legenstein & Maass, 2007; Boedecker et al., 2012; Matzner, 2017;
Kanamaru, Hensch & Aihara, 2023), situated within the broader "criticality hypothesis"
(Kauffman; Langton; reviewed in Roli, Villani, Filisetti & Serra, 2018)

## Setup
```bash
git clone <this-repo-url>
cd edge-of-chaos-rnn
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # installs the `eoc` package in editable mode
```

## What's implemented so far (Notebook 01)

### The model
A random recurrent network with a single scalar "gain" control parameter `g`:

```
x(t+1) = tanh( g * W @ x(t) + w_in * u(t) )
```

`W` is a fixed random matrix (entries ~ N(0, 1/N)) and `w_in` a fixed random input
vector, both drawn once when the network is created (from a seeded RNG, so results are
fully reproducible). See `src/eoc/network.py` for a heavily-commented, beginner-friendly
walkthrough of exactly why this specific setup was chosen.

### Three ways of measuring how close to criticality a network is 
3 independent criticality measures are implemented since the literature (see Roli et al., 2018, Section 5) explicitly warns that different operational definitions can disagree, a common source of unsound conclusions if left unexamined.

1. **Spectral radius of the Jacobian at the origin** (`spectral_radius_estimate`) —
   analytical, instantaneous, but only valid locally at x=0. Directly generalises the
   eigenvalue-based stability analysis to this discrete-time, N-dimensional system.
2. **Largest Lyapunov exponent** (`estimate_lyapunov_exponent`) — numerical, based on
   the network's actual nonlinear trajectory. The standard measure used throughout the
   reservoir-computing literature. Supports both *autonomous* (input-free) and *driven*
   (input-driven) measurement, since these give meaningfully different answers (input is known to suppress chaos).
3. **One-step perturbation-spreading factor** (`perturbation_spreading_factor`) — a
   simple, cheap, "Derrida-style" measure directly analogous to the sensitivity/damage-
   spreading measures used for Random Boolean Networks in the criticality literature.


### Linear memory capacity (Jaeger, 2002)

`memory.py` implements `compute_memory_capacity_spectrum`: for each delay k = 1..k_max,
fits a ridge-regularised linear readout of the network state to reconstruct the input
u(t-k), evaluated on **held-out test data** (never used for fitting — important for a
trustworthy estimate, especially as dynamics become richer near/past criticality).
`total_memory_capacity` sums the spectrum into a single MC(g) value.

### Tying it together: `experiments.py`

`run_gain_sweep` sweeps the gain g, and — for every (seed, gain) combination — measures
memory capacity, the driven Lyapunov exponent, and the driven perturbation-spreading
factor **using the exact same shared random input sequence**, removing "different random
input" as a possible confound when comparing the three curves. `run_network_size_sweep`
repeats the memory-capacity measurement (only, for speed) across several network sizes,
for the H4 robustness check. Every `SweepConfig` records all parameters explicitly, and
`save_sweep_results` / `load_sweep_results` persist results to disk together with the
exact configuration that produced them.

## Validated results (notebook 02, N=300, 8 seeds unless noted)

| Hypothesis | Finding |
| H1: MC(g) rises, peaks, falls | Confirmed — peak at g ≈ 1.74, MC_peak ≈ 7.5 |
| H2: peak at/just before driven g_c | Confirmed — peak ≈0.05–0.07 gain units *below* both driven-criticality estimates (g_c ≈ 1.79–1.81) |
| H3: short-delay memory outlives long-delay memory as g increases | Confirmed — clear storage-vs-transfer signature |
| H4: robust across seeds & network size | Confirmed — consistent peak location for N ∈ {100, 200, 300, 500} |



## Reproducibility notes
- Every `RandomRNN` is created from an explicit integer `seed`; identical seed +
  `n_units` always gives identical weights.
- All stochastic estimators (`estimate_lyapunov_exponent`,
  `perturbation_spreading_factor`) accept an explicit `numpy.random.Generator`; pass
  your own seeded generator for full reproducibility of any specific result.
- All sweeps in the notebook are repeated across multiple random seeds, and reported as
  mean ± SD, per the "repeat experiments" guideline.
- A warm-up/transient period is always discarded before measurement (see
  `n_transient` arguments) to avoid biasing results with the arbitrary all-zero initial
  condition.
- `tests/test_infrastructure.py` encodes the specific theoretical predictions checked
  in the notebook as automated regression tests — run these after any code change.



## References

- Sompolinsky, H., Crisanti, A., & Sommers, H. J. (1988). Chaos in random neural
  networks. *Physical Review Letters*, 61(3), 259.
- Rajan, K., Abbott, L. F., & Sompolinsky, H. (2010). Stimulus-dependent suppression of
  chaos in recurrent neural networks. *Physical Review E*, 82(1), 011903.
- Bertschinger, N., & Natschläger, T. (2004). Real-time computation at the edge of
  chaos in recurrent neural networks. *Neural Computation*, 16(7), 1413–1436.
- Legenstein, R., & Maass, W. (2007). Edge of chaos and prediction of computational
  performance for neural circuit models. *Neural Networks*, 20(3), 323–334.
- Boedecker, J., Obst, O., Lizier, J. T., Mayer, N. M., & Asada, M. (2012). Information
  processing in echo state networks at the edge of chaos. *Theory in Biosciences*,
  131(3), 205–213.
- Matzner, F. (2017). Neuroevolution on the edge of chaos. *GECCO 2017*, 465–472.
- Kanamaru, T., Hensch, T. K., & Aihara, K. (2023). Maximal memory capacity near the
  edge of chaos in balanced cortical E-I networks. *Neural Computation*, 35(8),
  1430–1462.
- Roli, A., Villani, M., Filisetti, A., & Serra, R. (2018). Dynamical criticality:
  overview and open questions. *Journal of Systems Science and Complexity*, 31,
  647–663.
- Jaeger, H. (2002). Short term memory in echo state networks. GMD Report 152.


























