# sim/ — ATLAS simulation code

See the top-level README for how ATLAS relates to JIPDA and what each
experiment does and doesn't compare.

## Quick start

```bash
pip install -r requirements.txt
python3 run_all.py
```

Reproduces every number, table, and figure in Section 5 of the paper,
end to end, in under 5 minutes. See `run_all.py`'s docstring for details.

## Module map

- `motion.py` — constant-velocity target model + Kalman filter (Eq. 12-17
  of Mušicki & Evans 2002), including the PDA-weighted soft-combination
  update (`KalmanFilter.pda_update`) used for all track state updates.
- `clutter.py` — Poisson clutter generator, matching the original paper's
  region size and two high-density patches.
- `jipda.py` — core JIPDA math (Eq. 1-11): gating with a proper elliptical
  gate area, clustering, joint event enumeration, existence probability
  (including the Markov Chain One survival prediction, Eq. 17), and beta
  association weights.
- `classifiers.py` — `CumulativeLLClassifier` (baseline) and
  `EMAClassifier` (ATLAS, Eq. 3-6), including the age-normalized
  threshold variant of the baseline used for the fairer comparison.
- `scenario.py` — single-target occlusion scenario (used by
  `run_experiment.py`'s Monte Carlo trials).
- `run_experiment.py` — occlusion-length sweep (Section 5.4).
- `calibrate_threshold.py` — ROC-based τ_novel calibration (Section 5.3).
- `crossing_experiment.py` — two-target crossing scenario (Section 5.5),
  reproducing the Mušicki & Evans Table 1 format.
- `ablations.py` — α and W_miss sensitivity sweeps (Section 5.6).
- `seeding.py` — deterministic seed generation. Python's built-in
  `hash()` is randomized per-process for security reasons and is NOT
  suitable for reproducible Monte Carlo seeding; this module uses
  `hashlib` instead, which is stable across processes and machines.
- `run_all.py` — single driver reproducing everything above in order.

## Known open items

See "Known limitations" in the top-level README.
