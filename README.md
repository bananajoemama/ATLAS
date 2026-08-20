# ATLAS: Anomaly Thresholding and Likelihood Averaging System

Code and paper source for **ATLAS**, a sequence-scoring classification
layer built on top of JIPDA (Joint Integrated Probabilistic Data
Association, Mušicki & Evans 2002) that uses an Exponential Moving
Average (EMA) of instantaneous log-likelihoods to avoid the "infinite
memory ruin" failure mode of standard cumulative log-likelihood
classifiers under occlusion.

**ATLAS is not an alternative to JIPDA.** It has no independent spatial
association mechanism; it depends on JIPDA (or an equivalent data
association algorithm) to supply track existence and measurement
association, and adds a classification/novelty-detection layer on top.
See Section 5 of the paper for the precise scope of what is and isn't
being compared in each experiment.

## Repository structure

- `sim/` — Python simulation code. See `sim/README.md` for module-level
  documentation.

## Reproducing all paper results

```bash
cd sim
pip install -r requirements.txt
python3 run_all.py
```

This reproduces every number, table, and figure in Section 5 of the
paper from a clean checkout in under 5 minutes, using deterministic
seeding (see `sim/seeding.py`) so output is bit-identical across
separate runs and machines given the pinned dependency versions.

## Known limitations

- τ_novel, α, and W_miss are calibrated to the specific clutter density,
  sensor noise, and class separation of the scenario simulated here —
  they are not universal constants and should be recalibrated (via the
  ROC procedure in `sim/calibrate_threshold.py`) for any other
  deployment scenario.
- The ablation study (Section 5.6) sweeps α and W_miss independently at
  a single occlusion length; their joint interaction, and their effect
  on clean-condition (non-occluded) performance, has not been evaluated.

## License

MIT (see LICENSE).
