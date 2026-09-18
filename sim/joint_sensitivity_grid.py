"""
Compact joint (alpha, W_miss) sensitivity grid.

The independent sweeps in ablations.py (Fig. 4 / Section V-F) vary alpha
and W_miss one at a time, holding the other at its default and tau_novel
at its Section V-C calibrated value. That does not characterize their
JOINT effect, which is what the reviewer asked for ("a more systematic
method for parameter selection or joint optimization").

This module is explicitly NOT a full multi-objective optimization
(that remains Future Work, Section VIII-3). It is a compact,
descriptive characterization: a small grid over (alpha, W_miss), at a
FIXED tau_novel per column of alpha (recalibrated via the same ROC
procedure as calibrate_threshold.py, since threshold and alpha
interact -- see the "Sensitivity and Ablation Analysis" text), reporting
two quantities per cell:
  - post-occlusion accuracy at a fixed, representative occlusion length
    (L=8, matching Fig. 4's operating point)
  - clean-condition discrimination (novelty ROC AUC)
so the accuracy/discrimination trade-off across the grid is visible
directly, rather than only assumed from the two independent 1-D sweeps.

Uses the same scalar feature channel as ablations.py / Fig. 4, for
direct comparability with the existing sensitivity results.
"""
import numpy as np
from run_experiment import run_single_trial, time_to_recover
from calibrate_threshold import calibrate
from classifiers import EMAClassifier
from seeding import deterministic_seed

ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.5)
W_MISS_VALS = (-4.0, -6.0, -8.0, -12.0, -16.0)


def joint_grid(alphas=ALPHAS, w_miss_vals=W_MISS_VALS, occ_len=8,
                n_trials_acc=100, n_trials_auc=100, n_scans=50,
                occlusion_start=15, novel_mean=7.0):
    """
    For each (alpha, W_miss) cell:
      1. Monkey-patch EMAClassifier.W_MISS = w (same technique as
         ablations.sweep_w_miss).
      2. Recalibrate tau_novel for THIS (alpha, W_miss) pair via ROC
         (n_trials_auc known + n_trials_auc novel trials), since the
         score scale depends on both parameters, not just alpha.
      3. Run the occlusion-recovery sweep at L=occ_len with the
         recalibrated threshold (n_trials_acc trials) to get
         post-occlusion accuracy.
    Returns: dict {(alpha, w_miss): {'auc': ..., 'tau_novel': ...,
                                      'post_acc_mean': ..., 'post_acc_std': ...}}
    """
    occ_end = occlusion_start + occ_len
    original_w_miss = EMAClassifier.W_MISS
    results = {}
    try:
        for a in alphas:
            for w in w_miss_vals:
                EMAClassifier.W_MISS = w
                # Step 1: calibrate tau_novel for this (alpha, w_miss) pair.
                ema_roc, _ = calibrate(n_trials=n_trials_auc, alpha=a, novel_mean=novel_mean)
                tau = ema_roc['best_threshold']
                auc = ema_roc['auc']

                # Step 2: post-occlusion accuracy at L=occ_len under this threshold.
                accs = []
                for trial in range(n_trials_acc):
                    seed = deterministic_seed(('joint_grid', a, w, trial))
                    res = run_single_trial(
                        occ_len, seed, n_scans=n_scans, occlusion_start=occlusion_start,
                        alpha=a, tau_novel_ema=tau, tau_novel_cum_norm=-1.0,
                    )
                    accs.append(np.mean(res['ema_correct'][occ_end:n_scans]))

                results[(a, w)] = {
                    'auc': auc, 'tau_novel': tau,
                    'post_acc_mean': float(np.mean(accs)),
                    'post_acc_std': float(np.std(accs)),
                }
                print(f"alpha={a:.2f}  W_miss={w:6.1f}  tau_novel={tau:6.2f}  "
                      f"AUC={auc:.3f}  post_acc(L={occ_len})={np.mean(accs):.3f}")
    finally:
        EMAClassifier.W_MISS = original_w_miss
    return results


if __name__ == '__main__':
    import json
    results = joint_grid()
    serializable = {f"{a}|{w}": v for (a, w), v in results.items()}
    json.dump(serializable, open('summary_joint_grid.json', 'w'), indent=2)
