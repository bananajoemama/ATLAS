"""
Ablation sweeps for ATLAS's two key hyperparameters:
  - alpha: EMA decay factor (Eq. 5)
  - W_miss: missed-detection survival penalty (Eq. 4)

Both swept independently at a fixed, representative occlusion length,
holding the other parameter at its paper-reported default, to justify
the chosen operating point rather than presenting it as unmotivated.

tau_novel is NOT hardcoded here. It must match the threshold calibrated
in calibrate_threshold.py (Section 5.3 / Fig. 2), since that's the value
used everywhere else in Section 5. Previously this module carried its
own stale literal (-2.689 / -2.903) left over from before the
calibration script was corrected, so the ablation numbers in Section 5.6
were quietly computed against a different, out-of-date novelty boundary
than the rest of the paper. If tau_novel isn't passed explicitly, this
module now re-derives it the same way run_all.py does, so it can't drift
out of sync again.
"""
import numpy as np
from run_experiment import run_single_trial
from seeding import deterministic_seed


def _calibrated_thresholds():
    """Recompute tau_novel exactly as Section 5.3 / run_all.py do
    (deterministic seeding => same result every call)."""
    from calibrate_threshold import calibrate
    ema_roc, cum_roc = calibrate(n_trials=150, novel_mean=7.0)
    return ema_roc['best_threshold'], cum_roc['best_threshold']


def sweep_alpha(alphas=(0.05, 0.1, 0.2, 0.3, 0.5), occ_len=8, n_trials=200,
                 n_scans=50, occlusion_start=15, tau_novel_ema=None,
                 tau_novel_cum_norm=None):
    if tau_novel_ema is None or tau_novel_cum_norm is None:
        cal_ema, cal_cum = _calibrated_thresholds()
        tau_novel_ema = cal_ema if tau_novel_ema is None else tau_novel_ema
        tau_novel_cum_norm = cal_cum if tau_novel_cum_norm is None else tau_novel_cum_norm
    occ_end = occlusion_start + occ_len
    results = {}
    for a in alphas:
        ema_acc = []
        for trial in range(n_trials):
            seed = deterministic_seed(('alpha_sweep', a, trial))
            res = run_single_trial(
                occ_len, seed, n_scans=n_scans, occlusion_start=occlusion_start,
                alpha=a, tau_novel_ema=tau_novel_ema, tau_novel_cum_norm=tau_novel_cum_norm,
            )
            ema_acc.append(np.mean(res['ema_correct'][occ_end:n_scans]))
        results[a] = {'mean': np.mean(ema_acc), 'std': np.std(ema_acc)}
        print(f"alpha={a:.2f}  post-occlusion ATLAS accuracy = {results[a]['mean']:.3f} "
              f"(+/- {results[a]['std']/np.sqrt(n_trials):.3f})")
    return results


def sweep_w_miss(w_miss_vals=(-4.0, -6.0, -8.0, -12.0, -16.0), occ_len=8,
                  n_trials=200, n_scans=50, occlusion_start=15, alpha=0.1,
                  tau_novel_ema=None, tau_novel_cum_norm=None):
    """
    W_miss is currently a class attribute constant on the classifier
    classes (CumulativeLLClassifier.W_MISS / EMAClassifier.W_MISS).
    We monkey-patch it per sweep value since it's not yet a constructor
    parameter -- this is the quickest correct way to sweep it without
    restructuring the classifier API.
    """
    if tau_novel_ema is None or tau_novel_cum_norm is None:
        cal_ema, cal_cum = _calibrated_thresholds()
        tau_novel_ema = cal_ema if tau_novel_ema is None else tau_novel_ema
        tau_novel_cum_norm = cal_cum if tau_novel_cum_norm is None else tau_novel_cum_norm
    from classifiers import EMAClassifier
    occ_end = occlusion_start + occ_len
    results = {}
    original = EMAClassifier.W_MISS
    try:
        for w in w_miss_vals:
            EMAClassifier.W_MISS = w
            ema_acc = []
            for trial in range(n_trials):
                seed = deterministic_seed(('wmiss_sweep', w, trial))
                res = run_single_trial(
                    occ_len, seed, n_scans=n_scans, occlusion_start=occlusion_start,
                    alpha=alpha, tau_novel_ema=tau_novel_ema, tau_novel_cum_norm=tau_novel_cum_norm,
                )
                ema_acc.append(np.mean(res['ema_correct'][occ_end:n_scans]))
            results[w] = {'mean': np.mean(ema_acc), 'std': np.std(ema_acc)}
            print(f"W_miss={w:.1f}  post-occlusion ATLAS accuracy = {results[w]['mean']:.3f} "
                  f"(+/- {results[w]['std']/np.sqrt(n_trials):.3f})")
    finally:
        EMAClassifier.W_MISS = original
    return results


if __name__ == '__main__':
    tau_ema, tau_cum = _calibrated_thresholds()
    print(f"Calibrated tau_novel: ATLAS={tau_ema:.3f}  cum-norm={tau_cum:.3f}  "
          f"(should match Section 5.3: -2.75 / -3.12)\n")

    print("=== alpha sweep (occ_len=8, W_miss=-8.0) ===")
    alpha_results = sweep_alpha(tau_novel_ema=tau_ema, tau_novel_cum_norm=tau_cum)
    print()
    print("=== W_miss sweep (occ_len=8, alpha=0.1) ===")
    wmiss_results = sweep_w_miss(tau_novel_ema=tau_ema, tau_novel_cum_norm=tau_cum)
