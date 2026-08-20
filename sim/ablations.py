"""
Ablation sweeps for ATLAS's two key hyperparameters:
  - alpha: EMA decay factor (Eq. 5)
  - W_miss: missed-detection survival penalty (Eq. 4)

Both swept independently at a fixed, representative occlusion length,
holding the other parameter at its paper-reported default, to justify
the chosen operating point rather than presenting it as unmotivated.
"""
import numpy as np
from run_experiment import run_single_trial
from seeding import deterministic_seed

def sweep_alpha(alphas=(0.05, 0.1, 0.2, 0.3, 0.5), occ_len=8, n_trials=200,
                 n_scans=50, occlusion_start=15, tau_novel_ema=-2.689,
                 tau_novel_cum_norm=-2.903):
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
                  tau_novel_ema=-2.689, tau_novel_cum_norm=-2.903):
    """
    W_miss is currently a class attribute constant on the classifier
    classes (CumulativeLLClassifier.W_MISS / EMAClassifier.W_MISS).
    parameter - this is the quickest correct way to sweep it without
    restructuring the classifier API.
    """
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
    print("=== alpha sweep (occ_len=8, W_miss=-8.0) ===")
    alpha_results = sweep_alpha()
    print()
    print("=== W_miss sweep (occ_len=8, alpha=0.1) ===")
    wmiss_results = sweep_w_miss()
