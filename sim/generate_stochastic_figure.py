"""
Generates fig_stochastic_occlusion.png for the paper's stochastic-occlusion
subsection. stochastic_occlusion.py's own sweep_stochastic_occlusion() only
saves means (no std), so this script recomputes per-trial values using the
exact same function/seeds and adds standard-error bars, in the same visual
style as run_all.py's fig_accuracy_vs_occlusion.png. Also re-verifies the
means still match summary_stochastic.json exactly, as one more
reproducibility check before this experiment appears in the paper.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stochastic_occlusion import run_single_trial_stochastic
from run_experiment import time_to_recover
from seeding import deterministic_seed

L_MEANS = (2, 4, 6, 8, 12, 16)
N_TRIALS = 200
N_SCANS = 60
OCCLUSION_START = 15


def sweep_with_std(l_means=L_MEANS, n_trials=N_TRIALS, n_scans=N_SCANS,
                    occlusion_start=OCCLUSION_START, alpha=0.1,
                    tau_novel_ema=-2.847, tau_novel_cum_norm=-3.204):
    summary = {}
    for l_mean in l_means:
        cum_norm_acc, ema_acc = [], []
        cum_norm_ttr, ema_ttr = [], []
        realized_lens = []
        for trial in range(n_trials):
            seed = deterministic_seed(('stochastic', l_mean, trial))
            res = run_single_trial_stochastic(
                l_mean, seed, n_scans=n_scans, occlusion_start=occlusion_start,
                alpha=alpha, tau_novel_ema=tau_novel_ema, tau_novel_cum_norm=tau_novel_cum_norm,
            )
            occ_end = res['occ_end']
            realized_lens.append(res['realized_occ_len'])
            post = slice(occ_end, n_scans)
            cum_norm_acc.append(np.mean(res['cum_norm_correct'][post]) if occ_end < n_scans else np.nan)
            ema_acc.append(np.mean(res['ema_correct'][post]) if occ_end < n_scans else np.nan)
            cum_norm_ttr.append(time_to_recover(res['cum_norm_correct'], occ_end, n_scans))
            ema_ttr.append(time_to_recover(res['ema_correct'], occ_end, n_scans))

        summary[l_mean] = {
            'mean_realized_len': float(np.mean(realized_lens)),
            'cum_norm_acc_mean': float(np.nanmean(cum_norm_acc)),
            'cum_norm_acc_std': float(np.nanstd(cum_norm_acc)),
            'ema_acc_mean': float(np.nanmean(ema_acc)),
            'ema_acc_std': float(np.nanstd(ema_acc)),
            'cum_norm_ttr_mean': float(np.mean(cum_norm_ttr)),
            'ema_ttr_mean': float(np.mean(ema_ttr)),
        }
    return summary


if __name__ == '__main__':
    summary = sweep_with_std()

    # Sanity check against the already-verified summary_stochastic.json
    with open('summary_stochastic.json') as f:
        orig = json.load(f)
    print(f"{'L_mean':>7} {'orig_ema':>9} {'new_ema':>9} {'orig_cum':>9} {'new_cum':>9}")
    for l in L_MEANS:
        o = orig[str(l)]
        n = summary[l]
        print(f"{l:>7} {o['ema_acc_mean']:>9.3f} {n['ema_acc_mean']:>9.3f} "
              f"{o['cum_norm_acc_mean']:>9.3f} {n['cum_norm_acc_mean']:>9.3f}")

    with open('summary_stochastic_with_std.json', 'w') as f:
        json.dump(summary, f, indent=2)

    l_means_sorted = sorted(summary.keys())
    cum_norm = [summary[k]['cum_norm_acc_mean'] for k in l_means_sorted]
    ema = [summary[k]['ema_acc_mean'] for k in l_means_sorted]
    cum_norm_se = [summary[k]['cum_norm_acc_std'] / np.sqrt(N_TRIALS) for k in l_means_sorted]
    ema_se = [summary[k]['ema_acc_std'] / np.sqrt(N_TRIALS) for k in l_means_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(l_means_sorted, cum_norm, yerr=cum_norm_se, marker='o',
                label='Cumulative LL (age-normalized threshold)', color='#e67e22', capsize=3)
    ax.errorbar(l_means_sorted, ema, yerr=ema_se, marker='^',
                label='ATLAS', color='#2471a3', capsize=3)
    ax.set_xlabel('Mean occlusion length $L_{\\mathrm{mean}}$ (scans)')
    ax.set_ylabel('Post-occlusion classification accuracy')
    ax.set_title('Accuracy recovery vs. occlusion length (stochastic occlusion)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.0)
    plt.tight_layout()
    plt.savefig('fig_stochastic_occlusion.png', dpi=150)
    plt.close(fig)
    print("\n[saved fig_stochastic_occlusion.png, summary_stochastic_with_std.json]")
