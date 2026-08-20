"""
run_all.py -- reproduces every number, table, and figure reported in
Section 5 of the ATLAS paper, in order, from a clean checkout.

Usage:
    pip install -r requirements.txt
    python3 run_all.py

Runtime: ~15-25 minutes on a single core (Monte Carlo trial counts match
what is reported in the paper; reduce n_trials in the calls below for a
faster smoke-test run, but note that reported numbers assume the trial
counts as configured here).

All randomness is seeded deterministically via seeding.deterministic_seed
(NOT Python's built-in hash(), which is randomized per-process) so this
script produces bit-identical output across separate runs/machines,
given the pinned dependency versions in requirements.txt.

Output:
    - Prints all summary statistics to stdout (matching the numbers
      quoted in Sections 5.3-5.6 of main.tex)
    - Writes summary_final.json (raw occlusion-sweep data)
    - Writes fig_roc_novelty.png, fig_accuracy_vs_occlusion.png,
      fig_ablations.png (the three figures embedded in main.tex)
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calibrate_threshold import calibrate
from run_experiment import sweep_occlusion_lengths
from crossing_experiment import run_crossing_experiment
from ablations import sweep_alpha, sweep_w_miss


def section_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # Section 5.3: threshold calibration via ROC
    # ------------------------------------------------------------------
    section_header("Section 5.3: Threshold calibration (ROC)")
    ema_roc, cum_roc = calibrate(n_trials=150, novel_mean=7.0)
    tau_ema = ema_roc['best_threshold']
    tau_cum = cum_roc['best_threshold']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ema_roc['fpr'], ema_roc['tpr'], label=f'ATLAS (AUC={ema_roc["auc"]:.3f})', color='#2471a3')
    ax.plot(cum_roc['fpr'], cum_roc['tpr'], label=f'Cumulative (norm.) (AUC={cum_roc["auc"]:.3f})', color='#e67e22')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    ax.scatter([ema_roc['best_fpr']], [ema_roc['best_tpr']], color='#2471a3', zorder=5, marker='*', s=150)
    ax.scatter([cum_roc['best_fpr']], [cum_roc['best_tpr']], color='#e67e22', zorder=5, marker='*', s=150)
    ax.set_xlabel('False positive rate (novel track misclassified as known)')
    ax.set_ylabel('True positive rate (known track correctly kept in-library)')
    ax.set_title('Novelty detection ROC (novel class mean=7, moderate separation)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_roc_novelty.png', dpi=150)
    plt.close(fig)
    print(f"\n[saved fig_roc_novelty.png]  tau_novel(ATLAS)={tau_ema:.3f}  tau_novel(cum-norm)={tau_cum:.3f}")

    # ------------------------------------------------------------------
    # Section 5.4: occlusion-length sweep, using calibrated thresholds
    # ------------------------------------------------------------------
    section_header("Section 5.4: Occlusion recovery sweep")
    summary = sweep_occlusion_lengths(n_trials=200, tau_novel_ema=tau_ema, tau_novel_cum_norm=tau_cum)
    with open('summary_final.json', 'w') as f:
        json.dump(summary, f, indent=2)

    occ_lens = sorted(summary.keys())
    cum_fixed = [summary[k]['cum_fixed_acc_mean'] for k in occ_lens]
    cum_norm = [summary[k]['cum_norm_acc_mean'] for k in occ_lens]
    ema = [summary[k]['ema_acc_mean'] for k in occ_lens]
    cum_norm_se = [summary[k]['cum_norm_acc_std'] / np.sqrt(200) for k in occ_lens]
    ema_se = [summary[k]['ema_acc_std'] / np.sqrt(200) for k in occ_lens]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(occ_lens, cum_fixed, marker='s', label='Cumulative LL (fixed threshold)', color='#c0392b', linestyle='--')
    ax.errorbar(occ_lens, cum_norm, yerr=cum_norm_se, marker='o', label='Cumulative LL (age-normalized threshold)', color='#e67e22', capsize=3)
    ax.errorbar(occ_lens, ema, yerr=ema_se, marker='^', label='ATLAS', color='#2471a3', capsize=3)
    ax.set_xlabel('Occlusion length (scans)')
    ax.set_ylabel('Post-occlusion classification accuracy')
    ax.set_title('Classification accuracy recovery vs. occlusion length')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.0)
    plt.tight_layout()
    plt.savefig('fig_accuracy_vs_occlusion.png', dpi=150)
    plt.close(fig)
    print("\n[saved fig_accuracy_vs_occlusion.png, summary_final.json]")

    # ------------------------------------------------------------------
    # Section 5.5: multi-target crossing comparison (JIPDA + ATLAS)
    # ------------------------------------------------------------------
    section_header("Section 5.5: Multi-target crossing comparison")
    outcomes, atlas_acc_given_a = run_crossing_experiment(
        n_trials=300, alpha=0.1, tau_novel=tau_ema
    )

    # ------------------------------------------------------------------
    # Section 5.6: ablations (alpha, W_miss sensitivity)
    # ------------------------------------------------------------------
    section_header("Section 5.6: Ablations (alpha, W_miss)")
    alpha_results = sweep_alpha(tau_novel_ema=tau_ema, tau_novel_cum_norm=tau_cum)
    wmiss_results = sweep_w_miss(tau_novel_ema=tau_ema, tau_novel_cum_norm=tau_cum)

    alphas = sorted(alpha_results.keys())
    alpha_acc = [alpha_results[a]['mean'] for a in alphas]
    alpha_se = [alpha_results[a]['std'] / np.sqrt(200) for a in alphas]

    wmiss = sorted(wmiss_results.keys())
    wmiss_acc = [wmiss_results[w]['mean'] for w in wmiss]
    wmiss_se = [wmiss_results[w]['std'] / np.sqrt(200) for w in wmiss]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(alphas, alpha_acc, yerr=alpha_se, marker='o', color='#2471a3', capsize=3)
    axes[0].axvline(0.1, linestyle='--', color='gray', alpha=0.6, label='Paper default (0.1)')
    axes[0].set_xlabel(r'EMA decay factor $\alpha$')
    axes[0].set_ylabel('Post-occlusion accuracy')
    axes[0].set_title(r'Sensitivity to $\alpha$ ($L=8$, $W_{miss}=-8.0$)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(wmiss, wmiss_acc, yerr=wmiss_se, marker='s', color='#c0392b', capsize=3)
    axes[1].axvline(-8.0, linestyle='--', color='gray', alpha=0.6, label='Paper default (-8.0)')
    axes[1].set_xlabel(r'Missed-detection penalty $W_{miss}$')
    axes[1].set_ylabel('Post-occlusion accuracy')
    axes[1].set_title(r'Sensitivity to $W_{miss}$ ($L=8$, $\alpha=0.1$)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_ablations.png', dpi=150)
    plt.close(fig)
    print("\n[saved fig_ablations.png]")

    elapsed = time.time() - t0
    section_header(f"Done in {elapsed/60:.1f} minutes")
    print("All figures and summary_final.json written to the current directory.")
    print("Compare printed numbers above against Sections 5.3-5.6 of main.tex.")


if __name__ == '__main__':
    main()
