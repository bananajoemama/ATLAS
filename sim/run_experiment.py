"""
Monte Carlo experiment driver.

Sweeps occlusion length and runs many trials per condition, comparing:
  - Cumulative LL classifier, fixed threshold  (dramatic baseline)
  - Cumulative LL classifier, age-normalized threshold  (fair baseline)
  - ATLAS EMA classifier  (proposed)

Metrics computed per condition:
  - post-occlusion classification accuracy (scans after occlusion ends)
  - time-to-recover: scans after occlusion ends until correct label returns
    (capped at remaining scan budget if never recovers)
"""
import numpy as np
from scenario import SingleTargetOcclusionScenario
from classifiers import ClassLibrary, CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed


def run_single_trial(occlusion_len, seed, n_scans=50, occlusion_start=15,
                      alpha=0.1, tau_novel_ema=-6.0, tau_novel_cum_fixed=-6.0,
                      tau_novel_cum_norm=-1.0, class_means={'A': 0.0, 'B': 5.0}):
    """
    Run one trial with all three classifier variants tracking the SAME
    underlying measurement stream (so comparisons are paired/fair).
    Returns dict of per-scan correctness lists for each variant.
    """
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = ClassLibrary(class_means, class_std=1.0)
    true_class = 'A'

    x_true = np.array([130.0, 35.0, 200.0, 0.0])
    track = Track(
        track_id=1,
        x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
        P=np.diag([25, 100, 25, 100]),
        P_existence=0.9, PD=0.9, PW=0.9999,
    )

    cum_clf = CumulativeLLClassifier(library)
    ema_clf = EMAClassifier(library, alpha=alpha)

    results = {
        'cum_fixed_correct': [], 'cum_norm_correct': [], 'ema_correct': [],
        'occlusion_start': occlusion_start, 'occlusion_len': occlusion_len,
    }

    for k in range(n_scans):
        x_true = model.step(x_true, rng)
        forced_miss = occlusion_start <= k < occlusion_start + occlusion_len

        clutter_meas = clutter.generate(rng)
        measurements = list(clutter_meas)

        detected = False
        feat = None
        if not forced_miss and rng.uniform() < track.PD:
            z = model.measure(x_true, rng)
            measurements.append(z)
            detected = True
            feat = library.sample_feature(true_class, rng)

        track.P_exist = predict_existence(track.P_exist)
        gated, x_pred, P_pred, S, V_t = gate_measurements(track, kf, measurements)
        meas_idx_set = set(i for i, z, d in gated)
        dens = {i: d for i, z, d in gated}

        gate_info = {
            track.id: {
                'PD': track.PD, 'PW': track.PW, 'P_exist': track.P_exist,
                'meas_idx_set': meas_idx_set, 'dens': dens,
                'm_t': max(len(meas_idx_set), 1), 'V_t': V_t,
            }
        }
        V = max(gate_info[track.id]['V_t'] for _ in [0])
        m_hat = estimate_clutter_count([track.id], gate_info, meas_idx_set)

        jipda_says_detected = False
        beta_det = 0.0
        best_idx = None
        if len(meas_idx_set) > 0:
            P_exist_post, betas = jipda_cluster_update(
                [track.id], gate_info, meas_idx_set, V, m_hat
            )
            track.P_exist = P_exist_post[track.id]
            beta0 = betas[track.id]['beta0']
            beta_i = betas[track.id]['beta_i']
            if beta_i:
                best_idx = max(beta_i, key=beta_i.get)
                beta_det = beta_i[best_idx]
                jipda_says_detected = beta_det > beta0
        else:
            # No measurements at all fell in the gate: apply the exact
            # zero-detection existence posterior (Eq. 8 degenerate case,
            # single-track cluster) rather than an ad hoc approximation.
            PD, PW, Pprior = track.PD, track.PW, track.P_exist
            denom = 1 - PD * PW * Pprior
            track.P_exist = ((1 - PD * PW) * Pprior / denom) if denom > 1e-12 else 0.0

        if jipda_says_detected:
            meas_list = [measurements[i] for i in beta_i.keys()]
            beta_list = [beta_i[i] for i in beta_i.keys()]
            track.x, track.P = kf.pda_update(x_pred, P_pred, meas_list, beta_list, beta0)
        else:
            track.x, track.P = x_pred, P_pred

        use_detected = detected and jipda_says_detected
        cum_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)
        ema_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)

        cpred_fixed, _ = cum_clf.predict(tau_novel=tau_novel_cum_fixed)
        cpred_norm, _ = cum_clf.predict_normalized(age=k + 1, tau_novel_per_scan=tau_novel_cum_norm)
        epred, _ = ema_clf.predict(tau_novel=tau_novel_ema)

        results['cum_fixed_correct'].append(cpred_fixed == true_class)
        results['cum_norm_correct'].append(cpred_norm == true_class)
        results['ema_correct'].append(epred == true_class)

    return results


def time_to_recover(correct_list, occlusion_end, n_scans):
    """Scans after occlusion_end until first correct classification (persisting)."""
    for k in range(occlusion_end, n_scans):
        if correct_list[k]:
            return k - occlusion_end
    return n_scans - occlusion_end  # never recovered, cap at budget


def sweep_occlusion_lengths(occlusion_lengths=(2, 4, 6, 8, 12, 16),
                             n_trials=200, n_scans=50, occlusion_start=15,
                             alpha=0.1, tau_novel_ema=-2.847, tau_novel_cum_norm=-3.204):
    """
    Run Monte Carlo trials across occlusion lengths, return summary stats.
    Default thresholds are calibrated via calibrate_threshold.py (ROC,
    novel_mean=7.0, moderate-difficulty separation) rather than guessed.
    """
    summary = {}
    for occ_len in occlusion_lengths:
        occ_end = occlusion_start + occ_len
        cum_fixed_acc, cum_norm_acc, ema_acc = [], [], []
        cum_fixed_ttr, cum_norm_ttr, ema_ttr = [], [], []

        for trial in range(n_trials):
            seed = deterministic_seed((occ_len, trial))
            res = run_single_trial(
                occ_len, seed, n_scans=n_scans,
                occlusion_start=occlusion_start, alpha=alpha,
                tau_novel_ema=tau_novel_ema, tau_novel_cum_norm=tau_novel_cum_norm,
            )
            post = slice(occ_end, n_scans)
            cum_fixed_acc.append(np.mean(res['cum_fixed_correct'][post]))
            cum_norm_acc.append(np.mean(res['cum_norm_correct'][post]))
            ema_acc.append(np.mean(res['ema_correct'][post]))

            cum_fixed_ttr.append(time_to_recover(res['cum_fixed_correct'], occ_end, n_scans))
            cum_norm_ttr.append(time_to_recover(res['cum_norm_correct'], occ_end, n_scans))
            ema_ttr.append(time_to_recover(res['ema_correct'], occ_end, n_scans))

        summary[occ_len] = {
            'cum_fixed_acc_mean': np.mean(cum_fixed_acc), 'cum_fixed_acc_std': np.std(cum_fixed_acc),
            'cum_norm_acc_mean': np.mean(cum_norm_acc), 'cum_norm_acc_std': np.std(cum_norm_acc),
            'ema_acc_mean': np.mean(ema_acc), 'ema_acc_std': np.std(ema_acc),
            'cum_fixed_ttr_mean': np.mean(cum_fixed_ttr),
            'cum_norm_ttr_mean': np.mean(cum_norm_ttr),
            'ema_ttr_mean': np.mean(ema_ttr),
        }
        print(f"occ_len={occ_len:>2}  "
              f"cum_fixed_acc={summary[occ_len]['cum_fixed_acc_mean']:.3f}  "
              f"cum_norm_acc={summary[occ_len]['cum_norm_acc_mean']:.3f}  "
              f"ema_acc={summary[occ_len]['ema_acc_mean']:.3f}  |  "
              f"ttr(fixed/norm/ema)="
              f"{summary[occ_len]['cum_fixed_ttr_mean']:.1f}/"
              f"{summary[occ_len]['cum_norm_ttr_mean']:.1f}/"
              f"{summary[occ_len]['ema_ttr_mean']:.1f}")
    return summary


if __name__ == '__main__':
    summary = sweep_occlusion_lengths(n_trials=200)
