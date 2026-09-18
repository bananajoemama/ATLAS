"""
Multi-class extension of run_experiment_multivariate.py.

Addresses the reviewer point that the simulation "does not clearly
define the known target classes... or demonstrate how ATLAS
distinguishes among multiple known classes." Section~V-... only ever
instantiated M=2 known classes (A, B). This module extends the
d=3 correlated-Gaussian library to M=4 known classes and:
  (1) reports a closed-set confusion matrix under clean (non-occluded)
      tracking, to directly demonstrate multi-class discrimination, and
  (2) reruns the occlusion-length sweep with M=4 classes competing,
      to check whether the recovery advantage established under M=2
      survives added classifier competition.

No changes to classifiers.py were needed: MultivariateGaussianLibrary
already takes an arbitrary dict of classes.
"""
import numpy as np
from classifiers import MultivariateGaussianLibrary, CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed
from run_experiment import time_to_recover

CLASS_IDS = ['A', 'B', 'C', 'D']


def make_library_multiclass(separation=5.0, correlation=0.5):
    """
    d=3 correlated feature library with M=4 known classes. Means are
    placed at 4 corners of a scaled tetrahedron-like spread in R^3 so
    that all pairwise separations are comparable (no pair of classes is
    trivially easier to distinguish than another), matching the
    controlled-separation design of the M=2 case in
    run_experiment_multivariate.py.
    """
    d = 3
    cov = np.full((d, d), correlation)
    np.fill_diagonal(cov, 1.0)
    s = separation
    class_means = {
        'A': np.array([0.0, 0.0, 0.0]),
        'B': np.array([s, s, 0.0]),
        'C': np.array([s, 0.0, s]),
        'D': np.array([0.0, s, s]),
    }
    return MultivariateGaussianLibrary(class_means, cov)


def run_single_trial_multiclass(occlusion_len, seed, true_class, n_scans=50,
                                 occlusion_start=15, alpha=0.1,
                                 tau_novel_ema=-4.53, tau_novel_cum_norm=-4.61,
                                 separation=5.0, correlation=0.5):
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = make_library_multiclass(separation=separation, correlation=correlation)

    x_true = np.array([130.0, 35.0, 200.0, 0.0])
    track = Track(
        track_id=1,
        x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
        P=np.diag([25, 100, 25, 100]),
        P_existence=0.9, PD=0.9, PW=0.9999,
    )

    cum_clf = CumulativeLLClassifier(library)
    ema_clf = EMAClassifier(library, alpha=alpha)

    results = {'cum_norm_correct': [], 'ema_correct': [],
               'ema_pred_final': None, 'cum_norm_pred_final': None}

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
            P_exist_post, betas = jipda_cluster_update([track.id], gate_info, meas_idx_set, V, m_hat)
            track.P_exist = P_exist_post[track.id]
            beta0 = betas[track.id]['beta0']
            beta_i = betas[track.id]['beta_i']
            if beta_i:
                best_idx = max(beta_i, key=beta_i.get)
                beta_det = beta_i[best_idx]
                jipda_says_detected = beta_det > beta0
        else:
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

        cpred_norm, _ = cum_clf.predict_normalized(age=k + 1, tau_novel_per_scan=tau_novel_cum_norm)
        epred, _ = ema_clf.predict(tau_novel=tau_novel_ema)

        results['cum_norm_correct'].append(cpred_norm == true_class)
        results['ema_correct'].append(epred == true_class)

    results['ema_pred_final'] = epred
    results['cum_norm_pred_final'] = cpred_norm
    return results


def confusion_matrix_clean(n_trials=150, n_scans=30, alpha=0.1,
                            tau_novel_ema=-4.53, separation=5.0, correlation=0.5):
    """
    Closed-set confusion matrix under clean (non-occluded) tracking:
    for each true class, run n_trials clean tracks and tally ATLAS's
    final predicted class. Directly answers "does ATLAS distinguish
    among multiple known classes" rather than only flagging novelty.
    """
    conf = {t: {p: 0 for p in CLASS_IDS + ['UNKNOWN']} for t in CLASS_IDS}
    for true_class in CLASS_IDS:
        for trial in range(n_trials):
            seed = deterministic_seed(('confusion', true_class, trial))
            res = run_single_trial_multiclass(
                occlusion_len=0, seed=seed, true_class=true_class,
                n_scans=n_scans, occlusion_start=n_scans + 1,  # no occlusion
                alpha=alpha, tau_novel_ema=tau_novel_ema,
                separation=separation, correlation=correlation,
            )
            pred = res['ema_pred_final']
            conf[true_class][pred if pred in conf[true_class] else 'UNKNOWN'] += 1
    accs = {t: conf[t][t] / n_trials for t in CLASS_IDS}
    overall_acc = sum(conf[t][t] for t in CLASS_IDS) / (n_trials * len(CLASS_IDS))
    print(f"Multi-class (M=4) clean closed-set accuracy: {overall_acc:.3f}")
    for t in CLASS_IDS:
        print(f"  true={t}: per-class acc={accs[t]:.3f}  row={conf[t]}")
    return conf, accs, overall_acc


def sweep_occlusion_lengths_multiclass(occlusion_lengths=(2, 4, 6, 8, 12, 16),
                                        n_trials_per_class=150, n_scans=50,
                                        occlusion_start=15, alpha=0.1,
                                        tau_novel_ema=-4.53, tau_novel_cum_norm=-4.61,
                                        separation=5.0, correlation=0.5):
    """
    Same occlusion-recovery sweep as run_experiment_multivariate.py's
    sweep_occlusion_lengths, but averaged over all M=4 true classes
    (n_trials_per_class trials per class per L) instead of a single
    fixed true class, so the reported numbers reflect performance
    under real multi-class competition rather than one class's
    idiosyncratic separation from the other three.
    """
    summary = {}
    for occ_len in occlusion_lengths:
        occ_end = occlusion_start + occ_len
        cum_norm_acc, ema_acc = [], []
        cum_norm_ttr, ema_ttr = [], []

        for true_class in CLASS_IDS:
            for trial in range(n_trials_per_class):
                seed = deterministic_seed(('mc_occ', occ_len, true_class, trial))
                res = run_single_trial_multiclass(
                    occ_len, seed, true_class, n_scans=n_scans, occlusion_start=occlusion_start,
                    alpha=alpha, tau_novel_ema=tau_novel_ema, tau_novel_cum_norm=tau_novel_cum_norm,
                    separation=separation, correlation=correlation,
                )
                post = slice(occ_end, n_scans)
                cum_norm_acc.append(np.mean(res['cum_norm_correct'][post]))
                ema_acc.append(np.mean(res['ema_correct'][post]))
                cum_norm_ttr.append(time_to_recover(res['cum_norm_correct'], occ_end, n_scans))
                ema_ttr.append(time_to_recover(res['ema_correct'], occ_end, n_scans))

        summary[occ_len] = {
            'cum_norm_acc_mean': np.mean(cum_norm_acc), 'cum_norm_acc_std': np.std(cum_norm_acc),
            'ema_acc_mean': np.mean(ema_acc), 'ema_acc_std': np.std(ema_acc),
            'cum_norm_ttr_mean': np.mean(cum_norm_ttr), 'ema_ttr_mean': np.mean(ema_ttr),
        }
        print(f"occ_len={occ_len:>2}  cum_norm_acc={summary[occ_len]['cum_norm_acc_mean']:.3f}  "
              f"ema_acc={summary[occ_len]['ema_acc_mean']:.3f}  |  "
              f"ttr(norm/ema)={summary[occ_len]['cum_norm_ttr_mean']:.1f}/{summary[occ_len]['ema_ttr_mean']:.1f}")
    return summary


if __name__ == '__main__':
    import json
    print("=== M=4 confusion matrix (clean tracking) ===")
    conf, accs, overall = confusion_matrix_clean(n_trials=150)
    print()
    print("=== M=4 occlusion sweep ===")
    summary = sweep_occlusion_lengths_multiclass(n_trials_per_class=150)
    json.dump({'confusion': conf, 'per_class_acc': accs, 'overall_acc': overall},
              open('summary_multiclass_confusion.json', 'w'), indent=2)
    json.dump(summary, open('summary_multiclass_occlusion.json', 'w'), indent=2)
