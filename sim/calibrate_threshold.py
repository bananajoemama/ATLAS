"""
tau_novel calibration.

To pick a defensible novelty threshold (rather than an eyeballed constant),
we need two populations of tracks:
  - "known" tracks: true class is in the library (e.g. class 'A')
  - "novel" tracks: true class is NOT in the library at all (a held-out
    class whose feature distribution the classifier has never seen)

For each classifier variant (EMA, cumulative-normalized), we run many
trials of each population, record the final score the classifier assigns
to its best-matching library class, then sweep a threshold over that
score and compute true positive rate (correctly keeping known tracks
in-library) vs false positive rate (incorrectly keeping novel tracks
in-library, i.e. failing to flag them UNKNOWN).

This produces a standard ROC curve; AUC summarizes overall separability,
and Youden's J statistic (tpr - fpr) picks a defensible single operating
point instead of an eyeballed constant.
"""
import numpy as np
from classifiers import ClassLibrary, CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed


def run_trial_get_best_score(true_class, seed, n_scans=30, alpha=0.1,
                              class_means={'A': 0.0, 'B': 5.0}, novel_mean=12.0):
    """
    Run one trial (no forced occlusion here -- calibration is done on
    "clean" tracking conditions, matching how tau_novel would be tuned
    in practice, separate from the occlusion-recovery experiment).

    true_class: 'A', 'B', or 'NOVEL' (feature drawn from novel_mean instead
    of any library class).

    Returns: (ema_best_score, cum_norm_best_score) -- each classifier's
    top library-class score at the end of the run.
    """
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = ClassLibrary(class_means, class_std=1.0)

    x_true = np.array([130.0, 35.0, 200.0, 0.0])
    track = Track(
        track_id=1,
        x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
        P=np.diag([25, 100, 25, 100]),
        P_existence=0.9, PD=0.9, PW=0.9999,
    )

    cum_clf = CumulativeLLClassifier(library)
    ema_clf = EMAClassifier(library, alpha=alpha)

    for k in range(n_scans):
        x_true = model.step(x_true, rng)
        clutter_meas = clutter.generate(rng)
        measurements = list(clutter_meas)

        detected = False
        feat = None
        if rng.uniform() < track.PD:
            z = model.measure(x_true, rng)
            measurements.append(z)
            detected = True
            if true_class == 'NOVEL':
                feat = rng.normal(novel_mean, 1.0)
            else:
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
            track.P_exist *= (1 - track.PD * track.PW)

        if jipda_says_detected:
            track.x, track.P = kf.update(x_pred, P_pred, measurements[best_idx])
        else:
            track.x, track.P = x_pred, P_pred

        use_detected = detected and jipda_says_detected
        cum_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)
        ema_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)

    ema_best = max(ema_clf.S.values())
    cum_norm_best = max(v / n_scans for v in cum_clf.LL.values())
    return ema_best, cum_norm_best


def compute_roc(known_scores, novel_scores, n_thresholds=200):
    """
    Sweep thresholds across the combined score range; a track is called
    'known' (in-library) if score >= threshold, else 'UNKNOWN'.
    TPR = fraction of known tracks correctly kept in-library.
    FPR = fraction of novel tracks incorrectly kept in-library.
    """
    all_scores = np.concatenate([known_scores, novel_scores])
    lo, hi = all_scores.min() - 1e-6, all_scores.max() + 1e-6
    thresholds = np.linspace(lo, hi, n_thresholds)

    tpr_list, fpr_list = [], []
    for t in thresholds:
        tpr = np.mean(known_scores >= t)
        fpr = np.mean(novel_scores >= t)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # AUC via trapezoidal rule (sort by fpr ascending)
    order = np.argsort(fpr_arr)
    auc = np.trapezoid(tpr_arr[order], fpr_arr[order])

    # Youden's J: best single operating point
    j_stat = tpr_arr - fpr_arr
    best_idx = np.argmax(j_stat)
    best_threshold = thresholds[best_idx]

    return {
        'thresholds': thresholds, 'tpr': tpr_arr, 'fpr': fpr_arr,
        'auc': auc, 'best_threshold': best_threshold,
        'best_tpr': tpr_arr[best_idx], 'best_fpr': fpr_arr[best_idx],
    }


def calibrate(n_trials=150, n_scans=30, alpha=0.1, novel_mean=7.0):
    """Run known vs novel populations, compute ROC/AUC for both classifiers."""
    ema_known, ema_novel = [], []
    cum_known, cum_novel = [], []

    for i in range(n_trials):
        seed_k = deterministic_seed(('known', i))
        ema_s, cum_s = run_trial_get_best_score('A', seed_k, n_scans=n_scans, alpha=alpha)
        ema_known.append(ema_s)
        cum_known.append(cum_s)

        seed_n = deterministic_seed(('novel', i))
        ema_s, cum_s = run_trial_get_best_score('NOVEL', seed_n, n_scans=n_scans, alpha=alpha, novel_mean=novel_mean)
        ema_novel.append(ema_s)
        cum_novel.append(cum_s)

    ema_roc = compute_roc(np.array(ema_known), np.array(ema_novel))
    cum_roc = compute_roc(np.array(cum_known), np.array(cum_novel))

    print(f"EMA classifier:        AUC={ema_roc['auc']:.3f}  "
          f"best_threshold={ema_roc['best_threshold']:.3f}  "
          f"(TPR={ema_roc['best_tpr']:.3f}, FPR={ema_roc['best_fpr']:.3f})")
    print(f"Cumulative (norm) clf: AUC={cum_roc['auc']:.3f}  "
          f"best_threshold={cum_roc['best_threshold']:.3f}  "
          f"(TPR={cum_roc['best_tpr']:.3f}, FPR={cum_roc['best_fpr']:.3f})")

    return ema_roc, cum_roc


if __name__ == '__main__':
    calibrate(n_trials=150)
