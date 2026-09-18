"""
tau_novel calibration for the d=3 multivariate feature extension
(run_experiment_multivariate.py).

This is the missing piece that caused run_experiment_multivariate.py's
default tau_novel_ema=-2.75 to be wrong: that value is the SCALAR channel's
calibrated threshold (from calibrate_threshold.py), reused here by mistake.
Score scale depends on feature dimensionality and covariance, so a
threshold calibrated for a 1-D feature does not transfer to a 3-D one --
this is explicitly called out in the paper (Section V-F): "tau_novel ...
were independently recalibrated for this feature model, since novelty
boundaries are not portable across feature representations." That
recalibration step was described in the paper but never actually implemented
in this script; this file implements it.

Mirrors calibrate_threshold.py's known-vs-novel ROC procedure exactly,
swapping in the d=3 correlated Gaussian library from
run_experiment_multivariate.make_library() instead of the scalar
ClassLibrary.
"""
import numpy as np
from classifiers import CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed
from run_experiment_multivariate import make_library
from calibrate_threshold import compute_roc


def run_trial_get_best_score(true_class, seed, n_scans=30, alpha=0.1,
                              separation=5.0, correlation=0.5,
                              novel_mean_scale=1.4):
    """
    Same structure as calibrate_threshold.run_trial_get_best_score, but
    using the d=3 correlated-Gaussian library instead of the scalar one.

    true_class: 'A', 'B', or 'NOVEL'. For 'NOVEL', the feature is drawn
    from a mean vector scaled away from the library means -- analogous to
    calibrate_threshold's novel_mean=7.0 with class means at 0/5 (a
    ~1.4x-separation offset), applied elementwise here since the feature
    is now a vector.
    """
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = make_library(separation=separation, correlation=correlation)

    x_true = np.array([130.0, 35.0, 200.0, 0.0])
    track = Track(
        track_id=1,
        x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
        P=np.diag([25, 100, 25, 100]),
        P_existence=0.9, PD=0.9, PW=0.9999,
    )

    cum_clf = CumulativeLLClassifier(library)
    ema_clf = EMAClassifier(library, alpha=alpha)

    novel_mean = library.class_means['B'] * novel_mean_scale

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
                d = library.d
                z_ = rng.normal(size=d)
                feat = novel_mean + z_  # unit-variance novel cloud, same spirit as scalar case
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
        beta_i = {}
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

    ema_best = max(ema_clf.S.values())
    cum_norm_best = max(v / n_scans for v in cum_clf.LL.values())
    return ema_best, cum_norm_best


def calibrate_multivariate(n_trials=150, n_scans=30, alpha=0.1,
                            separation=5.0, correlation=0.5, novel_mean_scale=1.4):
    ema_known, ema_novel = [], []
    cum_known, cum_novel = [], []

    for i in range(n_trials):
        seed_k = deterministic_seed(('mv_known', i))
        ema_s, cum_s = run_trial_get_best_score('A', seed_k, n_scans=n_scans, alpha=alpha,
                                                  separation=separation, correlation=correlation)
        ema_known.append(ema_s)
        cum_known.append(cum_s)

        seed_n = deterministic_seed(('mv_novel', i))
        ema_s, cum_s = run_trial_get_best_score('NOVEL', seed_n, n_scans=n_scans, alpha=alpha,
                                                  separation=separation, correlation=correlation,
                                                  novel_mean_scale=novel_mean_scale)
        ema_novel.append(ema_s)
        cum_novel.append(cum_s)

    ema_roc = compute_roc(np.array(ema_known), np.array(ema_novel))
    cum_roc = compute_roc(np.array(cum_known), np.array(cum_novel))

    print(f"[multivariate] EMA classifier:        AUC={ema_roc['auc']:.3f}  "
          f"best_threshold={ema_roc['best_threshold']:.3f}  "
          f"(TPR={ema_roc['best_tpr']:.3f}, FPR={ema_roc['best_fpr']:.3f})")
    print(f"[multivariate] Cumulative (norm) clf: AUC={cum_roc['auc']:.3f}  "
          f"best_threshold={cum_roc['best_threshold']:.3f}  "
          f"(TPR={cum_roc['best_tpr']:.3f}, FPR={cum_roc['best_fpr']:.3f})")

    return ema_roc, cum_roc


if __name__ == '__main__':
    calibrate_multivariate(n_trials=150)
