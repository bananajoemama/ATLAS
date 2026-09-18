"""
Multivariate feature extension of run_experiment.py.

Addresses the reviewer point that the scalar Gaussian ID channel used
elsewhere in this study "substantially simplifies realistic target
identification based on multidimensional ESM, RCS, or other sensor
signatures." This module reruns the same occlusion-recovery experiment
with a d=3 correlated feature vector standing in for a small stack of
ESM/RCS-derived channels (e.g. normalized mean amplitude, amplitude
spread, and a PRI-offset-like channel), instead of a single scalar.

CumulativeLLClassifier and EMAClassifier are unmodified: they only call
library.sample_feature(...) / library.log_likelihood(...), so swapping in
MultivariateGaussianLibrary is the only change needed to generalize the
architecture itself beyond 1D. This script exists to show that generalization
actually holds empirically, not just structurally.
"""
import numpy as np
from classifiers import MultivariateGaussianLibrary, CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed
from run_experiment import time_to_recover


def make_library(separation=5.0, correlation=0.5):
    """
    d=3 correlated feature library. Correlation is shared across classes
    (only the mean vectors differ) so that class separation, not
    covariance mismatch, is the controlled variable -- matching how the
    scalar-channel study isolated the memory mechanism from feature
    difficulty.
    """
    d = 3
    cov = np.full((d, d), correlation)
    np.fill_diagonal(cov, 1.0)
    class_means = {
        'A': np.array([0.0, 0.0, 0.0]),
        'B': np.array([separation, separation * 0.6, separation * 0.3]),
    }
    return MultivariateGaussianLibrary(class_means, cov)


def run_single_trial(occlusion_len, seed, n_scans=50, occlusion_start=15,
                      alpha=0.1, tau_novel_ema=-6.0, tau_novel_cum_norm=-1.0,
                      separation=5.0, correlation=0.5):
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = make_library(separation=separation, correlation=correlation)
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

    results = {'cum_norm_correct': [], 'ema_correct': []}

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

    return results


def sweep_occlusion_lengths(occlusion_lengths=(2, 4, 6, 8, 12, 16),
                             n_trials=200, n_scans=50, occlusion_start=15,
                             alpha=0.1, tau_novel_ema=-4.5299, tau_novel_cum_norm=-4.6098,
                             separation=5.0, correlation=0.5):
    summary = {}
    for occ_len in occlusion_lengths:
        occ_end = occlusion_start + occ_len
        cum_norm_acc, ema_acc = [], []
        cum_norm_ttr, ema_ttr = [], []

        for trial in range(n_trials):
            seed = deterministic_seed(('mv', occ_len, trial))
            res = run_single_trial(
                occ_len, seed, n_scans=n_scans, occlusion_start=occlusion_start,
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
    summary = sweep_occlusion_lengths(n_trials=200)
    with open('summary_multivariate.json', 'w') as f:
        json.dump(summary, f, indent=2)
