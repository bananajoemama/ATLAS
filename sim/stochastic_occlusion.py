"""
Stochastic occlusion robustness check.

The main paper (Section V) injects occlusions as a deterministic block of
exactly L consecutive missed detections. This module tests whether ATLAS's
recovery advantage survives when occlusion *duration* is instead drawn from
a stochastic process, as real terrain masking / fading / jamming episodes
would produce, rather than being fixed at exactly L.

Model: once an occlusion episode begins at `occlusion_start`, each scan
within it independently "recovers" (ends the occlusion) with probability
p_exit = 1 / L_mean, i.e. a two-state Markov chain (tracking <-> faded)
with a geometric sojourn time in the faded state with mean L_mean. This
preserves the paper's mean occlusion length L_mean as the sweep variable
(so results are directly comparable to Fig. 3) while making each trial's
actual realized duration random rather than fixed.

Everything else (JIPDA association, classifiers, thresholds, clutter,
motion model) is identical to run_experiment.py — only the occlusion-length
generator changes, isolating the effect of stochastic vs. deterministic
occlusion duration.
"""
import numpy as np
from classifiers import ClassLibrary, CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed
from run_experiment import time_to_recover


def sample_markov_occlusion_mask(rng, n_scans, occlusion_start, l_mean):
    """
    Returns a boolean array of length n_scans: True = forced missed
    detection at that scan. Occlusion begins at occlusion_start and each
    subsequent occluded scan independently ends the episode with
    probability p_exit = 1 / l_mean (geometric duration, mean = l_mean).
    Guarantees at least 1 occluded scan.
    """
    mask = np.zeros(n_scans, dtype=bool)
    p_exit = 1.0 / l_mean
    k = occlusion_start
    mask[k] = True
    k += 1
    while k < n_scans and rng.uniform() >= p_exit:
        mask[k] = True
        k += 1
    return mask, k  # k = first scan after occlusion ends (realized occ_end)


def run_single_trial_stochastic(l_mean, seed, n_scans=60, occlusion_start=15,
                                 alpha=0.1, tau_novel_ema=-2.847,
                                 tau_novel_cum_norm=-3.204,
                                 class_means={'A': 0.0, 'B': 5.0}):
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = ClassLibrary(class_means, class_std=1.0)
    true_class = 'A'

    occ_mask, occ_end = sample_markov_occlusion_mask(rng, n_scans, occlusion_start, l_mean)

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
        'realized_occ_len': occ_end - occlusion_start, 'occlusion_start': occlusion_start,
    }

    for k in range(n_scans):
        x_true = model.step(x_true, rng)
        forced_miss = occ_mask[k]

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
        V = gate_info[track.id]['V_t']
        m_hat = estimate_clutter_count([track.id], gate_info, meas_idx_set)

        jipda_says_detected = False
        beta_det = 0.0
        beta_i = {}
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

        cpred_fixed, _ = cum_clf.predict(tau_novel=tau_novel_cum_norm)  # unused variant, kept for parity
        cpred_norm, _ = cum_clf.predict_normalized(age=k + 1, tau_novel_per_scan=tau_novel_cum_norm)
        epred, _ = ema_clf.predict(tau_novel=tau_novel_ema)

        results['cum_fixed_correct'].append(cpred_fixed == true_class)
        results['cum_norm_correct'].append(cpred_norm == true_class)
        results['ema_correct'].append(epred == true_class)

    results['occ_end'] = occ_end
    return results


def sweep_stochastic_occlusion(l_means=(2, 4, 6, 8, 12, 16), n_trials=200,
                                n_scans=60, occlusion_start=15, alpha=0.1,
                                tau_novel_ema=-2.847, tau_novel_cum_norm=-3.204):
    """
    Same structure/metrics as run_experiment.sweep_occlusion_lengths, but
    with Markov (geometric, mean=l_mean) occlusion duration instead of a
    fixed block of exactly L scans. Directly comparable to Fig. 3/Table
    (accuracy, time-to-recover) for the deterministic case.
    """
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
            'ema_acc_mean': float(np.nanmean(ema_acc)),
            'cum_norm_ttr_mean': float(np.mean(cum_norm_ttr)),
            'ema_ttr_mean': float(np.mean(ema_ttr)),
        }
        print(f"L_mean={l_mean:>2} (realized avg={summary[l_mean]['mean_realized_len']:.1f})  "
              f"cum_norm_acc={summary[l_mean]['cum_norm_acc_mean']:.3f}  "
              f"ema_acc={summary[l_mean]['ema_acc_mean']:.3f}  |  "
              f"ttr(norm/ema)={summary[l_mean]['cum_norm_ttr_mean']:.1f}/"
              f"{summary[l_mean]['ema_ttr_mean']:.1f}")
    return summary


if __name__ == '__main__':
    import json
    summary = sweep_stochastic_occlusion(n_trials=200)
    with open('summary_stochastic.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nWrote summary_stochastic.json")
