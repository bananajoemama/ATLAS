"""
Validate the closed-form time-to-recover derivation (paper Eq. 5-8) against
the actual Monte Carlo simulation.

Step 1: measure mu_det = E[Delta LL_k | detected] for the true class,
        under clean (non-occluded) tracking, directly from the simulator.
Step 2: measure S_L = EMA score for the true class at the end of an
        L-scan occlusion, directly from the simulator.
Step 3: plug (alpha, W_miss, tau_novel, mu_det, S_L) into the closed-form
        n*(L) (Eq. 7) and compare to the empirically measured mean
        time-to-recover from sweep_occlusion_lengths.
"""
import numpy as np
from scenario import SingleTargetOcclusionScenario
from classifiers import ClassLibrary, CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed
from run_experiment import run_single_trial, sweep_occlusion_lengths, time_to_recover
from calibrate_threshold import calibrate

ALPHA = 0.1
W_MISS = -8.0
# tau_novel is NOT hardcoded here. run_all.py -- the canonical script that
# reproduces every number in Section 5 of the paper -- calibrates tau fresh
# via calibrate_threshold.calibrate(n_trials=150, novel_mean=7.0) and feeds
# that calibrated value into sweep_occlusion_lengths. An earlier version of
# this script used a stale hardcoded tau_novel_ema=-2.847, which silently
# diverged from the paper's actual calibrated threshold and produced
# time-to-recover numbers that did not match Table III. We fix that here by
# calibrating tau the same way run_all.py does, once, up front, and using
# that single value everywhere below.
_EMA_ROC, _CUM_ROC = calibrate(n_trials=150, novel_mean=7.0)
TAU_NOVEL = _EMA_ROC['best_threshold']
TAU_NOVEL_CUM_NORM = _CUM_ROC['best_threshold']


def measure_mu_det(n_trials=500, n_scans=50, occlusion_start=15, occlusion_len=0,
                    class_means={'A': 0.0, 'B': 5.0}, alpha=ALPHA):
    """
    Re-run the exact simulation pipeline (same JIPDA + measurement model as
    run_single_trial) with occlusion_len=0, i.e. clean tracking throughout,
    and record the instantaneous Delta LL actually applied to the *true*
    class's EMA update on every detected scan. This is mu_det as defined
    in the paper: E[Delta LL_k | detected], the top branch of Eq. (llk),
    measured empirically rather than assumed.
    """
    deltas = []
    library = ClassLibrary(class_means, class_std=1.0)
    true_class = 'A'

    for trial in range(n_trials):
        seed = deterministic_seed((occlusion_len, trial, 999))
        rng = np.random.default_rng(seed)
        model = ConstantVelocityModel()
        kf = KalmanFilter(model)
        clutter = ClutterModel()

        x_true = np.array([130.0, 35.0, 200.0, 0.0])
        track = Track(
            track_id=1,
            x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
            P=np.diag([25, 100, 25, 100]),
            P_existence=0.9, PD=0.9, PW=0.9999,
        )
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
                feat = library.sample_feature(true_class, rng)

            track.P_exist = predict_existence(track.P_exist)
            gated, x_pred, P_pred, S, V_t = gate_measurements(track, kf, measurements)
            meas_idx_set = set(i for i, z, d in gated)
            dens = {i: d for i, z, d in gated}
            gate_info = {track.id: {'PD': track.PD, 'PW': track.PW, 'P_exist': track.P_exist,
                                     'meas_idx_set': meas_idx_set, 'dens': dens,
                                     'm_t': max(len(meas_idx_set), 1), 'V_t': V_t}}
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

            prev_S_A = ema_clf.S['A']
            prev_LL_A = ema_clf.LL['A']
            ema_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)

            if use_detected:
                delta_true_class = ema_clf.LL['A'] - prev_LL_A
                deltas.append(delta_true_class)

    deltas = np.array(deltas)
    return deltas.mean(), deltas.std(), len(deltas)


def measure_S_at_occlusion_end(occlusion_lengths, n_trials=500, n_scans=50, occlusion_start=15,
                                class_means={'A': 0.0, 'B': 5.0}, alpha=ALPHA):
    """
    Measure S_0 (score just before occlusion) and S_L (score at end of the
    L-scan occlusion) for the true class, directly from the simulator,
    for comparison against the closed-form Eq. (score_at_occlusion_end).
    """
    library = ClassLibrary(class_means, class_std=1.0)
    true_class = 'A'
    results = {}

    for occ_len in occlusion_lengths:
        S0_list, SL_list = [], []
        for trial in range(n_trials):
            seed = deterministic_seed((occ_len, trial))
            rng = np.random.default_rng(seed)
            model = ConstantVelocityModel()
            kf = KalmanFilter(model)
            clutter = ClutterModel()
            x_true = np.array([130.0, 35.0, 200.0, 0.0])
            track = Track(track_id=1, x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
                           P=np.diag([25, 100, 25, 100]), P_existence=0.9, PD=0.9, PW=0.9999)
            ema_clf = EMAClassifier(library, alpha=alpha)
            occ_end = occlusion_start + occ_len

            for k in range(n_scans):
                x_true = model.step(x_true, rng)
                forced_miss = occlusion_start <= k < occ_end
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
                gate_info = {track.id: {'PD': track.PD, 'PW': track.PW, 'P_exist': track.P_exist,
                                         'meas_idx_set': meas_idx_set, 'dens': dens,
                                         'm_t': max(len(meas_idx_set), 1), 'V_t': V_t}}
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
                ema_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)

                if k == occlusion_start - 1:
                    S0_list.append(ema_clf.S['A'])
                if k == occ_end - 1:
                    SL_list.append(ema_clf.S['A'])

        results[occ_len] = (np.mean(S0_list), np.mean(SL_list))
    return results


def measure_effective_detection_rate(n_trials=500, n_scans=50,
                                      class_means={'A': 0.0, 'B': 5.0}, alpha=ALPHA):
    """
    Measure P(detected AND jipda_says_detected) per scan under clean
    (non-occluded) tracking -- the quantity used to build the
    mixture-corrected mu_eff. Also returns the trial-to-trial std of this
    rate as a sanity check on how stable it is.
    """
    library = ClassLibrary(class_means, class_std=1.0)
    true_class = 'A'
    per_trial_rates = []

    for trial in range(n_trials):
        seed = deterministic_seed((0, trial, 999))
        rng = np.random.default_rng(seed)
        model = ConstantVelocityModel()
        kf = KalmanFilter(model)
        clutter = ClutterModel()
        x_true = np.array([130.0, 35.0, 200.0, 0.0])
        track = Track(track_id=1, x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
                       P=np.diag([25, 100, 25, 100]), P_existence=0.9, PD=0.9, PW=0.9999)
        n_detect = 0
        for k in range(n_scans):
            x_true = model.step(x_true, rng)
            clutter_meas = clutter.generate(rng)
            measurements = list(clutter_meas)
            detected = False
            if rng.uniform() < track.PD:
                z = model.measure(x_true, rng)
                measurements.append(z)
                detected = True

            track.P_exist = predict_existence(track.P_exist)
            gated, x_pred, P_pred, S, V_t = gate_measurements(track, kf, measurements)
            meas_idx_set = set(i for i, z, d in gated)
            dens = {i: d for i, z, d in gated}
            gate_info = {track.id: {'PD': track.PD, 'PW': track.PW, 'P_exist': track.P_exist,
                                     'meas_idx_set': meas_idx_set, 'dens': dens,
                                     'm_t': max(len(meas_idx_set), 1), 'V_t': V_t}}
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

            if detected and jipda_says_detected:
                n_detect += 1

        per_trial_rates.append(n_detect / n_scans)

    per_trial_rates = np.array(per_trial_rates)
    return per_trial_rates.mean(), per_trial_rates.std()


def measure_ttr_distribution(occlusion_len, n_trials=200, n_scans=50, occlusion_start=15,
                              alpha=ALPHA):
    """
    Return the raw per-trial time-to-recover list (not just the mean) for
    a given occlusion length, using the paper's own calibrated tau_novel,
    so the spread of individual recovery times can be inspected directly
    rather than only reasoning about the Monte Carlo mean.
    """
    ttrs = []
    occ_end = occlusion_start + occlusion_len
    for trial in range(n_trials):
        seed = deterministic_seed((occlusion_len, trial))
        res = run_single_trial(occlusion_len, seed, n_scans=n_scans,
                                occlusion_start=occlusion_start, alpha=alpha,
                                tau_novel_ema=TAU_NOVEL, tau_novel_cum_norm=TAU_NOVEL_CUM_NORM)
        ttrs.append(time_to_recover(res['ema_correct'], occ_end, n_scans))
    return ttrs


if __name__ == '__main__':
    print("=" * 70)
    print(f"tau_novel (ATLAS, calibrated) = {TAU_NOVEL:.4f}  (AUC={_EMA_ROC['auc']:.3f})")
    print(f"tau_novel (Cum. norm., calibrated) = {TAU_NOVEL_CUM_NORM:.4f}  (AUC={_CUM_ROC['auc']:.3f})")
    print("(should match the paper: tau_ema=-2.75, AUC=0.826; tau_cum=-3.12, AUC=0.838)")
    print("=" * 70)
    print("STEP 1: measuring mu_det empirically")
    print("=" * 70)
    mu_det, mu_det_std, n = measure_mu_det(n_trials=500)
    print(f"mu_det = {mu_det:.4f}  (std={mu_det_std:.4f}, n={n} detected scans)")

    occ_lengths = [2, 4, 6, 8, 12, 16]

    print()
    print("=" * 70)
    print("STEP 2: measuring S_0, S_L at occlusion end, vs. closed-form Eq.(5)")
    print("=" * 70)
    sl_measured = measure_S_at_occlusion_end(occ_lengths, n_trials=300)
    for L in occ_lengths:
        S0, SL_sim = sl_measured[L]
        SL_pred = W_MISS + (1 - ALPHA) ** L * (S0 - W_MISS)
        print(f"L={L:>2}  S0={S0:6.3f}  S_L (sim)={SL_sim:7.3f}  S_L (closed-form)={SL_pred:7.3f}  "
              f"diff={SL_sim - SL_pred:+.3f}")

    print()
    print("=" * 70)
    print("STEP 3: predicted n*(L) [Eq.(7)] vs. measured mean time-to-recover")
    print("=" * 70)
    print("(running sweep_occlusion_lengths for measured ttr; this reproduces Table III)")
    summary = sweep_occlusion_lengths(occlusion_lengths=occ_lengths, n_trials=200,
                                       alpha=ALPHA, tau_novel_ema=TAU_NOVEL)
    print()
    for L in occ_lengths:
        S0, SL_sim = sl_measured[L]
        SL_pred = W_MISS + (1 - ALPHA) ** L * (S0 - W_MISS)
        ratio = (TAU_NOVEL - mu_det) / (SL_pred - mu_det)
        if ratio > 0:
            n_pred = np.log(ratio) / np.log(1 - ALPHA)
            n_pred = max(0, int(np.ceil(n_pred)))
        else:
            n_pred = float('nan')
        n_meas = summary[L]['ema_ttr_mean']
        print(f"L={L:>2}  n*(L) predicted={n_pred!s:>4}  measured ttr={n_meas:6.2f}  diff={n_pred - n_meas if isinstance(n_pred,int) else float('nan'):+.2f}")

    n_inf_ratio = (TAU_NOVEL - mu_det) / (W_MISS - mu_det)
    n_inf = np.ceil(np.log(n_inf_ratio) / np.log(1 - ALPHA)) if n_inf_ratio > 0 else float('nan')
    print()
    print(f"Predicted asymptotic ceiling n*_inf [Eq.(8), naive mu_det] = {n_inf}")

    print()
    print("=" * 70)
    print("STEP 4: mixture-corrected mu_eff and refined n*(L)")
    print("=" * 70)
    p_detect, p_detect_std = measure_effective_detection_rate(n_trials=500)
    mu_eff = p_detect * mu_det + (1 - p_detect) * W_MISS
    print(f"P(effective detection per post-occlusion scan) = {p_detect:.4f} (std across trials={p_detect_std:.4f})")
    print(f"mu_eff = p*mu_det + (1-p)*W_miss = {mu_eff:.4f}")
    print()
    for L in occ_lengths:
        S0, SL_sim = sl_measured[L]
        SL_pred = W_MISS + (1 - ALPHA) ** L * (S0 - W_MISS)
        ratio = (TAU_NOVEL - mu_eff) / (SL_pred - mu_eff)
        n_pred = int(np.ceil(np.log(ratio) / np.log(1 - ALPHA))) if ratio > 0 else float('nan')
        n_meas = summary[L]['ema_ttr_mean']
        print(f"L={L:>2}  n*(L) [mu_eff]={n_pred!s:>4}  measured ttr={n_meas:6.2f}  diff={n_pred - n_meas if isinstance(n_pred,int) else float('nan'):+.2f}")

    n_inf_ratio_eff = (TAU_NOVEL - mu_eff) / (W_MISS - mu_eff)
    n_inf_eff = np.ceil(np.log(n_inf_ratio_eff) / np.log(1 - ALPHA)) if n_inf_ratio_eff > 0 else float('nan')
    print()
    print(f"Predicted asymptotic ceiling n*_inf [mu_eff] = {n_inf_eff}")
    print(f"(measured ttr at L=16, the longest tested occlusion: {summary[16]['ema_ttr_mean']:.2f})")

    print()
    print("=" * 70)
    print("STEP 5: right-censoring check (scalar channel)")
    print("=" * 70)
    print("The paper's Section IV-C attributes the residual gap between the")
    print("mu_eff-corrected prediction and the measured mean ttr, from L=8")
    print("onward, to right-censoring at the scan budget (NOT to trials")
    print("crossing early), i.e. a growing fraction of trials never recover")
    print("a correct classification within the tested window at all:")
    for L, cap in ((8, 27), (12, 23), (16, 19)):
        ttrs = np.array(measure_ttr_distribution(L, n_trials=200))
        frac_capped = np.mean(ttrs >= cap)
        print(f"L={L:>2}  mean={ttrs.mean():.2f}  median={np.median(ttrs):.1f}  "
              f"std={ttrs.std():.2f}  min={ttrs.min()}  "
              f"remaining_budget_cap={cap}  frac_never_recovered={frac_capped:.2f}")
    print()
    print("(paper cites 43% at L=8, 56% at L=12, 82% at L=16 -- should match above)")

    print()
    print("=" * 70)
    print("STEP 6: right-censoring check (d=3 multivariate, Section V-F)")
    print("=" * 70)
    print("Confirms the same pattern holds, at a lower rate, once the")
    print("multivariate script's own tau_novel bug (see recover_multivariate_tau.py)")
    print("is fixed and the correct calibrated threshold is used.")
    from run_experiment_multivariate import run_single_trial as mv_run_single_trial
    from run_experiment_multivariate import time_to_recover as mv_time_to_recover
    for L, cap in ((8, 27), (12, 23), (16, 19)):
        ttrs = []
        for trial in range(200):
            seed = deterministic_seed(('mv', L, trial))
            res = mv_run_single_trial(L, seed, n_scans=50, occlusion_start=15)
            ttrs.append(mv_time_to_recover(res['ema_correct'], 15 + L, 50))
        ttrs = np.array(ttrs)
        print(f"L={L:>2}  mean={ttrs.mean():.2f}  remaining_budget_cap={cap}  "
              f"frac_never_recovered={np.mean(ttrs >= cap):.2f}")
    print()
    print("(paper cites 32% at L=8, 51% at L=12, 59% at L=16 for the multivariate case)")
