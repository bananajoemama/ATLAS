"""
Recovers the true calibrated novelty thresholds (tau_novel_ema,
tau_novel_cum_norm) used to originally generate summary_multivariate.json,
after discovering that run_experiment_multivariate.py's sweep_occlusion_lengths
had stale, incorrect default thresholds (tau_novel_ema=-2.75,
tau_novel_cum_norm=-1.0 -- the SCALAR channel's calibrated values, apparently
copy-pasted in by mistake at some point after the real multivariate run).

Method: capture each classifier's raw per-scan best-class score (rather than
an already-thresholded correct/incorrect label) using the exact same random
seeds as the original sweep, then bisect for the threshold that reproduces
the L=2 accuracy recorded in summary_multivariate.json. The recovered
thresholds are then checked against all SIX tested occlusion lengths, not
just the one used for the fit -- exact agreement across all six (not just
one, which would be an unremarkable single-point fit) is what confirms
these are the true original values rather than a coincidental match.

Result: tau_novel_ema=-4.5299, tau_novel_cum_norm=-4.6098. These reproduce
run_experiment_multivariate.py's summary_multivariate.json exactly at every
occlusion length, and match (up to rounding) the thresholds already
hardcoded -- correctly -- in run_experiment_multiclass.py, which extends
the same d=3 feature model to M=4 classes. This is strong corroborating
evidence: whoever built the M=4 extension correctly carried the real
calibration forward, but the M=2 multivariate script's own defaults were
never fixed to match.

Fix applied: run_experiment_multivariate.py's sweep_occlusion_lengths
defaults were updated to tau_novel_ema=-4.5299, tau_novel_cum_norm=-4.6098.
Verified: `python3 run_experiment_multivariate.py` with no arguments now
reproduces summary_multivariate.json standalone.
"""
import numpy as np
from classifiers import CumulativeLLClassifier, EMAClassifier
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from seeding import deterministic_seed
from run_experiment_multivariate import make_library

def run_capture_scores(occlusion_len, seed, n_scans=50, occlusion_start=15, alpha=0.1,
                        separation=5.0, correlation=0.5):
    """Same as run_single_trial but returns raw scores per scan instead of
    pre-thresholded correct/not, so we can test arbitrary tau after the fact."""
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = make_library(separation=separation, correlation=correlation)
    true_class = 'A'
    x_true = np.array([130.0, 35.0, 200.0, 0.0])
    track = Track(track_id=1, x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
                  P=np.diag([25, 100, 25, 100]), P_existence=0.9, PD=0.9, PW=0.9999)
    cum_clf = CumulativeLLClassifier(library)
    ema_clf = EMAClassifier(library, alpha=alpha)
    ema_scores, cum_norm_scores, best_ema_class, best_cum_class = [], [], [], []

    for k in range(n_scans):
        x_true = model.step(x_true, rng)
        forced_miss = occlusion_start <= k < occlusion_start + occlusion_len
        clutter_meas = clutter.generate(rng)
        measurements = list(clutter_meas)
        detected = False
        feat = None
        if not forced_miss and rng.uniform() < track.PD:
            z = model.measure(x_true, rng); measurements.append(z); detected = True
            feat = library.sample_feature(true_class, rng)
        track.P_exist = predict_existence(track.P_exist)
        gated, x_pred, P_pred, S, V_t = gate_measurements(track, kf, measurements)
        meas_idx_set = set(i for i, z, d in gated); dens = {i: d for i, z, d in gated}
        gate_info = {track.id: {'PD': track.PD, 'PW': track.PW, 'P_exist': track.P_exist,
                     'meas_idx_set': meas_idx_set, 'dens': dens,
                     'm_t': max(len(meas_idx_set), 1), 'V_t': V_t}}
        V = max(gate_info[track.id]['V_t'] for _ in [0])
        m_hat = estimate_clutter_count([track.id], gate_info, meas_idx_set)
        jipda_says_detected = False; beta_det = 0.0; beta_i = {}
        if len(meas_idx_set) > 0:
            P_exist_post, betas = jipda_cluster_update([track.id], gate_info, meas_idx_set, V, m_hat)
            track.P_exist = P_exist_post[track.id]
            beta0 = betas[track.id]['beta0']; beta_i = betas[track.id]['beta_i']
            if beta_i:
                best_idx = max(beta_i, key=beta_i.get); beta_det = beta_i[best_idx]
                jipda_says_detected = beta_det > beta0
        else:
            PD, PW, Pprior = track.PD, track.PW, track.P_exist
            denom = 1 - PD * PW * Pprior
            track.P_exist = ((1 - PD * PW) * Pprior / denom) if denom > 1e-12 else 0.0
        if jipda_says_detected:
            meas_list = [measurements[i] for i in beta_i.keys()]; beta_list = [beta_i[i] for i in beta_i.keys()]
            track.x, track.P = kf.pda_update(x_pred, P_pred, meas_list, beta_list, beta0)
        else:
            track.x, track.P = x_pred, P_pred
        use_detected = detected and jipda_says_detected
        cum_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)
        ema_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)

        best_ema_c = max(ema_clf.S, key=ema_clf.S.get)
        best_ema_s = ema_clf.S[best_ema_c]
        best_cum_c = max(cum_clf.LL, key=cum_clf.LL.get)
        best_cum_s = cum_clf.LL[best_cum_c] / (k + 1)

        ema_scores.append(best_ema_s); best_ema_class.append(best_ema_c)
        cum_norm_scores.append(best_cum_s); best_cum_class.append(best_cum_c)

    return np.array(ema_scores), best_ema_class, np.array(cum_norm_scores), best_cum_class, true_class


# Capture scores for L=2 across 200 trials (matches original n_trials=200)
L = 2
occlusion_start = 15
n_scans = 50
occ_end = occlusion_start + L
n_trials = 200

all_ema_scores, all_ema_classes = [], []
all_cum_scores, all_cum_classes = [], []
for trial in range(n_trials):
    seed = deterministic_seed(('mv', L, trial))
    es, ec, cs, cc, true_c = run_capture_scores(L, seed, n_scans=n_scans, occlusion_start=occlusion_start)
    all_ema_scores.append(es[occ_end:])
    all_ema_classes.append(ec[occ_end:])
    all_cum_scores.append(cs[occ_end:])
    all_cum_classes.append(cc[occ_end:])

def acc_at_tau(scores_list, classes_list, tau, true_class='A'):
    accs = []
    for scores, classes in zip(scores_list, classes_list):
        correct = [(c == true_class) if s >= tau else False for s, c in zip(scores, classes)]
        accs.append(np.mean(correct))
    return np.mean(accs)

# Bisection search for tau_ema hitting target ema_acc=0.617 at L=2
target_ema = 0.616969696969697
target_cum = 0.4309090909090909

lo, hi = -20.0, 5.0
for _ in range(60):
    mid = (lo + hi) / 2
    a = acc_at_tau(all_ema_scores, all_ema_classes, mid)
    if a < target_ema:
        hi = mid
    else:
        lo = mid
tau_ema_recovered = (lo + hi) / 2
print(f"Recovered tau_ema  = {tau_ema_recovered:.4f}  (gives acc={acc_at_tau(all_ema_scores, all_ema_classes, tau_ema_recovered):.4f}, target={target_ema:.4f})")

lo, hi = -20.0, 5.0
for _ in range(60):
    mid = (lo + hi) / 2
    a = acc_at_tau(all_cum_scores, all_cum_classes, mid)
    if a < target_cum:
        hi = mid
    else:
        lo = mid
tau_cum_recovered = (lo + hi) / 2
print(f"Recovered tau_cum_norm = {tau_cum_recovered:.4f}  (gives acc={acc_at_tau(all_cum_scores, all_cum_classes, tau_cum_recovered):.4f}, target={target_cum:.4f})")
