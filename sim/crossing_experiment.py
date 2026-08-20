"""
Multi-target crossing comparison, reproducing the scenario of Musicki &
Evans (2002) Section 4 / Table 1: two targets on intersecting trajectories,
tracked with JIPDA, tallying the five possible outcomes at scan 24.

Extended here with a classification-correctness dimension: each target
carries a distinct class-feature identity, and we additionally report
whether ATLAS (EMA) correctly classifies each surviving track's identity
at the crossing point, layered on top of JIPDA's existing association
output. This isolates whether the classification layer adds value beyond
JIPDA's association performance alone.
"""
import numpy as np
from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import (
    Track, gate_measurements, jipda_cluster_update, build_clusters,
    predict_existence, estimate_clutter_count,
)
from classifiers import ClassLibrary, EMAClassifier, CumulativeLLClassifier
from seeding import deterministic_seed


def make_two_target_trajectories(rng, n_scans=24):
    """
    Target 1: x'(1) = [130, 35, 200, 0], constant velocity, edges the
    high-clutter patches (from the JIPDA paper's scenario).
    Target 2: straight-line trajectory designed to cross target 1 at
    scan 19 with a 10-degree crossing angle.
    """
    model = ConstantVelocityModel()

    x1 = np.array([130.0, 35.0, 200.0, 0.0])
    traj1 = [x1.copy()]
    for k in range(n_scans - 1):
        x1 = model.step(x1, rng)
        traj1.append(x1.copy())
    traj1 = np.array(traj1)

    # Target 1's position at scan 19 (index 18) is the crossing point.
    cross_point = traj1[18, [0, 2]]
    # Target 1's heading at crossing, to compute a 10-degree offset heading for target 2.
    heading1 = np.arctan2(traj1[18, 3], traj1[18, 1])
    speed2 = 35.0
    heading2 = heading1 + np.radians(10.0)

    # Back-compute target 2's start so it reaches cross_point at scan 19 (t=18s).
    vx2 = speed2 * np.cos(heading2)
    vy2 = speed2 * np.sin(heading2)
    x2_start = cross_point[0] - vx2 * 18
    y2_start = cross_point[1] - vy2 * 18
    x2 = np.array([x2_start, vx2, y2_start, vy2])
    traj2 = [x2.copy()]
    for k in range(n_scans - 1):
        x2 = model.step(x2, rng)
        traj2.append(x2.copy())
    traj2 = np.array(traj2)

    return traj1, traj2


def run_crossing_trial(seed, n_scans=24, alpha=0.1, tau_novel=-2.753,
                        class_means={'T1': 0.0, 'T2': 5.0}):
    """
    Run one multi-target crossing trial with two confirmed tracks
    (initialized at truth, matching the paper's setup which considers
    only cases where both tracks are already confirmed and following
    each target prior to the crossing). Returns the outcome classification
    (a-e, per the paper's taxonomy) plus ATLAS classification correctness
    for each track at scan 24.
    """
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = ClassLibrary(class_means, class_std=1.0)

    traj1, traj2 = make_two_target_trajectories(rng, n_scans=n_scans)
    true_trajs = {'T1': traj1, 'T2': traj2}

    track_a = Track(1, traj1[0].copy(), np.diag([25, 100, 25, 100]),
                     P_existence=0.95, PD=0.9, PW=0.9999)
    track_b = Track(2, traj2[0].copy(), np.diag([25, 100, 25, 100]),
                     P_existence=0.95, PD=0.9, PW=0.9999)
    tracks = {1: track_a, 2: track_b}
    # Ground truth: which true target each track ID *started* following.
    track_truth = {1: 'T1', 2: 'T2'}

    ema_clfs = {1: EMAClassifier(library, alpha=alpha), 2: EMAClassifier(library, alpha=alpha)}

    for k in range(n_scans):
        clutter_meas = clutter.generate(rng)
        measurements = list(clutter_meas)
        meas_true_label = [None] * len(measurements)  # which true target (if any) generated this meas

        for label, traj in true_trajs.items():
            if rng.uniform() < 0.9:  # PD
                z = model.measure(traj[k], rng)
                measurements.append(z)
                meas_true_label.append(label)

        # Existence prediction (Markov Chain One, Eq. 17) BEFORE gating/update:
        # last scan's posterior must decay through the survival model to
        # become this scan's prior. Skipping this step was the main bug.
        for tr in tracks.values():
            if tr.alive:
                tr.P_exist = predict_existence(tr.P_exist)

        # Gate + cluster both tracks
        gate_info = {}
        track_gates = {}
        for tid, tr in tracks.items():
            if not tr.alive:
                continue
            gated, x_pred, P_pred, S, V_t = gate_measurements(tr, kf, measurements)
            meas_idx_set = set(i for i, z, d in gated)
            dens = {i: d for i, z, d in gated}
            gate_info[tid] = {
                'PD': tr.PD, 'PW': tr.PW, 'P_exist': tr.P_exist,
                'meas_idx_set': meas_idx_set, 'dens': dens,
                'm_t': max(len(meas_idx_set), 1), 'V_t': V_t,
                'x_pred': x_pred, 'P_pred': P_pred,
            }
            track_gates[tid] = meas_idx_set

        if not track_gates:
            continue

        clusters = build_clusters(track_gates)
        for cluster in clusters:
            cluster_meas = set()
            for t in cluster:
                cluster_meas |= gate_info[t]['meas_idx_set']
            V = max(gate_info[t]['V_t'] for t in cluster)
            m_hat = estimate_clutter_count(cluster, gate_info, cluster_meas)

            P_exist_post, betas = jipda_cluster_update(cluster, gate_info, cluster_meas, V, m_hat)

            for tid in cluster:
                tr = tracks[tid]
                tr.P_exist = P_exist_post[tid]
                beta0 = betas[tid]['beta0']
                beta_i = betas[tid]['beta_i']

                jipda_detected = False
                best_idx = None
                beta_det = 0.0
                if beta_i:
                    best_idx = max(beta_i, key=beta_i.get)
                    beta_det = beta_i[best_idx]
                    jipda_detected = beta_det > beta0

                x_pred, P_pred = gate_info[tid]['x_pred'], gate_info[tid]['P_pred']
                if jipda_detected and len(beta_i) > 0:
                    # PDA-weighted soft combination over ALL gated measurements
                    # (not just the single best one) -- this is what the JIPDA
                    # paper specifies for track estimation and is what prevents
                    # a single nearby clutter point from hijacking the update.
                    meas_list = [measurements[i] for i in beta_i.keys()]
                    beta_list = [beta_i[i] for i in beta_i.keys()]
                    tr.x, tr.P = kf.pda_update(x_pred, P_pred, meas_list, beta_list, beta0)
                    assoc_label = meas_true_label[best_idx]
                    feat = (
                        rng.normal(class_means[assoc_label], 1.0)
                        if assoc_label is not None
                        else rng.normal(2.5, 3.0)  # clutter: featureless/wide
                    )
                    ema_clfs[tid].update(True, feature_value=feat, beta_det=beta_det)
                else:
                    tr.x, tr.P = x_pred, P_pred
                    ema_clfs[tid].update(False)

                # Track termination: paper keeps a separate confirmed vs.
                # unconfirmed threshold; we use a single termination
                # threshold here since both tracks start pre-confirmed.
                if tr.P_exist < 0.05:
                    tr.alive = False

    # --- Outcome classification at scan 24, per paper's taxonomy ---
    # We determine "which true target" each surviving track is closest to
    # at the final scan, to decide switch/lost outcomes.
    final_true_pos = {label: traj[-1, [0, 2]] for label, traj in true_trajs.items()}

    def closest_target(track):
        d1 = np.linalg.norm(track.x[[0, 2]] - final_true_pos['T1'])
        d2 = np.linalg.norm(track.x[[0, 2]] - final_true_pos['T2'])
        boundary = 30.0  # meters, "predefined boundary of true target state"
        if d1 < boundary and d1 <= d2:
            return 'T1'
        elif d2 < boundary and d2 <= d1:
            return 'T2'
        return None  # false/lost

    followed = {}
    for tid, tr in tracks.items():
        if tr.P_exist < 0.1:
            followed[tid] = None
        else:
            followed[tid] = closest_target(tr)

    f1, f2 = followed[1], followed[2]
    orig1, orig2 = track_truth[1], track_truth[2]

    if f1 == orig1 and f2 == orig2:
        outcome = 'a'  # both continue original targets
    elif (f1 == orig1) != (f2 == orig2):
        outcome = 'b'  # only one continues
    elif f1 == orig2 and f2 == orig1:
        outcome = 'c'  # both switch
    elif (f1 == orig2 and f2 != orig1) or (f2 == orig1 and f1 != orig2):
        outcome = 'd'  # one switches, other false/lost
    else:
        outcome = 'e'  # both false/lost

    # ATLAS classification correctness at final scan for surviving tracks
    atlas_correct = {}
    for tid, tr in tracks.items():
        if followed[tid] is None:
            atlas_correct[tid] = None
            continue
        pred, _ = ema_clfs[tid].predict(tau_novel=tau_novel)
        true_label_for_track = followed[tid]  # what it's actually following now
        atlas_correct[tid] = (pred == true_label_for_track)

    return outcome, atlas_correct


def run_crossing_experiment(n_trials=300, n_scans=24, alpha=0.1, tau_novel=-2.753):
    outcomes = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
    atlas_correct_given_a = []  # classification accuracy conditioned on correct assoc. outcome

    for i in range(n_trials):
        seed = deterministic_seed(('crossing', i))
        outcome, atlas_correct = run_crossing_trial(seed, n_scans=n_scans, alpha=alpha, tau_novel=tau_novel)
        outcomes[outcome] += 1
        if outcome == 'a':
            vals = [v for v in atlas_correct.values() if v is not None]
            if vals:
                atlas_correct_given_a.append(np.mean(vals))

    print("Outcome tally (n_trials={}):".format(n_trials))
    for k in ['a', 'b', 'c', 'd', 'e']:
        print(f"  ({k}): {outcomes[k]}")
    if atlas_correct_given_a:
        print(f"ATLAS classification accuracy | outcome (a): {np.mean(atlas_correct_given_a):.3f}")
    return outcomes, atlas_correct_given_a


if __name__ == '__main__':
    run_crossing_experiment(n_trials=300)
