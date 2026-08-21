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


def initiate_track_two_point_diff(z0, z1, T, meas_std, track_id, PD=0.9, PW=0.9999):
    """
    Two-point differencing track initiation (as referenced, but not
    detailed, in Musicki & Evans Section 4, citing Musicki & Evans 1997/
    Submitted for track initialization via clutter map information). We
    use the standard two-point-differencing formulas: position from the
    most recent measurement, velocity from the finite difference between
    the two measurements, with the corresponding propagated covariance.

    This replaces initializing tracks directly at ground truth, which
    was the primary source of the reproduction gap against the original
    paper's Table 1 (Section 5.5): starting a track already-converged at
    the true state, with an arbitrary fixed covariance and an asserted
    P_exist=0.95, gives the filter an unrealistic head start relative to
    a track that had to actually earn confirmation from raw detections.

    Initial existence probability is set to a low, unconfirmed value
    (0.3) reflecting genuine uncertainty about whether this is a true or
    false (clutter-spawned) track at birth, rather than an asserted
    near-certainty.
    """
    x0, y0 = z0
    x1, y1 = z1
    vx = (x1 - x0) / T
    vy = (y1 - y0) / T
    x_init = np.array([x1, vx, y1, vy])

    # Two-point-difference covariance: position variance = measurement
    # variance; velocity variance = 2 * measurement variance / T^2
    # (standard finite-difference error propagation).
    pos_var = meas_std ** 2
    vel_var = 2 * (meas_std ** 2) / (T ** 2)
    P_init = np.diag([pos_var, vel_var, pos_var, vel_var])

    return Track(track_id, x_init, P_init, P_existence=0.3, PD=PD, PW=PW)


def run_crossing_trial(seed, n_scans=24, alpha=0.1, tau_novel=-2.753,
                        class_means={'T1': 0.0, 'T2': 5.0},
                        confirm_threshold=0.95,
                        term_threshold_unconfirmed=0.10,
                        term_threshold_confirmed=0.02,
                        checkpoint_scan=13):
    """
    Run one multi-target crossing trial with tracks initiated via
    two-point differencing from raw measurements (rather than at ground
    truth), matured through separate confirmed/unconfirmed termination
    thresholds (rather than a single blanket threshold), and filtered at
    a checkpoint scan matching the paper's own methodology: "only cases
    where two confirmed tracks were following each of the two targets
    were considered" (evaluated at scan 14, i.e. checkpoint_scan=13 in
    0-indexed scans, matching the paper's statement that "the true track
    situation is observed on scan 14 and then again on scan 24").

    Returns (outcome, atlas_correct, trial_counts) where trial_counts is
    None if the trial did not pass the checkpoint filter (and should be
    excluded from Table 1, exactly as in the original paper).
    """
    rng = np.random.default_rng(seed)
    model = ConstantVelocityModel()
    kf = KalmanFilter(model)
    clutter = ClutterModel()
    library = ClassLibrary(class_means, class_std=1.0)

    traj1, traj2 = make_two_target_trajectories(rng, n_scans=n_scans)
    true_trajs = {'T1': traj1, 'T2': traj2}

    # --- Two-point-difference initiation from the first two scans ---
    # Generate scan 0 and scan 1 detections for both targets (with real
    # detection noise/misses) to seed initial track state, rather than
    # starting at ground truth.
    z0 = {}
    z1 = {}
    for label, traj in true_trajs.items():
        z0[label] = model.measure(traj[0], rng) if rng.uniform() < 0.9 else None
        z1[label] = model.measure(traj[1], rng) if rng.uniform() < 0.9 else None

    # If either target failed to generate a usable two-scan detection
    # pair, fall back to a slightly-perturbed truth seed for that target
    # only (rare, PD=0.9 means ~1% chance both scans miss for one target)
    # so the trial can still proceed rather than being silently dropped
    # in a way that would bias the sample toward "easy" trials.
    def seed_pair(label, traj):
        a = z0[label] if z0[label] is not None else model.measure(traj[0], rng)
        b = z1[label] if z1[label] is not None else model.measure(traj[1], rng)
        return a, b

    a1, b1 = seed_pair('T1', traj1)
    a2, b2 = seed_pair('T2', traj2)

    track_a = initiate_track_two_point_diff(a1, b1, model.T, model.meas_std, track_id=1)
    track_b = initiate_track_two_point_diff(a2, b2, model.T, model.meas_std, track_id=2)
    tracks = {1: track_a, 2: track_b}
    # Ground truth: which true target each track ID *started* following.
    track_truth = {1: 'T1', 2: 'T2'}

    ema_clfs = {1: EMAClassifier(library, alpha=alpha), 2: EMAClassifier(library, alpha=alpha)}

    final_true_pos = {label: traj[-1, [0, 2]] for label, traj in true_trajs.items()}
    checkpoint_true_pos = {label: traj[checkpoint_scan, [0, 2]] for label, traj in true_trajs.items()}

    def closest_target(track_x, ref_positions, boundary=30.0):
        d1 = np.linalg.norm(track_x[[0, 2]] - ref_positions['T1'])
        d2 = np.linalg.norm(track_x[[0, 2]] - ref_positions['T2'])
        if d1 < boundary and d1 <= d2:
            return 'T1'
        elif d2 < boundary and d2 <= d1:
            return 'T2'
        return None

    checkpoint_passed = None  # set once we reach checkpoint_scan

    # Start the main loop from scan 2 onward, since scans 0-1 were
    # consumed by two-point-difference initiation above.
    for k in range(2, n_scans):
        clutter_meas = clutter.generate(rng)
        measurements = list(clutter_meas)
        meas_true_label = [None] * len(measurements)

        for label, traj in true_trajs.items():
            if rng.uniform() < 0.9:  # PD
                z = model.measure(traj[k], rng)
                measurements.append(z)
                meas_true_label.append(label)

        for tr in tracks.values():
            if tr.alive:
                tr.P_exist = predict_existence(tr.P_exist)

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

        if track_gates:
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
                        meas_list = [measurements[i] for i in beta_i.keys()]
                        beta_list = [beta_i[i] for i in beta_i.keys()]
                        tr.x, tr.P = kf.pda_update(x_pred, P_pred, meas_list, beta_list, beta0)
                        assoc_label = meas_true_label[best_idx]
                        feat = (
                            rng.normal(class_means[assoc_label], 1.0)
                            if assoc_label is not None
                            else rng.normal(2.5, 3.0)
                        )
                        ema_clfs[tid].update(True, feature_value=feat, beta_det=beta_det)
                    else:
                        tr.x, tr.P = x_pred, P_pred
                        ema_clfs[tid].update(False)

                    # Confirmation and separate confirmed/unconfirmed
                    # termination thresholds, matching the paper's
                    # description (Section 4: "Termination thresholds
                    # were kept separate for confirmed and unconfirmed
                    # tracks").
                    if not tr.confirmed and tr.P_exist >= confirm_threshold:
                        tr.confirmed = True
                    term_thresh = term_threshold_confirmed if tr.confirmed else term_threshold_unconfirmed
                    if tr.P_exist < term_thresh:
                        tr.alive = False

        # Checkpoint filter: record pass/fail once, at the paper's own
        # "scan 14" checkpoint, mirroring their methodology of only
        # considering trials where both tracks are already confirmed and
        # correctly following their targets before the crossing.
        if k == checkpoint_scan:
            both_confirmed = all(tr.confirmed and tr.alive for tr in tracks.values())
            if both_confirmed:
                f1 = closest_target(tracks[1].x, checkpoint_true_pos)
                f2 = closest_target(tracks[2].x, checkpoint_true_pos)
                checkpoint_passed = (f1 == track_truth[1] and f2 == track_truth[2])
            else:
                checkpoint_passed = False
            if not checkpoint_passed:
                return None, None  # trial excluded, matching paper's methodology

    # --- Outcome classification at scan 24 ---
    followed = {}
    for tid, tr in tracks.items():
        if not tr.alive or tr.P_exist < term_threshold_confirmed:
            followed[tid] = None
        else:
            followed[tid] = closest_target(tr.x, final_true_pos)

    f1, f2 = followed[1], followed[2]
    orig1, orig2 = track_truth[1], track_truth[2]

    if f1 == orig1 and f2 == orig2:
        outcome = 'a'
    elif (f1 == orig1) != (f2 == orig2):
        outcome = 'b'
    elif f1 == orig2 and f2 == orig1:
        outcome = 'c'
    elif (f1 == orig2 and f2 != orig1) or (f2 == orig1 and f1 != orig2):
        outcome = 'd'
    else:
        outcome = 'e'

    atlas_correct = {}
    for tid, tr in tracks.items():
        if followed[tid] is None:
            atlas_correct[tid] = None
            continue
        pred, _ = ema_clfs[tid].predict(tau_novel=tau_novel)
        atlas_correct[tid] = (pred == followed[tid])

    return outcome, atlas_correct


def run_crossing_experiment(n_trials=300, n_scans=24, alpha=0.1, tau_novel=-2.753):
    outcomes = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
    atlas_correct_given_a = []
    n_excluded = 0
    n_attempted = 0

    i = 0
    while sum(outcomes.values()) < n_trials:
        seed = deterministic_seed(('crossing_v2', i))
        i += 1
        n_attempted += 1
        outcome, atlas_correct = run_crossing_trial(seed, n_scans=n_scans, alpha=alpha, tau_novel=tau_novel)
        if outcome is None:
            n_excluded += 1
            continue
        outcomes[outcome] += 1
        if outcome == 'a':
            vals = [v for v in atlas_correct.values() if v is not None]
            if vals:
                atlas_correct_given_a.append(np.mean(vals))

    print(f"Outcome tally (n_trials={n_trials}, n_attempted={n_attempted}, "
          f"n_excluded_at_checkpoint={n_excluded}):")
    for k in ['a', 'b', 'c', 'd', 'e']:
        print(f"  ({k}): {outcomes[k]}")
    if atlas_correct_given_a:
        print(f"ATLAS classification accuracy | outcome (a): {np.mean(atlas_correct_given_a):.3f}")
    return outcomes, atlas_correct_given_a, n_attempted, n_excluded


if __name__ == '__main__':
    run_crossing_experiment(n_trials=300)
