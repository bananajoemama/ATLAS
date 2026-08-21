"""
JIPDA (Joint Integrated Probabilistic Data Association) core.

Implements Musicki & Evans (2002), "Joint Integrated Probabilistic Data
Association - JIPDA", Eq. 1-11.

This module handles, for one scan:
  1. Clustering tracks that share candidate measurements
  2. Enumerating joint association events within a cluster
  3. Computing a-posteriori joint event probabilities (Eq. 4)
  4. Marginalizing to get:
       - probability of target existence per track P^t_{k,k}  (Eq. 9)
       - data association coefficients beta^t_i per (track, measurement) (Eq. 10-11)

Track kinematic update (Kalman/PDA) is handled separately in motion.py;
this module only produces the association weights, matching the paper's
statement that "the use of the data association coefficients to update
each track's estimation is well covered [elsewhere]".
"""
import itertools
import numpy as np
from scipy.stats import multivariate_normal


class Track:
    """A single JIPDA track: kinematic state + existence probability."""

    def __init__(self, track_id, x, P, P_existence=0.5, PD=0.9, PW=0.9999, cls_id=None):
        self.id = track_id
        self.x = x            # state estimate (4,)
        self.P = P            # state covariance (4,4)
        self.P_exist = P_existence   # P^t_{k,k-1} going in, P^t_{k,k} coming out
        self.PD = PD
        self.PW = PW
        self.cls_id = cls_id  # ground-truth class, used only for scoring/labels
        self.confirmed = False
        self.age = 0
        self.consecutive_misses = 0
        self.alive = True


def predict_existence(P_exist_prev, p11=0.98, p12=0.02, p21=0.0, p22=1.0):
    """
    Markov Chain One existence propagation (Eq. 17 in Musicki & Evans):
    a target either continues to exist (p11) or ceases to exist (p12);
    once non-existent, it stays non-existent (p22=1, p21=0).

        P^t_{k,k-1} = p11 * P^t_{k-1,k-1} + p21 * (1 - P^t_{k-1,k-1})

    This MUST be applied before each scan's JIPDA update -- using last
    scan's posterior directly as this scan's prior (skipping this decay)
    silently overstates existence confidence during misses.
    """
    return p11 * P_exist_prev + p21 * (1 - P_exist_prev)


def gate_measurements(track, kf, measurements, gate_threshold=None):
    """
    Return list of (index, z, likelihood_density) for measurements falling
    within track's validation gate (window), using the innovation covariance.
    p^t_i is the truncated-Gaussian conditional density (Eq. row under Eq.4).
    """
    x_pred, P_pred = kf.predict(track.x, track.P)
    H, R = kf.model.H, kf.model.R
    S = H @ P_pred @ H.T + R
    S_inv = np.linalg.inv(S)
    det_S = np.linalg.det(S)

    # PW = 0.9999 -> use a large-but-finite gate (chi-square, 2 DOF)
    # For P_W ~ 0.9999 with 2 DOF, gate ~ chi2.ppf(0.9999, 2) ~ 18.4
    from scipy.stats import chi2
    gate = gate_threshold if gate_threshold is not None else chi2.ppf(track.PW, df=2)

    gated = []
    for i, z in enumerate(measurements):
        y = z - H @ x_pred
        d2 = y.T @ S_inv @ y
        if d2 <= gate:
            # truncated Gaussian density approx -> use plain Gaussian density
            dens = multivariate_normal.pdf(z, mean=H @ x_pred, cov=S)
            gated.append((i, z, dens))

    # Actual elliptical gate area (2D): V_t = pi * gate * sqrt(det(S))
    # (area of the ellipse {y : y^T S^-1 y <= gate})
    V_t = np.pi * gate * np.sqrt(det_S)

    return gated, x_pred, P_pred, S, V_t


def build_clusters(track_gates):
    """
    track_gates: dict track_idx -> set of measurement indices in its gate.
    Returns list of clusters, each a list of track indices, grouped by
    shared-measurement connectivity (union-find style).
    """
    track_ids = list(track_gates.keys())
    parent = {t: t for t in track_ids}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    meas_to_tracks = {}
    for t, meas_set in track_gates.items():
        for m in meas_set:
            meas_to_tracks.setdefault(m, []).append(t)

    for m, tlist in meas_to_tracks.items():
        for a, b in zip(tlist, tlist[1:]):
            union(a, b)

    clusters = {}
    for t in track_ids:
        r = find(t)
        clusters.setdefault(r, []).append(t)

    return list(clusters.values())


def compute_cluster_volume(tracks_in_cluster, gate_info, m_total):
    """
    Eq. 1-2: V_ap = (m / sum(m_t)) * sum(V_t),  V = max(V_max, V_ap)
    gate_info[t] must include 'V_t' (window area) and 'm_t' (num measurements
    in that track's window).
    """
    V_max = max(gate_info[t]['V_t'] for t in tracks_in_cluster)
    sum_mt = sum(gate_info[t]['m_t'] for t in tracks_in_cluster)
    sum_Vt = sum(gate_info[t]['V_t'] for t in tracks_in_cluster)
    if sum_mt == 0:
        V_ap = sum_Vt  # degenerate case, no measurements at all
    else:
        T = len(tracks_in_cluster)
        V_ap = (m_total / sum_mt) * sum_Vt if sum_mt > 0 else sum_Vt
    return max(V_max, V_ap)


def estimate_clutter_count(tracks_in_cluster, gate_info, cluster_measurements):
    """
    Eq. 3: a-priori estimated number of clutter measurements m_hat.
    Efficient approximation summed over measurements i in the cluster.
    """
    m_hat = 0.0
    for i in cluster_measurements:
        prod = 1.0
        for t in tracks_in_cluster:
            info = gate_info[t]
            mu_ti = 1 if i in info['meas_idx_set'] else 0
            if mu_ti == 1 and info['m_t'] > 0:
                term = 1 - (info['PD'] * info['PW'] * info['P_exist']) / info['m_t']
                prod *= term
        m_hat += prod
    return max(m_hat, 1e-6)


def enumerate_joint_events(tracks_in_cluster, gate_info):
    """
    Enumerate all valid joint events for a cluster:
      - each track assigned 0 or 1 measurement from its own gate
      - each measurement assigned to at most 1 track
    Returns list of dicts: {track_id: assigned_meas_idx_or_None}
    """
    # For each track, candidate assignments = [None] + its gated measurement indices
    options = []
    for t in tracks_in_cluster:
        meas_opts = [None] + list(gate_info[t]['meas_idx_set'])
        options.append(meas_opts)

    events = []
    for combo in itertools.product(*options):
        # check no two tracks share the same non-None measurement
        assigned = [m for m in combo if m is not None]
        if len(assigned) == len(set(assigned)):
            events.append(dict(zip(tracks_in_cluster, combo)))
    return events


def jipda_cluster_update(tracks_in_cluster, gate_info, cluster_measurements, V, m_hat):
    """
    Core JIPDA math for one cluster in one scan.

    gate_info[t] = {
        'PD', 'PW', 'P_exist' (prior, P^t_{k,k-1}),
        'meas_idx_set': set of measurement indices in track t's gate,
        'dens': dict meas_idx -> p^t_i (measurement likelihood density),
    }
    cluster_measurements: list of measurement indices belonging to this cluster
    V: cluster volume (Eq 1-2)
    m_hat: estimated clutter count (Eq 3)

    Returns:
      P_exist_post: dict track_id -> P^t_{k,k}  (Eq. 9)
      betas: dict track_id -> {meas_idx: beta^t_i, 0: beta^t_0}  (Eq. 10-11)
    """
    events = enumerate_joint_events(tracks_in_cluster, gate_info)

    # Eq. 4: unnormalized probability of each joint event
    raw_probs = []
    for ev in events:
        prob = 1.0
        for t in tracks_in_cluster:
            info = gate_info[t]
            PD, PW, Pexist = info['PD'], info['PW'], info['P_exist']
            assigned = ev[t]
            if assigned is None:
                prob *= (1 - PD * PW * Pexist)
            else:
                p_i = info['dens'][assigned]
                prob *= PD * PW * Pexist * p_i * V / m_hat
        raw_probs.append(prob)

    total = sum(raw_probs)
    if total <= 0:
        # Degenerate: no valid mass, fall back to uniform-ish handling
        norm_probs = [1.0 / len(events)] * len(events) if events else []
    else:
        norm_probs = [p / total for p in raw_probs]

    # Eq. 6-8: marginalize over joint events for each track
    P_exist_post = {}
    betas = {}
    for t in tracks_in_cluster:
        info = gate_info[t]
        PD, PW, Pexist = info['PD'], info['PW'], info['P_exist']

        # P{chi^t chi^t_0 | Z^k} : track exists AND no measurement (Eq. 8 numerator uses Eq.6)
        p_chi0 = sum(pr for ev, pr in zip(events, norm_probs) if ev[t] is None)
        denom = 1 - PD * PW * Pexist
        if denom > 1e-12:
            p_exist_and_no_meas = ((1 - PD * PW) * Pexist / denom) * p_chi0
        else:
            p_exist_and_no_meas = 0.0

        # P{chi^t chi^t_i | Z^k} for each measurement i assigned to t across events
        p_exist_and_meas = {}
        for i in info['meas_idx_set']:
            p_exist_and_meas[i] = sum(
                pr for ev, pr in zip(events, norm_probs) if ev[t] == i
            )

        P_t_kk = p_exist_and_no_meas + sum(p_exist_and_meas.values())
        P_exist_post[t] = min(max(P_t_kk, 0.0), 1.0)

        # Eq. 10-11: beta parameters (normalized by P^t_{k,k})
        if P_t_kk > 1e-12:
            beta0 = p_exist_and_no_meas / P_t_kk
            beta_i = {i: v / P_t_kk for i, v in p_exist_and_meas.items()}
        else:
            beta0 = 1.0
            beta_i = {i: 0.0 for i in info['meas_idx_set']}
        betas[t] = {'beta0': beta0, 'beta_i': beta_i}

    return P_exist_post, betas
