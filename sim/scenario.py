"""
Full simulation scenario: two-target crossing situation in clutter,
following Musicki & Evans (2002) Section 4 almost exactly, extended with:
  - a synthetic class-identity feature per target (for ATLAS classification)
  - controlled occlusion injection (forced consecutive missed detections)
  - ground truth bookkeeping for scoring classifier accuracy/recovery

This module runs ONE track (single-target) through JIPDA + a classifier,
to isolate the classification-layer effect cleanly, as agreed in the plan:
JIPDA/association stays fixed; only the classifier (cumulative vs EMA)
varies. Multi-target crossing (Table 1 style) can reuse these pieces.
"""
import numpy as np
from scipy.stats import chi2

from motion import ConstantVelocityModel, KalmanFilter
from clutter import ClutterModel
from jipda import Track, gate_measurements, jipda_cluster_update, predict_existence, estimate_clutter_count
from classifiers import ClassLibrary, CumulativeLLClassifier, EMAClassifier


class SingleTargetOcclusionScenario:
    """
    One true target moving through clutter under JIPDA tracking, with a
    forced occlusion window inserted at a chosen scan, followed by resumed
    detections. Used to directly measure classifier recovery behavior.
    """

    def __init__(
        self,
        n_scans=40,
        occlusion_start=15,
        occlusion_len=8,
        true_class='A',
        class_means={'A': 0.0, 'B': 5.0},
        class_std=1.0,
        PD=0.9,
        PW=0.9999,
        alpha=0.1,
        tau_novel=-6.0,
        initial_state=None,
        seed=0,
    ):
        self.n_scans = n_scans
        self.occlusion_start = occlusion_start
        self.occlusion_len = occlusion_len
        self.true_class = true_class
        self.PD = PD
        self.PW = PW
        self.alpha = alpha
        self.tau_novel = tau_novel

        self.model = ConstantVelocityModel()
        self.kf = KalmanFilter(self.model)
        self.clutter = ClutterModel()
        self.library = ClassLibrary(class_means, class_std=class_std)

        self.rng = np.random.default_rng(seed)
        self.initial_state = (
            initial_state if initial_state is not None
            else np.array([130.0, 35.0, 200.0, 0.0])
        )

    def _in_occlusion(self, k):
        return self.occlusion_start <= k < self.occlusion_start + self.occlusion_len

    def run(self):
        """
        Returns a dict of per-scan logs:
          cum_pred, ema_pred: predicted class label ('A'/'B'/'UNKNOWN') per scan
          cum_correct, ema_correct: bool, whether prediction == true_class
        """
        rng = self.rng
        x_true = self.initial_state.copy()

        track = Track(
            track_id=1,
            x=x_true + rng.normal(0, 5, size=4) * np.array([1, 0, 1, 0]),
            P=np.diag([25, 100, 25, 100]),
            P_existence=0.9,
            PD=self.PD,
            PW=self.PW,
        )

        cum_clf = CumulativeLLClassifier(self.library)
        ema_clf = EMAClassifier(self.library, alpha=self.alpha)

        log = {
            'cum_pred': [], 'ema_pred': [],
            'cum_correct': [], 'ema_correct': [],
            'detected': [], 'true_pos': [],
        }

        for k in range(self.n_scans):
            x_true = self.model.step(x_true, rng)
            forced_miss = self._in_occlusion(k)

            clutter_meas = self.clutter.generate(rng)
            measurements = list(clutter_meas)

            detected = False
            feat = None
            if not forced_miss and rng.uniform() < self.PD:
                z = self.model.measure(x_true, rng)
                measurements.append(z)
                detected = True
                feat = self.library.sample_feature(self.true_class, rng)

            track.P_exist = predict_existence(track.P_exist)
            gated, x_pred, P_pred, S, V_t = gate_measurements(track, self.kf, measurements)
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

            if len(meas_idx_set) > 0:
                P_exist_post, betas = jipda_cluster_update(
                    [track.id], gate_info, meas_idx_set, V, m_hat
                )
                track.P_exist = P_exist_post[track.id]
                beta0 = betas[track.id]['beta0']
                beta_i = betas[track.id]['beta_i']

                # Find which gated measurement (if any) corresponds to the
                # true detection we injected, to feed the classifier's
                # beta_det (use max-beta association as JIPDA's chosen one).
                if beta_i:
                    best_idx = max(beta_i, key=beta_i.get)
                    beta_det = beta_i[best_idx]
                    jipda_says_detected = beta_det > beta0
                else:
                    beta_det = 0.0
                    jipda_says_detected = False
            else:
                track.P_exist *= (1 - track.PD * track.PW)
                jipda_says_detected = False
                beta_det = 0.0

            # Kinematic update: simple nearest/PDA-weighted update to keep
            # the track near truth (not the focus of this experiment, but
            # needed so gating continues to work over time).
            if jipda_says_detected and detected:
                z_upd = measurements[best_idx] if best_idx < len(clutter_meas) else measurements[best_idx]
                track.x, track.P = self.kf.update(x_pred, P_pred, measurements[best_idx])
            else:
                track.x, track.P = x_pred, P_pred

            # --- Classifier layer ---
            use_detected = detected and jipda_says_detected
            cum_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)
            ema_clf.update(use_detected, feature_value=feat, beta_det=beta_det if use_detected else 1.0)

            cpred, _ = cum_clf.predict(tau_novel=self.tau_novel)
            epred, _ = ema_clf.predict(tau_novel=self.tau_novel)

            log['cum_pred'].append(cpred)
            log['ema_pred'].append(epred)
            log['cum_correct'].append(cpred == self.true_class)
            log['ema_correct'].append(epred == self.true_class)
            log['detected'].append(use_detected)
            log['true_pos'].append(x_true[[0, 2]].copy())

        return log
