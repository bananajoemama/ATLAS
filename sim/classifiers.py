"""
Sequence-scoring classifiers that sit on top of JIPDA's association output.

Implements two variants, evaluated against the same JIPDA association layer:

  1. CumulativeLLClassifier  - the "infinite memory ruin" baseline.
       LL_t = LL_{t-1} + log-likelihood contribution, unboundedly accumulated.
       Missed detections apply a fixed penalty each step, driving LL -> -inf
       over long occlusions with no recovery mechanism.

  2. EMAClassifier (ATLAS)  - Eq. 3-6 from the ATLAS paper.
       LL_t = LL_{t-1} + ln(sum_m pi(m) L(z|m)) + ln(beta_det)     [Eq. 3]
       delta_LL_t = LL_t - LL_{t-1}                 if detection
                  = W_miss (= -8.0)                  if missed detection  [Eq. 4]
       S_t = alpha * delta_LL_t + (1-alpha) * S_{t-1}              [Eq. 5]
       Novelty: S_t < tau_novel  =>  flag UNKNOWN
       Classification: softmax over S_c across candidate library models  [Eq. 6]

Both classifiers are evaluated per-track, per-candidate-class, using the
same beta_det (JIPDA association weight) and per-class measurement
likelihood, so the *only* difference under test is the memory mechanism.
"""
import numpy as np


class ClassLibrary:
    """
    A small library of candidate target classes, each defined by a
    measurement-feature likelihood model. For this simulation the
    'library model' likelihood L(z|m) is evaluated on a synthetic feature
    channel (e.g. RCS/amplitude-like scalar) attached to each measurement,
    distinct from the (x,y) kinematic channel used by JIPDA.
    """

    def __init__(self, class_means, class_std=1.0):
        """
        class_means: dict class_id -> mean feature value
        class_std: shared std of the per-class Gaussian feature likelihood
        """
        self.class_means = class_means
        self.class_std = class_std
        self.class_ids = list(class_means.keys())

    def sample_feature(self, cls_id, rng):
        mean = self.class_means[cls_id]
        return rng.normal(mean, self.class_std)

    def log_likelihood(self, feature_value, cls_id):
        mean = self.class_means[cls_id]
        var = self.class_std ** 2
        return -0.5 * np.log(2 * np.pi * var) - ((feature_value - mean) ** 2) / (2 * var)


class CumulativeLLClassifier:
    """Baseline: unbounded cumulative log-likelihood classifier per class."""

    W_MISS = -8.0

    def __init__(self, library: ClassLibrary):
        self.library = library
        self.LL = {c: 0.0 for c in library.class_ids}  # per-class cumulative LL

    def update(self, detected, feature_value=None, beta_det=1.0):
        for c in self.library.class_ids:
            if detected:
                inst_ll = self.library.log_likelihood(feature_value, c) + np.log(max(beta_det, 1e-12))
                self.LL[c] += inst_ll
            else:
                self.LL[c] += self.W_MISS
        return dict(self.LL)

    def predict(self, tau_novel=-6.0):
        best_c = max(self.LL, key=self.LL.get)
        best_score = self.LL[best_c]
        if best_score < tau_novel:
            return "UNKNOWN", self.LL
        return best_c, self.LL

    def softmax_conf(self):
        vals = np.array(list(self.LL.values()))
        vals = vals - np.max(vals)
        exp = np.exp(vals)
        probs = exp / np.sum(exp)
        return dict(zip(self.LL.keys(), probs))

    def predict_normalized(self, age, tau_novel_per_scan=-1.0):
        """
        Fairer baseline variant: normalize cumulative LL by track age
        (average LL per scan) before thresholding, rather than using a
        fixed raw threshold. Isolates the occlusion-recovery question
        from the general unbounded-drift issue.
        """
        age = max(age, 1)
        normed = {c: v / age for c, v in self.LL.items()}
        best_c = max(normed, key=normed.get)
        if normed[best_c] < tau_novel_per_scan:
            return "UNKNOWN", normed
        return best_c, normed


class EMAClassifier:
    """ATLAS: Exponential Moving Average sequence-scoring classifier."""

    W_MISS = -8.0

    def __init__(self, library: ClassLibrary, alpha=0.1):
        self.library = library
        self.alpha = alpha
        self.LL = {c: 0.0 for c in library.class_ids}       # cumulative LL, only used
                                                              # to compute delta between steps
        self.S = {c: 0.0 for c in library.class_ids}         # EMA sequence score (Eq. 5)

    def update(self, detected, feature_value=None, beta_det=1.0):
        for c in self.library.class_ids:
            prev_LL = self.LL[c]
            if detected:
                inst_ll = self.library.log_likelihood(feature_value, c) + np.log(max(beta_det, 1e-12))
                self.LL[c] = prev_LL + inst_ll
                delta = self.LL[c] - prev_LL
            else:
                delta = self.W_MISS
                self.LL[c] = prev_LL + delta
            self.S[c] = self.alpha * delta + (1 - self.alpha) * self.S[c]
        return dict(self.S)

    def predict(self, tau_novel=-6.0):
        best_c = max(self.S, key=self.S.get)
        best_score = self.S[best_c]
        if best_score < tau_novel:
            return "UNKNOWN", self.S
        return best_c, self.S

    def softmax_conf(self):
        vals = np.array(list(self.S.values()))
        vals = vals - np.max(vals)
        exp = np.exp(vals)
        probs = exp / np.sum(exp)
        return dict(zip(self.S.keys(), probs))
