"""Here we provide implementations as tf.keras.Model
of FedQUIT and its variations reported and documented in the paper
"""
import math
import numpy as np
import tensorflow as tf


def masked_softmax(logits, mask, axis=-1, tau=1.0, eps=1e-12):
    """Row-wise softmax over non-true classes only (mask=1 for kept entries)."""
    logits = logits / tau
    # subtract max for stability (over all classes, mask applied after exp)
    logits = logits - tf.reduce_max(logits, axis=axis, keepdims=True)
    exps = tf.exp(logits) * mask
    denom = tf.reduce_sum(exps, axis=axis, keepdims=True) + eps
    return exps / denom

class ModelFedQuitLogitDynamic(tf.keras.Model):
    """
    FedQUIT (logit masking) with optional dynamic v_i* per sample:
      (i) per-sample min logit: v_i = min_{c} z^g_{i}
      (ii) max-entropy teacher: v_i* = sum_{c!=y_i} q_{i,c} * z^g_{i,c}, where q = softmax(z^g over c!=y_i).
    If dynamic_v == False, a fixed v is applied
      (iii) v_i = v
    """
    def __init__(
        self,
        model: tf.keras.Model,
        global_model: tf.keras.Model,
        v: float = None,                  # if None and dynamic_v=True -> compute v_i* per sample
        model_type: str = "ResNet18",
        dynamic_v: bool = False,          # turn on per-sample entropy-maximizing and per-sample min v
        dynamic_type: str = "min",        # min || entropy
        tau: float = 1.0,                 # temperature for teacher/student softmax
        scale_kd_by_tau2: bool = False,   # optional KD scaling (tau^2) if you use T>1
    ):
        super().__init__()
        self.model = model
        self.global_model = global_model
        self.model_type = model_type
        self.v = v
        self.dynamic_v = dynamic_v
        self.dynamic_type = dynamic_type
        self.tau = float(tau)
        self.scale_kd_by_tau2 = bool(scale_kd_by_tau2)

    def _extract_logits(self, out):
        return out.logits if self.model_type in ["MitB0"] else out

    def train_step(self, data):
        x, y = data  # y is int class indices, shape [B]
        y = tf.cast(y, tf.int32)

        # --- Teacher (global) logits (no grad) ---
        g_out = self.global_model(x, training=False)
        z_g = tf.cast(self._extract_logits(g_out), tf.float32)
        z_g = tf.stop_gradient(z_g)  # do not backprop through the global model
        B = tf.shape(z_g)[0]
        C = tf.shape(z_g)[1]

        # --- Compute v (fixed or dynamic per-sample) ---
        if self.dynamic_v:
            if self.dynamic_type == "min":
                updates = tf.math.reduce_min(z_g, axis=1)
            elif self.dynamic_type == "entropy":
                # one-hot true-class mask
                if np.ndim(y) == 1:
                    oh = tf.one_hot(y, depth=C, dtype=z_g.dtype)               # [B,C]
                else:
                    oh = tf.cast(y, dtype=z_g.dtype)
                nontrue_mask = 1.0 - oh                                    # 1 for c!=y
                # q over non-true classes
                q_nontrue = masked_softmax(z_g, nontrue_mask, axis=1, tau=self.tau)  # [B,C]
                # dot over all classes is fine: q_nontrue at true class is ~0
                v_star = tf.reduce_sum(q_nontrue * z_g, axis=1)            # [B]
                updates = v_star
            else:
                print("Select the right dynamic type (min, entropy).")
                exit()
        else:
            # fixed v for all samples
            if self.v is None:
                raise ValueError("Either set a fixed v or enable dynamic_v=True.")
            updates = tf.fill([B], tf.cast(self.v, tf.float32))        # [B]

        # --- Build virtual-teacher logits by replacing true-class logit with v (or v_i*) ---
        batch_idx = tf.range(B, dtype=tf.int32)
        if np.ndim(y) > 1:
            y = tf.squeeze(y, axis=1)
        idx = tf.stack([batch_idx, y], axis=1)                                   # [B,2]
        z_virt = tf.tensor_scatter_nd_update(z_g, indices=idx, updates=updates)  # [B,C]

        # --- Soft teacher and student probabilities (temperature tau) ---
        p_virt = tf.nn.softmax(z_virt / self.tau, axis=1)               # [B,C]

        with tf.GradientTape() as tape:
            s_out = self.model(x, training=True)
            z_s = tf.cast(self._extract_logits(s_out), tf.float32)
            p_s = tf.nn.softmax(z_s / self.tau, axis=1)                 # [B,C]

            kd_loss = self.compiled_loss(
                p_virt,            # y_true: teacher probs
                p_s,               # y_pred: student probs
                regularization_losses=self.model.losses
            )
            if self.scale_kd_by_tau2 and self.tau != 1.0:
                kd_loss = (self.tau ** 2) * kd_loss

        # --- Update student (local) weights only ---
        trainable_vars = self.model.trainable_variables
        grads = tape.gradient(kd_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # --- Metrics: keep your compiled metrics (e.g., accuracy) vs true y ---
        self.compiled_metrics.update_state(y, p_s)  # p_s are probs; ok for tf.keras.metrics.CategoricalAccuracy if y is one-hot; for SparseCategoricalAccuracy use y as int

        # Return a dict of metric results
        results = {m.name: m.result() for m in self.metrics}
        results["kd_loss"] = kd_loss
        return results

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)
        s_out = self.model(x, training=False)
        z_s = tf.cast(self._extract_logits(s_out), tf.float32)
        p_s = tf.nn.softmax(z_s, axis=1)
        self.compiled_metrics.update_state(y, p_s)
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)


def _logsumexp(x, axis=-1):
    return tf.reduce_logsumexp(x, axis=axis)

class ModelFedQuitLogitDynamicAlternative(tf.keras.Model):
    """
    FedQUIT with per-sample v = min(logits) and non-true structure modes:
      - flatten:   all non-true logits -> log-mean-exp (mass-preserving)
      - topk:      keep top-K non-true, rest -> log-mean-exp of rest (mass-preserving)
      - rankonly_anchor (Option A): build a rank ladder with min == v and preserve non-true mass
    """
    def __init__(
        self,
        model: tf.keras.Model,
        global_model: tf.keras.Model,
        model_type: str = "ResNet18",
        tau: float = 1.0,
        scale_kd_by_tau2: bool = False,
        nontrue_mode: str = "flatten",    # 'flatten' | 'topk' | 'rankonly_anchor'
        topk: int = 0                     # used when nontrue_mode == 'topk'
    ):
        super().__init__()
        self.model = model
        self.global_model = global_model
        self.model_type = model_type
        self.tau = float(tau)
        self.scale_kd_by_tau2 = bool(scale_kd_by_tau2)
        self.nontrue_mode = nontrue_mode
        self.topk = int(topk)

    def _extract_logits(self, out):
        return out.logits if self.model_type in ["MitB0"] else out

    # ---------- helpers ----------
    # def _mask_nontrue(self, C, y):
    #     oh = tf.one_hot(y, depth=C, dtype=tf.bool)            # [B,C]
    #     return tf.logical_not(oh)                             # True on non-true

    def _rank_prob_ladder_nontrue(self, z_g, y, v):
        """
        Probability-ladder baseline (τ-aware, mass-preserving).
        Let M=C-1. After setting true logit to v, the true prob is p_y' = exp(v/τ) / (exp(v/τ)+S_sum).
        We assign non-true probabilities as q_r = p_y' + r*Δ, r=0..M-1,
        with the WORST non-true getting q_0 (= p_y'), BEST getting q_{M-1},
        and Δ chosen so Σ_r q_r = 1 - p_y'. Then z'_S = τ * (log(Z') + log q_r).
        This preserves non-true exp-mass and anchors min non-true logit at v.
        """
        tau = tf.cast(self.tau, z_g.dtype)
        B  = tf.shape(z_g)[0]
        C  = tf.shape(z_g)[1]

        mask_nt = self._mask_nontrue(C, y)                     # [B,C] bool
        very_neg = tf.constant(-1e9, z_g.dtype)

        # Non-true mass (τ-aware)
        z_nt = tf.where(mask_nt, z_g, very_neg)                # -inf on true
        logS = tf.reduce_logsumexp(z_nt / tau, axis=1)         # [B]
        S_sum = tf.exp(logS)                                   # [B]
        ev = tf.exp(v / tau)                                   # [B]
        Zp = ev + S_sum                                        # [B]
        py = ev / Zp                                           # [B]  (true-class prob after setting z_y <- v)

        # Δ step in probability: Δ = 2*(1 - (M+1)*py) / (M*(M-1))
        M_int = C - 1                                          # int tensor
        M = tf.cast(M_int, z_g.dtype)                          # float
        # Handle degenerate M<=1 (binary) → no change
        def degenerate():
            return z_g
        def proceed():
            Delta = 2.0 * (1.0 - (M + 1.0) * py) / (M * (M - 1.0))  # [B]
            # Clamp Δ if numerically negative (very rare) → keeps a valid ladder
            Delta = tf.maximum(Delta, tf.zeros_like(Delta))

            # Ranks over non-true: 0 = best (largest), ..., M-1 = worst
            order = tf.argsort(z_nt, axis=1, direction="DESCENDING")   # [B,C]
            ranks = tf.argsort(order, axis=1)                           # [B,C]
            ranks_nt = tf.where(mask_nt, ranks, tf.zeros_like(ranks))   # [B,C], 0..M-1 on non-true

            # We want worst → r'=0, best → r'=M-1  => r' = (M-1) - rank
            Mr = tf.cast(M_int - 1, ranks_nt.dtype)                     # int
            rprime = tf.cast(Mr - ranks_nt, z_g.dtype)                  # [B,C]

            # q = py + r'*Δ  (only meaningful on non-true)
            py_full    = tf.repeat(py[:, None], C, axis=1)
            Delta_full = tf.repeat(Delta[:, None], C, axis=1)
            q = py_full + rprime * Delta_full

            # numerics
            eps = tf.constant(1e-12, z_g.dtype)
            q = tf.clip_by_value(q, eps, 1.0 - eps)

            # logits over non-true: τ * (log(Z') + log q)
            logZp = tf.math.log(Zp)
            logZp_full = tf.repeat(logZp[:, None], C, axis=1)
            z_new_nt = tau * (logZp_full + tf.math.log(q))

            # write back only on non-true; keep true untouched (set later to v)
            return tf.where(mask_nt, z_new_nt, z_g)

        return tf.cond(tf.less_equal(M_int, 1), degenerate, proceed)


    def _mask_nontrue(self, C, y):
        # [B,C] bool: True where c != y
        return tf.not_equal(tf.range(C, dtype=y.dtype)[tf.newaxis, :], y[:, tf.newaxis])

    def _logsumexp_over_mask(self, z, mask):
        very_neg = tf.constant(-1e9, z.dtype)
        z_masked = tf.where(mask, z, very_neg)
        return _logsumexp(z_masked, axis=1)                   # [B]

    def _flatten_nontrue(self, z_g, y):
        # alpha = log-mean-exp over non-true (mass-preserving)
        B = tf.shape(z_g)[0]; C = tf.shape(z_g)[1]
        mask_nt = self._mask_nontrue(C, y)
        M = tf.cast(C - 1, z_g.dtype)
        logS = self._logsumexp_over_mask(z_g, mask_nt)        # [B]
        alpha = logS - tf.math.log(M)                         # [B]
        alpha_full = tf.repeat(alpha[:, None], C, axis=1)
        return tf.where(mask_nt, alpha_full, z_g)             # put alpha on non-true, keep true as-is (we'll set true->v later)

    def _topk_nontrue(self, z_g, y, K):
        B = tf.shape(z_g)[0]; C = tf.shape(z_g)[1]
        K = tf.minimum(K, C-1)
        mask_nt = self._mask_nontrue(C, y)
        very_neg = tf.constant(-1e9, z_g.dtype)

        # get top-K among non-true by setting true to -inf
        z_mask_nt_only = tf.where(mask_nt, z_g, very_neg)
        topv, topi = tf.math.top_k(z_mask_nt_only, k=K)       # [B,K]
        # build keep-mask for top-K
        row_ids = tf.repeat(tf.range(B)[:, None], K, axis=1)
        keep_mask = tf.scatter_nd(
            tf.stack([tf.reshape(row_ids, [-1]),
                      tf.reshape(topi,   [-1])], axis=1),
            tf.ones([B*K], dtype=tf.bool),
            tf.shape(z_g)
        )
        keep_mask = tf.logical_and(keep_mask, mask_nt)        # only non-true can be kept
        rest_mask = tf.logical_and(mask_nt, tf.logical_not(keep_mask))

        # alpha_K = log-mean-exp over "rest" (mass-preserving for the rest)
        logS_rest = self._logsumexp_over_mask(z_g, rest_mask)                 # [B]
        MR = tf.maximum(tf.cast(C-1-K, z_g.dtype), tf.constant(1.0, z_g.dtype))
        alphaK = logS_rest - tf.math.log(MR)                                   # [B]
        alphaK_full = tf.repeat(alphaK[:, None], C, axis=1)

        # assign: keep top-K original, set rest to alphaK, leave true untouched (set later)
        z_new = tf.where(rest_mask, alphaK_full, z_g)
        # z_new already keeps top-K (keep_mask) as z_g
        return z_new

    # ---- rankonly Option A: ladder with min == v and non-true mass preserved ----
    @staticmethod
    def _G_of_lambda(lmb, M):  # sum_{r=0}^{M-1} exp(lmb * r)
        if abs(lmb) < 1e-8:
            return float(M)
        a = math.exp(lmb)
        return (a**M - 1.0) / (a - 1.0)

    @staticmethod
    def _solve_lambda_np(target, M, iters=40):
        # target = S_sum / exp(v) >= M (since v is global min)
        lo, hi = 0.0, 1.0
        # expand hi until G(hi) >= target
        while ModelFedQuitLogitDynamicAlternative._G_of_lambda(hi, M) < target:
            hi *= 2.0
            if hi > 100.0: break
        for _ in range(iters):
            mid = 0.5*(lo+hi)
            if ModelFedQuitLogitDynamicAlternative._G_of_lambda(mid, M) < target:
                lo = mid
            else:
                hi = mid
        return 0.5*(lo+hi)

    def _rankonly_anchor_nontrue(self, z_g, y, v):
        """
        FedQUIT with per-sample v = min(logits) and non-true structure modes:
          - flatten:            all non-true logits -> log-mean-exp (mass-preserving)
          - topk:               keep top-K non-true, rest -> log-mean-exp of rest (mass-preserving)
          - rankonly_anchor:    geometric ladder in logit space, min == v, mass-preserving
          - rank_prob_ladder:   arithmetic ladder in PROBABILITY space, min prob == p_y', mass-preserving
        """

        B = tf.shape(z_g)[0]; C = tf.shape(z_g)[1]
        dtype = z_g.dtype
        mask_nt = self._mask_nontrue(C, y)

        # ranks over non-true (true class pushed to -inf so it ranks last)
        very_neg = tf.constant(-1e9, dtype)
        z_for_rank = tf.where(mask_nt, z_g, very_neg)
        order = tf.argsort(z_for_rank, axis=1, direction="DESCENDING")     # [B,C]
        ranks_full = tf.argsort(order, axis=1)                              # rank per class
        ranks_nt = tf.where(mask_nt, ranks_full, tf.zeros_like(ranks_full)) # 0..C-1 on non-true, 0 on true
        M = C - 1
        # since true got -inf, it sits at rank C-1; non-true ranks are 0..M-1 already
        r = tf.cast(ranks_nt, dtype)

        # S_sum / exp(v) as target for λ
        logS = self._logsumexp_over_mask(z_g, mask_nt)                      # [B]
        target = tf.exp(logS - v)                                           # [B], >= M

        # solve λ per-sample via numpy_function + map_fn (teacher side: no grads needed)
        def solve_one(t, m):
            # ensure the numpy side returns float32, matching Tout
            l = tf.numpy_function(
                func=lambda tt, mm: np.array(
                    ModelFedQuitLogitDynamicAlternative._solve_lambda_np(float(tt),
                                                                         int(mm)),
                    dtype=np.float32
                ),
                inp=[t, m],
                Tout=tf.float32
            )
            l.set_shape([])
            return tf.cast(l, dtype)  # cast to logits dtype (usually float32)

        lam = tf.map_fn(lambda t: solve_one(t, M), target)                  # [B]
        beta = v + lam * tf.cast(M - 1, dtype)                              # [B]

        beta_full = tf.repeat(beta[:, None], C, axis=1)
        lam_full  = tf.repeat(lam[:,  None], C, axis=1)

        z_ladder = beta_full - lam_full * r                                  # full [B,C]
        # apply ladder only on non-true; leave true as-is (set true->v later)
        return tf.where(mask_nt, z_ladder, z_g)

    # ---------- training ----------
    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)

        if np.ndim(y) > 1:
            y = tf.squeeze(y, axis=1)

        # Teacher logits
        g_out = self.global_model(x, training=False)
        z_g = tf.cast(self._extract_logits(g_out), tf.float32)
        z_g = tf.stop_gradient(z_g)

        B = tf.shape(z_g)[0]; C = tf.shape(z_g)[1]
        mask_nt = self._mask_nontrue(C, y)

        # --- per-sample v = min_c z_g[c] ---
        v_per = tf.reduce_min(z_g, axis=1)                                  # [B]

        # --- non-true reshaping (uses original z_g for mass) ---
        if self.nontrue_mode == "flatten":
            z_nt_mod = self._flatten_nontrue(z_g, y)
        elif self.nontrue_mode == "topk":
            z_nt_mod = self._topk_nontrue(z_g, y, self.topk)
        elif self.nontrue_mode == "rankonly_anchor":
            z_nt_mod = self._rankonly_anchor_nontrue(z_g, y, v_per)
        elif self.nontrue_mode == "rank_prob_ladder":
            z_nt_mod = self._rank_prob_ladder_nontrue(z_g, y, v_per)
        else:
            raise ValueError(f"Unknown nontrue_mode: {self.nontrue_mode}")

        # --- set true-class logit to v ---
        idx = tf.stack([tf.range(B, dtype=tf.int32), y], axis=1)
        z_virt = tf.tensor_scatter_nd_update(z_nt_mod, idx, v_per)          # [B,C]

        # --- KD ---
        p_virt = tf.nn.softmax(z_virt / self.tau, axis=1)
        with tf.GradientTape() as tape:
            s_out = self.model(x, training=True)
            z_s = tf.cast(self._extract_logits(s_out), tf.float32)
            p_s = tf.nn.softmax(z_s / self.tau, axis=1)
            kd_loss = self.compiled_loss(p_virt, p_s, regularization_losses=self.model.losses)
            if self.scale_kd_by_tau2 and self.tau != 1.0:
                kd_loss = (self.tau ** 2) * kd_loss

        grads = tape.gradient(kd_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # metrics
        self.compiled_metrics.update_state(y, tf.nn.softmax(z_s, axis=1))
        out = {m.name: m.result() for m in self.metrics}
        out["kd_loss"] = kd_loss
        return out

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)
        s_out = self.model(x, training=False)
        z_s = tf.cast(self._extract_logits(s_out), tf.float32)
        p_s = tf.nn.softmax(z_s, axis=1)
        self.compiled_metrics.update_state(y, p_s)
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self): return self.model.get_weights()
    def set_weights(self, weights): return self.model.set_weights(weights)


class IncompetentVirtualTeacher(tf.keras.Model):
    """ A virtual teacher that outputs fixed predictions, i.e.,
    1/num_classes for each class."""

    def __init__(self, num_classes,):
        super().__init__()
        self.num_classes = num_classes

    def call(self, inputs):
        x, y = inputs
        output = tf.fill([tf.shape(x)[0], self.num_classes],
                             1.0 / self.num_classes)
        return output

class ModelKLDiv(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
        virtual_model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model
        self.virtual_teacher = virtual_model

    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        virtual_output = self.virtual_teacher(data, training=True)

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            kd_loss = self.compiled_loss(
                # tf.nn.softmax(virtual_output, axis=1),
                virtual_output,
                tf.nn.softmax(local_output, axis=1),
                regularization_losses=self.model.losses
            )


        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(kd_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, local_output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Implement logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        """
        x, y = data
        y_pred = self.model(x, training=False)  # Forward pass
        # self.compiled_loss(y, y_pred, regularization_losses=self.local_model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        """Return the weights of the local model."""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Return the weights of the local model."""
        return self.model.set_weights(weights)


class ModelNoT(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model

    def test_step(self, data):
        """Implement logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        """
        x, y = data
        y_pred = self.model(x, training=False)  # Forward pass
        # self.compiled_loss(y, y_pred, regularization_losses=self.local_model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        """Return the weights of the local model."""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Return the weights of the local model."""
        return self.model.set_weights(weights)


