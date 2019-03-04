from typing import *
import math
import scipy.sparse.linalg as lin
import numpy as np
from sklearn.linear_model import Ridge

from . import utils


class ReservoirRandomFeatureConceptor:
    def __init__(
            self,
            F,
            G_star,
            W_bias,
            regressor,
            W_in,
            aperture=10,
            inp_scale=1.2,
            t_learn=400,
            t_learn_conceptor=2000,
            t_wash=200,
            c_adapt_rate=0.5,
            alpha_wload=0.01,
    ):
        """

        Args:
            F:
            G_star:
            W_bias:
            regressor: The model used to learn the mapping from the reservoir space to the outputs.
            aperture:
            inp_scale:
            t_learn: Number of learning time steps.
            t_learn_conceptor: Number of conceptor learning steps.
            t_wash: Number of washout time steps.
        """
        self.F = F
        self.G_star = G_star
        self.W_bias = W_bias

        self.M, self.N = F.shape
        """N: size of reservoir space
           M: size of feature space
        """

        self.aperture = aperture
        self.inp_scale = inp_scale
        self.regressor = regressor
        self.W_in = W_in
        self.conceptors = []
        self.t_learn_regressor = t_learn
        self.t_learn_conceptor = t_learn_conceptor
        self.t_wash = t_wash
        self.c_adapt_rate = c_adapt_rate
        self.n_patterns = None
        self.c_colls = None
        self.alpha_wload = alpha_wload
        self.history = {}

    @classmethod
    def init_random(
            cls,
            N=100,
            M=500,
            NetSR=1.4,
            bias_scale=0.2,
            aperture=8,
            inp_scale=1.2,
            t_learn=400,
            t_learn_conceptor=2000,
            t_wash=200,
    ):
        F, G_star, W_bias = ReservoirRandomFeatureConceptor._generate_connection_matrices(
            N, M, NetSR, bias_scale
        )
        W_in = inp_scale * np.random.randn(F.shape[1])

        return cls(
            F,
            G_star,
            W_bias,
            Ridge(alpha=1),
            W_in=W_in,
            aperture=aperture,
            inp_scale=inp_scale,
            t_learn=t_learn,
            t_learn_conceptor=t_learn_conceptor,
            t_wash=t_wash,
        )

    def fit(self, patterns: List[Callable[[int], float]]):
        """Load pattens into the reservoir.

        Args:
            patterns: List of functions producing patterns.
        """

        self._init_conceptors(patterns)

        self._init_history()
        self._drive_with_random_input(patterns)
        self._regularize_G()

        for i, pattern in enumerate(patterns):
            self._learn_one_pattern(pattern, i)

        self._train_regressor()

    # def fit(self, patterns: List[Callable[[int], float]]):
    #     """Load pattens into the reservoir.
    #
    #     Args:
    #         patterns: List of functions producing patterns.
    #     """
    #
    #     self._init_conceptors(patterns)
    #
    #     self._init_history()
    #     self.G = self.G_star
    #     for i, pattern in enumerate(patterns):
    #         self._learn_one_pattern(pattern, i)
    #
    #     self._train_regressor()
    #     self._regularize_G()

    def _init_conceptors(self, patterns):
        self.n_patterns = len(patterns)
        self.conceptors = np.ones([self.n_patterns, self.M])

    def _init_history(self):
        """Initialize arrays to write the training history to.

        u: input
        preactivations: recurrent_input + external_input.
        r: reservoir space
        c: conceptors
        z_scaled: Feature space after scaling by conceptor.
        """
        # These history objects follow the indexing scheme:
        # pattern, timestep[, vector_size].
        self.history["u"] = np.zeros([self.n_patterns, self.t_learn_regressor])
        self.history["preactivations"] = np.zeros(
            [self.n_patterns, self.t_learn_regressor, self.N]
        )
        self.history["r"] = np.zeros([self.n_patterns, self.t_learn_regressor, self.N])
        self.history["c"] = np.zeros([self.n_patterns, self.t_learn_conceptor, self.M])
        self.history["z_scaled"] = np.zeros(
            [self.n_patterns, self.t_learn_regressor, self.M]
        )

        self.history["z"] = np.zeros(
            [self.n_patterns, self.t_learn_regressor, self.M]
        )
        self.history["recurrent_input"] = np.zeros(
            [self.n_patterns, self.t_learn_regressor, self.N]
        )

    def _drive_with_random_input(self, patterns):
        for pattern_idx, _ in enumerate(patterns):
            z = np.zeros([self.M])
            for t in range(self.t_learn_regressor + self.t_learn_conceptor + self.t_wash):
                u = 2 * np.random.random() - 1  # TODO why not Gaussian?
                z_old = z
                recurrent_input = self.G_star @ z
                external_input = self.W_in * u
                # Compute reservoir space through non-linearity.
                r = np.tanh(recurrent_input + external_input + self.W_bias)
                # Project to feature space.
                z = self.F @ r
                # Scale by conception weights.
                in_regressor_learning_phase = (self.t_wash + self.t_learn_conceptor) < t
                if in_regressor_learning_phase:
                    offset = t - self.t_wash - self.t_learn_conceptor

                    self.history["z"][pattern_idx, offset] = z_old
                    self.history["recurrent_input"][pattern_idx, offset] = recurrent_input

    def _regularize_G(self):
        """Adapt weights to be able to generate output while driving with random input."""
        features = self.history["z"].reshape(
            -1, self.history["z"].shape[-1]
        )
        targets = self.history["recurrent_input"].reshape(
            -1, self.history["recurrent_input"].shape[-1]
        )
        self.G = Ridge(self.alpha_wload).fit(features, targets).coef_
        self.NRMSE_load = utils.NRMSE(self.G @ features.T, targets.T)
        print(f"Mean NRMSE per neuron for recomputing G = {np.mean(self.NRMSE_load)}")

    # def _regularize_G(self):
    #     """Adapt weights to be able to generate output while driving with random input."""
    #     features = self.history["z_scaled"].reshape(
    #         -1, self.history["z_scaled"].shape[-1]
    #     )
    #     targets = self.history["preactivations"].reshape(
    #         -1, self.history["preactivations"].shape[-1]
    #     )
    #     self.G = Ridge(self.alpha_wload).fit(features, targets).coef_
    #     self.NRMSE_load = utils.NRMSE(self.G @ features.T, targets.T)
    #     print(f"Mean NRMSE per neuron for recomputing G = {np.mean(self.NRMSE_load)}")

    def _learn_one_pattern(self, pattern, pattern_idx):
        self.z_scaled = np.zeros([self.M])
        for t in range(self.t_learn_regressor + self.t_learn_conceptor + self.t_wash):
            self._learn_one_step(pattern(t), t, pattern_idx)

    def _learn_one_step(self, u, t, pattern_idx):
        z_scaled_old = self.z_scaled
        recurrent_input = self.G @ self.z_scaled
        external_input = self.W_in * u
        # Compute reservoir space through non-linearity.
        r = np.tanh(recurrent_input + external_input + self.W_bias)
        # Project to feature space.
        z = self.F @ r
        # Scale by conception weights.
        self.z_scaled = self.conceptors[pattern_idx] * z
        in_conceptor_learning_phase = (
                self.t_wash < t <= (self.t_wash + self.t_learn_conceptor)
        )
        in_regressor_learning_phase = (self.t_wash + self.t_learn_conceptor) < t
        if in_conceptor_learning_phase:
            self._adapt_conceptor(pattern_idx)
            self.history["c"][pattern_idx, t - self.t_wash - 1] = self.conceptors[
                pattern_idx
            ]

        elif in_regressor_learning_phase:
            offset = t - self.t_wash - self.t_learn_conceptor
            self._write_history(
                offset,
                external_input,
                pattern_idx,
                r,
                recurrent_input,
                t,
                u,
                z_scaled_old,
            )

    def _adapt_conceptor(self, pattern_idx):
        self.conceptors[pattern_idx] += self.c_adapt_rate * (
                (self.z_scaled - self.conceptors[pattern_idx] * self.z_scaled)
                * self.z_scaled
                - (self.aperture ** -2) * self.conceptors[pattern_idx]
        )

    def _write_history(
            self,
            offset,
            external_input,
            pattern_idx,
            r,
            recurrent_input,
            t,
            u,
            z_scaled_old,
    ):
        self.history["r"][pattern_idx, offset] = r
        self.history["z_scaled"][pattern_idx, offset] = z_scaled_old
        self.history["u"][pattern_idx, offset] = u
        self.history["preactivations"][pattern_idx, offset] = (
                recurrent_input + external_input
        )

    def _train_regressor(self):
        """Output training with linear regression."""
        features = self.history["r"].reshape(-1, self.history["r"].shape[-1])
        targets = self.history["u"].reshape(-1)
        self.regressor.fit(features, targets)
        self.NRMSE_readout = utils.NRMSE(self.regressor.predict(features), targets)
        print(f"NRMSE for output training = {self.NRMSE_readout}")

    def recall(self, t_washout=200, t_recall=200):
        """Reproduce all learned patterns.

        Args:
            t_washout: Number of washout timesteps.
            t_recall: Number of timesteps to be recalled.

        Returns:
            The recalled patterns.
        """
        Y_recalls = []

        for i in range(self.n_patterns):
            Y_recalls.append(self._recall_pattern(i, t_washout, t_recall))
        return Y_recalls

    def _recall_pattern(self, pattern_index, t_washout, t_recall):
        conceptor = np.asarray(self.conceptors[pattern_index])
        z_scaled = 0.5 * np.random.randn(self.M)
        for t in range(t_washout):
            z_scaled, _ = self._run_conceptor(conceptor, z_scaled)

        y_recall = np.zeros([t_recall])
        for t in range(t_recall):
            z_scaled, r = self._run_conceptor(conceptor, z_scaled)
            y_recall[t] = self.regressor.predict(r[None, :])
        return y_recall

    def _run_conceptor(self, conceptor, z_scaled):
        recurrent_input = self.G @ z_scaled
        r = np.tanh(recurrent_input + self.W_bias)
        z_scaled = conceptor * (self.F @ r)
        return z_scaled, r

    @staticmethod
    def _generate_connection_matrices(N, M, NetSR, bias_scale):
        F_raw, G_star_raw, spectral_radius = ReservoirRandomFeatureConceptor._init_connection_weights(
            N, M
        )
        F_raw, G_star_raw = ReservoirRandomFeatureConceptor._rescale_connection_weights(
            F_raw, G_star_raw, spectral_radius
        )

        F = math.sqrt(NetSR) * F_raw
        G_star = math.sqrt(NetSR) * G_star_raw
        W_bias = bias_scale * np.random.randn(N)

        return F, G_star, W_bias

    @staticmethod
    def _init_connection_weights(N, M):
        success = False
        while not success:
            try:
                F_raw = np.random.randn(M, N)
                G_star_raw = np.random.randn(N, M)
                GF = G_star_raw @ F_raw
                spectral_radius, _ = np.abs(lin.eigs(GF, 1))
                success = True
            except Exception as ex:  # TODO what exceptions can happen here.
                print("Retrying to generate internal weights.")
                print(ex)

        return F_raw, G_star_raw, spectral_radius

    @staticmethod
    def _rescale_connection_weights(F_raw, G_star_raw, spectral_radius):
        F_raw = F_raw / math.sqrt(spectral_radius)
        G_star_raw = G_star_raw / math.sqrt(spectral_radius)
        return F_raw, G_star_raw
