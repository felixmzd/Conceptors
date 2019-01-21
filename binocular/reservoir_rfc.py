import math
import scipy.sparse.linalg as lin
import numpy as np
from sklearn.linear_model import Ridge

from . import utils


class ReservoirRandomFeatureConceptor:

    def __init__(self, F, G_star, W_bias, regressor, aperture=10, inp_scale=1.2, t_learn=400, t_learn_conceptor=2000,
                 t_wash=200, c_adapt_rate=0.5, alpha_wload=0.01):
        """

        Args:
            F:
            G_star:
            W_bias:
            regressor: The model used to learn the mapping from the reservoir space to the outputs.
            aperture:
            inp_scale:
        """
        self.F = F
        self.G_star = G_star
        self.W_bias = W_bias

        self.M, self.N = F.shape

        self.aperture = aperture
        self.inp_scale = inp_scale
        self.regressor = regressor
        self.C = []
        self.t_learn = t_learn
        self.t_learn_conceptor = t_learn_conceptor
        self.t_wash = t_wash
        self.c_adapt_rate = c_adapt_rate
        self.n_patterns = None
        self.n_input_dimensions = None
        self.c_colls = None
        self.alpha_wload = alpha_wload

    @classmethod
    def init_random(cls, N=100, M=500, NetSR=1.4, bias_scale=0.2, aperture=8, inp_scale=1.2):
        F, G_star, W_bias = ReservoirRandomFeatureConceptor._generate_connection_matrices(N, M, NetSR, bias_scale)
        return cls(F, G_star, W_bias, Ridge(alpha=1), aperture, inp_scale)

    def fit(self, patterns):
        """

        Args:
            patterns:
            self.t_learn: Number of learning time steps.
            self.t_learn_conceptor: Number of conceptor learning steps.
            self.t_wash: Number of washout time steps.
            TyA_wload:
            load:
            self.c_adapt_rate:

        Returns:

        """

        self.n_patterns = len(patterns)
        self.n_input_dimensions = 1
        self.W_in = self.inp_scale * np.random.randn(self.N, self.n_input_dimensions)
        self.c_colls = np.zeros([self.n_patterns, self.M, self.t_learn_conceptor])
        self.targets = np.zeros([self.n_input_dimensions,
                                 self.n_patterns * self.t_learn])  # TODO transpose dimensions so we have n_samples x n_features
        self.all_z_scaled_collected = np.zeros([self.M, self.n_patterns * self.t_learn])

        self.all_preactivations = np.zeros(
            [self.N, self.n_patterns * self.t_learn])  # Collects recurrent_input + external_input
        self.features = np.zeros([self.N, self.n_patterns * self.t_learn])
        for i, pattern in enumerate(patterns):
            self._learn_one_pattern(pattern, i)

        self._train_regressor()
        self._load_weight_matrix()

    def _train_regressor(self):
        """Output Training with linear regression."""
        self.regressor.fit(self.features.T, self.targets.flatten())
        self.NRMSE_readout = utils.NRMSE(self.regressor.predict(self.features.T)[None, :], self.targets)
        txt = f'NRMSE for output training = {self.NRMSE_readout}'
        print(txt)

    def _load_weight_matrix(self):
        """Adapt weights to be able to generate output while driving with random input."""
        self.G = Ridge(self.alpha_wload).fit(self.all_z_scaled_collected.T, self.all_preactivations.T).coef_
        self.NRMSE_load = utils.NRMSE(self.G @ self.all_z_scaled_collected, self.all_preactivations)
        txt = f'Mean NRMSE per neuron for recomputing G = {np.mean(self.NRMSE_load)}'
        print(txt)

    def _learn_one_pattern(self, pattern, i):
        self.z_scaled = np.zeros([self.M])
        self.r_collected = np.zeros([self.N, self.t_learn])
        self.c_collected = np.zeros([self.M, self.t_learn_conceptor])
        self.u_collected = np.zeros([self.n_input_dimensions, self.t_learn])
        self.z_scaled_collected = np.zeros([self.M, self.t_learn])
        self.preactivations_collected = np.zeros([self.N, self.t_learn])
        self.c = np.ones([self.M])
        for t in range(self.t_learn + self.t_learn_conceptor + self.t_wash):
            self._learn_one_step(pattern, t)

        self.C.append(self.c)
        collection_index = slice(i * self.t_learn, (i + 1) * self.t_learn)
        self.features[:, collection_index] = self.r_collected
        self.targets[:, collection_index] = self.u_collected
        self.all_z_scaled_collected[:, collection_index] = self.z_scaled_collected
        # needed to recompute G
        self.all_preactivations[:, collection_index] = self.preactivations_collected
        # plot adaptation of c?
        self.c_colls[i, ...] = self.c_collected

    def _learn_one_step(self, pattern, t):
        u = pattern(t)
        z_scaled_old = self.z_scaled
        recurrent_input = self.G_star @ self.z_scaled
        external_input = self.W_in @ u
        # Compute reservoir space through non-linearity.
        r = np.tanh(recurrent_input + external_input + self.W_bias)
        # Project to feature space.
        z = self.F @ r
        # Scale by conception weights.
        self.z_scaled = self.c * z
        # This is probably all state harvesting.
        in_conceptor_learning_phase = self.t_wash < t <= (self.t_wash + self.t_learn_conceptor)
        in_regressor_learning_phase = (self.t_wash + self.t_learn_conceptor) < t
        if in_conceptor_learning_phase:
            self.c += self.c_adapt_rate * (
                    (self.z_scaled - self.c * self.z_scaled) * self.z_scaled - (self.aperture ** -2) * self.c)
            self.c_collected[:, t - self.t_wash - 1] = self.c
        elif in_regressor_learning_phase:
            offset = t - self.t_wash - self.t_learn_conceptor

            self.r_collected[:, offset] = r
            self.z_scaled_collected[:, offset] = z_scaled_old
            self.u_collected[:, offset] = u
            self.preactivations_collected[:, offset] = recurrent_input + external_input

    def recall(self, t_washout=200, t_recall=200):

        Y_recalls = []

        for i in range(self.n_patterns):
            Y_recalls.append(
                self._recall_pattern(i, t_washout, t_recall)
            )
        return Y_recalls

    def _recall_pattern(self, pattern_index, t_washout, t_recall):
        c = np.asarray(self.C[pattern_index])
        z_scaled = 0.5 * np.random.randn(self.M)
        for t in range(t_washout):
            z_scaled, _ = self._run_conceptor(c, z_scaled)

        y_recall = np.zeros([t_recall, self.n_input_dimensions])
        for t in range(t_recall):
            z_scaled, r = self._run_conceptor(c, z_scaled)
            y_recall[t] = self.regressor.predict(r[None, :])
        return y_recall

    def _run_conceptor(self, c, z_scaled):
        recurrent_input = self.G @ z_scaled
        r = np.tanh(recurrent_input + self.W_bias)
        z_scaled = c * (self.F @ r)
        return z_scaled, r

    @staticmethod
    def _generate_connection_matrices(N, M, NetSR, bias_scale):
        F_raw, G_star_raw, spectral_radius = ReservoirRandomFeatureConceptor._init_connection_weights(N, M)
        F_raw, G_star_raw = ReservoirRandomFeatureConceptor._rescale_connection_weights(F_raw, G_star_raw,
                                                                                        spectral_radius)

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
                print('Retrying to generate internal weights.')
                print(ex)

        return F_raw, G_star_raw, spectral_radius

    @staticmethod
    def _rescale_connection_weights(F_raw, G_star_raw, spectral_radius):
        F_raw = F_raw / math.sqrt(spectral_radius)
        G_star_raw = G_star_raw / math.sqrt(spectral_radius)
        return F_raw, G_star_raw
