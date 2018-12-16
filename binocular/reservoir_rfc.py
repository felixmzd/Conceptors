import math
import scipy.sparse.linalg as lin
import numpy as np
from sklearn.linear_model import Ridge

from . import utils


class ReservoirRandomFeatureConceptor:

    def __init__(self, F, G_star, W_bias, regressor, alpha=10, inp_scale=1.2):
        """

        Args:
            F:
            G_star:
            W_bias:
            regressor: The model used to learn the mapping from the reservoir space to the outputs.
            alpha:
            inp_scale:
        """
        self.F = F
        self.G_star = G_star
        self.W_bias = W_bias

        self.M, self.N = F.shape

        self.alpha = alpha
        self.inp_scale = inp_scale
        self.regressor = regressor
        self.C = []
        self.n_patterns = None
        self.n_ip_dim = None
        self.c_colls = None

    @classmethod
    def init_random(cls, N=100, M=500, NetSR=1.4, bias_scale=0.2, alpha=8, inp_scale=1.2):
        F, G_star, W_bias = ReservoirRandomFeatureConceptor._generate_connection_matrices(N, M, NetSR, bias_scale)
        return cls(F, G_star, W_bias, Ridge(alpha=1), alpha, inp_scale)

    def fit(self, patterns, t_learn=400, t_learn_conceptor=2000, t_wash=200, TyA_wload=0.01,
            load=True, c_adapt_rate=0.5):
        """

        Args:
            patterns:
            t_learn: Number of learning time steps.
            t_learn_conceptor: Number of conceptor learning steps.
            t_wash: Number of washout time steps.
            TyA_wload:
            load:
            c_adapt_rate:

        Returns:

        """

        self.n_patterns = len(patterns)
        self.n_ip_dim = self._get_n_ip_dim(patterns)
        self.W_in = self.inp_scale * np.random.randn(self.N, self.n_ip_dim)
        self.c_colls = np.zeros([self.n_patterns, self.M, t_learn_conceptor])
        features = np.zeros([self.N, self.n_patterns * t_learn])
        targets = np.zeros([self.n_ip_dim, self.n_patterns * t_learn])
        old_features = np.zeros([self.M, self.n_patterns * t_learn])

        self.all_t = np.zeros([self.N, self.n_patterns * t_learn])
        for i, pattern in enumerate(patterns):
            self._learn_one_pattern(pattern, i, c_adapt_rate, t_learn, t_learn_conceptor, t_wash, features, targets,
                                    old_features)

        if load:
            self._train_regressor(features, targets)
            self._adapt_G(TyA_wload, old_features)

    # TODO this method might also be avoided by a unified pattern interface.
    def _get_n_ip_dim(self, patterns):

        if isinstance(patterns[0], np.ndarray):
            n_ip_dim = len(patterns[0][0])
        else:
            if isinstance(patterns[0](0), np.float64):
                n_ip_dim = 1
            else:
                n_ip_dim = len(patterns[0](0))
        return n_ip_dim

    def _adapt_G(self, TyA_wload, old_features):
        """Adapt weights to be able to generate output while driving with random input."""
        G_features = old_features
        G_targets = self.all_t
        self.G = utils.RidgeWload(G_features, G_targets, TyA_wload)
        self.NRMSE_load = utils.NRMSE(self.G @ old_features, G_targets)
        txt = f'Mean NRMSE per neuron for recomputing G = {np.mean(self.NRMSE_load)}'
        print(txt)

    def _train_regressor(self, features, targets):
        """Output Training with linear regression."""
        self.regressor.fit(features.T, targets.flatten())
        self.NRMSE_readout = utils.NRMSE(self.regressor.predict(features.T)[None, :], targets)
        txt = f'NRMSE for output training = {self.NRMSE_readout}'
        print(txt)

    def _learn_one_pattern(self, pattern, i, c_adapt_rate, t_learn, t_learn_conceptor, t_wash, features, targets,
                           old_features):
        self.z_scaled = np.zeros([self.M])
        self.r_collected = np.zeros([self.N, t_learn])
        self.c_collected = np.zeros([self.M, t_learn_conceptor])
        self.u_collected = np.zeros([self.n_ip_dim, t_learn])
        self.z_scaled_collected = np.zeros([self.M, t_learn])
        self.t_collected = np.zeros([self.N, t_learn])
        self.c = np.ones([self.M])
        for t in range(t_learn + t_learn_conceptor + t_wash):
            self._learn_one_step(pattern, t, t_learn_conceptor, t_wash, c_adapt_rate)
        self.C.append(self.c)
        collection_index = slice(i * t_learn, (i + 1) * t_learn)
        features[:, collection_index] = self.r_collected
        targets[:, collection_index] = self.u_collected
        old_features[:, collection_index] = self.z_scaled_collected
        # needed to recompute G
        self.all_t[:, collection_index] = self.t_collected
        # plot adaptation of c?
        self.c_colls[i, ...] = self.c_collected

    def _learn_one_step(self, pattern, t, t_learn_conceptor, t_wash, c_adapt_rate):
        u = self._get_pattern_value(pattern, t)  # TODO maybe this should be handled in a unified pattern interface.
        z_scaled_old = self.z_scaled
        recurrent_input = self.G_star @ self.z_scaled
        external_input = self.W_in @ u
        # Compute reservoir space through non-linearity.
        r = np.tanh(recurrent_input + external_input + self.W_bias)
        # Project to feature space.
        z = self.F @ r
        # Scale by conceptor weights.
        self.z_scaled = self.c * z
        # This is probably all state harvesting.
        in_conceptor_learning_phase = t_wash < t <= (t_wash + t_learn_conceptor)
        in_regressor_learning_phase = (t_wash + t_learn_conceptor) <= t
        if in_conceptor_learning_phase:
            self.c += c_adapt_rate * (
                    (self.z_scaled - self.c * self.z_scaled) * self.z_scaled - (self.alpha ** -2) * self.c)
            self.c_collected[:, t - t_wash - 1] = self.c
        # TODO shouldnt this be elif? Currently there is one step overlap because of the equals condition.
        if in_regressor_learning_phase:
            offset = t - t_wash - t_learn_conceptor

            self.r_collected[:, offset] = r
            self.z_scaled_collected[:, offset] = z_scaled_old
            self.u_collected[:, offset] = u
            self.t_collected[:, offset] = recurrent_input + external_input

    def _get_pattern_value(self, pattern, t):
        if isinstance(pattern, np.ndarray):
            value = pattern[t]
        else:
            value = np.reshape(pattern(t), self.n_ip_dim)
        return value

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

        y_recall = np.zeros([t_recall, self.n_ip_dim])
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
