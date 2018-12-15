import math
import scipy.sparse.linalg as lin
import numpy as np
from sklearn.linear_model import Ridge

from . import utils


def _generate_connection_matrices(N, M, NetSR, bias_scale):
    F_raw, G_star_raw, spectral_radius = _init_connection_weights(N, M)
    F_raw, G_star_raw = _rescale_connection_weights(F_raw, G_star_raw, spectral_radius)

    F = math.sqrt(NetSR) * F_raw
    G_star = math.sqrt(NetSR) * G_star_raw
    W_bias = bias_scale * np.random.randn(N)

    return F, G_star, W_bias


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


def _rescale_connection_weights(F_raw, G_star_raw, spectral_radius):
    F_raw = F_raw / math.sqrt(spectral_radius)
    G_star_raw = G_star_raw / math.sqrt(spectral_radius)
    return F_raw, G_star_raw


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
        F, G_star, W_bias = _generate_connection_matrices(N, M, NetSR, bias_scale)
        return cls(F, G_star, W_bias, Ridge(alpha=1), alpha, inp_scale)

    def run(self, patterns, t_learn=400, t_learnc=2000, t_wash=200, model=1, TyA_wload=0.01,
            load=True, c_adapt_rate=0.5):

        self.n_patterns = len(patterns)

        if isinstance(patterns[0], np.ndarray):
            self.n_ip_dim = len(patterns[0][0])
        else:
            if isinstance(patterns[0](0), np.float64):
                self.n_ip_dim = 1
            else:
                self.n_ip_dim = len(patterns[0](0))

        W_in = self.inp_scale * np.random.randn(self.N, self.n_ip_dim)

        features = np.zeros([self.N, self.n_patterns * t_learn])
        train_old_args = np.zeros([self.M, self.n_patterns * t_learn])
        all_t = np.zeros([self.N, self.n_patterns * t_learn])
        targets = np.zeros([self.n_ip_dim, self.n_patterns * t_learn])

        self.c_colls = np.zeros([self.n_patterns, self.M, t_learnc])

        for i, p in enumerate(patterns):

            c = np.ones([self.M])
            cz = np.zeros([self.M])
            rColl = np.zeros([self.N, t_learn])
            tColl = np.zeros([self.N, t_learn])
            czOldColl = np.zeros([self.M, t_learn])
            uColl = np.zeros([self.n_ip_dim, t_learn])
            cColl = np.zeros([self.M, t_learnc])

            for t in range(t_learn + t_learnc + t_wash):
                if isinstance(p, np.ndarray):
                    u = p[t]
                else:
                    u = np.reshape(p(t), self.n_ip_dim)

                cz_old = cz
                tmp = self.G_star @ cz + W_in @ u
                r = np.tanh(tmp + self.W_bias)
                z = self.F @ r
                cz = c * z

                if t_wash < t <= t_wash + t_learnc:
                    c = c + c_adapt_rate * ((cz - c * cz) * cz - math.pow(self.alpha, -2) * c)
                    cColl[:, (t - t_wash) - 1] = c

                if t_wash + t_learnc <= t:
                    offset = t - t_wash - t_learnc

                    rColl[:, offset] = r
                    czOldColl[:, offset] = cz_old
                    uColl[:, offset] = u
                    tColl[:, offset] = tmp

            self.C.append(c)

            features[:, i * t_learn:(i + 1) * t_learn] = rColl
            train_old_args[:, i * t_learn:(i + 1) * t_learn] = czOldColl
            targets[:, i * t_learn:(i + 1) * t_learn] = uColl

            # needed to recomute G
            all_t[:, i * t_learn:(i + 1) * t_learn] = tColl

            # plot adaptation of c
            self.c_colls[i, ...] = cColl

        if load:
            # Output Training with linear regression.
            self.regressor.fit(features.T, targets.flatten())
            self.NRMSE_readout = utils.NRMSE(self.regressor.predict(features.T)[None, :], targets)
            txt = f'NRMSE for output training = {self.NRMSE_readout}'
            print(txt)

            # Adapt weights to be able to generate output while driving with random input.
            G_args = train_old_args
            G_targs = all_t
            self.G = utils.RidgeWload(G_args, G_targs, TyA_wload)
            self.NRMSE_load = utils.NRMSE(self.G @ train_old_args, G_targs)
            txt = f'Mean NRMSE per neuron for recomputing G = {np.mean(self.NRMSE_load)}'
            print(txt)

    def recall(self, t_ctest_wash=200, t_recall=200):

        self.Y_recalls = []
        self.t_ctest_wash = t_ctest_wash
        self.t_recall = t_recall

        for i in range(self.n_patterns):
            c = np.asarray(self.C[i])
            cz = .5 * np.random.randn(self.M)
            for t in range(self.t_ctest_wash):
                r = np.tanh(self.G @ cz + self.W_bias)
                cz = c * (self.F @ r)

            y_recall = np.zeros([self.t_recall, self.n_ip_dim])

            for t in range(self.t_recall):
                r = np.tanh(self.G @ cz + self.W_bias)
                cz = c * (self.F @ r)
                y_recall[t] = self.regressor.predict(r[None, :])

            self.Y_recalls.append(y_recall)
