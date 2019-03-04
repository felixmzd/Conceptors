import math
import numpy as np
import scipy.sparse.linalg as lin

from binocular import utils
from .reservoir_rfc import ReservoirRandomFeatureConceptor
from sklearn.linear_model import Ridge


class ReservoirBinocular(ReservoirRandomFeatureConceptor):
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
            depth=3,
    ):
        super().__init__(
            F,
            G_star,
            W_bias,
            regressor,
            W_in,
            aperture,
            inp_scale,
            t_learn,
            t_learn_conceptor,
            t_wash,
            c_adapt_rate,
            alpha_wload,
        )

        self.depth = depth
        self.snr = 1.2
        self.trust_smooth_rate = 0.99
        self.trust_adapt_steepness = 8
        self.drift = 0.0001
        self.hypotheses_learning_rate = 0.002

    def fit(self, patterns):
        super().fit(patterns)
        self.patterns = patterns

    def binocular(self, t_run=4000):
        self._compute_signal_energies()
        self._init_mappings()
        self._init_states()
        self._init_binocular_history(t_run)

        for t in range(t_run):
            u = self._apply_topdown_feedback(t)
            u = self._add_noise(u)

            for layer in range(self.depth):
                self._run_layer(u, layer)

            self._write_binocular_history(t)

    def _run_layer(self, u, l):
        predicted_signal = self.D @ self.z_scaled[l]
        if l == 0:
            self.y[l] = u
        else:
            trust = self.trusts[l - 1]
            u = (1 - trust) + self.y[l - 1] + trust * predicted_signal

        recurrent_input = self.G_star @ self.z_scaled[l]
        external_input = self.W_in * u + self.W_bias
        # Compute reservoir space through non-linearity.
        r = np.tanh(recurrent_input + external_input)
        # Project to feature space.
        z = self.F @ r
        # Scale by conception weights.
        self.z_scaled[l] = self.mixed_conceptors[l] * z
        self.y[l + 1] = self.W_out @ r
        self.y_means[l] = (self.trust_smooth_rate * self.y_means[l]
                           + (1 - self.trust_smooth_rate) * self.y[l])
        self.y_variances[l] = (self.trust_smooth_rate * self.y_variances[l]
                               + (1 - self.trust_smooth_rate)
                               * (self.y[l] - self.y_means[l]) ** 2)
        self.unexplained[l] = predicted_signal - self.y[l]
        self.discrepancies[l] = (self.trust_smooth_rate * self.discrepancies[l]
                                 + ((1 - self.trust_smooth_rate)
                                    * self.unexplained[l] ** 2
                                    / self.y_variances[l]))
        self.auto_conceptors[l] += self.c_adapt_rate * (
                (self.z_scaled[l] - self.auto_conceptors[l] * self.z_scaled[l])
                * self.z_scaled[l]
                - (self.aperture ** -2) * self.auto_conceptors[l]
        )
        # Adapt trusts.
        if l > 0:
            self.trusts[l - 1] = (1 / (1 + (self.discrepancies[l] / self.discrepancies[l - 1])
                                       ** self.trust_adapt_steepness))
        # Calculate hypotheses.
        P_times_gamma = self.P @ (self.hypotheses[l] ** 2)
        hypo_adapt = (2  # TODO where is this 2 coming from?
                      * (self.z_scaled[l] ** 2 - P_times_gamma)
                      @ self.P
                      @ np.diag(self.hypotheses[l])
                      + self.drift * (0.5 - self.hypotheses[l]))
        self.hypotheses[l] += self.hypotheses_learning_rate * hypo_adapt
        self.hypotheses[l] = self.hypotheses[l] / self.hypotheses[l].sum()
        # Remix autoconceptors.
        if l != self.depth - 1:
            self.mixed_conceptors[l] = ((1 - self.trusts[l]) + self.auto_conceptors[l]
                                        + self.trusts[l] * self.auto_conceptors[l + 1])
        else:
            self.mixed_conceptors[l] = (P_times_gamma
                                        / (P_times_gamma
                                           + 1 / self.aperture ** 2))  # TODO ?

    def _init_states(self):
        self.noise_level = np.std(self.history["u"]) / self.snr
        self.y = np.zeros(self.depth + 1)
        self.z_scaled = np.zeros([self.depth, self.M])
        self.auto_conceptors = np.ones([self.depth, self.M])
        self.mixed_conceptors = np.ones([self.depth, self.M])
        self.unexplained = np.zeros(self.depth)
        self.hypotheses = np.ones([self.depth, self.n_patterns]) / self.n_patterns
        self.y_means = np.zeros(self.depth)
        self.y_variances = np.ones(self.depth)

        self.trusts = 0.5 * np.ones(self.depth - 1)
        self.discrepancies = 0.5 * np.ones(self.depth)

    def _write_binocular_history(self, t):
        self.history["y"][t] = self.y
        self.history["trusts"][t] = self.trusts
        self.history["hypotheses"][t] = self.hypotheses
        self.history["unexplained"][t] = self.unexplained
        self.history["patterns"][t] = [pattern(t) for pattern in self.patterns]

    def _compute_signal_energies(self):
        # Compute feature space energies for every pattern.
        # They are used to indirectly compose a weighted disjunction of the prototype conception weight vectors
        # together with the aperture the mean signal energies define a conception weight vector.
        # Feature space energy is a measure for how well the conceptor fits the activations.
        signal_energy = self.history["z_scaled"] ** 2
        # mean signal energy for every pattern.
        # [n_patterns, M]
        # where M is the mean of squared z_scaled.
        P_star = np.mean(signal_energy, axis=1)
        norms_P = np.linalg.norm(P_star, axis=-1)
        mean_norms_P = np.mean(norms_P)
        print("Mean feature space energies for every pattern = {0}".format(norms_P))
        # prototype mean signal energy vector matrix
        # [M, n_patterns]
        self.P: "[M, n_patterns]" = ((P_star.T @ np.diag(1.0 / norms_P)) * mean_norms_P)

    def _init_mappings(self):
        features = self.history["z_scaled"].reshape(
            -1, self.history["z_scaled"].shape[-1]
        )
        targets = self.history["u"].reshape(-1)
        self.D = Ridge(self.alpha_wload).fit(features, targets).coef_
        self.W_out = self.regressor.coef_

    def _apply_topdown_feedback(self, t):
        if self.hypotheses[-1, 0] > self.hypotheses[-1, 1]:
            u = self.patterns[1](t)
        else:
            u = self.patterns[0](t)

        return u

    def _add_noise(self, u):

        noise = np.random.normal(loc=0, scale=self.noise_level)
        return u + noise

    def _init_binocular_history(self, t_run):
        # These history objects follow the indexing scheme:
        # timestep, layer[, pattern].
        self.history["y"] = np.zeros([t_run, self.depth + 1])
        self.history["trusts"] = np.zeros([t_run, self.depth - 1])
        self.history["hypotheses"] = np.zeros([t_run, self.depth, self.n_patterns])
        self.history["unexplained"] = np.zeros([t_run, self.depth])
        self.history["patterns"] = np.zeros([t_run, self.n_patterns])
