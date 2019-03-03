import math
import numpy as np
import scipy.sparse.linalg as lin

from binocular import utils


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
            spectral_radius, eigenvecs = np.abs(lin.eigs(GF, 1))
            success = True
        except Exception as ex:  # TODO what exceptions can happen here.
            print('Retrying to generate internal weights.')
            print(ex)

    return F_raw, G_star_raw, spectral_radius


def _rescale_connection_weights(F_raw, G_star_raw, spectral_radius):
    F_raw = F_raw / math.sqrt(spectral_radius)
    G_star_raw = G_star_raw / math.sqrt(spectral_radius)
    return F_raw, G_star_raw


class ReservoirBinocular:

    def __init__(self, F, G_star, W_bias, alpha=10, inp_scale=1.2):

        self.F = F
        self.G_star = G_star
        self.W_bias = W_bias

        self.M, self.N = F.shape

        self.alpha = alpha
        self.inp_scale = inp_scale

    @classmethod
    def init_random(cls, N=100, M=700, NetSR=1.4, bias_scale=0.2, alpha=10, inp_scale=1.2):
        F, G_star, W_bias = _generate_connection_matrices(N, M, NetSR, bias_scale)
        return cls(F, G_star, W_bias, alpha, inp_scale)

    def fit(self, patterns, t_learn=600, t_learnc=2000, t_wash=200, TyA_wout=1, TyA_wload=0.01, c_adapt_rate=0.5):
        """Feed input, learn output weights, adapt weights."""
        self.patterns = patterns
        self.t_learn = t_learn
        self.t_learnc = t_learnc
        self.t_wash = t_wash
        self.TyA_wout = TyA_wout
        self.TyA_wload = TyA_wload
        self.c_adapt_rate = c_adapt_rate
        self.n_patts = len(self.patterns)

        ### variable name dictioanry: ###
        #
        # Variables:
        # p: input to the reservoir
        # r: state of the neurons in the reservoir pool
        # phi: result of mapping from reservoir to feature neurons by F
        # c: conception weights
        # c_phi: activations in feature space, resulting from applying conceptor
        #        weights c to phi.
        # z: result of mapping feature space activations to reservoir space by G
        #
        # Mappings:
        # F: from reservoir neuron pool to feature neuron pool
        # G: from feature neuron pool to reservoir neuron pool

        # space to store network related data from driving with random input
        # this is used to rescale G
        self.all_rand = {}
        self.all_rand['r'] = np.zeros([self.N, self.n_patts * self.t_learn])
        self.all_rand['p'] = np.zeros([1, self.n_patts * self.t_learn])
        self.all_rand['z'] = np.zeros([self.N, self.n_patts * self.t_learn])
        self.all_rand['old_cphi'] = np.zeros([self.M, self.n_patts * self.t_learn])

        # checks whether input is delivered in array or by function handle
        # and sets input dimensionality
        if isinstance(self.patterns[0], np.ndarray):
            self.n_ip_dim = len(patterns[0][0])
        else:
            if type(self.patterns[0](0)) == np.float64:
                self.n_ip_dim = 1
            else:
                self.n_ip_dim = len(self.patterns[0](0))

                # draw input weights according to input dimensionality
        self.W_in = self.inp_scale * np.random.randn(self.N, self.n_ip_dim)
        #######################################################################

        # drive reservoir with random input
        rando = {}
        for i, p in enumerate(self.patterns):
            rando['r'] = np.zeros([self.N, self.t_learn])
            rando['p'] = np.zeros([1, self.t_learn])
            rando['z'] = np.zeros([self.N, self.t_learn])
            rando['old_cphi'] = np.zeros([self.M, self.t_learn])
            cphi = np.zeros([self.M])

            for t in range(self.t_learn + self.t_learnc + self.t_wash):
                p = 2 * np.random.rand() - 1
                old_cphi = cphi
                z = self.G_star @ cphi
                r = np.tanh(z + np.squeeze(self.W_in * p) + self.W_bias)
                cphi = self.F @ r  # c is the identity in the random driving condition,
                # therefore dismissed here

                if t >= self.t_wash + self.t_learnc:
                    offset = (self.t_wash + self.t_learnc)
                    rando['r'][:, t - offset] = r
                    rando['p'][:, t - offset] = p
                    rando['z'][:, t - offset] = z
                    rando['old_cphi'][:, t - offset] = np.squeeze(old_cphi)

            self.all_rand['r'][:, i * self.t_learn:(i + 1) * self.t_learn] = rando['r']
            self.all_rand['p'][:, i * self.t_learn:(i + 1) * self.t_learn] = rando['p']
            self.all_rand['z'][:, i * self.t_learn:(i + 1) * self.t_learn] = rando['z']
            self.all_rand['old_cphi'][:, i * self.t_learn:(i + 1) * self.t_learn] = rando['old_cphi']

        # recompute G
        features = self.all_rand['old_cphi']
        targets = self.all_rand['z']
        self.G = utils.RidgeWload(features, targets, 0.1)
        nrmse_g = np.mean(utils.NRMSE(self.G @ features, targets))
        txt = 'NRMSE for recomputing G = {0}'.format(nrmse_g)
        print(txt)

        #######################################################################

        I = np.eye(self.N)
        self.C = []

        # space to store network related data from training
        self.all_train = {}
        self.all_train['r'] = np.zeros([self.N, self.n_patts * self.t_learn])
        self.all_train['old_r'] = np.zeros([self.N, self.n_patts * self.t_learn])
        self.all_train['cphi'] = np.zeros([self.M, self.n_patts * self.t_learn])
        self.all_train['old_cphi'] = np.zeros([self.M, self.n_patts * self.t_learn])
        self.all_train['z'] = np.zeros([self.N, self.n_patts * self.t_learn])
        self.all_train['p'] = np.zeros([1, self.n_patts * self.t_learn])

        self.cColls = np.zeros([self.n_patts, self.M, self.t_learnc])
        self.raw_Z = np.zeros([self.M, self.n_patts])
        train = {}
        for i, p in enumerate(self.patterns):
            train['r'] = np.zeros([self.N, self.t_learn])
            train['old_r'] = np.zeros([self.N, self.t_learn])
            train['phi'] = np.zeros([self.M, self.t_learn])
            train['cphi'] = np.zeros([self.M, self.t_learn])
            train['old_cphi'] = np.zeros([self.M, self.t_learn])
            train['p'] = np.zeros([1, self.t_learn])
            train['c'] = np.zeros([self.M, self.t_learnc])
            train['z'] = np.zeros([self.N, self.t_learn])

            r = np.zeros([self.N])
            c = np.ones([self.M])
            cphi = np.zeros([self.M])

            for t in range(self.t_learn + self.t_learnc + self.t_wash):
                if not type(p) == np.ndarray:
                    u = np.reshape(p(t), self.n_ip_dim)
                else:
                    u = p[t]
                old_cphi = cphi
                old_r = r
                z = self.G @ cphi
                r = np.tanh(z + self.W_in @ u + self.W_bias)
                phi = self.F @ r
                cphi = c * phi

                if self.t_wash < t <= self.t_learnc + self.t_wash:
                    c = c + self.c_adapt_rate * ((cphi - c * cphi) * cphi - math.pow(self.alpha, -2) * c)
                    train['c'][:, (t - self.t_wash) - 1] = c

                if t >= self.t_wash + self.t_learnc:
                    offset = t - (self.t_wash + self.t_learnc)

                    train['r'][:, offset] = r
                    train['old_r'][:, offset] = old_r
                    train['phi'][:, offset] = phi
                    train['cphi'][:, offset] = cphi
                    train['old_cphi'][:, offset] = old_cphi
                    train['p'][:, offset] = u
                    train['z'][:, offset] = z

            self.C.append(c)

            self.all_train['r'][:, i * self.t_learn:(i + 1) * self.t_learn] = train['r']
            self.all_train['old_r'][:, i * self.t_learn:(i + 1) * self.t_learn] = train['old_r']
            self.all_train['p'][:, i * self.t_learn:(i + 1) * self.t_learn] = train['p']
            self.all_train['z'][:, i * self.t_learn:(i + 1) * self.t_learn] = train['z']
            self.all_train['cphi'][:, i * self.t_learn:(i + 1) * self.t_learn] = train['cphi']
            self.all_train['old_cphi'][:, i * self.t_learn:(i + 1) * self.t_learn] = train['old_cphi']

            signal_energy = np.power(train['old_cphi'], 2)
            self.raw_Z[:, i] = np.mean(signal_energy, axis=1)

        # compute feature space energies for every pattern
        # they are used to indirectly compose a weighted disjunction of the prototype conception weight vectors
        # together with the aperture the mean signal energies define a conception weight vector
        # Feature space energy is a meaure for how well the conceptor fits the activations.
        # normalize
        norms_Z = np.sqrt(np.sum(np.power(self.raw_Z, 2), axis=0))
        mean_norms_Z = np.mean(norms_Z)
        txt = 'Mean feature space energys for every pattern = {0}'.format(norms_Z)
        print(txt)
        # prototype mean signal energy vector matrix
        self.Z = (self.raw_Z @ np.diag(1. / norms_Z)) * mean_norms_Z

        # Output Training with linear regression.
        features = self.all_train['r']
        targets = self.all_train['p']
        self.W_out = utils.RidgeWout(features, targets, self.TyA_wout)
        self.NRMSE_readout = utils.NRMSE(np.dot(self.W_out, features), targets)
        txt = 'NRMSE for output training = {0}'.format(self.NRMSE_readout)
        print(txt)

        # Loading
        # Adapt weights to be able to generate output while driving with random input.
        features = self.all_train['old_cphi']
        targets = self.all_train['p']
        self.D = utils.RidgeWload(features, targets, self.TyA_wload)
        self.NRMSE_load = utils.NRMSE(np.dot(self.D, features), targets)
        txt = 'Mean NRMSE per neuron for recomputing D = {0}'.format(np.mean(self.NRMSE_load))
        print(txt)

    def recall(self, t_washout=200, t_recall=200):
        """Get conceptor for each pattern and regenerate pattern according to conceptor.

        Args:
            t_recall: Number of timesteps to recall.
        """
        self.Y_recalls = []
        self.t_ctest_wash = t_washout
        self.t_recall = t_recall

        for i in range(self.n_patts):
            c = np.asarray(self.C[i])
            cphi = .5 * np.random.randn(self.M)
            r = .1 * np.random.randn(self.N)
            for t in range(self.t_ctest_wash):
                r = np.tanh(self.G @ cphi + self.W_in @ self.D @ cphi + self.W_bias)
                cphi = c * (self.F @ r)

            y_recall = np.zeros([self.t_recall, self.n_ip_dim])

            for t in range(self.t_recall):
                r = np.tanh(self.G @ cphi + self.W_in @ self.D @ cphi + self.W_bias)
                cphi = c * (self.F @ r)
                y_recall[t] = self.W_out @ r

            self.Y_recalls.append(y_recall)

        return self.Y_recalls

    def binocular(self, t_run=4000):

        # number of pattern templates for which conceptors have been learned beforehand
        self.n_templates = 2
        self.t_run = t_run

        # dict for collecting various variables over all patterns
        all = {}

        # level of the hierarchy are denoted by 1 (bottom-) 2 (mid-) 3 (upper-) layer

        # hypotheses for each level about which pattern is the current driver
        all['hypo1'] = np.zeros([self.n_templates, t_run])
        all['hypo2'] = np.zeros([self.n_templates, t_run])
        all['hypo3'] = np.zeros([self.n_templates, t_run])

        # trusts in the hypothesis for each level
        all['trusts1'] = np.zeros([1, t_run])
        all['trusts2'] = np.zeros([1, t_run])
        all['trusts3'] = np.zeros([1, t_run])

        all['unexplained1'] = np.zeros([1, t_run])
        all['unexplained2'] = np.zeros([1, t_run])
        all['unexplained3'] = np.zeros([1, t_run])

        all['trusts12'] = np.zeros([1, t_run])
        all['trusts23'] = np.zeros([1, t_run])

        # also collect binocular condition driver
        all['driver_sine1'] = np.zeros([1, t_run])
        all['driver_sine2'] = np.zeros([1, t_run])
        all['driver_noise'] = np.zeros([1, t_run])

        all['driver'] = np.zeros([1, t_run])

        all['real_input'] = np.zeros([1, t_run])
        all['real_input_w-o_noise'] = np.zeros([1, t_run])

        # output of all layers
        all['y1'] = np.zeros([1, t_run])
        all['y2'] = np.zeros([1, t_run])
        all['y3'] = np.zeros([1, t_run])

        # set equal probabilities to all hypotheses initially
        hypo1 = np.ones([1, self.n_templates])
        # hypo1[0][1] = 0
        hypo1 = hypo1 / np.sum(hypo1)
        # also on all levels
        hypo2 = hypo1
        hypo3 = hypo1

        # t????
        t1 = self.Z @ np.power(hypo1, 2).T
        t2 = self.Z @ np.power(hypo2, 2).T
        t3 = self.Z @ np.power(hypo3, 2).T

        c1 = t1 / (t1 + 1 / np.power(self.alpha, 2))
        c2 = t2 / (t2 + 1 / np.power(self.alpha, 2))
        c3 = t3 / (t3 + 1 / np.power(self.alpha, 2))

        c1_int = c1
        c2_int = c2
        c3_int = c3

        cphi1 = np.zeros([self.M, 1])
        cphi2 = np.zeros([self.M, 1])
        cphi3 = np.zeros([self.M, 1])

        y1var = 1
        y2var = 1
        y3var = 1

        y1mean = 0
        y2mean = 0
        y3mean = 0

        y3 = 0

        trust12 = 0.5
        trust23 = 0.5

        discrepancy1 = 0.5
        discrepancy2 = 0.5
        discrepancy3 = 0.5

        # SNR = 0.5
        SNR = 1.2
        noise_level = np.std(self.all_train['p']) / SNR

        trust_smooth_rate = 0.99
        trust_adapt_steepness12 = 8
        trust_adapt_steepness23 = 8
        drift = 0.0001
        hypo_adapt_rate = 0.002

        # WORKAROUND
        self.W_bias = np.reshape(self.W_bias, (self.N, 1))

        ### Dictionary of Variables ###
        # inext
        # inaut

        # testing trusts
        # zer = np.zeros([1,200])
        # on = np.ones([1, 200])
        # tru = [zer], [on]
        # truu = np.squeeze(np.tile(tru, 10))
        # print(np.shape(truu))
        inaut = 0
        for t in range(t_run):
            # TODO this is the part where the special part for binocular comes in.
            # TODO however are we not just showing the other pattern once the system thinks
            # is is one pattern pattern? this would mean the whole thing is just equivalent to what
            # Jäger did...

            # only the part of the stimulus is fed into the system that can not be explained by the current hypothesis
            if hypo3[0][0] > hypo3[0][1]:
                u = 1.0 * self.patterns[1](t)
            else:
                u = 1.0 * self.patterns[0](t)


            # u = self.patterns[0](t) #+ self.patterns[1](t)) #- inaut  # Does not work maybe because of phase shift.
            # u = self.patterns[1](t)
            # u = self.patterns[0](t)*hypo3[0][1] + self.patterns[1](t)*hypo3[0][0]  # Creates to fast switching and is not plausible as well.
            # noise = noise_level * np.random.randn()  # TODO is this noise really Gaussian?
            noise = np.random.normal(loc=0, scale=noise_level)

            # LEVEL 1 #
            inext = u + noise

            # real input to the system
            all['real_input'][:, t] = inext
            all['real_input_w-o_noise'][:, t] = u

            # input without feedback loop
            all['driver_sine1'][:, t] = self.patterns[0](t)
            all['driver_sine2'][:, t] = self.patterns[1](t)
            all['driver_noise'][:, t] = noise
            all['driver'][:, t] = self.patterns[1](t) + self.patterns[0](t) + noise

            inaut = self.D @ cphi1

            r1 = np.tanh(self.G @ cphi1 + self.W_in * inext + self.W_bias)
            cphi1 = c1 * (self.F @ r1)
            y1 = self.W_out @ r1

            y1mean = trust_smooth_rate * y1mean + (1 - trust_smooth_rate) * y1
            y1var = trust_smooth_rate * y1var + (1 - trust_smooth_rate) * (y1 - y1mean) ** 2
            unexplained1 = inaut - inext
            discrepancy1 = trust_smooth_rate * discrepancy1 + (1 - trust_smooth_rate) * unexplained1 ** 2 / y1var
            c1_int = c1_int + self.c_adapt_rate * (
                    (cphi1 - c1_int * cphi1) * cphi1 - 1 / np.power(self.alpha, 2) * c1_int)

            # LEVEL 2 #
            inaut = self.D @ cphi2
            inext = y1
            r2 = np.tanh(self.G @ cphi2 + self.W_in * ((1 - trust12) * inext + trust12 * inaut) + self.W_bias)
            cphi2 = c2 * (self.F @ r2)
            y2 = self.W_out @ r2

            y2mean = trust_smooth_rate * y2mean + (1 - trust_smooth_rate) * y2
            y2var = trust_smooth_rate * y2var + (1 - trust_smooth_rate) * (y2 - y2mean) ** 2
            unexplained2 = inaut - inext
            discrepancy2 = trust_smooth_rate * discrepancy2 + (1 - trust_smooth_rate) * unexplained2 ** 2 / y2var
            c2_int = c2_int + self.c_adapt_rate * (
                    (cphi2 - c2_int * cphi2) * cphi2 - 1 / np.power(self.alpha, 2) * c2_int)

            # LEVEL 3 #
            inaut = self.D @ cphi3
            inext = y2
            r3 = np.tanh(self.G @ cphi3 + self.W_in * ((1 - trust23) * inext + trust23 * inaut) + self.W_bias)
            cphi3 = c3 * (self.F @ r3)
            y3 = self.W_out @ r3

            y3mean = trust_smooth_rate * y3mean + (1 - trust_smooth_rate) * y3
            y3var = trust_smooth_rate * y3var + (1 - trust_smooth_rate) * (y3 - y3mean) ** 2
            unexplained3 = inaut - inext
            discrepancy3 = trust_smooth_rate * discrepancy3 + (1 - trust_smooth_rate) * unexplained3 ** 2 / y3var

            trust12 = (1 / (1 + (discrepancy2 / discrepancy1) ** trust_adapt_steepness12))
            trust23 = (1 / (1 + (discrepancy3 / discrepancy2) ** trust_adapt_steepness23))

            #
            # trust12 = truu[t]
            # trust23 = truu[t]

            t1 = self.Z @ np.power(hypo1, 2).T
            d_hypo1 = 2 * (np.power(cphi1, 2) - t1).T @ self.Z @ np.diag(np.squeeze(hypo1)) + drift * (0.5 - hypo1)
            hypo1 = hypo1 + hypo_adapt_rate * d_hypo1
            hypo1 = hypo1 / np.sum(hypo1)

            t2 = self.Z @ np.power(hypo2, 2).T
            d_hypo2 = 2 * (np.power(cphi2, 2) - t2).T @ self.Z @ np.diag(np.squeeze(hypo2)) + drift * (0.5 - hypo2)
            hypo2 = hypo2 + hypo_adapt_rate * d_hypo2
            hypo2 = hypo2 / np.sum(hypo2)
            # page 129
            t3 = self.Z @ np.power(hypo3, 2).T
            d_hypo3 = 2 * (np.power(cphi3, 2) - t3).T @ self.Z @ np.diag(np.squeeze(hypo3)) + drift * (0.5 - hypo3)
            hypo3 = hypo3 + hypo_adapt_rate * d_hypo3
            hypo3 = hypo3 / np.sum(hypo3)

            c3 = t3 / (t3 + 1 / np.power(self.alpha, 2))
            c2 = trust23 * c3 + (1 - trust23) * c2_int
            c1 = trust12 * c2 + (1 - trust12) * c1_int

            # Save #
            all['hypo1'][:, t] = hypo1
            all['hypo2'][:, t] = hypo2
            all['hypo3'][:, t] = hypo3

            all['unexplained1'][:, t] = unexplained1
            all['unexplained2'][:, t] = unexplained2
            all['unexplained3'][:, t] = unexplained3

            all['y1'][:, t] = y1
            all['y2'][:, t] = y2
            all['y3'][:, t] = y3

            all['trusts1'][:, t] = discrepancy1
            all['trusts2'][:, t] = discrepancy2
            all['trusts3'][:, t] = discrepancy3

            all['trusts12'][:, t] = trust12
            all['trusts23'][:, t] = trust23

            self.all = all
