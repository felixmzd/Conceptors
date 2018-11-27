from binocular import utils
import scipy.sparse.linalg as lin
import math
import numpy as np


class ReservoirRandomFeatureConceptor:

    def __init__(self, N=100, M=500, alpha=8, NetSR=1.4, bias_scale=0.2, inp_scale=1.2):

        self.N = N
        self.M = M
        self.alpha = alpha
        self.NetSR = NetSR
        self.bias_scale = bias_scale
        self.inp_scale = inp_scale

        succ = False
        while not succ:
            try:
                F_raw = np.random.randn(self.M, self.N)
                G_StarRaw = np.random.randn(self.N, self.M)
                GF = G_StarRaw @ F_raw
                specRad, eigenvecs = np.abs(lin.eigs(GF, 1))
                succ = True
            except:
                print('Retrying to generate internal weights.')
                pass

        F_raw = F_raw / math.sqrt(specRad)
        G_StarRaw = G_StarRaw / math.sqrt(specRad)

        self.F = math.sqrt(self.NetSR) * F_raw
        self.G_Star = math.sqrt(self.NetSR) * G_StarRaw
        self.W_bias = self.bias_scale * np.random.randn(self.N)

    def run(self, patterns, t_learn=400, t_learnc=2000, t_wash=200, TyA_wout=1, TyA_wload=0.01,
            gradient_load=False, load=True, c_adapt_rate=0.5, ):

        self.patterns = patterns
        self.t_learn = t_learn
        self.t_learnc = t_learnc
        self.t_wash = t_wash
        self.TyA_wout = TyA_wout
        self.TyA_wload = TyA_wload
        self.gradient_load = gradient_load
        self.c_adapt_rate = c_adapt_rate
        self.n_patts = len(self.patterns)

        if type(self.patterns[0]) == np.ndarray:
            self.n_ip_dim = len(patterns[0][0])
        else:
            if type(self.patterns[0](0)) == np.float64:
                self.n_ip_dim = 1
            else:
                self.n_ip_dim = len(self.patterns[0](0))

        self.W_in = self.inp_scale * np.random.randn(self.N, self.n_ip_dim)

        self.C = []

        self.TrainArgs = np.zeros([self.N, self.n_patts * self.t_learn])
        self.TrainOldArgs = np.zeros([self.M, self.n_patts * self.t_learn])
        self.allT = np.zeros([self.N, self.n_patts * self.t_learn])
        TrainOuts = np.zeros([self.n_ip_dim, self.n_patts * self.t_learn])
        I = np.eye(self.N)

        self.cColls = np.zeros([self.n_patts, self.M, self.t_learnc])

        for i, p in zip(range(self.n_patts), self.patterns):

            z = np.zeros([self.N])
            c = np.ones([self.M])
            cz = np.zeros([self.M])
            rColl = np.zeros([self.N, self.t_learn])
            tColl = np.zeros([self.N, self.t_learn])
            czOldColl = np.zeros([self.M, self.t_learn])
            uColl = np.zeros([self.n_ip_dim, self.t_learn])
            cColl = np.zeros([self.M, self.t_learnc])

            for t in range(self.t_learn + self.t_learnc + self.t_wash):
                if not type(p) == np.ndarray:
                    u = np.reshape(p(t), self.n_ip_dim)
                else:
                    u = p[t]

                czOld = cz
                tmp = self.G_Star @ cz + self.W_in @ u
                r = np.tanh(tmp + self.W_bias)
                z = self.F @ r
                cz = c * z

                if t <= self.t_learnc + self.t_wash and t > self.t_wash:
                    c = c + self.c_adapt_rate * ((cz - c * cz) * cz - math.pow(self.alpha, -2) * c)
                    cColl[:, (t - self.t_wash) - 1] = c

                if t >= self.t_wash + self.t_learnc:
                    offset = t - (self.t_wash + self.t_learnc)

                    rColl[:, offset] = r
                    czOldColl[:, offset] = czOld
                    uColl[:, offset] = u
                    tColl[:, offset] = tmp

            self.C.append(c)

            self.TrainArgs[:, i * self.t_learn:(i + 1) * self.t_learn] = rColl
            self.TrainOldArgs[:, i * self.t_learn:(i + 1) * self.t_learn] = czOldColl
            TrainOuts[:, i * self.t_learn:(i + 1) * self.t_learn] = uColl

            # needed to recomute G
            self.allT[:, i * self.t_learn:(i + 1) * self.t_learn] = tColl

            # plot adaptation of c
            self.cColls[i, :, :] = cColl

        if load:
            """Output Training."""

            self.W_out = utils.RidgeWout(self.TrainArgs, TrainOuts, self.TyA_wout)
            self.NRMSE_readout = utils.NRMSE(np.dot(self.W_out, self.TrainArgs), TrainOuts);
            txt = 'NRMSE for output training = {0}'.format(self.NRMSE_readout)
            print(txt)

            """ Loading """

            G_args = self.TrainOldArgs
            G_targs = self.allT
            self.G = utils.RidgeWload(G_args, G_targs, self.TyA_wload)
            self.NRMSE_load = utils.NRMSE(np.dot(self.G, self.TrainOldArgs), G_targs)
            txt = 'Mean NRMSE per neuron for recomputing G = {0}'.format(np.mean(self.NRMSE_load))
            print(txt)

    def recall(self, t_ctest_wash=200, t_recall=200):

        self.Y_recalls = []
        self.t_ctest_wash = t_ctest_wash
        self.t_recall = t_recall

        for i in range(self.n_patts):
            c = np.asarray(self.C[i])
            cz = .5 * np.random.randn(self.M)
            for t in range(self.t_ctest_wash):
                r = np.tanh(self.G @ cz + self.W_bias)
                cz = c * (self.F @ r)

            y_recall = np.zeros([self.t_recall, self.n_ip_dim])

            for t in range(self.t_recall):
                r = np.tanh(self.G @ cz + self.W_bias)
                cz = c * (self.F @ r)
                y_recall[t] = self.W_out @ r

            self.Y_recalls.append(y_recall)
