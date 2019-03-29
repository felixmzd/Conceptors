import numpy as np
import scipy.interpolate as interpolate


def conc_sv(C):
    SVS = []
    for i, c in zip(range(len(C)), C):
        SVS.append(np.linalg.svd(c, full_matrices=True)[1])
    return SVS


def plot_interpolate_1d(patterns, Y_recalls, overSFac=20, plotrange=30):
    Driver_int = np.zeros([(plotrange - 1) * overSFac])
    Recall_int = np.zeros([(len(Y_recalls[0]) - 1) * overSFac])
    NRMSEsAlign = np.zeros([len(patterns)])

    allDriverPL = np.zeros([len(patterns), plotrange])
    allRecallPL = np.zeros([len(patterns), plotrange])

    for i, p in zip(range(len(patterns)), patterns):

        p = np.vectorize(p)

        Driver = p(np.linspace(0, plotrange - 1, plotrange, dtype=int))
        Recall = np.squeeze(Y_recalls[i])

        fD = interpolate.interp1d(range(plotrange), Driver, kind='cubic')
        fR = interpolate.interp1d(range(len(Recall)), Recall, kind='cubic')

        Driver_int = fD(np.linspace(0, (len(Driver_int) - 1.) / overSFac, len(Driver_int)))
        Recall_int = fR(np.linspace(0, (len(Recall_int) - 1.) / overSFac, len(Recall_int)))

        L = len(Recall_int)
        M = len(Driver_int)

        phasematches = np.zeros([L - M])

        for s in range(L - M):
            phasematches[s] = np.linalg.norm(Driver_int - Recall_int[s:s + M])

        pos = np.argmin(phasematches)
        Recall_PL = Recall_int[np.linspace(pos, pos + overSFac * (plotrange - 1), plotrange).astype(int)]
        Driver_PL = Driver_int[np.linspace(0, overSFac * (plotrange - 1) - 1, plotrange).astype(int)]

        NRMSEsAlign[i] = NRMSE(np.reshape(Recall_PL, (1, len(Recall_PL))), np.reshape(Driver_PL, (1, len(Driver_PL))))

        allDriverPL[i, :] = Driver_PL
        allRecallPL[i, :] = Recall_PL

    txt = f'NRMSE for aligned reconstruction of signals = {NRMSEsAlign}'
    print(txt)
    return allDriverPL, allRecallPL, NRMSEsAlign


# TODO scikit learn

def NRMSE(output, target):
    combinedVar = 0.5 * (np.var(target) + np.var(output))
    error = output - target

    return np.sqrt(np.mean(error ** 2) / combinedVar)


def RidgeWout(TrainArgs, TrainOuts, TychonovAlpha):
    n_n = len(TrainArgs[:, 1])
    return np.transpose(np.dot(np.linalg.inv(np.dot(TrainArgs, TrainArgs.T) + TychonovAlpha * np.eye(n_n)),
                               np.dot(TrainArgs, TrainOuts.T)))


def RidgeWload(TrainOldArgs, W_targets, TychonovAlpha):
    n_n = len(TrainOldArgs[:, 1])
    return np.transpose(np.dot(np.linalg.pinv(np.dot(TrainOldArgs, TrainOldArgs.T) + TychonovAlpha * np.eye(n_n)),
                               np.dot(TrainOldArgs, W_targets.T)))
