import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as lin


def Lorenz(delta, subsampleRate, sampleLength, washout):
    ls = np.array([[10.036677794959058], [9.98674414052542], [29.024692318601613]]) + 0.01 * np.random.randn(3, 1)
    sigma = 10.0
    b = 8.0 / 3
    r = 28.0
    series = np.zeros((2, sampleLength))

    for n in range(washout):
        ls = ls + delta * np.array([[sigma * (ls[1, 0] - ls[0, 0])],
                                    [r * ls[0, 0] - ls[1, 0] - ls[0, 0] * ls[2, 0]],
                                    [ls[0, 0] * ls[1, 0] - b * ls[2, 0]]])

    for n in range(sampleLength):
        for k in range(subsampleRate):
            ls = ls + delta * np.array([[sigma * (ls[1, 0] - ls[0, 0])],
                                        [r * ls[0, 0] - ls[1, 0] - ls[0, 0] * ls[2, 0]],
                                        [ls[0, 0] * ls[1, 0] - b * ls[2, 0]]])
        series[:, n] = [ls[0], ls[2]];
    # normalize range
    maxval = series.max(1)
    minval = series.min(1)

    series = np.linalg.inv(np.diag(maxval - minval)) @ (series - np.tile(np.array([minval]).T, (1, sampleLength)));
    return series


def IntWeights(N, M, connectivity):
    succ = False
    while not succ:
        try:
            W_raw = sparse.rand(N, M, format='lil', density=connectivity)
            rows, cols = W_raw.nonzero()
            for row, col in zip(rows, cols):
                W_raw[row, col] = np.random.randn()
            specRad, eigenvecs = np.abs(lin.eigs(W_raw, 1))
            W_raw = np.squeeze(np.asarray(W_raw / specRad))
            succ = True
            return W_raw
        except:
            print('Retrying to generate internal weights.')
            pass


def NRMSE(output, target):
    combinedVar = 0.5 * (np.var(target, 1) + np.var(output, 1))
    error = output - target

    return np.sqrt(np.mean(error ** 2, axis=1) / combinedVar)


def RidgeWout(TrainArgs, TrainOuts, TychonovAlpha):
    n_n = len(TrainArgs[:, 1])
    return np.transpose(np.dot(np.linalg.inv(np.dot(TrainArgs, TrainArgs.T) + TychonovAlpha * np.eye(n_n)),
                               np.dot(TrainArgs, TrainOuts.T)))


def RidgeWload(TrainOldArgs, W_targets, TychonovAlpha):
    n_n = len(TrainOldArgs[:, 1])
    return np.transpose(np.dot(np.linalg.pinv(np.dot(TrainOldArgs, TrainOldArgs.T) + TychonovAlpha * np.eye(n_n)),
                               np.dot(TrainOldArgs, W_targets.T)))
