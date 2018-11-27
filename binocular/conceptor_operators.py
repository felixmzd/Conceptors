"""Logical operators defined on conceptors."""
import numpy as np


def AND(C, B):
    dim = len(C)
    tol = 1e-12

    Uc, Sc, Vc = np.linalg.svd(C, full_matrices=True)
    Ub, Sb, Vb = np.linalg.svd(B, full_matrices=True)

    if np.diag(Sc[Sc > tol]).size:
        numRankC = np.linalg.matrix_rank(np.diag(Sc[Sc > tol]))
    else:
        numRankC = 0
    if np.diag(Sb[Sb > tol]).size:
        numRankB = np.linalg.matrix_rank(np.diag(Sb[Sb > tol]))
    else:
        numRankB = 0

    Uc0 = Uc[:, numRankC:]
    Ub0 = Uc[:, numRankB:]

    W, Sig, Wt = np.linalg.svd(np.dot(Uc0, Uc0.T) + np.dot(Ub0, Ub0.T), full_matrices=True)
    if np.diag(Sig[Sig > tol]).size:
        numRankSig = np.linalg.matrix_rank(np.diag(Sig[Sig > tol]))
    else:
        numRankSig = 0
    Wgk = W[:, numRankSig:]
    arg = np.linalg.pinv(C, tol) + np.linalg.pinv(B, tol) - np.eye(dim)

    return np.dot(np.dot(Wgk, np.linalg.inv(np.dot(Wgk.T, np.dot(arg, Wgk)))), Wgk.T)


def NOT(C):
    dim = len(C)
    return np.eye(dim) - C


def OR(C, B):
    return NOT(AND(NOT(C), NOT(B)))


def scalarPHI(c, gamma):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if gamma == 0:
            if c[i] < 1:
                d[i] = 0
            if c[i] == 1:
                d[i] = 1
        else:
            d[i] = c[i] / (c[i] + (gamma ** -2) * (1 - c[i]))
    return d


def scalarNOT(c):
    return 1 - c


def scalarAND(c, b):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if c[i] == 0 and b[i] == 0:
            d[i] = 0
        else:
            d[i] = c[i] * b[i] / (c[i] + b[i] - c[i] * b[i])
    return d


def scalarOR(c, b):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if c[i] == 1 and c[i] == b[i]:
            d[i] = 1
        else:
            d[i] = (c[i] + b[i] - 2 * c[i] * b[i]) / (1 - c[i] * b[i])
    return d


def phi(C, gamma):
    return np.dot(C, np.linalg.inv(C + gamma ** (-2) * (np.eye(len(C)) - C)))
