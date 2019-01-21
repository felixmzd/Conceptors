from binocular import utils
import numpy as np
import scipy.sparse.linalg as lin
import math
from .reservoir_rfc import ReservoirRandomFeatureConceptor
from typing import *

class ReservoirHierarchicalRandomFeatureConceptor:

    def __init__(self, rfcs: List[ReservoirRandomFeatureConceptor]):
        self.depth = len(rfcs)
        self.trusts = 0.5 * np.ones(self.depth - 1)
        self.discrepancies = 0.5 * np.ones(self.depth)
        self.means = np.zeros(self.depth)
        self.variances = np.ones(self.depth)

    def fit(self, patterns):
        n_patterns = len(patterns)
        self.hypotheses = np.ones(shape=(self.depth, n_patterns)) / n_patterns


    def recall(self, t_washout=200, t_recall=200):
        ...

    def denoise(self):
        ...
