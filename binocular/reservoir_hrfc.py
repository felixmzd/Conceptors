from binocular import utils
import numpy as np
import scipy.sparse.linalg as lin
import math
from .reservoir_rfc import ReservoirRandomFeatureConceptor
from typing import *


class HierarchicalRandomFeatureConceptor:

    def __init__(self, rfcs: List[ReservoirRandomFeatureConceptor], depth, trusts, discrepancies, means, variances):
        self.rfcs = rfcs
        self.depth = depth

        # State variables.
        self.trusts = trusts
        self.discrepancies = 0.5 * np.ones(self.depth)
        self.hypotheses = None

        # Needed for calculation of trust updates.
        self.means = np.zeros(self.depth)
        self.variances = np.ones(self.depth)
        # decisivenes12
        # decisivenes23

        # drift_force
    @classmethod
    def init_random(cls, depth: int):
        rfcs = [ReservoirRandomFeatureConceptor.init_random() for _ in range(depth)]
        trusts = 0.5 * np.ones(depth - 1)
        discrepancies = 0.5 * np.ones(depth)
        means = np.zeros(depth)
        variances = np.ones(depth)

        return cls(rfcs, depth, trusts, discrepancies, means, variances)

    def fit(self, patterns):
        """Initial loading of the patterns."""
        n_patterns = len(patterns)
        self.hypotheses = np.ones(shape=(self.depth, n_patterns)) / n_patterns
        for rfc in self.rfcs:
            rfc.fit(patterns)

    def recall(self, t_washout=200, t_recall=200):
        ...

    def denoise(self, pattern, t_steps: int):
        for t in range(t_steps):
            u = pattern(t)
            for module_idx in range(self.depth):
                u = self.rfcs[module_idx].forward(u, self.rfcs[module_idx + 1].conceptor, self.trusts[module_idx])
