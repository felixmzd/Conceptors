import pickle

import numpy as np

from binocular.reservoir_rfc import ReservoirRandomFeatureConceptor
from binocular import pattern_functions
from binocular import utils


def test_rfc_result(shared_datadir):
    np.random.seed(0)

    patterns = []
    for p in [53, 54, 10, 36]:
        patterns.append(pattern_functions.patterns[p])

    reservoir = ReservoirRandomFeatureConceptor.init_random()
    reservoir.run(patterns)
    reservoir.recall()

    allDriverPL, allRecallPL, NRMSE = utils.plot_interpolate_1d(patterns, reservoir.Y_recalls)

    results = [allDriverPL, allRecallPL, reservoir.C, reservoir.c_colls]

    original_pickle = pickle.load(open(shared_datadir / 'FigureObject.fig_rfc_old.pickle', 'rb'))
    for arr_old, arr_new in zip(original_pickle, results):
        print('old')

        assert np.array_equal(arr_old, arr_new)
