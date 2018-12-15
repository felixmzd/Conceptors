import pickle

import numpy as np

def test_same_binocular_pickle(shared_datadir):
    """Test that binocular_run.py still gives same results."""
    original_pickle = pickle.load(open(shared_datadir / 'FigureObject.fig_loading_old.pickle', 'rb'))
    new_pickle = pickle.load(open(shared_datadir / 'FigureObject.fig_loading.pickle', 'rb'))
    for arr_old, arr_new in zip(original_pickle, new_pickle):
        assert np.array_equal(arr_old, arr_new)
