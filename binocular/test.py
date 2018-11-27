import pickle

import numpy as np

def test_same_binocular_pickle():
    """Test that binocular_run.py still gives same results."""
    original_pickle = pickle.load(open('FigureObject.fig_loading_old.pickle', 'rb'))
    new_pickle = pickle.load(open('FigureObject.fig_loading.pickle', 'rb'))
    for arr_old, arr_new in zip(original_pickle, new_pickle):
        assert np.array_equal(arr_old, arr_new)

def test_same_rfc_pickle():
    """Test that rfc.py still gives same results."""
    original_pickle = pickle.load(open('FigureObject.fig_rfc_old.pickle', 'rb'))
    new_pickle = pickle.load(open('FigureObject.fig_rfc.pickle', 'rb'))
    for arr_old, arr_new in zip(original_pickle, new_pickle):
        assert np.array_equal(arr_old, arr_new)

