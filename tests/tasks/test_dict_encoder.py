# Author: Lars Buitinck <L.J.Buitinck@uva.nl> and Peter Prettenhofer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from ModelingMachine.engine.tasks.dict_encoder import DictOneHotEncoder as DictVectorizer


def assert_true(a):
    assert a


def assert_false(a):
    assert not a


def test_dictvectorizer():
    D = [{"foo": 1, "bar": 1},
         {"bar": 4, "baz": 2},
         {"bar": 1, "quux": 1, "quuux": 2}]

    for sparse in (True, False):
        for dtype in (int, np.float32, np.int16):
            v = DictVectorizer(sparse=sparse, dtype=dtype)
            X = v.fit_transform(D)

            assert_equal(sp.issparse(X), sparse)
            assert_equal(X.shape, (3, 6))
            assert_equal(X.sum(), 7)

            if sparse:
                # CSR matrices can't be compared for equality
                assert_array_equal(X.A, v.transform(D).A)
            else:
                assert_array_equal(X, v.transform(D))


def test_one_of_k():
    """Smoke test for one hot encoding"""
    D_in = [{"version": "1", "ham": 2},
            {"version": "2", "spam": 1},
            {"version": True, "spam": -1}]
    v = DictVectorizer()
    X = v.fit_transform(D_in)
    assert_equal(X.shape, (3, 6))

    D_out = v.inverse_transform(X)
    assert D_out[0] == {"version=1": 1.0, "ham=2": 1.0}, '%r not expected' % D_out[0]

    names = v.get_feature_names()
    assert_true("version=2" in names)
    assert_false("version" in names)

def test_different_dtypes():
    """Test that shows that different dtypes might map to the same OH fx. """
    D_in = [{"version": "1", "ham": 2},
            {"version": 1, "ham": 1},
            ]
    v = DictVectorizer()
    X = v.fit_transform(D_in)
    assert X.shape == (2, 3)

def test_unseen_or_no_features():
    """Test if unseen features are skipped. """
    D = [{"camelot": 0, "spamalot": 1}]
    for sparse in [True, False]:
        v = DictVectorizer(sparse=sparse).fit(D)

        X = v.transform({"push the pram a lot": 2})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))

        X = v.transform({})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))
