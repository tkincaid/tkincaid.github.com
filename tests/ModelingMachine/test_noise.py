'''Unit tests for noise transformer.

What do we use this for?

'''

import unittest
import pytest

import numpy as np
import pandas as pd

from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.tasks.transformers import Noise
from ModelingMachine.engine.tasks.transformers import Noise
from ModelingMachine.engine.tasks.transformers import Noise

class TestNOISE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_noise_repeatable(self):
        X = Container()
        X.add(np.arange(50).reshape(10, 5))
        Y = pd.Series(np.arange(10))
        Z = Partition(10, folds=5, reps=1, total_size=10)

        noise = Noise()
        noise.fit(X, Y, Z)

        trans = noise.transform(X, Y, Z)

        # Assert - statically check the output
        reference = [0, 1, 2, 3, 4, 0.60210196]
        check = trans(r=0, k=0)[0, :]

        np.testing.assert_almost_equal(reference, check)

