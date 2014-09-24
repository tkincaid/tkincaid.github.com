#########################################################
#
#       Unit Test for blenders
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest
import logging
import copy
import numpy as np
import pandas as pd

import pytest
from mock import patch

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.blend import AvgModel
from ModelingMachine.engine.tasks.blend import MedianModel
from ModelingMachine.engine import metrics
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition


class TestBlend(BaseTaskTest):

    def test_avg_blend(self):
        X = np.array([[1, 2], [2, 3]], dtype=np.float)
        y = np.array([1, 2], dtype=np.float)

        est = AvgModel(transform='none')
        est.fit(X, y)
        np.testing.assert_array_equal(est.predict(X), [1.5, 2.5])

        est = AvgModel(transform='exp')
        est.fit(X, y)
        np.testing.assert_array_equal(est.predict(X), np.exp([1.5, 2.5]))

        est = AvgModel(transform='foobar')
        est.fit(X, y)
        np.testing.assert_array_equal(est.predict(X), 1/(1+np.exp([-1.5, -2.5])))

    def test_median_blend(self):
        X = np.array([[1, 2, 3], [2, 3, 9]], dtype=np.float)
        y = np.array([-1, -1], dtype=np.float)

        est = MedianModel(transform='none')
        est.fit(X, y)
        np.testing.assert_array_equal(est.predict(X), [2, 3])

        est = MedianModel(transform='exp')
        est.fit(X, y)
        np.testing.assert_array_equal(est.predict(X), np.exp([2, 3]))

        est = MedianModel(transform='foobar')
        est.fit(X, y)
        np.testing.assert_array_equal(est.predict(X), 1/(1+np.exp([-2, -3])))

    def test_glm_bsr_blender(self):
        """Check the backwards stepwise regression blender. """
        X, Y, Z = self.create_bin_data()
        reference = [ 0.65025429,  0.43288926,  0.84738552]
        t = self.check_task('GLMB logitx;sw_b=2', X, Y, Z, transform=True,
                            standardize=False, reference=reference)
