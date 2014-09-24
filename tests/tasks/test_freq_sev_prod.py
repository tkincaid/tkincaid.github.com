############################################
#
# Unit tests for frequency severity modeling
#
# Author: Mark Steadman
#
# Copyright: 2013
#
############################################

import unittest
import numpy as np
import pandas as pd

from base_task_test import BaseTaskTest
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.tasks.frequency_severity_combiner import FreqSevEstimator, FreqSevCombiner
class TestFreqSevProd(BaseTaskTest):

    def test_freq_sev_est_fit(self):
        """
        Test estimator fit function
        """
        xcol1 = np.random.random((30, 1))
        xcol2 = np.random.randint(0, 2, (30, 1))
        xdata = np.hstack((xcol1, xcol2))
        ydata = np.random.randint(1, 8, (30, 1)) * xcol2 + 0.2
        est = FreqSevEstimator()
        out = est.fit(xdata, ydata)
        self.assertIsInstance(out, FreqSevEstimator)
        self.assertEqual(out.yMin, 0.2)

    def test_freq_sev_est_predict(self):
        """
        Test estimator predict function
        """
        xcol1 = np.random.random((30, 1))
        xcol2 = np.random.randint(0, 2, (30, 1))
        xdata = np.hstack((xcol1, xcol2))
        ydata = np.random.randint(1, 8, (30, 1)) * xcol2 + 0.2
        est = FreqSevEstimator()
        est = est.fit(xdata, ydata)
        x2col1 = np.random.random((30, 1))
        x2col2 = np.random.randint(0, 2, (30, 1))
        x2data = np.hstack((xcol1, xcol2))
        out = est.predict(x2data)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape[0], x2data.shape[0])
        desired = 0.2 + x2data[:, 0] * xdata[:, 1]
        np.testing.assert_array_almost_equal_nulp(out, desired)

    def test_freq_sev_task_fit(self):
        """
        Test task fit function
        """
        xcol1 = np.random.random((30, 1))
        xcol2 = np.random.randint(0, 2, (30, 1))
        xdata = np.hstack((xcol1, xcol2))
        xcont = Container()
        xcont.add(xdata)
        Z = Partition(30, folds=5, reps=5, total_size=30)
        Z.set(max_folds=0, max_reps=1)
        ydata = np.random.randint(1, 8, (30, 1)) * xcol2 + 0.2
        est = FreqSevCombiner()
        out = est.fit(xcont, ydata, Z)
        self.assertIsInstance(out, FreqSevCombiner)

    def test_freq_sev_task_predict(self):
        """
        Test task predict function
        """
        xcol1 = np.random.random((30, 1))
        xcol2 = np.random.randint(0, 2, (30, 1))
        xcont1 = Container()
        xcont2 = Container()
        xdata = np.hstack((xcol1, xcol2))
        ydata = np.random.randint(1, 8, (30, 1)) * xcol2 + 0.2
        sevcols = {'importances-sev': [['sev-col1', 1], ['sev-col2', 2], ['sev-col3', 3]]}
        freqcols = {'coefficients-freq': [['freq-col1', 1], ['freq-col2', 2]]}
        xcont1.add(xcol1, metadata=sevcols, r=0, k=-1)
        xcont2.add(xcol2, metadata=freqcols,r=0, k=-1)
        xcont1.add(xcol1, metadata=sevcols, r=1, k=-1)
        xcont2.add(xcol2, metadata=freqcols,r=1, k=-1)
        print xcont1.metadata(r=0, k=-1)
        xcont = xcont1 + xcont2
        print xcont.metadata(r=0, k=-1)
        Z = Partition(30, folds=5, reps=5, total_size=30)
        Z.set(max_folds=0, max_reps=2)
        est = FreqSevCombiner()
        est = est.fit(xcont, ydata, Z)
        x2col1 = np.random.random((30, 1))
        x2col2 = np.random.randint(0, 2, (30, 1))
        x2data = np.hstack((xcol1, xcol2))
        x2cont = Container()
        x2cont.add(x2data)
        Z = Partition(30, folds=5, reps=5, total_size=30)
        Z.set(max_folds=0, max_reps=2)
        ydata = np.random.randint(1, 8, (30, 1)) * xcol2 + 0.2
        out = est.predict(x2cont, ydata, Z)
        self.assertIsInstance(out, Container)
        for p in out:
            key = (p['r'], p['k'])
            print key
            self.assertEqual(out(**p).shape[1], 1)
            self.assertEqual(out(**p).shape[0], x2data.shape[0])
            desired = 0.2 + x2data[:, 0] * xdata[:, 1]
            np.testing.assert_array_almost_equal_nulp(desired.ravel(), out(**p).ravel())
            self.assertTrue(hasattr(est, 'coefficients'))
            self.assertTrue(hasattr(est, 'coefficients2'))
            self.assertGreater(len(est.coefficients), 0)
            self.assertGreater(len(est.coefficients2), 0)
            self.assertTrue(est.coefficients[key][0][0].startswith('sev') or est.coefficients[key][0][0].startswith('freq'))
            if est.coefficients[key][0][0].startswith('sev'):
                self.assertEqual(sevcols['importances-sev'], est.coefficients[key])
                self.assertEqual(freqcols['coefficients-freq'], est.coefficients2[key])
            else:
                self.assertEqual(sevcols['importances-sev'], est.coefficients2[key])
                self.assertEqual(freqcols['coefficients-freq'], est.coefficients[key])
