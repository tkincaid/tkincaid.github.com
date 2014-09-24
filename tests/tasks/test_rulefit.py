#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np

#import pytest
from sklearn import datasets
from sklearn.cross_validation import train_test_split

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.rulefit import _RuleFitClassifier
from ModelingMachine.engine.tasks.rulefit import _RuleFitRegressor
from ModelingMachine.engine.tasks.rulefit import RuleFitC
from ModelingMachine.engine.tasks.rulefit import RuleFitR
from ModelingMachine.engine.tasks.base_modeler import Hotspot
from ModelingMachine.engine.tasks.base_modeler import VisualizableHotspot
from ModelingMachine.engine.tasks.base_modeler import RegressionHotspotScore
from ModelingMachine.engine import metrics
from common.insights import Insights


class TestRuleFit(BaseTaskTest):

    def test_arguments(self):
        """Test if arguments are passed to wrapped estimator"""
        xt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        yt = np.array([0, 1, 0])
        self.check_arguments(RuleFitC, _RuleFitClassifier, xt, yt)
        self.check_arguments(RuleFitR, _RuleFitRegressor, xt, yt)

    def test_default_arguments(self):
        """Test if task defaults are the same as sklearn/tesla defaults.

        Ignores max_features because we use a different default here.
        """
        self.check_default_arguments(RuleFitC, _RuleFitClassifier, ignore_params=('random_state',))
        self.check_default_arguments(RuleFitR, _RuleFitRegressor, ignore_params=('random_state',))

    #@pytest.mark.dscomp
    def test_RULEFITC_reproducible(self):
        """Smoke test for classification. """
        X, Y, Z = self.create_bin_data()

        reference = np.array([0.92671833, 0.28697295, 0.92671833])
        task_args = ['n=200', 'md=[3, 4]', 'lr=0.002', 't_f=0.2',
                     't_m={metrics.AUC}'.format(metrics=metrics)]
        task_desc = 'RULEFITC ' + ';'.join(task_args)
        t = self.check_task(task_desc, X, Y, Z, transform=False,
                            standardize=False, reference=reference,
                            insights=Insights.HOTSPOTS)
        self.assertTrue((0, -1) in t.hotspots)

    #@pytest.mark.dscomp
    def test_RULEFITR_reproducible(self):
        """Smoke test for regression. """
        X, Y, Z = self.create_reg_data()
        # no reference for RULEFITR because no random seed :/
        t = self.check_task('RULEFITR logy;n=10;a=0.01', X, Y, Z, transform=False,
                            standardize=False, insights=Insights.HOTSPOTS)
        hs = t.hotspots[(0, -1)][0]
        # if inv-transform was not applied this will be way smaller than 50
        self.assertGreater(hs['score']['mean_response'], 50)

    def test_rulefit_hotspots_regression(self):
        boston = datasets.load_boston()
        X, y = boston.data, boston.target
        fx_names = boston.feature_names
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                           random_state=1)
        est = _RuleFitRegressor(n_estimators=20, max_depth=2, main_effects=True)
        est.fit(X_train, y_train)
        hotspots = est.hotspots(X_test, y_test, fx_names)
        self.assertTrue(all(isinstance(hs, VisualizableHotspot) for hs in hotspots))
        hotspots = [hs.hotspot for hs in hotspots]
        self.assertTrue(all(isinstance(hs.rule, basestring) for hs in hotspots))
        self.assertTrue(all(hs.support is None for hs in hotspots
                           if hs.rule.startswith('linear:')))
        self.assertTrue(all(h.score.lift == np.abs(h.score.mean_relative_response - 1.0)
                            for h in hotspots))

        # test hotspots with unicode
        fx_names = map(unicode, fx_names)
        fx_names = [u'äüö' + n for n in fx_names]
        hotspots = est.hotspots(X_test, y_test, fx_names)
        for hs in hotspots:
            self.assertTrue(isinstance(hs.hotspot.rule, unicode))

    def test_rulefit_hotspots_classification(self):
        X, y = datasets.make_hastie_10_2(n_samples=1000)
        # remap labels to {0, 1} -- expected by hotspots
        _, y = np.unique(y, return_inverse=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                           random_state=1)
        est = _RuleFitClassifier(n_estimators=20, max_depth=2, main_effects=True)
        est.fit(X_train, y_train)
        hotspots = est.hotspots(X_test, y_test)
        self.assertTrue(all(isinstance(hs, VisualizableHotspot) for hs in hotspots))
        hotspots = [hs.hotspot for hs in hotspots]
        self.assertTrue(all(isinstance(hs.rule, basestring) for hs in hotspots))
        self.assertTrue(all(hs.support is None for hs in hotspots
                           if hs.rule.startswith('linear:')))

        self.assertTrue(all(h.score.lift == np.abs(h.score.mean_relative_response - 1.0)
                            for h in hotspots))

    def test_hotspots_asdict(self):
        score = RegressionHotspotScore(0, 0, 0, 0, 0, 0, 0, 0)
        hs = Hotspot(score, 0, 'the rule', 0.0, 1.0)
        hs_dct = hs._asdict()
        self.assertTrue(isinstance(hs_dct['score'], dict))


    def test_subsample_heuristic(self):
        """Test subsample=auto heuristic"""
        task = RuleFitC('s=auto')
        X, y =  datasets.make_hastie_10_2(n_samples=112)
        task._modify_parameters(X, y)
        self.assertEqual(task.parameters['subsample'], 1.0)

        task = RuleFitC('s=auto')
        X, y =  datasets.make_hastie_10_2(n_samples=500)
        task._modify_parameters(X, y)
        self.assertEqual(round(task.parameters['subsample'], 2), 0.49)
