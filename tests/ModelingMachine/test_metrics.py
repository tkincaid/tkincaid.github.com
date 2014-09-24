######################################################
#
#   test metrics computations
#
#   Author: Xavier Conort
#
#   Copyright DataRobot LLC, 2013
#
######################################################
import pytest
import unittest
import numpy as np
import pandas as pd

from rpy2 import robjects
from functools import partial
from mock import patch

import common.engine.metrics
import ModelingMachine.engine.metrics as metrics
from ModelingMachine.engine.metrics import poisson_deviance, gamma_deviance, tweedie_deviance
from ModelingMachine.engine.metrics import gini_score, gini_norm, auc, gini_norm_w, auc_w
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.new_partition import NewPartition



class Testmetrics(unittest.TestCase):

    # test Poisson
    def test_poisson(self,size=10):
        x = np.random.poisson(size=size)
        d = poisson_deviance(x,np.ones(size)*x.mean())
        robjects.globalenv['x']=robjects.FloatVector(x)
        robjects.globalenv['size']=size
        rcode = """ d_from_glm <- glm(x~1,data=data.frame(x=x),family=poisson())$deviance/size
                    """
        robjects.r(rcode)
        d_from_glm = robjects.globalenv['d_from_glm'][0]
        self.assertEqual(round(d_from_glm,8), round(d,8))

    # test Tweedie
    def test_tweedie(self,size=10):
        np.random.seed(12345)
        x = np.random.poisson(size=size)
        pred = np.ones(size)*x.mean()
        # test p = 1.5
        p = 1.5
        d = tweedie_deviance(x, pred, p=p)
        # hard copy because of potential GPL issues
        self.assertEqual(2.27302105, round(d,8))
        # test p = 1
        p = 1
        d = tweedie_deviance(x, pred, p=p)
        self.assertEqual(1.64057352, round(d,8))
        # test p = 2
        x[x==0] = 1
        pred = np.ones(size)*x.mean()
        p = 2
        d = tweedie_deviance(x, pred, p=p)
        self.assertEqual(0.2870163, round(d,8))

    # test Gamma
    def test_gamma(self,size=10):
        np.random.seed(13)
        x = np.random.gamma(shape=1,size=size)
        d = gamma_deviance(x,np.ones(size)*x.mean())
        robjects.globalenv['x']=robjects.FloatVector(x)
        robjects.globalenv['size']=size
        rcode = """ d_from_glm <- glm(x~1,data=data.frame(x=x),family=Gamma())$deviance/size
                    """
        robjects.r(rcode)
        d_from_glm = robjects.globalenv['d_from_glm'][0]
        self.assertEqual(round(d_from_glm,8), round(d,8))

    def test_pred_zeros(self, size=10):
        rs = np.random.RandomState(13)
        x = rs.poisson(size=size)
        pred = np.zeros_like(x)
        d = tweedie_deviance(x, pred, p=1.5)
        self.assertEqual(np.isfinite(d), True)
        d = poisson_deviance(x, pred)
        self.assertEqual(np.isfinite(d), True)
        x = rs.lognormal(size=size)
        d = gamma_deviance(x, pred)
        self.assertEqual(np.isfinite(d), True)

    def test_gini(self):
        """Test if gini gives correct results.

        Test cases adopted from
        http://www.kaggle.com/c/ClaimPredictionChallenge/forums/t/703/code-to-calculate-normalizedgini?page=2
        """
        np.testing.assert_almost_equal(gini_score([1, 2, 3], [10, 20, 30]), 0.111111, decimal=5)
        np.testing.assert_almost_equal(gini_score([1, 2, 3], [-30, -20, -10]), 0.111111, decimal=5)
        np.testing.assert_almost_equal(gini_score([1, 2, 3], [30, 20, 10]), -0.111111, decimal=5)
        np.testing.assert_almost_equal(gini_score([1, 2, 3], [0, 0, 0]), -0.111111, decimal=5)
        np.testing.assert_almost_equal(gini_score([3, 2, 1], [0, 0, 0]), 0.111111, decimal=5)
        np.testing.assert_almost_equal(gini_score([1, 2, 4, 3], [0, 0, 0, 0]), -0.1, decimal=5)
        np.testing.assert_almost_equal(gini_score([2, 1, 4, 3], [0, 0, 2, 1]), 0.125, decimal=5)
        np.testing.assert_almost_equal(gini_score([0, 20, 40, 0, 10], [40, 40, 10, 5, 5]), 0,
                                       decimal=5)
        np.testing.assert_almost_equal(gini_score([40, 0, 20, 0, 10], [1000000, 40, 40, 5, 5]),
                                       0.171428, decimal=5)
        np.testing.assert_almost_equal(gini_score([40, 20, 10, 0, 0], [40, 20, 10, 0, 0]), 0.285714,
                                       decimal=5)
        np.testing.assert_almost_equal(gini_score([1, 1, 0, 1], [0.86, 0.26, 0.52, 0.32]), -0.041666,
                                       decimal=5)

        # test case with negative act, we have gini_score>0 for a positive correlation
        self.assertEqual(gini_score([-3, -2, -1], [0, 2, 3])>0,True)

        # test if gini_norm unchanged after translation
        np.testing.assert_almost_equal(gini_norm(np.array([-3, -2, -1]), np.array([0, 0, 0])),
            gini_norm(np.array([1, 2, 3]), np.array([0, 0, 0])), decimal=5)

        # failing test-case

        # this has gini_score -inf
        np.testing.assert_almost_equal(gini_score([-1, 0, 1], [30, 20, 10]), -0.222222, decimal=5)
        np.testing.assert_almost_equal(gini_norm(np.array([-1, 0, 1]), np.array([30, 20, 10])),
            gini_norm(np.array([1, 2, 3]), np.array([30, 20, 10])), decimal=5)

    def test_gini_vs_auc(self):
        pred, act = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.42, 0.44]), np.array([0, 1, 0, 1, 1, 1, 0])
        np.testing.assert_almost_equal(gini_norm(act, pred), 2 * auc(act, pred) - 1)
        np.testing.assert_almost_equal(gini_norm(act, pred), 0.3333333)

    def test_gini_vs_auc_w(self):
        # numeric example from Kaggle
        #https://www.kaggle.com/c/liberty-mutual-fire-peril/forums/t/9880/update-on-the-evaluation-metric
        pred, act = np.array([0.1, 0.4, 0.3, 1.2, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        wei=np.array([1.0 ,2.0 ,5.0 ,4.0 ,3.0]);
        np.testing.assert_almost_equal(gini_norm(act, pred), 2 * auc(act, pred) - 1)
        np.testing.assert_almost_equal(gini_norm_w(act, pred), 2 * auc_w(act, pred) - 1)
        np.testing.assert_almost_equal(metrics.gini_norm_w(act, pred,weight=wei), -0.8214285714)


    def test_gini_norm_logpred(self):
        pred, act = np.array([1, 2, 3]), np.array([10, 20, 30])
        gini_untransformed = gini_norm(act, pred)
        gini_transformed = gini_norm(act, np.log(pred))
        np.testing.assert_almost_equal(gini_untransformed, gini_transformed)

        rng = np.random.RandomState(13)
        pred = 10 + rng.rand(10)
        act = 10 + rng.rand(10)
        gini_untransformed = gini_norm(act, pred)
        gini_transformed = gini_norm(act,np.log(pred))
        np.testing.assert_almost_equal(gini_untransformed, gini_transformed)

    def test_all_metrics_have_match_func(self):
        for k,v in metrics._metric_map.iteritems():
            self.assertIsNotNone(v['match_func'],
                                 'Metric {} has no match function'.format(k))

    def test_logloss_weighted_version_is_logloss_w(self):
        self.assertEqual(
            metrics._metric_map[metrics.LOGLOSS]['weighted_version'],
            metrics._metric_map[metrics.LOGLOSS_W]['short_name'])

    def test_value_error_infinite(self):
        act = np.array([np.nan, np.inf, 0.0, 1.0])
        pred = np.zeros_like(act)

        for metric_name in metrics.ALL_METRICS:
            metric = metrics.metric_by_name(metric_name)
            with self.assertRaises(ValueError) as cm:
                metric(act, pred)

    def test_ndcg(self):
        act = np.array([3, 2, 3, 0, 1, 2])
        pred = np.arange(act.shape[0])[::-1]
        groups = np.ones_like(act).astype(np.int)
        np.testing.assert_almost_equal(metrics.ndcg(act, pred, groups=groups, rank=20), 0.94881075)

    def test_ace(self):
        # make sure all metrics are assigned an ACE normalization
        unsupported_metrics = set(metrics.ALL_METRICS) - set(metrics.metric_normalization.keys())
        self.assertSetEqual(unsupported_metrics, set())

        # make sure all normalized scores are in [0, 1]
        for metric in metrics.ALL_METRICS:
            if metrics.direction_by_name(metric) == metrics.DIRECTION_DESCENDING:
                info = metrics.normalize_ace_score(metric, 0.8, 0.2)
            else:
                info = metrics.normalize_ace_score(metric, 0.2, 0.8)
            self.assertLessEqual(info, 1)
            self.assertGreater(info, -0.1)

class TestLogLossWMetric(unittest.TestCase):
    def test_hand_verified_case(self):
        act = np.array([0.2, 0.8, 0.1, 0.1, 0.5])
        pred = np.array([0.25, 0.25, 0.25, 0.2, 0.25])
        weights = np.array([1 for i in act])
        out = metrics.logloss_w(act, pred, weight=weights)
        np.testing.assert_almost_equal(out, 0.65405618)

    def test_weight_scale(self):
        act = np.array([0.2, 0.8, 0.1, 0.1, 0.5])
        pred = np.array([0.25, 0.25, 0.25, 0.2, 0.25])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.logloss_w(act, pred, weight=weights1)
        out2 = metrics.logloss_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([0.2, 0.8, 0.1, 0.1, 0.5])
        pred = np.array([0.25, 0.25, 0.25, 0.2, 0.25])
        weights1 = np.array([1.9, 0.5, 1.1, 0.2, 2.8])
        weights2 = weights1*0.1
        out1 = metrics.logloss_w(act, pred, weight=weights1)
        out2 = metrics.logloss_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([0.2, 0.8, 0.1, 0.1, 0.5])
        pred = np.array([0.25, 0.25, 0.25, 0.2, 0.25])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.logloss_w(act, pred, weight=weights1)
        out2 = metrics.logloss(act, pred)
        np.testing.assert_almost_equal(out1, out2)


class TestAUCWMetric(unittest.TestCase):
    def test_hand_verified_case(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights = np.array([1 for i in act])
        out = metrics.auc_w(act, pred, weight=weights)
        np.testing.assert_almost_equal(out, 0.66666667)

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.auc_w(act, pred, weight=weights1)
        out2 = metrics.auc_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1.9, 0.5, 1.1, 0.2, 2.8])
        weights2 = weights1*0.1
        out1 = metrics.auc_w(act, pred, weight=weights1)
        out2 = metrics.auc_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.auc_w(act, pred, weight=weights1)
        out2 = metrics.auc(act, pred)
        np.testing.assert_almost_equal(out1, out2)

class TestGiniWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        pred = np.array([1, 2, 3, 4, 5])
#        act = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        weights = np.array([1 for i in act])
#        out = metrics.gini_w(pred, act, weight=weights)
#        out = metrics.gini(pred, act)
#        np.testing.assert_almost_equal(out, 0.4206349)


    def test_weight_scale(self):
        pred = np.array([1, 2, 3, 4, 5])
        act = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.gini_w(act, pred, weight=weights1)
        out2 = metrics.gini_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        pred = np.array([1, 2, 3, 4, 5])
        act = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1.9, 0.5, 1.1, 0.2, 2.8])
        weights2 = weights1*0.1
        out1 = metrics.gini_w(act, pred, weight=weights1)
        out2 = metrics.gini_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        pred = np.array([1, 2, 3, 4, 5])
        act = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        out1 = metrics.gini_w(act, pred, weight=weights1)
        out2 = metrics.gini(act, pred)*2 #SY gini is 1/2 of gini_w
        np.testing.assert_almost_equal(out1, out2)



class TestAMS15Metric(unittest.TestCase):
    def test_hand_verified_case(self):
        act = np.array([1,1,1,0,0,0,1,1])
        pred = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
        weights = np.array([0.1,2,3,0.1,0.2,0.3,0.3,0.3])
        out = metrics.ams15(act, pred, weight=weights)
        np.testing.assert_almost_equal(out, 331.23585,decimal=5)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.ams15(act, pred, weight=weights1)
        out2 = metrics.ams15(act, pred)
        np.testing.assert_almost_equal(out1, out2)

class TestAMSoptMetric(unittest.TestCase):
    def test_hand_verified_case(self):
        act = np.array([1,1,1,0,0,0,1,1])
        pred = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
        weights = np.array([0.1,2,3,0.1,0.2,0.3,0.3,0.3])
        out = metrics.amsopt(act, pred, weight=weights)
        np.testing.assert_almost_equal(out, 754.626451,decimal=5)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.amsopt(act, pred, weight=weights1)
        out2 = metrics.amsopt(act, pred)
        np.testing.assert_almost_equal(out1, out2)


class TestRSquaredWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.r_squared_w(act, pred, weight=weights1)
        out2 = metrics.r_squared_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1.9, 0.5, 1.1, 0.2, 2.8])
        weights2 = weights1*0.1
        out1 = metrics.r_squared_w(act, pred, weight=weights1)
        out2 = metrics.r_squared_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.r_squared_w(act, pred, weight=weights1)
        out2 = metrics.r_squared(act, pred)
        np.testing.assert_almost_equal(out1, out2)

class TestRMSEWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.rmse_w(act, pred, weight=weights1)
        out2 = metrics.rmse_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1.9, 0.5, 1.1, 0.2, 2.8])
        weights2 = weights1*0.1
        out1 = metrics.rmse_w(act, pred, weight=weights1)
        out2 = metrics.rmse_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.rmse_w(act, pred, weight=weights1)
        out2 = metrics.rmse(act, pred)
        np.testing.assert_almost_equal(out1, out2)

class TestMADWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.mad_w(act, pred, weight=weights1)
        out2 = metrics.mad_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.mad_w(act, pred, weight=weights1)
        out2 = metrics.mad_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.mad_w(act, pred, weight=weights1)
        out2 = metrics.mad(act, pred)
        np.testing.assert_almost_equal(out1, out2)

class TestRMSLEWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.rmsle_w(act, pred, weight=weights1)
        out2 = metrics.rmsle_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.rmsle_w(act, pred, weight=weights1)
        out2 = metrics.rmsle_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.rmsle_w(act, pred, weight=weights1)
        out2 = metrics.rmsle(act, pred)
        np.testing.assert_almost_equal(out1, out2)

class TestPoissonDevianceWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.poisson_deviance_w(act, pred, weight=weights1)
        out2 = metrics.poisson_deviance_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.poisson_deviance_w(act, pred, weight=weights1)
        out2 = metrics.poisson_deviance_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.poisson_deviance_w(act, pred, weight=weights1)
        out2 = metrics.poisson_deviance(act, pred)
        np.testing.assert_almost_equal(out1, out2)

class TestTweedieDevianceWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.tweedie_deviance_w(act, pred, p=1.2, weight=weights1)
        out2 = metrics.tweedie_deviance_w(act, pred, p=1.2, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.tweedie_deviance_w(act, pred, p=1.2, weight=weights1)
        out2 = metrics.tweedie_deviance_w(act, pred, p=1.2, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.tweedie_deviance_w(act, pred, p=1.2, weight=weights1)
        out2 = metrics.tweedie_deviance(act, pred, p=1.2)
        np.testing.assert_almost_equal(out1, out2)

class TestGammaDevianceWMetric(unittest.TestCase):

#    def test_hand_verified_case_2(self):
#        act = np.array([1, 0, 1, 0, 1])
#        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
#        weights = np.array([1, 5, 8, 3, 1])
#        out = metrics.auc_w(act, pred, weight=weights)
#        np.testing.assert_almost_equal(out, 0.2)


    def test_weight_scale(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([10 for i in act])
        out1 = metrics.gamma_deviance_w(act, pred, weight=weights1)
        out2 = metrics.gamma_deviance_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_scale_2(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.gamma_deviance_w(act, pred, weight=weights1)
        out2 = metrics.gamma_deviance_w(act, pred, weight=weights2)
        np.testing.assert_almost_equal(out1, out2)

    def test_weight_vs_no_weights(self):
        act = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.5, 0.1, 0.2, 0.8])
        weights1 = np.array([1 for i in act])
        weights2 = np.array([0.1 for i in act])
        out1 = metrics.gamma_deviance_w(act, pred, weight=weights1)
        out2 = metrics.gamma_deviance(act, pred)
        np.testing.assert_almost_equal(out1, out2)


class TestMAPEMetric(unittest.TestCase):

    def test_easy_regression_check(self):
        act = np.ones(10)
        pred = np.ones(10)

        out = metrics.mape(act, pred)
        self.assertEqual(out, 0)

    def test_hand_verified_case(self):
        act = np.array([1.2, 1.3, 1.2, 1.1, 1.2])
        pred = np.array([1.25, 1.25, 1.25, 1.2, 1.25])

        out = metrics.mape(act, pred)
        np.testing.assert_almost_equal(out, 5.0874125874)

    def test_weights_make_sense_for_even_weight(self):
        act = np.array([1.2, 1.3, 1.2, 1.1, 1.2])
        pred = np.array([1.25, 1.25, 1.25, 1.2, 1.25])
        weights = np.array([0.2 for i in act])
        out = metrics.mape_w(act, pred, weight=weights)
        np.testing.assert_almost_equal(5.0874125874, out)

    def test_weights_make_sense_for_weights_that_dont_total_1(self):
        act = np.array([1.2, 1.3, 1.2, 1.1, 1.2])
        pred = np.array([1.25, 1.25, 1.25, 1.2, 1.25])
        weights = np.array([0.1 for i in act])
        out = metrics.mape_w(act, pred, weight=weights)
        np.testing.assert_almost_equal(5.0874125874, out)

    def test_weights_make_sense_for_single_important_sample(self):
        act = np.array([1.2, 1.3, 1.2, 1.1, 1.2])
        pred = np.array([1.25, 1.25, 1.25, 1.2, 1.25])
        weights = np.array([0, 0, 0, 0, 1])
        out = metrics.mape_w(act, pred, weight=weights)
        np.testing.assert_almost_equal(4.1666666667, out)

    def test_obvious_case(self):
        act = np.array([100])
        pred = np.array([105])

        out = metrics.mape(act, pred)
        self.assertEqual(5, out)

    def test_obvious_case_2(self):
        act = np.array([100])
        pred = np.array([200])

        out = metrics.mape(act, pred)
        self.assertEqual(100, out)

    def test_another_easy_one_but_with_many_samples(self):
        act = np.array([100] * 100)
        pred = np.array([105] * 100)

        out = metrics.mape(act, pred)
        np.testing.assert_almost_equal(5, out)

    def test_from_actual_data(self):
        act = np.array([6.,8.,26.,46.,54.,82.,86.,92.,96.,
                        104.,112.,120.,158.,164.,172.,178.])
        pred=np.array([[6.02632155],
                       [8.02570358],
                       [26.02014187],
                       [46.0139622],
                       [54.01149033],
                       [82.00283879],
                       [86.00160285],
                       [91.99974895],
                       [95.99851302],
                       [103.99604115],
                       [111.99356928],
                       [119.99109741],
                       [157.97935603],
                       [163.97750213],
                       [171.97503026],
                       [177.97317635]])
        out = metrics.mape(act, pred)
        np.testing.assert_almost_equal(0.0605982759, out)



class GroupedMetricsTest(unittest.TestCase):

    def test_smoke(self):
        act = np.array([1, 1, 2, 2])
        pred = np.array([0.9, 1.1, 1.9, 2.1])
        groups = np.array([0, 0, 1, 1])
        rmse = metrics.metric_by_name(metrics.RMSE)
        score = rmse(act, pred, groups=groups)
        np.testing.assert_almost_equal(score, (rmse(act[:2], pred[:2]) + rmse(act[2:], pred[2:])) / 2.0)

    def test_ndcg(self):
        act = np.array([0, 1, 0, 0])
        pred = np.array([0, 1, 1, 0])
        groups = np.array([0, 0, 1, 1])
        # perfect ndcg for 0 and worst for 1
        np.testing.assert_almost_equal(metrics.ndcg(act, pred, groups=groups), 0.5)

    def test_ndcg_worst(self):
        """NDCG has no clear worst score... """
        act = np.array([1, 2, 3, 4])
        pred = np.array([4, 3, 2, 1])
        groups = np.array([0, 0, 0, 0])
        np.testing.assert_almost_equal(metrics.ndcg(act, pred, groups=groups),
                                       0.60209052070893998)

    def test_ndcg_skip(self):
        """Test skipping singleton groups """
        act = np.array([0, 1, 0, 0])
        pred = np.array([0, 1, 1, 0])
        groups = np.array([0, 0, 1, 2])
        # perfect ndcg for 0 and worst for 1
        np.testing.assert_almost_equal(metrics.ndcg(act, pred, groups=groups), 1.0)

    def test_ndcg_constant(self):
        """Test constant group gives worst NDCG """
        act = np.array([1, 1, 2, 2])
        pred = np.array([0, 1, 1, 0])
        groups = np.array([0, 0, 1, 1])
        # perfect ndcg for 0 and worst for 1
        np.testing.assert_almost_equal(metrics.ndcg(act, pred, groups=groups), 0.0)

    def test_ndcg_negative(self):
        act = np.array([-10.0, 10.0, 10.0, -1.0])
        pred = np.array([0, 1, 1, 0])
        groups = np.array([0, 0, 0, 0])
        np.testing.assert_almost_equal(metrics.ndcg(act, pred, groups=groups), 1.0)

    def test_ndcg_negative2(self):
        act = np.array([-10.0, 10.0, 10.0, -2.0])
        pred = np.array([1, 0, 0, 2])
        groups = np.array([0, 0, 0, 0])
        np.testing.assert_almost_equal(metrics.ndcg(act, pred, groups=groups),
                                       0.57008955947784912)


class TestMetricsFamily(unittest.TestCase):

    def test_each_regression_metric_has_a_family(self):
        for metric in common.engine.metrics.REGRESSION_METRICS:
            self.assertIn(metric,
                          common.engine.metrics.metrics_family['Regression'])

    def test_each_classification_metric_has_a_family(self):
        for metric in common.engine.metrics.BINARY_METRICS:
            self.assertIn(metric,
                          common.engine.metrics.metrics_family['Binary'])


class MetricsReportTest(unittest.TestCase):

    PartitionClass = Partition
    size = 100

    def assertMetricReportEqual(self, out, reference):
        self.assertEqual(sorted(out.keys()), sorted(reference.keys()))
        self.assertEqual(sorted(out['metrics']), sorted(reference['metrics']))
        for key in reference['metrics']:
            try:
                np.testing.assert_array_almost_equal(out[key], reference[key], decimal=2)
            except:
                assert False, 'metric %r mismatch: %r != %r' % (key, out[key], reference[key])

    def test_first_cv_fold_reg(self):
        rs = np.random.RandomState(13)
        parts = [{'r': 0, 'k': -1}]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.rand(self.size)
        for p in Z:
            pred.add(rs.rand(self.size), **p)

        out = metrics.metrics_report(parts, metrics.REGRESSION_METRICS, pred, y, Z)
        reference = {'Gamma Deviance': [1.17616],
                     'Gini': [0.02723],
                     'Gini Norm': [0.14242],
                     'MAD': [0.3068],
                     'MAPE': [207.75575],
                     'Poisson Deviance': [0.35665],
                     'R Squared': [-0.46058],
                     'R Squared 20/80': [-3832.21439],
                     'RMSE': [0.38125],
                     'RMSLE': [0.26495],
                     'Tweedie Deviance': [0.61957],
                     'Weighted Gamma Deviance': [1.17616],
                     'Weighted Gini': [0.02723],
                     'Weighted Gini Norm': [0.14242],
                     'Weighted MAD': [0.3068],
                     'Weighted MAPE': [207.75575],
                     'Weighted Poisson Deviance': [0.35665],
                     'Weighted R Squared': [-0.46058],
                     'Weighted RMSE': [0.38125],
                     'Weighted RMSLE': [0.26495],
                     'Weighted Tweedie Deviance': [0.61957],
                     'labels': ['(0,-1)'],
            'metrics': ('Gini',
                        'Gini Norm',
                        'R Squared',
                        'R Squared 20/80',
                        'RMSLE',
                        'RMSE',
                        'MAD',
                        'MAPE',
                        'Poisson Deviance',
                        'Tweedie Deviance',
                        'Gamma Deviance',
                        'Weighted Gini',
                        'Weighted Gini Norm',
                        'Weighted R Squared',
                        'Weighted RMSLE',
                        'Weighted RMSE',
                        'Weighted MAD',
                        'Weighted MAPE',
                        'Weighted Poisson Deviance',
                        'Weighted Tweedie Deviance',
                        'Weighted Gamma Deviance')}
        self.assertMetricReportEqual(out, reference)

    def test_first_cv_fold_clf(self):
        rs = np.random.RandomState(13)
        parts = [{'r': 0, 'k': -1}]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.randint(0, 2, self.size)
        for p in Z:
            pred.add(rs.randint(0, 2, self.size), **p)

        out = metrics.metrics_report(parts, metrics.BINARY_METRICS, pred, y, Z)
        reference = {'AUC': [0.37363],
                     'Gini': [-0.11071],
                     'Gini Norm': [-0.34066],
                     'Ians Metric': [-0.25275],
                     'LogLoss': [20.72355],
                     'Rate@Top10%': [0.22222],
                     'Rate@Top5%': [0.22222],
                     'RMSE': [0.7746],
                     'Weighted AUC': [0.37363],
                     'Weighted Gini': [-0.11071],
                     'Weighted Gini Norm': [-0.34066],
                     'Weighted LogLoss': [20.72355],
                     'Weighted RMSE': [0.7746],
                     'AMS@15%tsh': [0],
                     'AMS@opt_tsh': [215.89706],
                     'labels': ['(0,-1)'],
            'metrics': ('LogLoss',
                        'AUC',
                        'Ians Metric',
                        'Gini',
                        'Gini Norm',
                        'RMSE',
                        'Weighted LogLoss',
                        'Weighted AUC',
                        'Weighted Gini',
                        'Weighted Gini Norm',
                        'Weighted RMSE',
                        'Rate@Top10%',
                        'Rate@Top5%',
                        'AMS@15%tsh',
                        'AMS@opt_tsh')}

        self.assertMetricReportEqual(out, reference)

    # FIXME unmark once we settled the issue on how 5-fold cv results are presented
    @pytest.mark.skip
    def test_five_cv_fold_reg(self):
        rs = np.random.RandomState(13)
        parts = [{'r': i, 'k': -1} for i in range(5)]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.rand(self.size)
        for p in Z:
            pred.add(rs.rand(self.size), **p)
        out = metrics.metrics_report(parts, metrics.REGRESSION_METRICS, pred, y, Z)

        reference = {'Gamma Deviance': [1.17616, 5.03282],
            'Gini': [0.02723, 0.0142],
            'Gini Norm': [0.14242, 0.07714],
            'MAD': [0.3068, 0.30731],
            'MAPE': [207.75575, 8861.26092],
            'Poisson Deviance': [0.35665, 0.53566],
            'R Squared': [-0.46058, -0.91838],
            'R Squared 20/80': [-3832.21439, -4492.46728],
            'RMSE': [0.38125, 0.40081],
            'RMSLE': [0.26495, 0.27676],
            'Tweedie Deviance': [0.61957, 1.71256],
            'Weighted Gamma Deviance': [1.17616, 5.03282],
            'Weighted Gini': [0.02723, 0.0142],
            'Weighted Gini Norm': [0.14242, 0.07714],
            'Weighted MAD': [0.3068, 0.30731],
            'Weighted MAPE': [207.75575, 8861.26092],
            'Weighted Poisson Deviance': [0.35665, 0.53566],
            'Weighted R Squared': [-0.46058, -0.91838],
            'Weighted RMSE': [0.38125, 0.40081],
            'Weighted RMSLE': [0.26495, 0.27676],
            'Weighted Tweedie Deviance': [0.61957, 1.71256],
            'labels': ['(0,-1)', '(.,-1)'],
            'metrics': ('Gini',
                        'Gini Norm',
                        'R Squared',
                        'R Squared 20/80',
                        'RMSLE',
                        'RMSE',
                        'MAD',
                        'MAPE',
                        'Poisson Deviance',
                        'Tweedie Deviance',
                        'Gamma Deviance',
                        'Weighted Gini',
                        'Weighted Gini Norm',
                        'Weighted R Squared',
                        'Weighted RMSLE',
                        'Weighted RMSE',
                        'Weighted MAD',
                        'Weighted MAPE',
                        'Weighted Poisson Deviance',
                        'Weighted Tweedie Deviance',
                        'Weighted Gamma Deviance')}

        self.assertMetricReportEqual(out, reference)

    @pytest.mark.skip
    def test_five_cv_fold_clf(self):
        rs = np.random.RandomState(13)
        parts = [{'r': i, 'k': -1} for i in range(5)]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.randint(0, 2, self.size)
        for p in Z:
            pred.add(rs.randint(0, 2, self.size), **p)
        out = metrics.metrics_report(parts, metrics.BINARY_METRICS, pred, y, Z)

        reference = {'AUC': [0.37363, 0.46919],
                     'Gini': [-0.11071, -0.03786],
                     'Gini Norm': [-0.34066, -0.19151],
                     'Ians Metric': [-0.25275, -0.06162],
                     'LogLoss': [20.72355, 18.30575],
                     'Rate@Top10%': [0.22222, 0.45652],
                     'Rate@Top5%': [0.22222, 0.45652],
                     'Weighted AUC': [0.37363, 0.46919],
                     'Weighted Gini': [-0.11071, -0.03786],
                     'Weighted Gini Norm': [-0.34066, -0.19151],
                     'Weighted LogLoss': [20.72355, 18.30575],
                     'AMS@15%tsh': [0, 212.69869],
                     'AMS@opt_tsh': [215.89706, 422.06775],
                     'labels': ['(0,-1)', '(.,-1)'],
            'metrics': ('LogLoss',
                        'AUC',
                        'Ians Metric',
                        'Gini',
                        'Gini Norm',
                        'Rate@Top10%',
                        'Rate@Top5%',
                        'Weighted LogLoss',
                        'Weighted AUC',
                        'Weighted Gini',
                        'Weighted Gini Norm',
                        'AMS@15%tsh',
                        'AMS@opt_tsh')}
        self.assertMetricReportEqual(out, reference)

    def test_eval_100pct_reg(self):
        """Above 64%. """
        rs = np.random.RandomState(13)
        parts = [{'r': -1, 'k': -1}]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.rand(self.size)
        for p in Z:
            pred.add(rs.rand(self.size), **p)
        out = metrics.metrics_report(parts, metrics.REGRESSION_METRICS, pred, y, Z)

        reference = {'Gamma Deviance': [1.17616, 4.00879],
            'Gini': [0.02723, -0.02455],
            'Gini Norm': [0.14242, -0.13339],
            'MAD': [0.3068, 0.37685],
            'MAPE': [207.75575, 9751.87082],
            'Poisson Deviance': [0.35665, 0.59348],
            'R Squared': [-0.46058, -1.22237],
            'R Squared 20/80': [-3832.21439, -4125.19428],
            'RMSE': [0.38125, 0.45318],
            'RMSLE': [0.26495, 0.31098],
            'Tweedie Deviance': [0.61957, 1.32457],
            'Weighted Gamma Deviance': [1.17616, 4.00879],
            'Weighted Gini': [0.02723, -0.02455],
            'Weighted Gini Norm': [0.14242, -0.13339],
            'Weighted MAD': [0.3068, 0.37685],
            'Weighted MAPE': [207.75575, 9751.87082],
            'Weighted Poisson Deviance': [0.35665, 0.59348],
            'Weighted R Squared': [-0.46058, -1.22237],
            'Weighted RMSE': [0.38125, 0.45318],
            'Weighted RMSLE': [0.26495, 0.31098],
            'Weighted Tweedie Deviance': [0.61957, 1.32457],
            'labels': ['(0,-1)', '(.,-1)'],
            'metrics': ('Gini',
                        'Gini Norm',
                        'R Squared',
                        'R Squared 20/80',
                        'RMSLE',
                        'RMSE',
                        'MAD',
                        'MAPE',
                        'Poisson Deviance',
                        'Tweedie Deviance',
                        'Gamma Deviance',
                        'Weighted Gini',
                        'Weighted Gini Norm',
                        'Weighted R Squared',
                        'Weighted RMSLE',
                        'Weighted RMSE',
                        'Weighted MAD',
                        'Weighted MAPE',
                        'Weighted Poisson Deviance',
                        'Weighted Tweedie Deviance',
                        'Weighted Gamma Deviance')}

        self.assertMetricReportEqual(out, reference)

    def test_eval_100pct_clf(self):
        rs = np.random.RandomState(13)
        parts = [{'r': -1, 'k': -1}]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.randint(0, 2, self.size)
        for p in Z:
            pred.add(rs.randint(0, 2, self.size), **p)
        out = metrics.metrics_report(parts, metrics.BINARY_METRICS, pred, y, Z)

        reference = {'AUC': [0.37363, 0.44098],
            'Gini': [-0.11071, -0.05663],
            'Gini Norm': [-0.34066, -0.22209],
            'Ians Metric': [-0.25275, -0.11805],
            'LogLoss': [20.72355, 19.34196],
            'Rate@Top10%': [0.22222, 0.43636],
            'Rate@Top5%': [0.22222, 0.43636],
            'RMSE': [0.7746, 0.74833],
            'Weighted AUC': [0.37363, 0.44098],
            'Weighted Gini': [-0.11071, -0.05663],
            'Weighted Gini Norm': [-0.34066, -0.22209],
            'Weighted LogLoss': [20.72355, 19.34196],
            'Weighted RMSE': [0.7746, 0.7483],
            'AMS@15%tsh': [0, 91.15265],
            'AMS@opt_tsh': [215.89706, 305.16717],
            'labels': ['(0,-1)', '(.,-1)'],
            'metrics': ('LogLoss',
                        'AUC',
                        'Ians Metric',
                        'Gini',
                        'Gini Norm',
                        'Rate@Top10%',
                        'Rate@Top5%',
                        'RMSE',
                        'Weighted LogLoss',
                        'Weighted AUC',
                        'Weighted Gini',
                        'Weighted Gini Norm',
                        'Weighted RMSE',
                        'AMS@15%tsh',
                        'AMS@opt_tsh')}

        self.assertMetricReportEqual(out, reference)

    @patch('ModelingMachine.engine.blueprint_interpreter.BuildData')
    def test_eval_100pct_ordreg_builddata(self, MockBuildData):
        rs = np.random.RandomState(13)

        build_data = MockBuildData()
        build_data.dataframe.return_value = pd.DataFrame(rs.randint(0, 2, self.size))

        parts = [{'r': -1, 'k': -1}]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.randint(0, 4, self.size)
        for p in Z:
            pred.add(rs.randint(0, 4, self.size), **p)
        out = metrics.metrics_report(parts, [metrics.CURMSE, metrics.NDCG],
                                     pred, y, Z, build_data=build_data)

        reference = {'Coldstart RMSE': [0.0, 1.79165],
                     'NDCG': [0.71045, 0.24772],
                     'labels': ['(0,-1)', '(.,-1)'],
                     'metrics': ['Coldstart RMSE', 'NDCG']}

        self.assertMetricReportEqual(out, reference)

    @patch('ModelingMachine.engine.blueprint_interpreter.BuildData')
    def test_eval_no_coldstart(self, MockBuildData):
        rs = np.random.RandomState(13)

        build_data = MockBuildData()


        parts = [{'r': 0, 'k': -1}]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.rand(self.size)
        for p in Z:
            pred.add(rs.rand(self.size), **p)

        coldstart_col = pd.DataFrame([0] * self.size)  # only one group
        build_data.dataframe.return_value =coldstart_col
        out = metrics.metrics_report(parts, [metrics.CURMSE, metrics.CUMAD],
                                     pred, y, Z, build_data=build_data)

        reference = {'Coldstart MAD': [0.0],
                     'Coldstart RMSE': [0.0],
                     'labels': ['(0,-1)'],
                     'metrics': ['Coldstart RMSE', 'Coldstart MAD']}

        self.assertMetricReportEqual(out, reference)

    @patch('ModelingMachine.engine.blueprint_interpreter.BuildData')
    def test_eval_all_coldstart(self, MockBuildData):
        rs = np.random.RandomState(13)

        build_data = MockBuildData()
        parts = [{'r': 0, 'k': -1}]
        Z = self.PartitionClass(size=self.size, seed=0, folds=5, reps=5)
        Z.set(partitions=[(p['r'], p['k']) for p in parts])
        pred = Container()
        y = rs.rand(self.size)
        for p in Z:
            pred.add(rs.rand(self.size), **p)

        coldstart_col = pd.DataFrame(range(self.size))  # all different groups
        build_data.dataframe.return_value = coldstart_col
        out = metrics.metrics_report(parts, [metrics.CURMSE, metrics.CUMAD],
                                     pred, y, Z, build_data=build_data)

        reference = metrics.metrics_report(parts, [metrics.RMSE, metrics.MAD],
                                           pred, y, Z)

        self.assertEqual(out[metrics.CURMSE], reference[metrics.RMSE])
        self.assertEqual(out[metrics.CUMAD], reference[metrics.MAD])


# FIXME -- those are different; different partitioning?
# class NewPartitionMetricsReportTest(MetricsReportTest):

#     PartitionClass = partial(NewPartition, cv_method='RandomCV')



if __name__ == '__main__':
    unittest.main()
