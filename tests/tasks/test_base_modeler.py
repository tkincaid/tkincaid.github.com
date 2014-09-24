import unittest
import copy
import logging
import numpy as np
import pandas as pd

from ModelingMachine.engine.container import Container
from ModelingMachine.engine.new_partition import NewPartition
from ModelingMachine.engine.tasks.base_modeler import BaseModeler
from ModelingMachine.engine.tasks.glm import GLMG, GLMB
from ModelingMachine.engine.tasks.rf import RFC, RFR
from ModelingMachine.engine.tasks.lasso_ridge import RegL2
from ModelingMachine.engine.tasks.sgd import SGDC, SGDR
from ModelingMachine.engine.tasks.cart import CARTClassifier, CARTRegressor
from ModelingMachine.engine.tasks.svc import SVMC, SVMR
from ModelingMachine.engine.tasks import gbm
from common.engine import metrics
from common.logger import logger as getLogger

from base_task_test import BaseTaskTest


logger = getLogger()


def create_correlated_test_data(reverse=False):
    ''' Create a test Container for X and ndarray for Y.
        One of the columns in X is a perfect predictor of Y.
    '''
    np.random.seed(1234)
    X = Container()
    X.add(np.random.rand(100), colnames=['rand'])
    if reverse:
        X.add(np.arange(100)[::-1], colnames=['linear'])
        Y = np.arange(100)[::-1]
    else:
        X.add(np.arange(100), colnames=['linear'])
        Y = np.arange(100)
    X.add(np.ones(100), colnames=['ones'])
    return (X, Y)


def create_semi_correlated_test_data():
    ''' Create a test Container for X and ndarray for Y.
        One of the columns in X is correlated with Y.
    '''
    np.random.seed(1234)
    X = Container()
    X.add(np.concatenate((np.ones(500), np.arange(500) + 500)), colnames=['rand'])
    X.add(np.concatenate((np.arange(500) - 1, np.arange(500) + 501)), colnames=['linear'])
    Y = np.arange(1000)
    X.add(np.ones(1000), colnames=['ones'])
    return (X, Y)


def create_zero_data():
    ''' Create a test Container for X and ndarray for Y.
        The only predictor is zero
    '''
    np.random.seed(1234)
    Y = np.random.choice([0, 1, 0, 0], 5000)
    X = Container()
    X.add(np.zeros(5000), colnames=['zeros'])
    return (X, Y)

def create_rd_data(n=1000):
    ''' Create a test Container for X and ndarray for Y.
        The only predictor is zero
    '''
    np.random.seed(1234)
    Y = np.random.choice([0, 1], n)
    X = Container()
    rng = np.random.RandomState(158)
    X.add(np.transpose((Y/2+rng.rand(1, n))), colnames=['rd'])
    return (X, Y)

class TestEstimator(object):

    ''' An estimator that just predicts the value of the
        'pred_output' parameter.  Useful for testing grid search. '''

    def __init__(self, **args):
        self.args = args

    def fit(self, X, Y, weights=None, offsets=None, **args):
        pass

    def predict(self, X, Y=None, weights=None, offsets=None, **args):
        ''' output the value of parameter 'pred_output' '''
        return np.repeat(self.args['pred_output'], X.shape[0])

    def get_params(self, deep):
        args = copy.deepcopy(self.args)
        return args

    def set_params(self, **args):
        self.args.update(args)


class TestModel(BaseModeler):

    ''' The BaseModeler part of TestEstimator '''

    sparse_input_support = False
    estimator = TestEstimator
    description = "Test Model"
    arguments = {
        'p': {'name': 'pred_output', 'type': 'intgrid', 'values': [1, 3], 'default': '1'},
    }

    def _weight_and_offset(self, other_args):
        out = {}
        if 'weight' in other_args:
            out['weights'] = other_args.pop('weight').astype(float)
        if 'offset' in other_args:
            out['offsets'] = other_args.pop('offset').astype(float)
        return out


class TestBaseModeler(BaseTaskTest):

    def test_arguments(self):
        """Checks all base modeler argument combinations. """
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(TestModel, TestEstimator, xt, yt, skip_bm_args=False)

    def test_glm(self):
        ''' test GLM fit and predict '''
        model = GLMG()
        # X contains a column that predicts Y perfectly to make verifying output easier
        X, Y = create_correlated_test_data()
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        pred = model.predict(X, Y, Z)
        # The prediction for each partition should match Y
        for p in pred:
            key = (p['r'], p['k'])
            np.testing.assert_array_almost_equal(pred(**p).flatten(), Y)
            np.testing.assert_array_almost_equal(model.pred_stack[key], Y)

    def test_glm_offset(self):
        ''' test fitting a GLM with offsets '''
        model = GLMG()
        X, Y = create_correlated_test_data()
        X.initialize({'offset': pd.Series(np.ones(100))})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # test predict with offset -  should match Y
        pred = model.predict(X, Y, Z)
        for p in pred:
            key = (p['r'], p['k'])
            np.testing.assert_array_almost_equal(pred(**p).flatten(), Y)
            np.testing.assert_array_almost_equal(model.pred_stack[key], Y)
        # re-create test data, but don't add offset so we can see if the
        # offset if being used by checking that the prediction is Y-1
        X, Y = create_correlated_test_data()
        pred = model.predict(X, Y, Z)
        for p in pred:
            key = (p['r'], p['k'])
            np.testing.assert_array_almost_equal(pred(**p).flatten(), Y - 1)

    def test_glm_weights(self):
        ''' test fitting a GLM with weights '''
        model = GLMG()
        X, Y = create_correlated_test_data()
        # First half of weights are one and second half are one hundred
        X.initialize({'weight': pd.Series(np.concatenate((np.repeat(1, 50), np.repeat(100, 50))))})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # test predict with weights -  should match Y
        pred = model.predict(X, Y, Z)
        for p in pred:
            key = (p['r'], p['k'])
            np.testing.assert_array_almost_equal(pred(**p).flatten(), Y)
            np.testing.assert_array_almost_equal(model.pred_stack[key], Y)

        # First half of Y is X[1]-1, second half is X[1]+1
        Y[:50] = X()[:50, 1] - 1
        Y[50:] = X()[50:, 1] + 1
        # fit model with updated Y
        model = GLMG()
        model.fit(X, Y, Z)
        pred = model.predict(X, Y, Z)
        # pred[51:100] should be closer to Y than pred[0:50] since it
        # was weighted much higher
        for p in pred:
            key = (p['r'], p['k'])
            # check predictions
            diff_first_half = np.sum(np.abs(pred(**p).flatten()[:50] - Y[:50]))
            diff_second_half = np.sum(np.abs(pred(**p).flatten()[50:] - Y[50:]))
            self.assertGreater(diff_first_half, diff_second_half)
            # check stacked predictions
            diff_first_half = np.sum(np.abs(model.pred_stack[key][:50] - Y[:50]))
            diff_second_half = np.sum(np.abs(model.pred_stack[key][50:] - Y[50:]))
            self.assertGreater(diff_first_half, diff_second_half)

    def test_glm_newdata(self):
        ''' test GLM fit and predict with new_data '''
        model = GLMG()
        # X contains a column that predicts Y perfectly to make verifying output easier
        X, Y = create_correlated_test_data()
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # Make "new" data
        X, Y = create_correlated_test_data(reverse=True)
        pred = model.predict(X, Y, Z)
        # The prediction for each partition should match Y
        for p in pred:
            np.testing.assert_array_almost_equal(pred(**p).flatten(), Y)

    def test_grid_search(self):
        ''' test that grid search chooses the best parameter '''
        model = TestModel('p=[1,2,3]')
        X, Y = create_correlated_test_data()
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # grid search should choose 3 since it yields the lowest error
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 3)

        # try with different Y to make sure the last test didn't pass by chance
        Y = np.repeat(2, X.shape[0])
        model = TestModel('p=[1,2,3]')
        model.fit(X, Y, Z)
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 2)

    #
    # disabled until #3382 is merged
    #
    def x_test_weighted_grid_search(self):
        ''' test that weighted grid search chooses the best parameter '''
        model = TestModel('p=[1,2,3];t_m=Weighted RMSE')
        X, Y = create_correlated_test_data()
        weights = np.ones(X.shape[0])
        # put all weight on Y=0
        weights[0] = 99999
        X.initialize({'weight': pd.Series(weights)})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        # Just do one rep since the test will fail if row 0 is in the test set
        Z.set(max_folds=0, max_reps=1)
        model.fit(X, Y, Z)
        # grid search should choose 1 since we put a huge
        # weight on Y=0
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 1)

        # try with different weights to make sure the last test didn't pass by chance
        X, Y = create_correlated_test_data()
        weights = np.ones(X.shape[0])
        # put all weight on Y=0
        weights[2] = 99999
        X.initialize({'weight': pd.Series(weights)})
        model = TestModel('p=[1,2,3];t_m=Weighted RMSE')
        model.fit(X, Y, Z)
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 2)

    def test_weighted_check_stack(self):
        model = GLMB()
        X, Y = create_zero_data()
        weights = 1 - Y + 1
        X.initialize({'weight': pd.Series(weights)})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        for stack in model.pred_stack.values():
            self.assertTrue(np.all(np.abs(stack - 0.1425) < 0.01))

    def test_weighted_check_stack_skw_classification(self):
        X, Y = create_rd_data(1000)
        weights = 2 - Y
        X.initialize({'weight': pd.Series(weights)})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0, max_reps=1)
        for model in [
                GLMB(), RFC('nt=20;ss=[2,5]'), RegL2('a=[0.1,0.01]'),
                SGDC('a=[0.1,0.01]'), CARTClassifier(),
                gbm.ESGBC('n=10;md=1'),
                gbm.GBC('n=10;md=1'),
                gbm.RGBC('n=10;md=1'),
                ]:
            logger.info('_' * 80)
            logger.info('n partitions: %d', sum(1 for _ in Z))
            model.fit(X, Y, Z)
            logger.info('_' * 80)
            for stack in model.pred_stack.values():
                self.assertTrue(stack.mean() < 0.45, '%r >= 0.45 for %r' %
                                (stack.mean(), model))
        #TODO test SVMC too. Currently it is too long and fails to deliver expected results

    def test_weighted_check_stack_skw_regression(self):
        X, Y = create_rd_data(1000)
        weights = 2 - Y
        X.initialize({'weight': pd.Series(weights)})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0, max_reps=1)
        for model in [
                GLMG(), RFR('nt=20;ss=[2,5]'), RegL2('a=[0.1,0.01]'),
                SGDR('a=[0.1,0.01]'), CARTRegressor(),
                gbm.ESGBR('n=10;md=1'),
                gbm.GBR('n=10;md=1'),
                gbm.RGBR('n=10;md=1'),
                SVMR('g=0.1;C=1.0'),
                ]:
            logger.info('_' * 80)
            logger.info('n partitions: %d', sum(1 for _ in Z))
            model.fit(X, Y, Z)
            logger.info('_' * 80)
            for stack in model.pred_stack.values():
                self.assertTrue(stack.mean() < 0.45, '%r >= 0.45 for %r' %
                                (stack.mean(), model))

    def test_backwards_stepwise_regression(self):
        model = GLMG('sw_b=1')
        # X contains a column that predicts Y perfectly to make verifying output easier
        X, Y = create_correlated_test_data()
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # BSR should drop all useless columns (ie. 0 & 2)
        for value in model.use_cols.values():
            self.assertEqual(value, [1])

    #
    # disabled until #3382 is merged
    #
    def x_test_weighted_backwards_stepwise_regression(self):
        model = GLMG('sw_b=1;t_m=Weighted RMSE')
        # X contains a column that is correlated with Y which
        # should be retained by BSR
        X, Y = create_semi_correlated_test_data()
        # making the weights equal to column 0 should give the column 0 value
        X.initialize({'weight': pd.Series(X()[:, 0])})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # BSR should drop all useless columns (ie. 2)
        for value in model.use_cols.values():
            self.assertEqual(value, [0, 1])

    def test_additional_grid_search(self):
        ''' test additional grid search chooses the best parameter '''
        model = TestModel('p=[1,2,3]')
        X, Y = create_correlated_test_data()
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # grid search should choose 3 since it yields the lowest error
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 3)
        original_model = model
        # expand grid
        model = TestModel('p=[1,2,3,4,5,6]')
        # simulate vertex update
        model.best_parameters = original_model.best_parameters
        model.grid_scores = original_model.grid_scores
        model.old__parameters = original_model.parameters
        # refit model with expanded grid
        model.fit(X, Y, Z)
        # additional grid search should choose 6 since it yields the lowest error
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 6)

    #
    # disabled until #3382 is merged
    #
    def x_test_weighted_additional_grid_search(self):
        ''' test weighted additional grid search chooses the best parameter '''
        model = TestModel('p=[1,2,3];t_m=Weighted RMSE')
        X, Y = create_correlated_test_data()
        X.initialize({'weight': pd.Series(X()[:, 0])})
        Z = NewPartition(X.nrow, cv_method='RandomCV')
        Z.set(max_folds=0)
        model.fit(X, Y, Z)
        # grid search should choose 3 since it yields the lowest error
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 3)
        original_model = model
        # expand grid
        model = TestModel('p=[1,2,3,4,5,6]')
        # simulate vertex update
        model.best_parameters = original_model.best_parameters
        model.grid_scores = original_model.grid_scores
        model.old__parameters = original_model.parameters
        # refit model with expanded grid
        model.fit(X, Y, Z)
        # additional grid search should choose 6 since it yields the lowest error
        for key in model.best_parameters:
            self.assertEqual(model.best_parameters[key]['pred_output'], 6)

if __name__ == '__main__':
    unittest.main()
