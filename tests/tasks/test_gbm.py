#########################################################
#
#       Unit Test for tasks/gbm.py
#
#       Author: Tom de Godoy, Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import logging
import copy
import numpy as np
import pandas as pd

import pytest
from mock import patch

from tesla.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from tesla.ensemble._r_gbm import GBMClassifier, GBMRegressor

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.gbm import GBC
from ModelingMachine.engine.tasks.gbm import GBR
from ModelingMachine.engine.tasks.gbm import ESGBC
from ModelingMachine.engine.tasks.gbm import ESGBR
from ModelingMachine.engine.tasks.gbm import MultinomialGBR, _MultinomialGBR
from ModelingMachine.engine.tasks.gbm import RGBC
from ModelingMachine.engine.tasks.gbm import RGBR
from ModelingMachine.engine.tasks.partial_dependence import PartialDependencePlot
from ModelingMachine.engine.tasks.cat_encoders import OrdinalEncoder
from ModelingMachine.engine import metrics
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition

from sklearn.datasets import make_hastie_10_2

class TestTeslaGBRT(BaseTaskTest):

    def test_arguments(self):
        """Test if arguments are passed to wrapped estimator"""
        xt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        yt = np.array([0, 1, 0])
        self.check_arguments(GBC, GradientBoostingClassifier, xt, yt)
        self.check_arguments(GBR, GradientBoostingRegressor, xt, yt)
        self.check_arguments(ESGBC, GradientBoostingClassifier, xt, yt)
        self.check_arguments(ESGBR, GradientBoostingRegressor, xt, yt)
        # Multinomial GBR has an additional argument so can't compare
        # with the tesla version
        self.check_arguments(MultinomialGBR, _MultinomialGBR, xt, yt)

    def test_default_arguments(self):
        """Test if task defaults are the same as sklearn/tesla defaults.

        Ignores max_features because we use a different default here.
        """
        self.check_default_arguments(GBC, GradientBoostingClassifier, ignore_params=('random_state',))
        self.check_default_arguments(GBR, GradientBoostingRegressor, ignore_params=('random_state',))
        self.check_default_arguments(ESGBC, GradientBoostingClassifier, ignore_params=('random_state',))
        self.check_default_arguments(ESGBR, GradientBoostingRegressor, ignore_params=('random_state',))

    @pytest.mark.dscomp
    def test_GBC_reproducible(self):
        """Smoke test for classification. """
        X, Y, Z = self.create_bin_data()
        #reference = [0.41451537,  0.41154491,  0.41451537]
        reference = None
        task_args = ['md=[3, 4]', 'ls=[2, 4]', 'lr=0.002', 't_f=0.2',
                     't_m={metrics.AUC}'.format(metrics=metrics), 't_a=1']
        task_desc = 'GBC '+';'.join(task_args)
        t = self.check_task(task_desc, X, Y, Z, transform=True,
                            standardize=False, reference=reference)
        # check default of 1
        self.assertEqual(t.gridsearch_parameters['step'], 1)
        np.testing.assert_array_almost_equal(t.pred_stack[(0, -1)][:5],
                                             np.array([0.40618405, 0.40618405,
                                                    0.40618405, 0.40639949, 0.41059348]))

    @pytest.mark.dscomp
    def test_ESGBC_reproducible(self):
        """Smoke test for classification early stopping. """
        X, Y, Z = self.create_bin_data()
        #reference = [0.42249643, 0.40776091, 0.42249643]
        reference = None
        task_args = ['md=[3, 4]', 'ls=[2, 4]', 'lr=0.002', 't_f=0.2',
                     't_m={metrics.AUC}'.format(metrics=metrics), 't_a=1',
                     't_sp=10']
        task_desc = 'ESGBC '+';'.join(task_args)
        t = self.check_task(task_desc, X, Y, Z, transform=True,
                            standardize=False, reference=reference)
        # check if higher step works too
        self.assertEqual(t.gridsearch_parameters['step'], 10)
        np.testing.assert_array_almost_equal(t.pred_stack[(0, -1)][:5],
                                             np.array([0.39338353, 0.39338353,
                                                       0.39338353, 0.39490888,
                                                       0.40311164]))

    def test_GBR_reproducible(self):
        """Smoke test for regression. """
        X, Y, Z = self.create_reg_data()
        reference = [ 182.44433653,  163.98684038,  182.44433653]
        t = self.check_task('GBR n=2', X, Y, Z, transform=True, standardize=False,
                            reference=reference)
        self.assertIsNone(t.model.values()[0].max_leaf_nodes)
        np.testing.assert_array_almost_equal(t.pred_stack[(0, -1)][:4],
                                             np.array([175.33700529, 159.54769827,
                                                       175.33700529, 166.06402382]))

    def test_ESGBR_reproducible(self):
        """Smoke test for regression early stopping. """
        X, Y, Z = self.create_reg_data()
        reference = [ 182.44433653,  163.98684038,  182.44433653]
        t = self.check_task('ESGBR n=2', X, Y, Z, transform=True, standardize=False,
                            reference=reference)
        self.assertIsNone(t.model.values()[0].max_leaf_nodes)
        np.testing.assert_array_almost_equal(t.pred_stack[(0, -1)][:4],
                                             np.array([175.33700529, 159.54769827,
                                                       175.33700529, 166.06402382]))

    def test_MNGBR_reproducible(self):
        """Smoke test for multinomial regression. """
        X, Y, Z = self.create_reg_data()
        # need to have fewer than ten values
        reference = [ 3.0788119 ,  3.24363511,  3.24899394]
        self.check_task('MNGBR n=2', X, np.round(np.log(Y)), Z,
                        transform=True, standardize=False, reference=reference)

    @pytest.mark.dscomp
    def test_clf_early_stop(self):
        """Test if early-stopping is in effect. """
        X, Y, Z = self.create_bin_data()
        # this should overfit enough
        t = self.check_task('ESGBC s=0.5;n=100;md=6;ls=1;lr=0.4', X, Y, Z,
                            transform=True, standardize=False)
        est = t.model[(0, -1)]
        # NOTE: this might change if you change the GBM implementation
        # early stopped after 28 iterations
        self.assertEqual(est.n_estimators, 100)
        self.assertEqual(est.max_features, None)
        self.assertEqual(np.unique(est.learning_rate_).shape[0], 1)
        self.assertEqual(est.learning_rate_[0], 0.4)
        self.assertEqual(est.min_samples_leaf, 1)
        self.assertEqual(est.max_depth, 6)

    @pytest.mark.dscomp
    def test_clf_early_stop_gridsearch_logloss(self):
        """Test clf if early-stopping is in effect when doing gridsearch with LogLoss. """
        X, Y, Z = self.create_bin_data()
        # this should overfit enough
        t = self.check_task('ESGBC s=1.0;n=100;md=[2];ls=1;lr=[0.1, 0.000001];'
                            't_m={metrics.LOGLOSS}'.format(metrics=metrics),
                            X, Y, Z, transform=True, standardize=False)
        self.assertEqual(metrics.LOGLOSS, t.gridsearch_parameters['metric'])
        est = t.model[(0, -1)]
        self.assertEqual(np.unique(est.learning_rate_).shape[0], 1)
        # best parameters
        self.assertEqual(est.learning_rate_[0], 0.1)
        self.assertEqual(est.min_samples_leaf, 1)
        self.assertEqual(est.max_depth, 2)
        # ES in effect?
        self.assertLess(est.estimators_.shape[0], 100)
        # check default of 1
        self.assertEqual(t.gridsearch_parameters['step'], 1)
        self.assertTrue((0, -1) in t.partial_dependence_plots)

    @pytest.mark.dscomp
    @patch('ModelingMachine.engine.metrics._logloss', autospec=True)
    def test_clf_early_stop_gridsearch_weights(self, mocklogloss):
        """Test clf passes weights to the loss function if early-stopping is in effect when doing gridsearch. """
        def weight_loss(actual, pred, weights):
            print "Test"
            if np.all(weights[actual == 1] == 10.) and \
               np.all(weights[actual == -1] == 1.):
                raise ValueError("Weights passed successfully")
            else:
                assert(False)
                return np.sum(pred) - 50.0
        mocklogloss.method = weight_loss
        x, Y = make_hastie_10_2(n_samples=300, random_state=41)
        X = Container()
        X.add(x)
        Z = Partition(X.shape[0], max_reps=2, max_folds=0)
        Z.set(max_reps=1, max_folds=1)
        wt = {'weight': pd.Series(2.0 + 9.0 * (Y == 1).astype(float))}

        # Add weights to container
        X.initialize(wt)
        task = ESGBC('s=1;n=10;md=[2];ls=1;lr=[0.1, 0.000001];t_m=Weighted LogLoss')

        task.fit(X, Y, Z)
        # Assert the patched loss function was passed the weights
        self.assertTrue(mocklogloss.called)
        # The third argument is weight, we should be passed two values
        passed_weights = mocklogloss.call_args[0][2]
        passed_actuals = mocklogloss.call_args[0][0]
        self.assertEqual(len(np.unique(passed_weights)), 2)
        print passed_weights
        self.assertTrue(np.all(passed_weights[passed_actuals == -1] == 2))
        self.assertTrue(np.all(passed_weights[passed_actuals == 1] == 11))

    @pytest.mark.dscomp
    def test_clf_early_stop_gridsearch_auc(self):
        """Test clf if early-stopping is in effect when doing gridsearch with AUC. """
        X, Y, Z = self.create_bin_data()
        # this should overfit enough
        t = self.check_task('ESGBC n=100;md=[2];ls=1;lr=[0.1, 0.000001];'
                            't_m={metrics.AUC}'.format(metrics=metrics),
                            X, Y, Z, transform=True, standardize=False)
        self.assertEqual(metrics.AUC, t.gridsearch_parameters['metric'])
        est = t.model[(0, -1)]
        self.assertEqual(np.unique(est.learning_rate_).shape[0], 1)
        # best parameters
        self.assertEqual(est.learning_rate_[0], 0.1)
        self.assertEqual(est.min_samples_leaf, 1)
        self.assertEqual(est.max_depth, 2)
        # ES in effect?
        self.assertLess(est.estimators_.shape[0], 100)
        # check default of 1
        self.assertEqual(t.gridsearch_parameters['step'], 1)
        self.assertTrue((0, -1) in t.partial_dependence_plots)

    @pytest.mark.dscomp
    def test_reg_early_stop_gridsearch_rmsle(self):
        """Test reg if early-stopping is in effect when doing gridsearch with RMSLE. """
        X, Y, Z = self.create_reg_syn_data()
        # this should overfit enough
        t = self.check_task('ESGBR n=100;md=[6];ls=1;lr=[0.4, 0.000001];s=1.0;t_m={metrics.RMSLE}'.format(
            metrics=metrics), X, Y, Z, transform=True, standardize=False)
        self.assertEqual(metrics.RMSLE, t.gridsearch_parameters['metric'])
        est = t.model[(0, -1)]
        self.assertEqual(np.unique(est.learning_rate_).shape[0], 1)
        # best parameters
        self.assertEqual(est.learning_rate_[0], 0.4)
        self.assertEqual(est.min_samples_leaf, 1)
        self.assertEqual(est.max_depth, 6)
        # ES in effect?
        self.assertLess(est.estimators_.shape[0], 100)
        # check default of 1
        self.assertEqual(t.gridsearch_parameters['step'], 1)
        self.assertTrue((0, -1) in t.partial_dependence_plots)

    @pytest.mark.dscomp
    def test_reg_early_stop_gridsearch_rsquared(self):
        """Test reg if early-stopping is in effect when doing gridsearch with R Squared. """
        X, Y, Z = self.create_reg_syn_data()
        # this should overfit enough
        t = self.check_task(('ESGBR n=100;ml=[8];ls=1;lr=[0.4, 0.000001];s=1.0;'
                             't_m={metrics.R_SQUARED}').format(metrics=metrics),
                            X, Y, Z, transform=True, standardize=False)
        self.assertEqual(metrics.R_SQUARED, t.gridsearch_parameters['metric'])
        est = t.model[(0, -1)]
        self.assertEqual(np.unique(est.learning_rate_).shape[0], 1)
        # best parameters
        self.assertEqual(est.learning_rate_[0], 0.4)
        self.assertEqual(est.min_samples_leaf, 1)
        self.assertEqual(est.max_leaf_nodes, 8)
        # ES in effect?
        self.assertLess(est.estimators_.shape[0], 100)
        # check default of 1
        self.assertEqual(t.gridsearch_parameters['step'], 1)
        self.assertTrue((0, -1) in t.partial_dependence_plots)

    @pytest.mark.dscomp
    def test_reg_early_stop_gridsearch_smaller_than_step(self):
        """Test _fit_grid if stopped before step.

        A regression test for https://trello.com/c/CIY6vGG2/230-gbm-error-for-fastiron-sample-400.
        """
        X, Y, Z = self.create_reg_syn_data()
        # this should overfit enough
        t = self.check_task(('ESGBR n=120;md=[6];ls=1;lr=[1.0, 0.000001];s=1.0;rs=12;'
                             't_m={metrics.R_SQUARED};t_sp=40').format(metrics=metrics),
                            X, Y, Z, transform=True, standardize=False)
        self.assertEqual(metrics.R_SQUARED, t.gridsearch_parameters['metric'])
        est = t.model[(0, -1)]
        self.assertEqual(np.unique(est.learning_rate_).shape[0], 1)
        # best parameters
        self.assertEqual(est.learning_rate_[0], 1.0)
        self.assertEqual(est.min_samples_leaf, 1)
        self.assertEqual(est.max_depth, 6)
        # ES in effect?
        self.assertLess(est.estimators_.shape[0], 100)
        self.assertTrue((0, -1) in t.partial_dependence_plots)

    #@pytest.mark.skip
    def test_multinomial_reg_more_than_ten_value_error(self):
        """Test that large number of values raise an error. """
        X, Y, Z = self.create_reg_data()
        # need to have fewer than ten values
        self.assertRaises(ValueError, self.check_task, 'MNGBR', X, Y, Z, transform=True, standardize=False)

    def test_multinomial_reg_predictions_max_match_unique_values(self):
        """Test that large number of values raise an error. """
        X, Y, Z = self.create_reg_data()
        Y = np.round(np.log(Y))
        # need to have fewer than ten values
        task = self.check_task('MNGBR n=10;cm=max;md=[3, 4];t_n=1;t_f=0.15', X, Y, Z, transform=True, standardize=False)
        est = task.model[(0, -1)]
        Z = Partition(size=X.dataframe.shape[0], folds=5, reps=5, total_size=X.dataframe.shape[0])
        Z.set(max_reps=1, max_folds=0)
        C = Container()
        C.add(X.dataframe.values)
        # task.model[(-1, -1)] = task.model[(0, -1)]
        preds = task.predict(C, Y, Z)
        for predval in np.unique(preds(r=0, k=-1)):
            assert(predval in np.unique(Y))

    def test_multinomial_reg_predictions_mean_in_range_unique_values(self):
        """Test that large number of values raise an error. """
        X, Y, Z = self.create_reg_data()
        Y = np.round(np.log(Y)) + 5  # Greater than 1 to ensure bounds don't work for probabilities)
        # need to have fewer than ten values
        task = self.check_task('MNGBR n=10;cm=mean;md=[3, 4];t_n=1;t_f=0.15', X, Y, Z, transform=True, standardize=False)
        est = task.model[(0, -1)]
        Z = Partition(size=X.dataframe.shape[0], folds=5, reps=5, total_size=X.dataframe.shape[0])
        Z.set(max_reps=1, max_folds=0)
        C = Container()
        C.add(X.dataframe.values)
        # task.model[(-1, -1)] = task.model[(0, -1)]
        preds = task.predict(C, Y, Z)(r=0, k=-1)
        assert(np.unique(Y).max() >= preds.max())
        assert(np.unique(Y).min() <= preds.min())

    def test_multinomial_reg_predictions_median_in_range_unique_values(self):
        """Test that large number of values raise an error. """
        X, Y, Z = self.create_reg_data()
        Y = np.round(np.log(Y)) + 5  # Greater than 1 to ensure bounds don't work for probabilities
        # need to have fewer than ten values
        task = self.check_task('MNGBR n=10;cm=median;md=[3, 4];t_n=1;t_f=0.15', X, Y, Z, transform=True, standardize=False)
        est = task.model[(0, -1)]
        Z = Partition(size=X.dataframe.shape[0], folds=5, reps=5, total_size=X.dataframe.shape[0])
        Z.set(max_reps=1, max_folds=0)
        C = Container()
        C.add(X.dataframe.values)
        # task.model[(-1, -1)] = task.model[(0, -1)]
        preds = task.predict(C, Y, Z)(r=0, k=-1)
        assert(np.unique(Y).max() >= preds.max())
        assert(np.unique(Y).min() <= preds.min())

    def test_poisson_tweedie_loss(self):
        """Run both a poisson and a tweedie gbm (with alpha close to 1) and
        check if pred stack is roughly the same.
        """
        X, Y, Z = self.create_reg_count_syn_data()
        t_p = self.check_task('GBR l=poisson;n=100;md=1;lr=0.1', X, Y, Z,
                            transform=True, standardize=False)
        est_p = t_p.model[(0, -1)]
        self.assertEqual(est_p.loss, 'poisson')

        t_t = self.check_task('GBR l=tweedie;a=1.000001;n=100;md=1;lr=0.1', X, Y, Z,
                            transform=True, standardize=False)
        est_t = t_t.model[(0, -1)]
        self.assertEqual(est_t.loss, 'tweedie')
        np.testing.assert_array_almost_equal(t_p.pred_stack[(0, -1)],
                                             t_t.pred_stack[(0, -1)], decimal=4)


class TestRGBMWrapper(BaseTaskTest):

    def create_reg_data(self, reps=1):
        X = copy.deepcopy(self.ds2)
        Y = X.pop('Claim_Amount').values
        Z = Partition(size=X.shape[0], folds=5, reps=5, total_size=X.shape[0])
        Z.set(max_reps=reps, max_folds=0)
        return X, Y, Z

    def test_arguments(self):
        """Test if arguments are passed to wrapped estimator"""
        xt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        yt = np.array([0, 1, 0])
        self.check_arguments(RGBC, GBMClassifier, xt, yt)
        self.check_arguments(RGBR, GBMRegressor, xt, yt)

    def test_default_arguments(self):
        """Test if task defaults are the same as sklearn/tesla defaults.

        Ignores max_features because we use a different default here.
        """
        self.check_default_arguments(RGBC, GBMClassifier, ['column_types', 'random_state'])
        self.check_default_arguments(RGBR, GBMRegressor, ['column_types', 'random_state'])

    def x_test_as_factor(self):
        """Tests if factors are recorded properly.

        We will fit RGBC with one cat variable with 3 levels a and b are negative
        c is positive. We then predict on a container where only two levels a and c
        are present -- if c will be predicted as negative we don't store factors
        properly.
        """
        X_train = pd.DataFrame({'cat': ['a', 'a', 'b', 'c', 'c', 'a', 'a', 'b']})
        Y_train = np.array([0, 0, 0, 1, 1, 0, 0, 0])
        Z_train = Partition(size=X_train.shape[0], folds=5, reps=5, total_size=X_train.shape[0])
        Z_train.set(max_reps=1, max_folds=0)
        # create a container -- we have 3 levels in 'cat'
        C_train = Container()
        C_train.add(X_train.values, colnames=['cat'], coltypes=[3])
        task = RGBC('n=10;md=2;s=1.0')
        task.fit(C_train, Y_train, Z_train)

        # now omit level b -- this should map level c to 2 which is the
        # same index that b had before
        X_test = pd.DataFrame({'cat': ['a', 'a', 'c', 'c', 'a', 'a', 'c']})
        Y_test = np.array([0, 0, 1, 1, 0, 0, 1])
        Z_test = Partition(size=X_test.shape[0], folds=5, reps=5, total_size=X_test.shape[0])
        Z_test.set(max_reps=1, max_folds=0)
        C_test = Container()
        C_test.add(X_test.values, colnames=['cat'], coltypes=[2])

        # test if predictions match
        pred = task.predict(C_test, Y_test, Z_test)
        pred = pred(**next(iter(Z_test))).ravel()
        np.testing.assert_array_equal(Y_test, (pred >= 0.5).astype(np.int))

    @pytest.mark.dscomp
    def test_RGBR_reproducible(self):
        """Smoke test for classification. """
        X, Y, Z = self.create_bin_data()
        reference = [0.54181945, 0.33011115, 0.54181945]
        task_args = ['n=200', 'md=3', 'ls=4', 'lr=0.002', 't_f=0.2',
                     't_m={metrics.AUC}'.format(metrics=metrics), 't_a=1']
        task_desc = 'RGBR ' + ';'.join(task_args)
        t = self.check_task(task_desc, X, Y, Z, transform=False,
                            standardize=False, reference=reference)
        # 200 / 100 == 2
        self.assertEqual(t.gridsearch_parameters['step'], 2)

    @pytest.mark.skip('Because it is not reproducible yet')
    @pytest.mark.dscomp
    def test_RGBC_reproducible(self):
        """Smoke test for classification. """
        X, Y, Z = self.create_bin_data()
        reference = [0.53904452, 0.33198552, 0.53904452]
        task_args = ['n=200', 'md=3', 'ls=4', 'lr=0.002', 't_f=0.2',
                     't_m={metrics.AUC}'.format(metrics=metrics), 't_a=1']

        task_desc = 'RGBC ' + ';'.join(task_args)
        t = self.check_task(task_desc, X, Y, Z, transform=False,
                            standardize=False, reference=reference)
        self.assertEqual(t.gridsearch_parameters['step'], 2)

    def test_reg_cat_r(self):
        """Test for RGBR regression using categorical variables """
        X, Y, Z = self.create_reg_data()
        C = Container()
        cat_cols = [c for c in X.columns if X[c].dtype not in (np.int64, np.float64)]

        # ordinal encode categoricals - R gbm will treat them as cats based on coltype
        enc = OrdinalEncoder(columns=cat_cols, min_support=1)
        X = enc.fit_transform(X)

        def cardinality(col):
            # get cardinality of each categorical column
            if col in cat_cols:
                return X[col].nunique()
            else:
                return 0

        coltypes = [cardinality(c) for c in X.columns]
        C.add(X.values.astype(np.float), colnames=X.columns.tolist(), coltypes=coltypes)
        task = RGBR('md=1;s=1.0')
        task.fit(C, Y, Z)
        pred = task.predict(C, Y, Z)
        pred = pred(**next(iter(Z))).ravel()
        assert pred.shape[0] == Y.shape[0]
        rmse = np.sqrt(np.mean((pred - Y) ** 2.0))
        # should be around 260
        assert rmse < 300.0

    def test_reg_cat_tesla(self):
        """Test for GBR regression with categorical variables. """
        X, Y, Z = self.create_reg_data()
        C = Container()
        cat_cols = [c for c in X.columns if X[c].dtype not in (np.int64, np.float64)]

        def cardinality(col):
            # get cardinality of each categorical column
            if col in cat_cols:
                return X[col].nunique()
            else:
                return 0

        coltypes = [cardinality(c) for c in X.columns]

        for c in cat_cols:
            levels, enc = np.unique(X[c], return_inverse=True)
            X[c] = enc

        C.add(X.values, colnames=X.columns.tolist(), coltypes=coltypes)
        task = GBR('md=1;s=1.0')
        task.fit(C, Y, Z)

        self.assertTrue((0, -1) in task.partial_dependence_plots)
        pred = task.predict(C, Y, Z)
        pred = pred(**next(iter(Z))).ravel()
        assert pred.shape[0] == Y.shape[0]
        rmse = np.sqrt(np.mean((pred - Y) ** 2.0))
        # shoudl be around 360 - R much better with CAT and small trees
        assert rmse < 400.0

    @patch('ModelingMachine.engine.tasks.gbm.logger')
    def test_high_cardinality(self, mock_logger):
        """Tests if GBM falls back to numerical for high cardinaity features. """
        rs = np.random.RandomState(13)
        n = 1025
        X = pd.DataFrame({'cat': np.arange(n)})
        Y = np.array(rs.randint(2, size=n))
        Z = Partition(size=X.shape[0], folds=5, reps=5, total_size=X.shape[0])
        Z.set(max_reps=1, max_folds=0)

        C = Container()
        C.add(X.values, colnames=['cat'], coltypes=[n])
        task = RGBC('n=1;md=2;s=1.0')

        task.fit(C, Y, Z)
        # must warn because 1025 levels > 1024
        self.assertTrue(mock_logger.info.called)

    def test_unbalanced_tree_consistency(self):
        """Check if RGBM and Tesla give similar results when using max_leaf_nodes. """
        X, Y, Z = self.create_bin_data()
        reference = [0.72322707, 0.28756388, 0.72102573, 0.24320669, 0.31403745,
                     0.64779349, 0.28756388, 0.39719372, 0.44241538, 0.47666953]
        task_args = ['n=10', 's=1.0', 'ml=6', 'md=None', 'ls=2', 'lr=0.1']
        task_desc = 'GBC ' + ';'.join(task_args)
        t_py = self.check_task(task_desc, X, Y, Z, transform=False,
                               standardize=False, reference=reference, decimal=5)

        task_desc = 'RGBC ' + ';'.join(task_args)
        t_r = self.check_task(task_desc, X, Y, Z, transform=False,
                              standardize=False, reference=reference, decimal=5)



class TestGBRTPartialDependence(BaseTaskTest):
    """Test case for partial dependence plots for GBRT. """

    def test_pdp_reg_cat_num(self):
        """Test PDP for GBR regression with categorical and numeric variables. """
        X, Y, Z = self.create_reg_data()
        X = X.dataframe
        C = Container()
        cat_cols = [c for c in X.columns if X[c].dtype not in (np.int64, np.float64)]

        def cardinality(col):
            # get cardinality of each categorical column
            if col in cat_cols:
                return X[col].nunique()
            else:
                return 0

        coltypes = [cardinality(c) for c in X.columns]

        for c in cat_cols:
            levels, enc = np.unique(X[c], return_inverse=True)
            X[c] = enc

        C.add(X.values, colnames=X.columns.tolist(), coltypes=coltypes)
        task = GBR('n=10;md=1;s=1.0;rs=1')
        task.fit(C, Y, Z)
        colnames = C.colnames()

        self.assertEqual(set(task.partial_dependence_plots.keys()), {(0, -1)})
        self.assertTrue((0, -1) in task.partial_dependence_plots)
        pdps = task.partial_dependence_plots[(0, -1)]

        for pdp in pdps:
            self.assertEqual(set(pdp.keys()), set(PartialDependencePlot._fields))
            self.assertTrue(all(col in colnames for col in pdp['colnames']))
            self.assertTrue(all(pdp['coltypes'][i] == coltypes[ci]
                                for i, ci in enumerate(pdp['colindices'])))

            print(pdp)
            if pdp['coltypes'][0] > 0:
                self.assertEqual(len(pdp['freqs'][0]), coltypes[pdp['colindices'][0]])
            else:
                self.assertEqual(len(pdp['freqs'][0]), len(pdp['axes'][0]))
                # might be less than 100 if less unique vals (we also filter extreme
                # cases
                self.assertLess(len(pdp['freqs'][0]), 101)


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
