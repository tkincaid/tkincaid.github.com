#########################################################
#
#       Unit Test for base classes in base_model_class.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import numpy as np
import scipy as sp
import pandas as pd
import logging
import copy
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

from mock import patch

from joblib.my_exceptions import JoblibException
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit

from ModelingMachine.engine.tasks.base_modeler import NamedArguments, BaseModeler
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.response import Response
from ModelingMachine.engine.gridsearch import BaseGridSearch
from ModelingMachine.engine.gridsearch import ExhaustiveGridSearch
from ModelingMachine.engine.metrics import logloss1D
import ModelingMachine
from base_task_test import BaseTaskTest
from common.exceptions import TaskError

class TestBaseModelClass(BaseTaskTest):

    def test_string_to_number(self):
        task = NamedArguments()
        self.assertEqual(task.string_to_number('1'),1)
        self.assertEqual(task.string_to_number('1.5'),1.5)
        self.assertEqual(task.string_to_number('1e10'),1e10)

    def test_parse_value(self):
        task = NamedArguments()
        self.assertEqual(task.parse_value('asdf'),'asdf')
        self.assertEqual(task.parse_value('1',int),1)
        self.assertEqual(task.parse_value('2',float),2.0)
        self.assertIsInstance(task.parse_value('1',int),int)
        self.assertIsInstance(task.parse_value('2',float),float)
        self.assertEqual(task.parse_value('["a","b"]'),['a','b'])
        self.assertEqual(task.parse_value("['a','b']"),['a','b'])
        self.assertEqual(task.parse_value("[a,b]"),['a','b'])
        self.assertEqual(task.parse_value('[1,2]'),['1','2'])
        self.assertEqual(task.parse_value('[1,2]',int),[1,2])
        self.assertEqual(task.parse_value('[1,2]',float),[1.0,2.0])
        self.assertIsInstance(task.parse_value('[1,2]',int)[0],int)
        self.assertIsInstance(task.parse_value('[1,2]',float)[0],float)

    def create_testdata(self):
        """ create some test data to help in the tests """
        x= np.array([[1,2,3],[4,5,6],[7,8,9]])
        # the syntax is (data,(rows, cols)), shape=(nrows,ncols)
        s= sp.sparse.coo_matrix( ([3,2],([0,2],[1,6])),shape=(3,10))
        return x, s

    def test_check_sparse(self):
        """ test the _check_sparse helper function """
        x,s = self.create_testdata()
        task = BaseModeler()
        #check that a dense array x is passed thru unchanged
        check = task._check_sparse(x)
        self.assertEqual(np.all(check==x),True)
        #check that a sparse matrix s is converted to a numpy array
        check = task._check_sparse(s)
        self.assertIsInstance(check,np.ndarray)
        self.assertEqual(np.all(check==s.todense()),True)

    def test_prep_fit(self):
        task = BaseModeler()
        self.assertEqual(task._prep_fit(1,2),1)
        self.assertEqual(task._prep_apply(1,2),1)
        self.assertEqual(task._translate_coef(1,2),1)

    def test_create_estimator(self):
        #fake estimator class
        class Testest():
            def __init__(self,a,b):
                self.a=a
                self.b=b
            def get_parms(self):
                return {'a':self.a,'b':self.b}
        #task class
        task = BaseModeler()
        task.estimator = Testest
        parameters = {'a':1,'b':2}
        est = task._create_estimator(parameters)
        self.assertIsInstance(est,Testest)
        self.assertEqual(est.get_parms(),parameters)

    def create_fake_task(self):
        class FakeTask(BaseModeler):
            arguments = {
                'nt':{'name':'n_estimators','type':'int','values':[1,10000],'default':'500'},
                'c' :{'name':'criterion','type':'select','values':['gini','entropy'],'default':'0'},
                'ls':{'name':'min_samples_leaf','type':'intgrid','values':[1,1000],'default':'5'}
            }
            estimator = ExtraTreesClassifier
        return FakeTask

    def test_get_other_args(self):
        FakeTask = self.create_fake_task()
        task = FakeTask('wa_bw=1')
        weight = pd.Series(np.arange(10))
        X = Container()
        X.initialize({'weight':weight})
        Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        X.balance('weight', Y)
        expected = {'balanced_weight': np.array([0, 3.5, 7, 10.5, 14, 5, 6, 7, 8, 9]),
                    'weight': np.array([ 0, 3.5, 7, 10.5, 14, 5, 6, 7, 8, 9])}
        np.testing.assert_array_equal(expected['balanced_weight'], task.get_other_args(X)['balanced_weight'])
        np.testing.assert_array_equal(expected['weight'], task.get_other_args(X)['weight'])

    @patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_train')
    def test_fit_with_balanced_weights(self, tmock):
        X,Y,Z = self.create_bin_data(3)
        weight = pd.Series(np.arange(Y.shape[0]))
        X.initialize({'weight':weight})
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()
        task = FakeTask('wa_bw=1')
        try:
            task.fit(X,Y,Z)
        except:
            pass
        train_weights = tmock.call_args[1]['other_args']['weight']
        train_Y = Y[Z.T(r=0, k=-1)]
        # weight sums will be a little off since we are only checking the 1st CV training data
        self.assertLess(np.abs(np.sum(train_weights[train_Y==0]) - np.sum(train_weights[train_Y==1])), 250)

    @patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_train')
    @patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_check_stack')
    @patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_get_importances')
    def test_fit_with_capped_response(self, gimock, csmock, tmock):
        X,Y,Z = self.create_reg_data(100)
        Z.set(max_reps=1, max_folds=1)
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()
        task = FakeTask('pr_capy=0.8')
        task.fit(X,Y,Z)
        # Get capped rows using mock
        Y_cap = tmock.call_args[0][1].ravel()
        train_rows = Z.T(r=0, k=0)
        train_Y = Y[train_rows].ravel()
        unchanged = Y_cap  < np.percentile(train_Y, 80)
        np.testing.assert_array_almost_equal(Y_cap[unchanged], train_Y[unchanged])
        changed = np.invert(unchanged)
        np.testing.assert_array_almost_equal(Y_cap[changed], np.percentile(train_Y, 80) * np.ones(len(train_Y[changed])))

    def test_capped_response_max_200(self):
        rng = np.random.RandomState(200)
        Y = rng.exponential(size=2000)
        Y_copy = copy.deepcopy(Y)
        FakeTask = BaseModeler()

        Y_capped = FakeTask._cap_response(Y, max_prcnt=.5, max_number=200)
        not_eq = np.not_equal(Y_copy, Y_capped)
        print not_eq
        # Number 200 is the same since they are capped to the 200th
        # highest value
        self.assertEqual(len(Y_capped.shape), 1)
        self.assertEqual(np.count_nonzero(not_eq), 199)
        np.testing.assert_array_equal(np.sort(np.nonzero(not_eq)).flatten(), np.sort(np.argsort(Y_copy)[1801:]))

    def test_separate_parameters(self):
        FakeTask = self.create_fake_task()
        #no args
        task = FakeTask()
        out = task._separate_parameters('a')
        self.assertEqual(out,({},{'n_estimators':500,'criterion':'gini','min_samples_leaf':5}))
        #fixed args
        task = FakeTask('nt=10;c=1')
        out = task._separate_parameters('a')
        self.assertEqual(out,({},{'n_estimators':10,'criterion':'entropy','min_samples_leaf':5}))
        #fixed and tuning args
        task = FakeTask('nt=10;ls=[1,2,3,4,5]')
        out = task._separate_parameters('a')
        self.assertEqual(out,({'min_samples_leaf':[1,2,3,4,5]},{'n_estimators':10,'criterion':'gini'}))
        #refit old grid with new grid
        task = FakeTask('nt=10;ls=[1,2,3,4,5]')
        task.old__parameters = {'n_estimators':10,'min_samples_leaf':[1,3]}
        out = task._separate_parameters('a')
        self.assertEqual(out,({'min_samples_leaf':[2,4,5]},{'n_estimators':10,'criterion':'gini'}))
        #refit single value with new grid
        task = FakeTask('nt=10;ls=[1,2,3,4,5]')
        task.old__parameters = {'n_estimators':10,'min_samples_leaf':3}
        out = task._separate_parameters('a')
        self.assertEqual(out,({'min_samples_leaf':[1,2,4,5]},{'n_estimators':10,'criterion':'gini'}))

    def test_remove_extra_GBMtrees(self):
        FakeTask = self.create_fake_task()
        task = FakeTask()
        task.grid_scores['x'] = [
            ({ 'p1': 1, 'p2': 2, 'n_estimators': 10 }, 0.3 ),
            ({ 'p1': 1, 'p2': 2, 'n_estimators': 20 }, 0.4 ),
            ({ 'p1': 1, 'p2': 3, 'n_estimators': 10 }, 0.2 ),
            ({ 'p1': 1, 'p2': 3, 'n_estimators': 20 }, 0.1 ),
        ]
        task.best_parameters = {'x': { 'p1': 1, 'p2': 3, 'n_estimators': 20 } }
        task._remove_extra_nestimators('x')
        expected = [
            ({ 'p1': 1, 'p2': 3, 'n_estimators': 10 }, 0.2 ),
            ({ 'p1': 1, 'p2': 3, 'n_estimators': 20 }, 0.1 ),
            ({ 'p1': 1, 'p2': 2, 'n_estimators': 10 }, 0.3 ),
        ]
        self.assertEqual(expected,task.grid_scores['x'])

    def test_grid_search(self):
        X,Y,Z = self.create_bin_data()
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()

        task = FakeTask('cs_l=-500;cs_r=500')
        expected = set(['n_estimators','criterion','min_samples_leaf'])
        self.assertEqual(set(task.parameters.keys()), expected)
        expected = set(['CV_folds','algorithm','max_iterations','stratified',
                'step','random_state', 'stacked_coeffs'])
        self.assertEqual(set(task.gridsearch_parameters.keys()),expected)
        expected = set(['left_censoring','right_censoring','stack','stack_folds', 'stage'])
        self.assertEqual(set(task.post_processing_parameters.keys()),expected)
        estimator = task.estimator()
        parameters = {'min_samples_leaf':[3,4,5],'max_features':np.arange(1,10,dtype=float)}
        cv = Z.get_cv(size=len(Y),yt=Y,folds=5,random_state=0,stratified=True)
        gs = task._grid_search_object(estimator,parameters,X(),Y,cv)
        self.assertIsInstance(gs, BaseGridSearch)
        self.assertIsInstance(gs.cv, StratifiedKFold)

        task = FakeTask('t_a=0;t_s=0;t_n=10')
        cv = Z.get_cv(size=len(Y),yt=Y,folds=10,random_state=0,stratified=False)
        gs = task._grid_search_object(estimator,parameters,X(),Y,cv)
        self.assertIsInstance(gs, ExhaustiveGridSearch)
        self.assertIsInstance(gs.cv, KFold)
        self.assertEqual(gs.cv.n_folds,10)

        task = FakeTask('t_a=1;t_f=0.1;t_mi=10;t_sp=1;t_m=LogLoss')
        cv = Z.get_cv(size=len(Y),yt=Y,validation_pct=0.1,random_state=0,stratified=False)
        gs = task._grid_search_object(estimator,parameters,X(),Y,cv)
        self.assertIsInstance(gs, BaseGridSearch)
        self.assertIsInstance(gs.cv, ShuffleSplit)
        self.assertEqual(gs.cv.test_size, 0.1)
        self.assertEqual(gs.max_iter,10)
        self.assertEqual(gs.step,1)
        self.assertEqual(gs.loss_func(Y,Y),logloss1D(Y,Y))

    @patch('test_base_model_class.ExtraTreesClassifier.fit')
    def test_grid_search_joblib_exception(self, mock_fit):

        mock_fit.side_effect = TaskError('foobar', client_message='barfoo', error_code=201)

        X,Y,Z = self.create_bin_data()
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()

        task = FakeTask('ls=[5,10];t_a=1;t_f=0.1;t_mi=10;t_sp=1;t_m=LogLoss')
        try:
            task.fit(X, Y, Z)
            assert False
        except Exception as e:
            self.assertIsInstance(e, JoblibException)
            self.assertIsInstance(e, TaskError)

    @pytest.mark.dscomp
    def test_train(self):
        X,Y,Z = self.create_bin_data()
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()
        #no grid search
        task = FakeTask('nt=10')
        cv = Z.get_cv(size=len(Y),yt=Y,folds=5,random_state=0)
        mdl = task._train(X(),Y,None,cv,'a')
        self.assertIsInstance(mdl, ExtraTreesClassifier)
        parms = mdl.get_params()
        self.assertEqual(parms['n_estimators'],10)
        self.assertEqual(len(mdl.estimators_),10)

        #grid search
        task = FakeTask('nt=10;ls=[1,2,3,4,5]')
        #task.grid_scores = {'a':[]}
        #task.best_parameters = {'a':{}}
        mdl = task._train(X(),Y,None,cv,'a')
        self.assertIsInstance(mdl, ExtraTreesClassifier)
        self.assertIsInstance(task.best_parameters,dict)
        self.assertIn('a',task.best_parameters)
        self.assertIsInstance(task.grid_scores,dict)
        self.assertIn('a',task.grid_scores)
        self.assertGreater(len(task.grid_scores['a']),0)


    def fit_task(self,reps=1, task_args='nt=10'):
        X,Y,Z = self.create_bin_data(reps)
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()
        #no grid search
        task = FakeTask(task_args)
        task.fit(X,Y,Z)
        return task,X,Y,Z,FakeTask

    def test_fit(self):
        task,X,Y,Z,FakeTask = self.fit_task(reps=5)
        self.assertEqual(task.pred_stack.values()[0].shape[0], Y.shape[0])

    def test_predict(self):
        task,X,Y,Z,FakeTask = self.fit_task()
        out = task.predict(X,Y,Z)
        self.assertIsInstance(out,Container)
        expected = (X().shape[0],1)
        for p in out:
            self.assertEqual(out(**p).shape, expected)

    def test_censoring(self):
        task,X,Y,Z,FakeTask = self.fit_task(task_args='cs_l=20')
        out = task.predict(X,Y,Z)
        self.assertIsInstance(out,Container)
        for p in out:
            self.assertEqual(np.all(out(**p) >= 20), True)
        task,X,Y,Z,FakeTask = self.fit_task(task_args='cs_r=0.5')
        out = task.predict(X,Y,Z)
        self.assertIsInstance(out,Container)
        for p in out:
            self.assertEqual(np.all(out(**p) <= 0.5), True)

    def test_transform(self):
        task,X,Y,Z,FakeTask = self.fit_task(reps=5)
        out = task.transform(X,Y,Z)
        self.assertIsInstance(out,Container)

    def test_transform2(self):
        task,X,Y,Z,FakeTask = self.fit_task()
        out = task.transform2(X,Y,Z)
        self.assertIsInstance(out,Container)

    def test_stack(self):
        X,Y,Z = self.create_bin_data()
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()
        task = FakeTask('nt=10;ls=[3,5]')
        task.fit(X,Y,Z)
        out = task.stack(X,Y,Z,inputs={'N'})
        self.assertIsInstance(out,Container)
        expected = (X().shape[0],1)
        for p in out:
            self.assertEqual(out(**p).shape, expected)

    def test_report(self):
        task,X,Y,Z,FakeTask = self.fit_task()
        out = task.report()
        self.assertIsInstance(out,dict)
        self.assertEqual(set(out.keys()),set([(0,-1)]))
        self.assertGreaterEqual(set(out[0,-1].keys()),set(['time_cpu','time_real','extras']))
        self.assertEqual(set(['importance']),set(out[0,-1]['extras'].keys()))
        self.assertIsInstance(out[0,-1]['extras']['importance'],list)
        for i in out[0,-1]['extras']['importance']:
            self.assertIsInstance(i,tuple)
            self.assertEqual(len(i),2)
            self.assertIsInstance(i[0],str)
            self.assertIsInstance(i[1],float)

        print out

    def test_transform_multi_partition_doesnt_exit_early(self):
        task, X, Y, Z, FakeTask = self.fit_task(reps=5)
        with patch.object(task, '_choose_cols') as fake_choose_cols:
            fake_choose_cols.side_effect = [[1], [], [3], [4], [5]]
            out = task.transform(X, Y, Z)

            #Only the second one should have crapped out
            n_nones = 0
            for p in Z:
                if out(**p).shape[1] == 0:
                    n_nones += 1
            self.assertEqual(n_nones, 1)

    def test_mask_smoketest(self):
        """Test that masks work
        """
        X,Y,Z = self.create_bin_data(3)
        vertex = Vertex(['NI'],'id')
        X = vertex.fit_transform(X,Y,Z)
        FakeTask = self.create_fake_task()
        mask = np.random.randint(0, 2, X.shape[0])
        X.set_mask(mask)
        FakeTask = self.create_fake_task()
        task = FakeTask('nt=10;ls=[3,5]')
        task.fit(X, Y, Z)


def remember_x(xt, yt, wt, cv, key, fit_args, **kwargs):
    '''We needed a way to keep track of what _train was called with,
    so we'll just patch it to use this - it will return the input data
    in xt.  That way, the ``model`` saved from this fit
    is what we can use to check the size of the arrays that were used
    to fit

    This function is meant for TestMasking
    '''
    return xt, yt, wt


class TestMasking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_fit_on_dynamic_data_trains_on_subset_if_mask_exists(self):
        bm = BaseModeler()
        data = np.arange(10)
        X = Container()
        Y = np.arange(10)
        Y = Response.from_array(Y)
        Z = Partition(10, folds=2, reps=0, total_size=10)
        for p in Z:
            X.add(data, r=p['r'], k=p['k'])
            X.set_mask([True]*5 + [False]*5, r=p['r'], k=p['k'])

        with patch.object(bm, '_train', new=remember_x) as fake_train:
            with patch.object(bm, '_check_stack' ) as fake_cs:
                bm.fit(X, Y, Z)
                for key in bm.model.keys():
                    train_data = bm.model[key]
                    self.assertTrue(np.all(train_data[0] < 5))

    def test_fit_on_dynamic_data_trains_on_everything_if_no_mask(self):
        bm = BaseModeler()
        data = np.arange(10)
        X = Container()
        Y = np.arange(10)
        Y = Response.from_array(Y)
        Z = Partition(10, folds=2, reps=0, total_size=10)
        for p in Z:
            X.add(data, r=p['r'], k=p['k'])

        with patch.object(bm, '_train', new=remember_x) as fake_train:
            with patch.object(bm, '_check_stack' ) as fake_cs:
                bm.fit(X, Y, Z)
                for key in bm.model.keys():
                    self.assertEqual(len(bm.model[key][0]), 5)


if __name__ == '__main__':
    unittest.main()
