#########################################################
#
#       Unit Test for newgridsearch.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import numpy as np
import itertools
import copy
import json

from collections import namedtuple
from sklearn.svm import SVC, SVR
from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit

#-locals
from base_task_test import BaseTaskTest
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.gridsearch import BaseGridSearch, fit_one_point, LeaderboardRef
from ModelingMachine.engine.tasks.svc import SVMC, SVMR
from ModelingMachine.engine.metrics import logloss
from ModelingMachine.engine.tasks.gbm import GradientBoostingClassifier, GBC
from ModelingMachine.engine.tasks.logistic import LogRegL1

from common.engine.progress import Progress, ProgressState, ProgressSink
from config.test_config import db_config as config
from common.wrappers import database

class TestGridSearch(BaseTaskTest):
    @classmethod
    def setUpClass(cls):
        super(TestGridSearch,cls).setUpClass()
        cls.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

    @classmethod
    def tearDownClass(cls):
        super(TestGridSearch,cls).tearDownClass()
        cls.persistent.destroy(table='leaderboard')

    def startup1(self):
        X,Y,Z = self.create_bin_data()
        vertex = Vertex(['NI','ST'],'id')
        X = vertex.fit_transform(X,Y,Z)
        x = X()
        estimator = SVC(kernel='poly',probability=True)
        parameters = {  'C'    :np.logspace(0,4,base=10,num=5),
                        'gamma':np.logspace(-4,1,base=10,num=6) }
        #cv = KFold(n=len(Y), n_folds=5)
        cv = StratifiedKFold(y=Y, n_folds=2)
        return x,Y,estimator,parameters,cv

    def startup2(self):
        X,Y,Z = self.create_bin_data()
        vertex = Vertex(['NI','ST'],'id')
        X = vertex.fit_transform(X,Y,Z)
        x = X()
        estimator = RegL2(kernel='poly',probability=True)
        parameters = {  'C'    :np.logspace(0,4,base=10,num=5),
                        'gamma':np.logspace(-4,1,base=10,num=6) }
        #cv = KFold(n=len(Y), n_folds=5)
        cv = StratifiedKFold(y=Y, n_folds=2)
        return x,Y,estimator,parameters,cv

    def test_helpers(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        self.assertEqual(gs._par_to_pos({'C':10  ,'gamma':1    }), (1,4))
        self.assertEqual(gs._par_to_pos({'C':1   ,'gamma':0.1  }), (0,3))
        self.assertEqual(gs._par_to_pos({'C':100 ,'gamma':0.01 }), (2,2))

        self.assertEqual(gs._pos_to_par((1,4)), {'C':10  ,'gamma':1     })
        self.assertEqual(gs._pos_to_par((0,3)), {'C':1   ,'gamma':0.1   })
        self.assertEqual(gs._pos_to_par((2,2)), {'C':100 ,'gamma':0.01  })
        self.assertEqual(gs._pos_to_par((0,0)), {'C':1   ,'gamma':0.0001})

    def assert_equal_lists(self,a,b):
        #self.assertEqual(len(a),len(b))
        self.assertListEqual(sorted(a),sorted(b))
        #for i in range(len(a)):
        #    self.assertEqual(a[i],b[i])

    def test_select_points(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        mat = np.empty((6,6))
        mat.fill(np.Inf)
        out = gs._select_points((1,1),1,True,mat)
        expected = [i for i in itertools.product([0,1,2],[0,1,2])]
        self.assert_equal_lists(out,expected)

        out = gs._select_points((1,1),1,False,mat)
        expected = [i for i in itertools.product([0,2],[0,2])]
        expected.append((1,1))
        self.assert_equal_lists(out,expected)

        out = gs._select_points((1,1),2,True,mat)
        expected = [i for i in itertools.product([1,3],[1,3])]
        self.assert_equal_lists(out,expected)

        out = gs._select_points((1,1),2,False,mat)
        expected = [i for i in itertools.product([3],[3])]
        expected.append((1,1))
        self.assert_equal_lists(out,expected)


    def test_neighbors(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        mat = np.empty((6,6))
        mat.fill(np.Inf)
        out = gs._neighbors((1,1))
        expected = [i for i in itertools.product([0,2],[0,2])]
        expected.append((1,1))
        self.assert_equal_lists(out,expected)

        mat = np.empty((6,6))
        mat.fill(np.Inf)
        mat[0,0] = 1
        mat[1,1] = 1
        out = gs._neighbors((0,0),score_matrix=mat)
        expected = [i for i in itertools.product([0,1],[0,1]) if i not in [(0,0),(1,1)]]
        self.assert_equal_lists(out,expected)

    def test_initial_net(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        net = gs._initial_net()
        self.assertEqual(gs._initial_net(), list(set([(1,1),(1,4),(3,1),(3,4)])))
        self.assertEqual(gs._initial_net((6,6)), list(set([(1,1),(1,4),(4,1),(4,4)])))
        self.assertEqual(gs._initial_net((7,7)), list(set([(1,1),(1,5),(5,1),(5,5)])))
        self.assertEqual(gs._initial_net((8,8)), list(set([(2,2),(2,5),(5,2),(5,5)])))
        self.assertEqual(gs._initial_net((9,9)), list(set([(2,2),(2,6),(6,2),(6,6),(4,4)])))

        estimator = GBC
        parameters = {'max_depth':[3]}
        gs = BaseGridSearch(estimator,parameters,cv,logloss)
        self.assertEqual(gs._initial_net(), [(0,)])

        parameters = {'max_depth':[3],'learning_rate':[0.1]}
        gs = BaseGridSearch(estimator,parameters,cv,logloss)
        self.assertEqual(gs._initial_net(), [(0,0)])

    def test_find_best_pos(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        gs.score_matrix[1,2] = -1
        out = gs._find_best_pos()
        expected = (1,2)
        self.assertEqual(out,expected)

        mat = np.empty((6,6))
        mat.fill(np.Inf)
        mat[1,1] = -1
        mat[2,2] = -2
        mat[3,3] = -4
        mat[4,4] = -3
        out = gs._find_best_pos(mat)
        expected = (3,3)
        self.assertEqual(out,expected)

    def test_next_steps(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        out = gs._next_steps()
        net = gs._initial_net()
        expected = [gs._pos_to_par(i) for i in net]
        self.assert_equal_lists(out,expected)

        gs.score_matrix[4,1]=-3
        gs.score_matrix[1,4]=-1
        gs.score_matrix[1,1]=-2
        out = gs._next_steps()
        net = gs._neighbors((4,1))
        expected = [gs._pos_to_par(i) for i in net]
        self.assert_equal_lists(out,expected)

    def check_fit_output(self,out,Estimator,y):
        for j in out:
            i = j['mdl'],j['pred']
            self.assertEqual(len(i),2)
            self.assertIsInstance(i[0], Estimator)
            self.assertIsInstance(i[1], np.ndarray)
            self.assertEqual(len(i[1]),len(y))
            self.assertIsInstance(i[1][0], float)

    def test_fit_one(self):
        x,y,estimator,parameters,cv = self.startup1()

        parameters = {'C':1,'gamma':0.1}
        out = []
        for train,test in cv:
            mdl = fit_one_point(x,y,estimator,parameters,train,test,logloss,1,None,None,{})
            out.append(mdl)
        self.check_fit_output(out,SVC,y)

        estimator = SVR(kernel='rbf')
        out = []
        for train,test in cv:
            mdl = fit_one_point(x,y,estimator,parameters,train,test,logloss,1,None,None,{})
            out.append(mdl)
        self.check_fit_output(out,SVR,y)

    def test_distribute_jobs(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        parameter_list = [{'C':10,'gamma':0.1},{'C':10,'gamma':1}]
        out = gs._distribute_jobs(x,y,parameter_list,cv,logloss,{})

        self.check_fit_output(out,SVC,y)

        preds = gs._stack_predictions(out,parameter_list,cv)
        self.assertIsInstance(preds,dict)
        expected = map(gs._par_to_pos, parameter_list)
        self.assertEqual(set(preds.keys()),set(expected))
        for k in preds:
            self.assertIsInstance(preds[k],np.ndarray)
            self.assertEqual(len(preds[k]),len(y))
            self.assertIsInstance(preds[k][0],float)
            self.assertGreater(preds[k].var(),0)

        scores = gs._compute_scores(preds, logloss, y, None, cv)

    def test_stack_coefficents(self):
        """Unittest to check whether stack coefficients behaves correctly
        """
        x,y,a,b,cv = self.startup1()
        estimator = GradientBoostingClassifier(n_estimators=100,learning_rate=0.03)
        estimator.coef_ = [1, 2]
        parameters = {'max_depth':[3,5,7,9,11],'min_samples_leaf':range(1,6)}
        loss_func = logloss

        cv = ShuffleSplit(n=len(y),n_iter=1,test_size=0.10,random_state=1234)
        #cv = StratifiedKFold(y=y, n_folds=5)
        #cv = KFold(n=len(y), n_folds=5, random_state=1234)

        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['max_depth','min_samples_leaf'],step=5)
        parameter_list = [{'max_depth':3, 'min_samples_leaf':1}, {'max_depth':3, 'min_samples_leaf':3}, {'max_depth':5, 'min_samples_leaf':1}, {'max_depth':5, 'min_samples_leaf':3}]
        out = gs._distribute_jobs(x,y,parameter_list,cv,logloss,{})
        for o in out:
            o['mdl'].coef_ = [1, 2]
        coeffs_inds, coeffs_rows = gs._stack_coefficients(out, parameter_list, cv)
        self.assertEqual(len(parameter_list) * len(cv), len(coeffs_inds))
        self.assertEqual(len(cv) * len(parameter_list), len(coeffs_rows.keys()))
        for key in coeffs_inds:
            self.assertEqual(coeffs_inds[key].shape[0], x.shape[0])
            self.assertEqual(len(cv), len(np.unique(coeffs_inds[key][~np.isnan(coeffs_inds[key])])))
            self.assertEqual(coeffs_inds[key].shape[1], 1)
        for key in coeffs_rows:
            self.assertEqual(coeffs_rows[key].shape[1], 2)

    def test_get_stack_coefficients_return_coeffs_returns_coeffs_true(self):
        """ Unittest that get_stack_coefficients returns coeffs if return coeffs is true
        """
        x,y,a,b,cv = self.startup1()
        task = LogRegL1()
        estimator = task.estimator()
        parameters = {'C':[0.001,0.01,0.1,1]}
        loss_func = logloss

        # cv= ShuffleSplit(n=len(y),n_iter=1,test_size=0.10,random_state=1234)
        #cv = StratifiedKFold(y=y, n_folds=5)
        cv = KFold(n=len(y), n_folds=5, random_state=1234)

        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C'],step=5, return_coeffs=True)
        parameter_list = [{'C':0.001}, {'C':0.01}, {'C':0.1}, {'C':1}]
        out = gs._distribute_jobs(x,y,parameter_list,cv,logloss,{})
        bp = gs._find_best_pos()
        coeffs_inds, coeffs_rows = gs._get_stacked_coefficients(out, parameter_list, cv, bp)
        self.assertEqual(coeffs_inds.shape[0], x.shape[0])
        self.assertEqual(coeffs_inds.shape[1], 1)
        self.assertEqual(sorted(coeffs_rows.keys()), sorted([str(i) for i in range(len(cv) * len(parameter_list))]))
        for key in coeffs_rows:
            self.assertEqual(coeffs_rows[key].shape[1], x.shape[1])
            self.assertTrue(np.NaN not in coeffs_rows[key])
        self.assertTrue(np.NaN not in coeffs_inds)

    def test_get_stack_coefficients_return_coeffs_returns_coeffs_false(self):
        """ Unittest that get_stack_coefficients returns coeffs if return coeffs is false
        """
        x,y,a,b,cv = self.startup1()
        task = LogRegL1()
        estimator = task.estimator()
        parameters = {'C':[0.001,0.01,0.1,1]}
        loss_func = logloss

#        cv= ShuffleSplit(n=len(y),n_iter=1,test_size=0.10,random_state=1234)
        #cv = StratifiedKFold(y=y, n_folds=5)
        cv = KFold(n=len(y), n_folds=5, random_state=1234)

        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C'],step=5, return_coeffs=False)
        parameter_list = [{'C':0.001}, {'C':0.01}, {'C':0.1}, {'C':1}]
        out = gs._distribute_jobs(x,y,parameter_list,cv,logloss,{})
        bp = gs._find_best_pos()
        coefs = gs._get_stacked_coefficients(out, parameter_list, cv, bp)
        self.assertTrue(coefs is None)

    def check_find_minimum(self,parameters,tuneorder=None):
        x,y,estimator,_,cv = self.startup1()

        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=tuneorder)

        gs._find_minimum(x,y,None,{})

        lowest_score = gs.score_matrix.min()
        self.assertLess( lowest_score, np.Inf )
        best_position = gs._find_best_pos()
        self.assertEqual(gs.score_matrix[best_position],lowest_score)
        self.assertEqual(gs.best_params_,gs._pos_to_par(best_position))

        grid = gs.grid_scores_
        self.assertIsInstance(grid,list)
        for i in grid:
            self.assertIsInstance(i,tuple)
            self.assertEqual(len(i),2)
            self.assertIsInstance(i[0],dict)
            self.assertIsInstance(i[1],float)
            self.assertEqual(set(i[0].keys()),set(parameters.keys()))
            for k in parameters:
                self.assertIn(i[0][k],parameters[k])

    @pytest.mark.dscomp
    def test_find_minimum_1d(self):
        parameters = {  'gamma'    :np.logspace(-4,1,base=10,num=20)}
        self.check_find_minimum(parameters)

        parameters = {  'gamma'    :np.logspace(-4,1,base=10,num=20),
                        'C'        :[10]}
        self.check_find_minimum(parameters)

    @pytest.mark.dscomp
    def test_find_minimum_2d(self):
        parameters = {  'C'    :np.logspace(0,4,base=10,num=9),
                        'gamma':np.logspace(-4,1,base=10,num=9) }
        self.check_find_minimum(parameters)

    @pytest.mark.dscomp
    def test_find_minimum_3d(self):
        parameters = {  'C'    :np.logspace(0,4,base=10,num=9),
                        'gamma':np.logspace(-4,1,base=10,num=9),
                        'degree':[3,5,7]}
        self.check_find_minimum(parameters,tuneorder=['C','gamma','degree'])

    @pytest.mark.dscomp
    def test_fit(self):
        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'])

        out = gs.fit(x,y)
        self.assertIsInstance(out,BaseGridSearch)

        self.assertIsInstance(gs.best_estimator_,estimator.__class__)

        params = gs.best_estimator_.get_params()
        for k in gs.best_params_:
            self.assertEqual(gs.best_params_[k],params[k])

    @pytest.mark.dscomp
    def test_with_gbm(self):
        x,y,a,b,cv = self.startup1()
        estimator = GradientBoostingClassifier(n_estimators=100,learning_rate=0.03)
        parameters = {'max_depth':[3,5,7,9,11],'min_samples_leaf':range(1,6)}
        loss_func = logloss

        cv= ShuffleSplit(n=len(y),n_iter=1,test_size=0.10,random_state=1234)
        #cv = StratifiedKFold(y=y, n_folds=5)
        #cv = KFold(n=len(y), n_folds=5, random_state=1234)

        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['max_depth','min_samples_leaf'],step=5)

        out = gs.fit(x,y)
        self.assertIsInstance(out,BaseGridSearch)

        lowest_score = gs.score_matrix.min()
        self.assertLess( lowest_score, np.Inf )
        best_position = gs._find_best_pos()
        self.assertEqual(gs.score_matrix[best_position],lowest_score)

        grid = gs.grid_scores_
        self.assertIsInstance(grid,list)
        for i in grid:
            self.assertIsInstance(i,tuple)
            self.assertEqual(len(i),2)
            self.assertIsInstance(i[0],dict)
            self.assertIsInstance(i[1],float)
            for k in parameters:
                self.assertIn(i[0][k],parameters[k])

    @pytest.mark.dscomp
    def test_update_leaderboard(self):
        initial_report = {'pid':'asdf'}
        key = (0,1)
        progress = ProgressSink()

        lid_oid = self.persistent.create(initial_report,table='leaderboard')
        lref = LeaderboardRef(lid_oid, self.persistent, key, progress)

        query = self.persistent.read(condition={'_id':lid_oid}, table='leaderboard', result=[])
        report = query[0]
        self.assertEqual(report,initial_report)

        x,y,estimator,parameters,cv = self.startup1()
        gs = BaseGridSearch(estimator,parameters,cv,logloss,tuneorder=['C','gamma'], leaderboard_ref = lref)

        for i in gs._search_iterator(x,y,None,{}):

            c = [gs._par_to_pos(j) for j in i]
            partialgrid = gs._get_grid_scores(c)
            self.assertEqual(len(c),len(partialgrid))

            grid = gs.grid_scores_

            query = self.persistent.read(condition={'_id':lid_oid}, table='leaderboard', result=[])
            report = query[0]

            self.assertEqual(len(query),1)
            self.assertEqual(set(report.keys()), set(['_id','pid','grid_scores']))
            for (p,s),(pcheck,scheck) in zip(grid,report['grid_scores'][str(key)]):
                self.assertEqual(p,pcheck)
                self.assertAlmostEqual(s,scheck)







if __name__ == '__main__':
    unittest.main()
