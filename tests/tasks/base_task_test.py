#########################################################
#
#       Base Unit Test Case for tasks
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import copy
import os
import pandas as pd
import numpy as np
import itertools
import logging
import json
import random

from collections import OrderedDict
from sklearn.datasets import make_friedman1
from sklearn.utils import check_random_state
from sklearn.utils import check_arrays

from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.response import Response

from common.validation import accepts

TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../testdata')
EXTRA_KEYS = {'coefficients','importance','grid_scores', 'hotspots',
              'partial_dependence_plots'}


def syn_counts(n_samples=50, offset=0.0, xv=(1., -0.5, 1.0), random_state=None):
    """Synthetic count data generator with len(xv) - 1 features.

    Returns
    -------
    X : np.array, shape=(n_samples, len(xv) - 1)
        The features
    y = np.array, shape=(n_samples,)
        The response
    """
    rs = check_random_state(random_state)
    xv, = check_arrays(xv)
    p = xv.shape[0] - 1
    X = np.c_[np.ones(n_samples), rs.normal(size=n_samples * p).reshape((n_samples, p))]
    xb = np.dot(X, xv)
    exb = np.exp(xb + offset)
    py = rs.poisson(lam=exb, size=n_samples)
    return X[:, 1:], py


class BaseTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ds = pd.read_csv(
                os.path.join(TESTDATA_DIR, 'credit-sample-200.csv'))
        self.ds2 = pd.read_csv(
                os.path.join(TESTDATA_DIR, 'allstate-nonzero-200.csv'))
        self.ds3 = pd.read_csv(
                os.path.join(TESTDATA_DIR, 'credit-train-small.csv'))

    def create_bin_data(self,reps=1, rows=None):
        X = copy.deepcopy(self.ds)
        if rows is not None and rows < X.shape[0]:
            X = X[:rows]
        Y = X.pop('SeriousDlqin2yrs').values
        Y = Response.from_array(Y)
        Z = Partition(size=X.shape[0],folds=5,reps=reps,total_size=X.shape[0])
        Z.set(max_reps=reps,max_folds=0)
        X = Container(dataframe=X)
        return X,Y,Z

    def create_reg_data(self,reps=1, rows=None, categoricals=False):
        X = copy.deepcopy(self.ds2)
        if rows is not None and rows < X.shape[0]:
            X = X[:rows]
        Y = X.pop('Claim_Amount').values
        if not categoricals:
            X = X.take(range(21,29),axis=1)
        else:
            X = X.take(range(5,29),axis=1)
        Y = Response.from_array(Y)
        Z = Partition(size=X.shape[0],folds=5,reps=reps,total_size=X.shape[0])
        Z.set(max_reps=reps,max_folds=0)
        X = Container(dataframe=X)
        return X,Y,Z

    def create_reg_syn_data(self, reps=1, rows=None):
        """Create synthetic data using friedman1 sample generator.

        Returns
        -------
        X : pd.DataFrame
            The input as a pd.DataFrame.
        Y : np.ndarray
            The targets as a ndarray
        Z : Partition
            The partition object holding 5-folds.
        """
        if rows is None:
            rows = 1000
        X, y = make_friedman1(n_samples=rows, random_state=13)
        X = pd.DataFrame(data=X, columns=map(unicode, range(X.shape[1])))
        Y = Response.from_array(y)
        Z = Partition(size=X.shape[0], folds=5, reps=reps,total_size=X.shape[0])
        Z.set(max_reps=reps,max_folds=0)
        X = Container(dataframe=X)
        return X, Y, Z

    def create_reg_count_syn_data(self, reps=1):
        X, y = syn_counts(n_samples=500, random_state=13)
        X = pd.DataFrame(data=X, columns=map(unicode, range(X.shape[1])))
        Y = Response.from_array(y)
        Z = Partition(size=X.shape[0], folds=5, reps=reps,total_size=X.shape[0])
        Z.set(max_reps=reps,max_folds=0)
        X = Container(dataframe=X)
        return X, Y, Z

    def create_bin_large_data(self,reps=1):
        X = copy.deepcopy(self.ds3)
        Y = X.pop('SeriousDlqin2yrs').values
        Y = Response.from_array(Y)
        Z = Partition(size=X.shape[0],folds=5,reps=reps,total_size=X.shape[0])
        Z.set(max_reps=reps,max_folds=0)
        X = Container(dataframe=X)
        return X,Y,Z

    def create_cf_syn_data(self, n_samples=1000, n_users=500, n_items=10, reps=1, folds=0,
                           categoricals=False):
        rs = np.random.RandomState(13)
        user_ids = rs.randint(0, n_users, size=n_samples)
        if categoricals:
            user_ids = map(lambda ui: 'UID-%d' % ui, user_ids)
        item_ids = rs.randint(0, n_items, size=n_samples)
        if categoricals:
            item_ids = map(lambda ii: 'IID-%d' % ii, item_ids)

        X = pd.DataFrame(data=OrderedDict([('user_id', user_ids),
                                           ('item_id', item_ids)]))
        # set special columns to adhere blueprint_interpreter.BuildData.dataframe
        X.special_columns = {'I': 'item_id', 'U': 'user_id'}
        Y = Response.from_array(rs.randint(1, 5, size=n_samples))
        Z = Partition(size=X.shape[0],folds=5,reps=reps,total_size=X.shape[0])
        Z.set(max_reps=reps,max_folds=folds)
        if reps == -1 and folds == -1:
            Z.partitions = [(-1, -1)]
        X = Container(dataframe=X)
        return X, Y, Z

    def check_predictions(self, pred, X, predtype, reference=None, decimal=6):
        """
        pred = predictions Container
        X = X data
        reference = first N values to check against (checks for regressions
        or RNG) numpy.ndarray
        """
        self.assertIsInstance(pred,Container)
        for p in pred:
            print p
            a = pred(**p)
            self.assertIsInstance(a,np.ndarray)
            self.assertLessEqual(a.shape[1],1)
            self.assertEqual(a.shape[0],X().shape[0])
            for i in a.flatten():
                self.assertIsInstance(i,predtype)
                self.assertNotEqual(i,np.NaN)
            if reference is not None and p['r'] == 0:
                np.testing.assert_almost_equal(
                        reference, a.flatten()[:len(reference)], decimal=decimal)

    def check_transform(self,out,X):
        """
        out = output Container
        X = X data
        """
        self.assertIsInstance(out,Container)
        for p in out:
            a = out(**p)
            self.assertIsInstance(a,np.ndarray)
            self.assertLessEqual(a.shape[1],X().shape[1])
            self.assertEqual(a.shape[0],X().shape[0])

    def check_transform2(self,out,X,task):
        self.check_transform(out,X)
        if task.raw_importances:
            imp = task.raw_importances[0,-1]
            expected = [i[0] for i in sorted(zip(X.colnames(),imp),key=lambda x:-x[1]) if i[1]>0]
        else:
            imp = task.raw_coefficients[0,-1]
            expected = [i[0] for i in sorted(zip(X.colnames(),np.abs(imp)),key=lambda x:-x[1]) if i[1]>0]
        for p in out:
            check = out.colnames(**p)
            self.assertEqual(check,expected)
            self.assertEqual(out(**p).shape[1],len(expected))

    def check_default_arguments(self, Task, Estimator, ignore_params=None):
        """Test if task defaults are the same as sklearn ``Estimator`` defaults.

        This will create a new instance of ``Task`` w/o args, call ``_create_estimator``
        and then checks if the resulting BaseEstimator has the same parameters
        as an instance of ``Estimator``.

        Parameters
        ----------
        Task : instance of BaseModeler
            The class of the task
        Estimator : sklearn.base.BaseEstimator
            The class of the sklearn estimator
        ignore_params : seq of str
            The estimator parameters that should be ignored for the check.
        """
        task = Task()
        task._parse_parameters('')
        tuneparms, fixedparms = task._separate_parameters('a')
        task_est = task._create_estimator(fixedparms)
        est = Estimator()
        est_params = est.get_params()
        task_est_params = task_est.get_params()
        if ignore_params is None:
            ignore_params = []
        for param in ignore_params:
            est_params.pop(param, None)
            task_est_params.pop(param, None)
        self.assertDictEqual(est_params, task_est_params)

    def check_arguments(self,Task, Estimator=None, xt=None, yt=None, skip_bm_args=True):
        '''
        try to call task._create_estimator with all possible argument combinations
        '''
        args = {}
        for cd in Task.arguments:
            val = Task.arguments[cd]
            if val['type'] == 'select':
                args[cd] = map(str, range(len(val['values'])))
            if val['type'] == 'multi':
                args[cd] = val['values']['select']

        keys = args.keys()

        # arguments on the BaseModeler level -- dont search over them
        bm_level_args = [Task.gridsearch_arguments,
                         Task.post_processing_arguments,
                         Task.stepwise_arguments,
                         # weight arguments are optional
                         Task.pre_processing_arguments,]

        def shrink_vals(vals, key):
            """Shrink the values of BaseModeler arguments;

            Only pick one element from them.
            """
            # randomly select an argument but deterministic based on Task and argument.
            random.seed(hash(Task.__class__.__name__ + key))

            if any(key in bm_args for bm_args in bm_level_args):
                vals = [random.choice(vals)]
            return vals

        if skip_bm_args:
            values = [shrink_vals(args[k], k) for k in keys]
        else:
            values = [args[k] for k in keys]

        for i in itertools.product(*values):
            arg = ';'.join(map(lambda x: x[0]+'='+x[1],zip(keys,i)))
            #print arg
            task = Task(arg)
            task._modify_parameters(xt,yt)
            a, b = task._separate_parameters('x')
            for i in b.values():
                if type(i) in [str,unicode]:
                    self.assertNotIn(',',i)
            if Estimator is None:
                mdl = task._create_estimator(b)
            else:
                mdl = Estimator(**b)
            self.assertNotEqual(mdl, None)

    def check_arguments_min(self, Task, Estimator, xt, yt):
        '''Similar to check_arguments but doesn't check as many configurations

        The other version does a full cartesian product of all possible
        settings of the parameters.  In some cases, that is like 21120
        configurations.  In the case that you feel that such a test is overkill
        you can use this one instead.

        Or perhaps you use the other one while actively developing on your
        task, and switch to this one before committing
        '''
        args = {}
        for cd in Task.arguments:
            val = Task.arguments[cd]
            if val['type'] == 'select':
                args[cd] = map(str,range(len(val['values'])))
            if val['type'] == 'multi':
                args[cd] = val['values']['select']

        for key in args.keys():
            vals = args[key]
            for arg_option in vals:
                fixed_args = [
                    (arg, val[0]) for arg, val in args.iteritems() if arg != key]
                fixed_args.append((key, arg_option))
                arg = ';'.join([x[0]+'='+x[1] for x in fixed_args])
                #print arg
                task = Task(arg)
                task._modify_parameters(xt,yt)
                a,b = task._separate_parameters('x')
                for i in b.values():
                    if type(i) in [str,unicode]:
                        self.assertNotIn(',',i)
                if Estimator is None:
                    mdl = task._create_estimator(b)
                else:
                    mdl = Estimator(**b)
                self.assertNotEqual(mdl, None)

    @accepts(None, None, Container, np.ndarray, None)
    def check_task(self, taskname, X, Y, Z, transform=False, standardize=False,
                   predtype=np.float64, predict=True, reference=None, decimal=6,
                   insights=None):
        '''Optionally provide reference to compare to output

        reference : numpy.ndarray
            A vector of values to compare the output of the predict function's
            first fold

        '''
        Y = Response.from_array(Y)
        tasks = ['NI','ST'] if standardize else ['NI']
        vertex = Vertex(tasks,'id')
        X = vertex.fit_transform(X,Y,Z)
        vertex = Vertex([taskname],'id', insights=insights)
        #fit and predict
        vertex.fit(X,Y,Z)
        task, xfunc, yfunc = vertex.steps[-1]
        if predict:
            out= vertex.predict(X,Y,Z)
            self.check_predictions(out,X,predtype, reference, decimal=decimal)
            #report
            out= task.report()
            self.assertIsInstance(out,dict)
            self.assertIn((0,-1),out.keys())
            for key in out[0,-1]:
                self.assertIsInstance(key,str)
                if key == 'extras':
                    self.assertLessEqual(set(out[0,-1]['extras'].keys()),
                                         EXTRA_KEYS)
            json.dumps(out[0,-1])

            #check gridsearch results
            if task.grid_scores:
                print
                for i,n in enumerate(task.grid_scores[0,-1]):
                    print n
                print
                print ' ---- best model parameters = %s'%task.best_parameters
                print
        #transform
        if transform:
            logging.debug(' ----- testing %s  transform -----'%taskname)
            out= vertex.transform(X,Y,Z)
            self.check_transform(out,X)
            logging.debug(' ----- testing %s  transform2 -----'%taskname)
            out= vertex.transform2(X,Y,Z)
            self.check_transform2(out,X,task)

        return task

    """
    def test_01(self):
        '''
        example test for the fit and predict methods
        '''
        X,Y,Z = self.create_bin_data()
        self.check_task('RFC',X,Y,Z)
        """




if __name__ == '__main__':
    unittest.main()
