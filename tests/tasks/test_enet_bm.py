#########################################################
#
#       Unit Test for tasks/enet.py
#
#       Author: Mark Steadman
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import copy
import os
import pandas
import numpy as np
from mock import Mock, patch, DEFAULT

from sklearn.linear_model import SGDClassifier

from base_task_test import BaseTaskTest
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.tasks.base_modeler import BaseModeler
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.tasks.enet_bm import EnetLog, EnetLogWeighted, SGDLog
import ModelingMachine

class TestLogistic(BaseTaskTest):
    @classmethod
    def setUpClass(cls):
        alphagrid = np.logspace(0,-4,num=50, base=10).tolist()
        argstring = 'p=elasticnet;b=0.85;a=0.001' # + ';a=' + str(alphagrid)
        cls.test_instance_EL1 = EnetLog(argstring)
        cls.test_instance_EL1_gs = EnetLog('p=elasticnet;b=0.85;a=auto')
        cls.test_instance_EL1_bc = EnetLog('p=elasticnet;b=0.85;a=0.001')
        cls.test_instance_EL1_bc_rgs = EnetLog('p=elasticnet;b=[0.9,0.5,0.1];a=[0.001, 0.002]')
        cls.test_instance_ELW = EnetLogWeighted(argstring)
        cls.xt_mp = np.arange(75).reshape(15,5)
        cls.yt_mp = pandas.Series(np.hstack((np.ones((5,)) + 1, np.ones((10,)))).flatten())

    @patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_check_stack')
    def test_enet_fits_and_predicts_overdetermined(self,*args):
        """
        Test to see that it creates predictions on a large class of inputs.
        """
        xt = np.arange(500).reshape(100,5)
        yt = pandas.Series(np.random.randint(2, size=(1,xt.shape[0])).flatten()).values
        X = Container()
        Z = Partition(xt.shape[0],total_size=xt.shape[0])
        X.add(xt)
        self.test_instance_EL1.fit(X, yt, Z)
        predCont = Container()
        predarr = np.array([[5,6,7,8,9],[10,11,12,13,14]])
        predCont.add(predarr)
        predZ = Partition(predarr.shape[0],total_size=predarr.shape[0])
        pred = self.test_instance_EL1.predict(predCont, pandas.Series(np.array([1, 2]).flatten()), predZ)
        report = self.test_instance_EL1.report()
        self.test_instance_ELW.fit(X,yt,Z)
        pred_log = self.test_instance_ELW.predict(predCont, pandas.Series(np.array([1, 2]).flatten()), predZ)

    def test_enet_modify_parameters(self):
        """
        Tests to see that modify parameters correctly handles removing rho and replacing it by alpha
        """
        xt = self.xt_mp
        yt = self.yt_mp
        self.test_instance_EL1_bc._modify_parameters(xt, yt)
        (dparams, sparams) = self.test_instance_EL1_bc._separate_parameters('')
        self.assertEqual(sparams['l1_ratio'], 0.85)
        self.assertEqual(sparams['loss'], 'log')
        self.assertTrue('rho' not in sparams)

        test_instance_log_auto = EnetLog('p=elasticnet;a=auto;b=auto')
        test_instance_log_auto._modify_parameters(xt, yt)
        (dparams, sparams) = test_instance_log_auto._separate_parameters('')
        np.testing.assert_array_almost_equal_nulp(np.array(dparams['alpha']),  np.logspace(0,-4,num=50, base=10))
        self.assertEqual(sparams['l1_ratio'], 0.5)

    def test_enet_modify_parameters_backwardCompatability_creates_proper_alpha(self):
        """
        Tests El1 with the 'auto' option for the grid search over alpha to see that it creates the correct parameters.
        """
        self.test_instance_EL1_bc_rgs._modify_parameters(self.xt_mp, self.yt_mp)
        (dparamsrgs, sparamsrgs) = self.test_instance_EL1_bc_rgs._separate_parameters('')
        np.testing.assert_allclose(np.array(dparamsrgs['l1_ratio']), np.array([0.1, 0.5, 0.9]))

    def test_enet_modify_parameters_SGD_runs_correctly(self):
        """
        Test to see that the SGD correctly modifies parameters
        """
        #Test SGDLog _modify parameters
        sgd = SGDLog()
        sgd._modify_parameters(self.xt_mp,self.yt_mp)
        (dparamsSGD, sparamsSGD) = sgd._separate_parameters('')
        self.assertEqual(sparamsSGD['penalty'], 'l2')
        self.assertEqual(sparamsSGD['alpha'], 0)
        self.assertEqual(sparamsSGD['l1_ratio'], 0)

    def test_enet_fits_smaller_array(self):
        """
        Test to see if it works irrespective of whether rho or alpha is specified for backwards compatibility

        Also, if oversampling is fixed, this produces a reproducible response that can be used to test that
        the predictions are correct.
        """
        xt = np.arange(75).reshape(15,5)
        yt = pandas.Series(np.hstack((np.zeros((7,)), np.ones((8,)))).flatten()).values
        X = Container()
        Z = Partition(xt.shape[0],total_size=xt.shape[0])
        X.add(xt)
        self.test_instance_EL1._modify_parameters(xt,yt)
        (dparams, sparams) =  self.test_instance_EL1._separate_parameters('')
        predCont = Container()
        predarr = np.array([[5,6, 7, 8, 9]])
        predCont.add(predarr)
        predZ = Partition(predarr.shape[0],total_size=predarr.shape[0])
        self.test_instance_EL1_bc._modify_parameters(xt, yt)
        (dparamsbc, sparamsbc) =  self.test_instance_EL1_bc._separate_parameters('')
        for param in dparams: #Test that same params are passed after _modify_parameters()
            self.assertEqual(dparamsbc[param], dparams[param])
        for param in sparams:
            self.assertEqual(sparamsbc[param], sparams[param])
        #   self.test_instance_EL1_bc.fit(X, yt, Z)#this fails due to cross-validating on folds containing single class label responses

        # self.test_instance_EL1_bc_gs.fit(X, yt, Z)
#        pred = self.test_instance_EL1_bc.predict(predCont, pandas.Series(np.array([2]).flatten()).values, predZ)

        # Test elastic net weighted log regression fit and predict
        argstring = 'p=elasticnet;b=0.85;a=0.001' # + ';a=' + str(alphagrid)
        self.test_instance_ELW = EnetLogWeighted(argstring)
        self.test_instance_ELW.fit(X, yt, Z)
 #       pred_log = self.test_instance_ELW.predict(predCont, pandas.Series(np.array([2]).flatten()), predZ)

if __name__ == '__main__':
    unittest.main()
