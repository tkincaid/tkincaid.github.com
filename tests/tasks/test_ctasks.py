#########################################################
#
#       Unit Test for tasks/svc.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import logging
import os
import copy
import numpy as np
import pandas
from base_task_test import BaseTaskTest, TESTDATA_DIR
from ModelingMachine.engine.tasks.cgbm import GBMCC, GBMCR, GBMWrapper, TreeWrapper
from ModelingMachine.engine.tasks.crf import RFCC,RFCR
from ModelingMachine.engine.tasks.glm import GLMC,GLMR,GLMWrapper
from ModelingMachine.engine.tasks.elnet import ElnetC,ElnetR,ElasticNetWrapper
from ModelingMachine.engine.tasks.GLM import GLM


def calc_residuals(pred,act,dist):
    if dist == 'Poisson':
        return 2*(act*np.log(np.where(act==0,np.repeat(1,act.shape[0]), act/pred)) - (act - pred))
    if dist == "Gaussian":
        return np.square(act-pred)
    if dist == "Gamma":
        return -2*(np.log(np.where(act==0,np.repeat(1,act.shape[0]), act/pred)) - (act - pred)/pred)
    if dist == "Bernoulli":
        return 2*(act*np.log(np.where(act==0,np.repeat(1,act.shape[0]), act/pred))+(1-act)*np.log(np.where(act==1,np.repeat(1,act.shape[0]),(1-act)/(1-pred))));

class TestCTasks(BaseTaskTest):

    @pytest.mark.dscomp
    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(GBMCC, GBMWrapper, xt,yt)
        self.check_arguments(GBMCR, GBMWrapper, xt,yt)
        self.check_arguments(RFCC, TreeWrapper, xt,yt)
        self.check_arguments(RFCR, TreeWrapper, xt,yt)
        self.check_arguments(GLMC, GLMWrapper, xt,yt)
        self.check_arguments(GLMR, GLMWrapper, xt,yt)
        self.check_arguments(ElnetC, GLMWrapper, xt,yt)
        self.check_arguments(ElnetR, GLMWrapper, xt,yt)

    def test_01a(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('GBMCC',X,Y,Z,transform=False,standardize=True)
        self.check_task('GBMCR d=1',X,Y,Z,transform=False,standardize=True)

    def test_01b(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('RFCC',X,Y,Z,transform=False,standardize=True)
        self.check_task('RFCR',X,Y,Z,transform=False,standardize=True)

    @pytest.mark.skip('GLMC produces NANs in stacked predictions')
    def test_01c(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('GLMC',X,Y,Z,transform=False,standardize=True)
        self.check_task('GLMR d=0',X,Y,Z,transform=False,standardize=True)

    @pytest.mark.dscomp
    @pytest.mark.skip('ELNC produces NANs in stacked predictions')
    def test_01d(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('ELNC',X,Y,Z,transform=False,standardize=True)
        self.check_task('ELNR d=1',X,Y,Z,transform=False,standardize=True)

    def test_02a(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('GBMCC md=2',X,Y,Z,transform=True,standardize=True)
        self.check_task('GBMCR md=2',X,Y,Z,transform=True,standardize=True)

    def test_02b(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('RFCC tc=100;mn=9',X,Y,Z,transform=True,standardize=True)
        self.check_task('RFCR tc=100;mn=9',X,Y,Z,transform=True,standardize=True)

    def test_02c(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('GLMC',X,Y,Z,transform=True,standardize=True)
        self.check_task('GLMR',X,Y,Z,transform=True,standardize=True)

    @pytest.mark.dscomp
    def skip_test_02d(self):
        # skipping this for now since it fails due to a bug in grid search where
        # grid_scores contains the average of alpha values over CV folds while
        # best_parameters has the alpha associated with the lowest score causing
        # additional grid search to fail.
        # We are not using ELNC or ELNR in the app anyways
        X,Y,Z = self.create_bin_data()
        self.check_task('ELNC',X,Y,Z,transform=True,standardize=True)
        self.check_task('ELNR',X,Y,Z,transform=True,standardize=True)

    def test_reg_residuals(self):
        ds = pandas.read_csv(os.path.join(TESTDATA_DIR,'allstate-nonzero-200.csv'))
        X = copy.deepcopy(ds)
        Y = X.pop('Claim_Amount').values.astype(float)
        X = X[['Calendar_Year','Model_Year','Var5','Var6','Var7','Var8']].values.astype(float)
        mdl = GLM()
        for dist in ("Poisson","Gamma","Gaussian"):
            mdl.fit(X, Y, distribution = dist)
            pred = mdl.predict(X)
            self.assertTrue(np.all(np.abs(mdl.residuals-calc_residuals(pred,Y,dist))<0.0000001))
            self.assertTrue(np.all(np.abs(mdl.fitted_values-pred)<0.0000001))

    def test_cls_residuals(self):
        here = os.path.dirname(os.path.abspath(__file__))
        ds = pandas.read_csv(os.path.join(TESTDATA_DIR, 'credit-sample-200.csv'))
        X = copy.deepcopy(ds)
        Y = X.pop('SeriousDlqin2yrs').values.astype(float)
        X = X[['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio']].values.astype(float)
        mdl = GLM()
        mdl.fit(X, Y, distribution = "Bernoulli")
        pred = mdl.predict(X)
        self.assertTrue(np.all(np.abs(mdl.residuals-calc_residuals(pred,Y,"Bernoulli"))<0.0000001))
        self.assertTrue(np.all(np.abs(mdl.fitted_values-pred)<0.0000001))

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
