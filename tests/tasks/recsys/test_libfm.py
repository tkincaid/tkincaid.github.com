import numpy as np
from pylibfm import pylibfm

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.recsys.libfm import LibFMRegressor, LibFMClassifier
from ModelingMachine.engine.tasks.converters import DesignMatrix2


class TestLibFM(BaseTaskTest):

    def test_arguments(self):
        """Test if arguments are passed to wrapped estimator"""
        xt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        yt = np.array([0, 1, 0], dtype=np.float64)
        self.check_arguments(LibFMRegressor, pylibfm.FMRegressor, xt, yt)

    def test_default_arguments(self):
        self.check_default_arguments(LibFMRegressor, pylibfm.FMRegressor, ignore_params=('seed', 'verbose', 'task'))

    def test_regressor(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=1000, n_users=100, n_items=10)

        ohe = DesignMatrix2()
        ohe.fit(X, Y, Z)
        c = ohe.transform(X, Y, Z)

        task = LibFMRegressor('nc=[4];ni=[2];v=0;eta=[0.01,0.001];is=[0.1,0.01];rs=1145;'
                              't_n=1;t_f=0.10;t_m=RMSE')
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

    def test_classification(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=1000, n_users=100, n_items=10)
        Y = (Y > 3).astype(float)

        ohe = DesignMatrix2()
        ohe.fit(X, Y, Z)
        c = ohe.transform(X, Y, Z)

        task = LibFMClassifier('nc=[4];ni=[4];v=0;eta=[0.1,0.01];is=[0.01];rs=1145;'
                              't_n=1;t_f=0.10;t_m=AUC')
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)
