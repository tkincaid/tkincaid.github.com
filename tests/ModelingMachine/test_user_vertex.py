# -*- coding: utf-8 -*-

import unittest
import pandas
from numpy.random import randn
from rpy2 import robjects
from mock import patch
from ModelingMachine.engine.user_vertex import UserVertex
import ModelingMachine.engine.user_vertex as user_vertex
from ModelingMachine.engine.monitor import FakeMonitor
from ModelingMachine.engine.partition import Partition

class UserVertexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # cls.monitor_zmq_patch = patch('ModelingMachine.engine.monitor.zmq')
        # cls.monitor_zmq_mock = cls.monitor_zmq_patch.start()
        # cls.uv_monitor_patch = patch('ModelingMachine.engine.user_vertex.Monitor', FakeMonitor)
        # cls.uv_monitor_mock = cls.uv_monitor_patch.start()

        cls.task_map = {
            '12345r': {
                'modeltype': 'R',
                'modelfit': 'function(response,data) {\n  rdata=data; rdata$y=response$y;\n lm("y ~ .",rdata);\n}\n',
                'modelpredict': 'function(lm, data) {\n  predict.lm(lm,data)\n}'
            },
            '12345p': {
                'modeltype': 'Python',
                'classname': 'CustomModel',
                'modelsource': '''
import numpy as np
class CustomModel(object):
    def fit(self, X, Y):
        return self
    def predict(self, X):
        return np.ones(len(X))
                '''
            }
        }
        cls.r_xdata = robjects.r('data.frame(x1 = rnorm(350,5000), x2 = rnorm(350, 1000))')
        cls.r_ydata = robjects.r('data.frame(y = rnorm(350, 2000),z=rnorm(350,23))')
        cls.r_ydata_neg = robjects.r('data.frame(y = rnorm(350, 0),z=rnorm(350,23))')
        cls.p_xdata = pandas.DataFrame({'x1':randn(350), 'x2':randn(350)})
        cls.p_ydata = pandas.DataFrame({'y':randn(350)+2000})
        cls.p_ydata_neg = pandas.DataFrame({'y':randn(350)})
        cls.Z = Partition(350,total_size=400)
        cls.Z.set(max_folds=0, max_reps=5)
        cls.Z1 = Partition(350,total_size=400)
        cls.Z1.set(max_folds=0, max_reps=1)

    @classmethod
    def tearDownClass(cls):
        super(UserVertexTest, cls).tearDownClass()
        # cls.monitor_zmq_patch.stop()
        # cls.uv_monitor_patch.stop()

    def test_R_Model(self):
        rvertex = UserVertex(['USERTASK id=12345r'], self.task_map, '1a13b1')
        pred = rvertex._fit_and_act(self.r_xdata, self.r_ydata, self.Z, method='predict')
        for p in pred:
            robjects.globalenv['p'] = pred(**p)
            self.assertLess(abs(2000-robjects.r('mean(p)')[0]),1)

    def test_R_Model_logx(self):
        rvertex = UserVertex(['USERTASK id=12345r;logx'], self.task_map, '1a13b1')
        pred = rvertex._fit_and_act(self.r_xdata, self.r_ydata, self.Z, method='predict')
        for p in pred:
            robjects.globalenv['p'] = pred(**p)
            self.assertLess(abs(2000-robjects.r('mean(p)')[0]),1)

    def test_R_Model_logy(self):
        rvertex = UserVertex(['USERTASK id=12345r;logy'], self.task_map, '1a13b1')
        pred = rvertex._fit_and_act(self.r_xdata, self.r_ydata, self.Z, method='predict')
        for p in pred:
            robjects.globalenv['p'] = pred(**p)
            self.assertLess(abs(2000-robjects.r('mean(p)')[0]),1)

    def test_R_Model_logy_error(self):
        # check for log by putting negative values in Y
        rvertex = UserVertex(['USERTASK id=12345r;logy'], self.task_map, '1a13b1')
        pred = rvertex._fit_and_act(self.r_xdata, self.r_ydata_neg, self.Z, method='predict')
        for p in pred:
            robjects.globalenv['p'] = pred(**p)
            self.assertTrue(robjects.r('any(is.nan(p))'))

    def test_Python_Model(self):
        pvertex = UserVertex(['USERTASK id=12345p'], self.task_map, 'a11b31')
        pred = pvertex._fit_and_act(self.p_xdata, self.p_ydata, self.Z, method='predict')
        for p in pred:
            self.assertEqual(sum(pred(**p)),350)

    def test_Python_Model_logx(self):
        pvertex = UserVertex(['USERTASK logx;id=12345p'], self.task_map, 'a11b31')
        pred = pvertex._fit_and_act(self.p_xdata, self.p_ydata, self.Z, method='predict')
        for p in pred:
            self.assertEqual(sum(pred(**p)),350)

    def test_Python_Model_logx(self):
        pvertex = UserVertex(['USERTASK log1+x;id=12345p'], self.task_map, 'a11b31')
        pred = pvertex._fit_and_act(self.p_xdata, self.p_ydata, self.Z, method='predict')
        for p in pred:
            print sum(pred(**p))
            self.assertEqual(sum(pred(**p)),350)

    def test_Python_Model_logy(self):
        pvertex = UserVertex(['USERTASK logx;id=12345p'], self.task_map, 'a11b31')
        pred = pvertex._fit_and_act(self.p_xdata, self.p_ydata, self.Z, method='predict')
        for p in pred:
            self.assertEqual(sum(pred(**p)),350)

    def test_Python_Model_logy_error(self):
        pvertex = UserVertex(['USERTASK logy;id=12345p'], self.task_map, 'a11b31')
        pred = pvertex._fit_and_act(self.p_xdata, self.p_ydata_neg, self.Z, method='predict')
        for p in pred:
            self.assertAlmostEqual(sum(pred(**p)),601.39863996)

    def test_R_dump_and_load(self):
        rvertex1 = UserVertex(['USERTASK id=12345r'], self.task_map, '1a13b1')
        rvertex2 = UserVertex(['USERTASK id=12345r'], self.task_map, '1a13b1')
        pred = rvertex1._fit_and_act(self.r_xdata, self.r_ydata, self.Z, method='predict')
        self.assertItemsEqual(rvertex1.dirty_parts,self.Z.items)
        rvertex2.update(rvertex1.dump((0,-1)))
        self.assertItemsEqual(rvertex2.loaded_parts,((0,-1),))
        pred = rvertex2._act(self.r_xdata, self.r_ydata, self.Z1, method='predict')
        robjects.globalenv['p'] = pred(r=0,k=-1)
        self.assertLess(abs(2000-robjects.r('mean(p)')[0]),1)

    def test_Python_dump_and_load(self):
        pvertex1 = UserVertex(['USERTASK id=12345p'], self.task_map, 'a11b31')
        pvertex2 = UserVertex(['USERTASK id=12345p'], self.task_map, 'a11b31')
        pred = pvertex1._fit_and_act(self.p_xdata, self.p_ydata, self.Z, method='predict')
        self.assertItemsEqual(pvertex1.dirty_parts,self.Z.items)
        pvertex2.update(pvertex1.dump((0,-1)))
        self.assertItemsEqual(pvertex2.loaded_parts,((0,-1),))
        pred = pvertex2._act(self.p_xdata, self.p_ydata, self.Z1, method='predict')
        self.assertEqual(sum(pred(r=0,k=-1)),350)

class TestConvertToRDataFrame(unittest.TestCase):

    UNICODE_WORD = u'Ꮦ'

    def test_convert_unicode_column_names(self):
        ser1 = pandas.DataFrame(range(5), columns=[self.UNICODE_WORD])
        ser2 = pandas.DataFrame(range(5, 10), columns=['other'.decode('utf-8')])
        df = pandas.concat([ser1, ser2], axis=1)
        rdf = user_vertex.convert_to_r_dataframe(df)
        self.assertEqual(df.columns[0], rdf.colnames[0])
        self.assertEqual(df.columns[1], rdf.colnames[1])

    def test_convert_ascii_column_names_unchanged(self):
        ser1 = pandas.DataFrame(range(5), columns=['happiness'])
        ser2 = pandas.DataFrame(range(5, 10), columns=['other'])
        df = pandas.concat([ser1, ser2], axis=1)
        rdf = user_vertex.convert_to_r_dataframe(df)
        self.assertEqual(df.columns[0], rdf.colnames[0])
        self.assertEqual(df.columns[1], rdf.colnames[1])


if __name__ == '__main__':
    unittest.main()
