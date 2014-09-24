############################################################################
#
#       unit test for Eda
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import pandas
import numpy as np
import unittest
import os
import json

from ModelingMachine.engine.histogram import varProfile
from ModelingMachine.engine.pandas_data_utils import parse_column
from MMApp.entities.db_conn import DBConnections
from tests.IntegrationTests.storage_test_base import StorageTestBase
import config.test_config


class TestEDA(StorageTestBase):
    """
    """
    def np_equal(self,a,b):
        """ assert two numpy arrays are equal (even if they have nan's)
        """
        try:
            np.testing.assert_equal(a,b)
        except AssertionError:
            return False
        return True

    @classmethod
    def setUpClass(cls):
        super(TestEDA, cls).setUpClass()
        cls.dbs = DBConnections()
        cls.get_collection = cls.dbs.get_collection

        cls.datasets = [
            {'filename':'credit-sample-200.csv','target': 'SeriousDlqin2yrs'},
            {'filename':'allstate-nonzero-200.csv','target': 'Claim_Amount'},
            {'filename':'kickcars-sample-200.csv','target': 'IsBadBuy'},
            {'filename':'nfl_stats_200.csv','target': 'OffGreaterThanDef'}
        ]

        cls.test_directory, _ = cls.create_test_files()

    @classmethod
    def tearDownClass(cls):
        super(TestEDA, cls).tearDownClass()
        cls.dbs.destroy_database()

    def check_varProfile(self,dataset):
        dsn = dataset['filename']
        yn = dataset['target']
        filename = os.path.join(self.test_directory,dsn)
        ds = pandas.read_csv(filename)
        y = ds[yn]
        for xn in ds:
            col,vtype = parse_column(ds[xn])
            out = varProfile(ds[xn],y,vtype=vtype)
            #check output dict keys
            answer1 = ['plot', 'name', 'miss_count', 'y', 'plot2', 'type', 'miss_ymean', 'weight'] #for num vars
            answer2 = ['y', 'plot', 'type', 'name', 'weight'] #for cat vars
            self.assertIsInstance(out,dict)
            self.assertIn(set(out.keys()),(set(answer1),set(answer2)))
            #check plot
            plot = out.get('plot')
            self.assertIsInstance(plot,list)
            for i in plot:
                self.assertIn(type(i),(list,tuple))
                for j in i:
                    self.assertIn(type(j),(str,unicode,float))
            #check plot2
            plot2 = out.get('plot2')
            if plot2 is not None:
                self.assertIsInstance(plot2,list)
                for i in plot2:
                    self.assertIn(type(i),(list,tuple))
                    for j in i:
                        self.assertIn(type(j),(str,unicode,float))
            #check other keys
            self.assertIsInstance(out.get('name'),str)
            self.assertIsInstance(out.get('type'),str)
            self.assertIsInstance(out.get('y'),str)
            miss_count = out.get('miss_count')
            if miss_count:
                self.assertIsInstance(miss_count,float)
            miss_ymean = out.get('miss_ymean')
            if miss_ymean:
                self.assertIsInstance(miss_ymean,float)
            #check json conversion
            check = json.dumps(out)
        return ds

    def test_on_credit_data(self):
        ds = self.check_varProfile(self.datasets[0])

    def test_on_kickcars_data(self):
        self.check_varProfile(self.datasets[2])

    def test_on_allstate_data(self):
        self.check_varProfile(self.datasets[1])

    def test_varProfile(self):
        constant_weight = 2
        x = {'a':map(str,np.arange(20)%5), 'b':np.zeros(20)+constant_weight, 'c': np.arange(20)%5}
        x = pandas.DataFrame(x)

        #1. test categorical variable
        out1 = varProfile(x['a'])
        print out1['plot']
        out2 = varProfile(x['a'], weight = x['b'])
        print out2['plot']
        #compare results with and without weights
        for i,j in zip(out1['plot'], out2['plot']):
            self.assertEqual(i[0], j[0])
            self.assertEqual(i[1]*constant_weight, j[1])

        #2. test numeric variable
        out1 = varProfile(x['c'])
        print out1['plot']
        out2 = varProfile(x['c'], weight = x['b'])
        print out2['plot']
        #compare results with and without weights
        for i,j in zip(out1['plot'], out2['plot']):
            self.assertEqual(i[0], j[0])
            self.assertEqual(i[1]*constant_weight, j[1])

    def test_varProfile_with_y(self):
        constant_weight = 2
        x = {'a':map(str,np.arange(20)%5), 'b':np.zeros(20)+constant_weight, 'c': np.arange(20)%5}
        x = pandas.DataFrame(x)
        y = pandas.Series(np.arange(20), name='target')

        #1. test categorical variable
        out1 = varProfile(x['a'], target_column=y)
        print out1['plot']
        out2 = varProfile(x['a'], target_column=y, weight = x['b'])
        print out2['plot']
        #compare results with and without weights
        for i,j in zip(out1['plot'], out2['plot']):
            self.assertEqual(i[0], j[0])
            self.assertEqual(i[1]*constant_weight, j[1])
            self.assertEqual(i[2]/i[1], j[2]/j[1]) #true because weight is constant

        #2. test numeric variable
        out1 = varProfile(x['c'], target_column=y)
        print out1['plot']
        out2 = varProfile(x['c'], target_column=y, weight = x['b'])
        print out2['plot']
        #compare results with and without weights
        for i,j in zip(out1['plot'], out2['plot']):
            self.assertEqual(i[0], j[0])
            self.assertEqual(i[1]*constant_weight, j[1])
            self.assertEqual(i[2]/i[1], j[2]/j[1]) #true because weight is constant





if __name__ == '__main__':
    unittest.main()
