############################################################################
#
#       unit test for ModelingMachine/tests/test_meta.py
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import pandas
import numpy as np
import sys
import unittest
import hashlib
from bson.objectid import ObjectId
import os
import shutil
import copy
import time


from ModelingMachine.tests import test_meta
from config.test_config import db_config as config
from common.wrappers import database
from ModelingMachine.metablueprint.quickmb import QuickMetablueprint
test_meta.scoreboard_config = config
def func(proc,interval,retval):
    a = proc[0]
    a()
    return [1.,2.,3.],0
test_meta.memory_usage = func

class TestTestMeta(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        assert config['persistent']['host'] == 'localhost'
        assert config['tempstore']['host'] == 'localhost'
        assert config['persistent']['dbname'] == 'unitTesting'

        here = os.path.dirname(os.path.abspath(__file__))
        self.testdatadir = os.path.join(here,'../testdata')
        self.tempstore = database.new("tempstore", host=config['tempstore']['host'],
                port=config['tempstore']['port'])
        self.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

    @classmethod
    def tearDownClass(self):
        self.persistent.db_connection.drop_database('unitTesting')
        self.tempstore.conn.flushdb()

    def setUp(self):
        self.persistent.db_connection.drop_database('unitTesting')
        self.tempstore.conn.flushdb()

    def tearDown(self):
        pass

    def test_run_tests(self):
        testdatasets = [{'filename':'credit-sample-200.csv','response':'SeriousDlqin2yrs'}]
        test_meta.run_tests(testdatasets, QuickMetablueprint, testdatadir=self.testdatadir)
        cursor = self.persistent.conn['testmeta_models'].find()
        query = [i for i in cursor]
        self.assertEqual(len(query),4)
        for i in query:
            self.assertIn('build_error',i.keys())
            if i['build_error'] == 'None':
                self.assertIn('metrics',i.keys())
        cursor = self.persistent.conn['testmeta_summary'].find()
        query = [i for i in cursor]
        self.assertEqual(len(query),1)



if __name__ == '__main__':
    unittest.main()
