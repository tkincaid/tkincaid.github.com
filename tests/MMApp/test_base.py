############################################################################
#
#       unit test base class for MMApp
#
#       Author: Ulises Reyes
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import unittest
import os
import sys
from mock import Mock
from bson import ObjectId

mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))
import config.test_config
import logging
from MMApp.entities.db_conn import DBConnections

class TestBase(unittest.TestCase):

    test_pid = str(ObjectId())
    test_qid = str(ObjectId())
    test_uid = str(ObjectId())
    test_lid = str(ObjectId())

    @classmethod
    def setUpClass(self):
        dbs = DBConnections()
        self.redis_conn = dbs.get_redis_connection()
        self.get_collection = dbs.get_collection

        self.logger = logging.getLogger(name="datarobot.user")
        self.logger.addHandler(logging.StreamHandler())

        self.logger.debug('Redis port: %s' % self.redis_conn.info()['tcp_port'])
        self.logger.debug('Mongo database name: %s' % dbs.mongo_db_name)

    @classmethod
    def tearDownClass(self):
        DBConnections().destroy_database()
        self.redis_conn.flushdb()

    def assertContains(self, larger, smaller):
        for key in smaller:
            self.assertIn(key, larger)
            self.assertEqual(smaller[key], larger[key])