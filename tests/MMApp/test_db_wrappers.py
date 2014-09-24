####################################################################
#
#       Modeling Machine Database Wrapper Tests
#
#       Author: David Lapointe
#
#       Copyright (C) 2013 DataRobot Inc.
####################################################################

import sys
import os
import re
import unittest
import time
import pymongo
from mock import patch, call
import pytest

import config.test_config
from common.wrappers import database

@pytest.mark.db
class GenericDatabaseTestCase(unittest.TestCase):
    def test_abstract_db(self):
        """ Expect abstract class initialization to raise TypeError """
        with self.assertRaises(TypeError):
            from common.wrappers.dbs import generic_db
            db = generic_db.DB()
    def test_invalid_newdb(self):
        with self.assertRaises(ValueError):
            db = database.new("noexists")

@pytest.mark.db
class DBTestMethods(object):
    dictEntry = {
        "abc": 123,
        "def": 456,
        "pqr": [1, 4, "g"]
    }
    listEntry = [1, 2, 3]
    strEntry = "qwerty"

    def test_CRUD(self):
        #test_create
        self.assertNotEqual(self.db.create(keyname="testid", index="12345dict",
            values=self.dictEntry), None)
        self.assertNotEqual(self.db.create(keyname="testid", index="12345list",
            values=self.listEntry), None)
        self.assertNotEqual(self.db.create(keyname="testid", index="12345str",
            values=self.strEntry), None)
        #test_read
        ret = self.db.read(keyname="testid", index="12345dict",
            fields=self.dictEntry.keys(), result={})
        self.assertTrue(set([re.sub("u'","'",i) for i in map(str, ret.keys())]).issuperset(
            set(map(str, self.dictEntry.keys()))))
        self.assertTrue(set([re.sub("u'","'",i) for i in map(str, ret.values())]).issuperset(
            set(map(str, self.dictEntry.values()))))
        self.assertEqual(map(str, sorted(self.db.read(keyname="testid",
            index="12345list", result=[]))), map(str, sorted(self.listEntry)))
        self.assertEqual(self.db.read(keyname="testid", index="12345str",
            result=""), self.strEntry)
        #test_destroy
        self.assertNotEqual(self.db.destroy(keyname="testid", index="12345dict"),
            None)
        self.assertNotEqual(self.db.destroy(keyname="testid", index="12345list"),
            None)
        self.assertNotEqual(self.db.destroy(keyname="testid", index="12345str"),
            None)
        #test_read after deletion
        self.assertEqual(self.db.read(keyname="testid", index="12345dict",
            result={}), {})
        self.assertEqual(self.db.read(keyname="testid", index="12345list",
            result=[]), [])
        self.assertEqual(self.db.read(keyname="testid", index="12345str",
            result=""), "")

    def test_transaction(self):
        with self.assertRaises(RuntimeError):
            self.db.commit()
        with self.assertRaises(RuntimeError):
            self.db.rollback()
        self.assertEqual(self.db.count(keyname="testid", index="12345str"), 0)
        self.db.start_transaction()
        self.db.create(keyname="testid", index="12345str", values=self.strEntry)
        self.db.rollback()
        self.assertEqual(self.db.count(keyname="testid", index="12345str"), 0)
        self.db.start_transaction()
        self.db.create(keyname="testid", index="12345str", values=self.strEntry)
        self.db.commit()
        self.assertEqual(self.db.count(keyname="testid", index="12345str"), 1)
        self.db.destroy(keyname="testid", index="12345str")

    def test_lists(self):
        self.assertNotEqual(self.db.create(keyname="testid", index="12345list",
            values=self.listEntry), None)
        self.assertEqual(self.db.count(keyname="testid", index="12345list"),
            len(self.listEntry))
        self.assertNotEqual(self.db.destroy(keyname="testid", index="12345list"),
            None)
        self.assertEqual(self.db.count(keyname="testid", index="12345list"), 0)

    def test_delete(self):
        self.assertNotEqual(self.db.create(keyname="testid", index="12345dict",
            values=self.dictEntry), None)
        ret = self.db.read(keyname="testid", index="12345dict",
            fields=self.dictEntry.keys(), result={})
        self.assertTrue(set(map(str, ret.keys())).issuperset(
            set(map(str, self.dictEntry.keys()))))
        self.assertTrue(set([re.sub("u'","'",i) for i in map(str, ret.values())]).issuperset(
            set(map(str, self.dictEntry.values()))))
        self.assertNotEqual(self.db.destroy(keyname="testid", index="12345dict",
            fields=["abc", "def"]), None)
        self.assertTrue(set(map(str, ret.keys())).issuperset(
            set(["pqr"])))
        self.assertNotEqual(self.db.destroy(keyname="testid", index="12345dict"),
            None)


@pytest.mark.db
class RedisDBTestCase(unittest.TestCase, DBTestMethods):
    @classmethod
    def setUpClass(self):
        self.db = database.new("tempstore", host="localhost", port=6482)
        self.db.conn.flushdb()

    def setUp(self):
        self.db.conn.flushdb()

    def test_make_key(self):
        pass

    def test_destroy_with_expiry_valid(self):
        '''re: Issue 343 - doesn't actually care if the destroy takes place,
        just that passing a float as the ``at`` parameter get handled by the
        db wrapper
        '''
        self.db.create(keyname='testkey',index='testindex', values='Test')
        self.db.destroy(keyname='testkey',index='testindex', at= 1)
        check = self.db.read(keyname='testkey',index='testindex')
        self.assertEqual(check, 'Test')

    def test_removing_call_returns_something(self):
        self.db.create(keyname='key',
                       index='index',
                       values=['test'])
        result = self.db.read(keyname='key',
                              index='index',
                              result=[],
                              remove=True)
        self.assertIsNotNone(result)

    def test_read_during_transaction_does_not_error(self):
        self.db.create(keyname='key',
                      index='index',
                      values=['test'])
        self.db.start_transaction()
        self.db.read(keyname='key',
                     index='index',
                     result=[])
        result = self.db.commit()
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)
        self.assertEqual(result[0][0], 'test')

    def test_hdel_succeeds_fields_as_tuple(self):
        self.db.create(keyname='key',
                       index='index',
                       values={'test':'value'})
        ret = self.db.destroy(keyname='key',
                              index='index',
                              fields=('test',))
        self.assertTrue(ret)

    def test_hdel_succeeds_fields_as_str(self):
        self.db.create(keyname='key',
                       index='index',
                       values={'1':'a','2':'b'})
        ret = self.db.destroy(keyname='key',
                              index='index',
                              fields='1')
        self.assertTrue(ret)

@pytest.mark.db
class MongoDBTestCase(unittest.TestCase, DBTestMethods):
    @classmethod
    def setUpClass(self):
        self.db = database.new("persistent", dbname="Testing")
        self.db.conn.Default.drop()

    @patch('common.wrappers.dbs.mongo_db.time', autospec = True)
    @patch('pymongo.MongoClient', autospec = True)
    def test_reconnects(self, MockMongoClient, MockTime):
        self.db = database.new("persistent", dbname="Testing", connect = False)

        MockMongoClient.side_effect = pymongo.errors.AutoReconnect('Boom!')


        retry_conn = 5
        with patch.dict(self.db.config, {'retry_conn' : retry_conn}):
            self.assertRaises(pymongo.errors.AutoReconnect, self.db.connect)

            MockTime.sleep.assert_has_calls([
                call(pow(2,0)),
                call(pow(2,1)),
                call(pow(2,2)),
                call(pow(2,3)),
                call(pow(2,4)),
                ])


if __name__ == '__main__':
    unittest.main()
