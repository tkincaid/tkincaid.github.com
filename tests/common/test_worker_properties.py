import unittest
import pytest

from config.test_config import db_config
from config.engine import EngConfig
from common.wrappers import database

import common.broker.workers as wm

class TestWorkerProperties(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempstore = database.new('tempstore')

    def setUp(self):
        self.tempstore.conn.flushdb()
        self.worker_id = '01359random10932'  # Actually a randomly generated ID

    def test_constructor(self):
        wp = wm.WorkerProperties(worker_id=self.worker_id,
                                 size='>30',
                                 service_id='5223deadbeefdeadbeef1234')
        self.assertEqual(wp.worker_id, self.worker_id)
        self.assertEqual(wp.size, '>30')
        self.assertEqual(wp.service_id, '5223deadbeefdeadbeef1234')

    def test_default_constructor(self):
        wp = wm.WorkerProperties.new()
        self.assertEqual(len(wp.worker_id), 36)
        self.assertEqual(wp.size, EngConfig.get('WORKER_SIZE'))
        self.assertIsNone(wp.service_id)  # Only for localhosts, though

    def test_to_dict(self):
        wp = wm.WorkerProperties.new()
        data = wp.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['size'], wp.size)
        self.assertEqual(data['worker_id'], wp.worker_id)
        self.assertEqual(data['service_id'], wp.service_id)

    def test_save_goes_to_db(self):
        wp = wm.WorkerProperties(worker_id=self.worker_id,
                                 size='>30',
                                 service_id='5223deadbeefdeadbeef1234')
        wp.save(self.tempstore)
        redis_key = 'worker:properties:{}'.format(self.worker_id)
        size = self.tempstore.conn.hget(redis_key, 'size')
        service_id = self.tempstore.conn.hget(redis_key, 'service_id')
        self.assertEqual(size, wp.size)
        self.assertEqual(service_id, wp.service_id)
        worker_id = self.tempstore.conn.hget(redis_key, 'worker_id')
        self.assertIsNone(worker_id)

    def test_read_comes_from_db(self):
        redis_key = 'worker:properties:{}'.format(self.worker_id)
        size = '>30'
        service_id = '5223deadbeefdeadbeef1234'
        self.tempstore.conn.hset(redis_key, 'size', size)
        self.tempstore.conn.hset(redis_key, 'service_id', service_id)
        wp = wm.WorkerProperties.from_db(worker_id=self.worker_id,
                                         tempstore=self.tempstore)

        self.assertEqual(size, wp.size)
        self.assertEqual(service_id, wp.service_id)

    def test_deletion(self):
        redis_key = 'worker:properties:{}'.format(self.worker_id)
        size = '>30'
        service_id = '5223deadbeefdeadbeef1234'
        self.tempstore.conn.hset(redis_key, 'size', size)
        self.tempstore.conn.hset(redis_key, 'service_id', service_id)
        wm.WorkerProperties.delete(self.worker_id, self.tempstore)

        vals = self.tempstore.conn.hgetall(redis_key)
        self.assertEqual(vals, {})


class TestWorkerPropertiesComparisons(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempstore = database.new('tempstore')

    def test_service_id_compare_when_match(self):
        wp1 = wm.WorkerProperties(worker_id='1',
                                  service_id='5223deadbeefdeadbeef1234',
                                  size='>30')
        wp2 = wm.WorkerProperties(worker_id='2',
                                  service_id='5223deadbeefdeadbeef1234',
                                  size='>30')
        result = wp1.does_service_id_match(wp2)
        self.assertTrue(result)

    def test_service_id_compare_when_mismatch(self):
        wp1 = wm.WorkerProperties(worker_id='1',
                                  service_id='5223deadbeefdeadbeef1234',
                                  size='>30')
        wp2 = wm.WorkerProperties(worker_id='2',
                                  service_id='5223deadbeefdeadbeef1235',
                                  size='>30')
        result = wp1.does_service_id_match(wp2)
        self.assertFalse(result)

    def test_size_compare_when_match(self):
        wp1 = wm.WorkerProperties(worker_id='1',
                                  service_id='5223deadbeefdeadbeef1234',
                                  size='>30')
        wp2 = wm.WorkerProperties(worker_id='2',
                                  service_id='5223deadbeefdeadbeef1234',
                                  size='>30')
        result = wp1.does_size_match(wp2)
        self.assertTrue(result)

    def test_size_compare_when_mismatch(self):
        wp1 = wm.WorkerProperties(worker_id='1',
                                  service_id='5223deadbeefdeadbeef1234',
                                  size='>30')
        wp2 = wm.WorkerProperties(worker_id='2',
                                  service_id='5223deadbeefdeadbeef1234',
                                  size=None)
        result = wp1.does_size_match(wp2)
        self.assertFalse(result)

