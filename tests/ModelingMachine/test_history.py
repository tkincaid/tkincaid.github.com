import unittest
import pytest
import json
from bson import ObjectId

import ModelingMachine.request_history as rh
from config.test_config import db_config
from common.wrappers import database


class TestWorkerHistoryRecord(unittest.TestCase):

    def setUp(self):
        self.pid = ObjectId('5223deadbeefdeadbeef1234')
        self.request_id = 4557

    def test_serialize_inverse_of_from_serialized(self):
        data = {'pid': str(self.pid), 'request_id': str(self.request_id)}
        serialized = json.dumps(data)
        record = rh.WorkerHistoryRecord.from_serialized(serialized)
        back = record.serialize()
        self.assertEqual(serialized, back)

    def test_from_serialized_inverse_of_serialize(self):
        record = rh.WorkerHistoryRecord(self.pid, self.request_id)
        serialized = record.serialize()
        record_back = rh.WorkerHistoryRecord.from_serialized(serialized)
        self.assertEqual(record_back.request_id, record.request_id)
        self.assertEqual(record_back.pid, record.pid)


class TestProjectHistoryRecord(unittest.TestCase):

    def setUp(self):
        self.worker_id = ObjectId('5223deadbeefdeadbeef1234')
        self.request_id = 4557

    def test_serialize_inverse_of_from_serialized(self):
        data = {'worker_id': str(self.worker_id),
                'request_id': str(self.request_id)}
        serialized = json.dumps(data)
        record = rh.ProjectHistoryRecord.from_serialized(serialized)
        back = record.serialize()
        self.assertEqual(serialized, back)

    def test_from_serialized_inverse_of_serialize(self):
        record = rh.ProjectHistoryRecord(self.worker_id, self.request_id)
        serialized = record.serialize()
        record_back = rh.ProjectHistoryRecord.from_serialized(serialized)
        self.assertEqual(record_back.request_id, record.request_id)
        self.assertEqual(record_back.worker_id, record.worker_id)


class TestWorkerHistory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempstore = database.new('tempstore')

    def setUp(self):
        self.tempstore.conn.flushdb()
        worker_id = 4557
        self.worker_history = rh.WorkerHistory(worker_id, self.tempstore)

    def test_add_and_read_are_inverses_and_in_order(self):
        pid0 = ObjectId('5223deadbeefdeadbeef0000')
        pid1 = ObjectId('5223deadbeefdeadbeef0001')

        request_order = []
        for request_id in range(10, 20):
            if 0 == request_id % 2:
                request_order.append({'pid': pid0, 'request_id': request_id})
            else:
                request_order.append({'pid': pid1, 'request_id': request_id})

        for r in request_order:
            self.worker_history.add(**r)

        back = self.worker_history.read()
        for (reference, compare) in zip(request_order, back):
            print compare  # Tests the __repr__ command
            self.assertEqual(str(reference['pid']), str(compare.pid))
            self.assertEqual(str(reference['request_id']),
                             str(compare.request_id))


class TestProjectHistory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempstore = database.new('tempstore')

    def setUp(self):
        self.tempstore.conn.flushdb()
        pid = ObjectId('5223deadbeefdeadbeef0000')
        self.project_history = rh.ProjectHistory(pid, self.tempstore)

    def test_add_and_read_are_inverses_and_in_order(self):
        worker_id0 = 455
        worker_id1 = 456

        request_order = []
        for request_id in range(10, 20):
            if 0 == request_id % 2:
                request_order.append({'worker_id': worker_id0,
                                      'request_id': request_id})
            else:
                request_order.append({'worker_id': worker_id1,
                                      'request_id': request_id})

        for r in request_order:
            self.project_history.add(**r)

        back = self.project_history.read()
        for (reference, compare) in zip(request_order, back):
            print compare  # Tests the __repr__ command
            self.assertEqual(str(reference['worker_id']),
                             str(compare.worker_id))
            self.assertEqual(str(reference['request_id']),
                             str(compare.request_id))
