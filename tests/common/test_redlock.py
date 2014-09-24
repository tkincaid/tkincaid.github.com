import unittest
from bson import ObjectId

import config.test_config
from common.services.redlock import Redlock
from common.wrappers import database


class TestRedlock(unittest.TestCase):

    def setUp(self):
        self.tempstore = database.new('tempstore')
        self.resource = 'redlock:{}'.format(ObjectId())

    def test_only_one_process_locks(self):
        locker = Redlock(self.tempstore.conn, self.resource)
        self.assertTrue(locker.lock())

        contender = Redlock(self.tempstore.conn, self.resource, max_tries = 1)
        self.assertFalse(contender.lock())

        locker.unlock()

    def test_process_can_lock_only_once(self):
        locker = Redlock(self.tempstore.conn, self.resource)
        self.assertTrue(locker.lock())
        self.assertFalse(locker.lock())

    def test_unlock(self):
        locker = Redlock(self.tempstore.conn, self.resource)
        locker.lock()
        locker.unlock()

        contender = Redlock(self.tempstore.conn, self.resource, max_tries = 1)
        self.assertTrue(contender.lock())
