from abc import ABCMeta
import json
import time


from bson.objectid import ObjectId
import pytest

import config.test_config
from common.engine.billing import ChargeRecord
from common.wrappers import database


class AbstractTestCharges(object):
    __metaclass__ = ABCMeta
    targetClass = ChargeRecord
    def setUp(self):
        self.record = self.newRecord()
        self.uid = ObjectId()

        self.persistent = database.new('persistent')
        self.persistent.destroy(table='charges')

    def newRecord(self):
        return self.targetClass()

    @pytest.mark.unit
    def test_invalid_alone(self):
        self.assertFalse(self.targetClass.valid({}))
        self.assertFalse(self.targetClass.valid(self.record.to_dict()))

    @pytest.mark.unit
    def test_valid_with_uid(self):
        self.record.uid = self.uid
        self.assertTrue(self.targetClass.valid(self.record.to_dict()))
        self.record.uid = 'garbage'
        self.assertFalse(self.targetClass.valid(self.record.to_dict()))

    @pytest.mark.unit
    def test_valid_with_rid(self):
        self.record.uid = self.uid
        self.record.rid = ObjectId(None)
        self.assertTrue(self.targetClass.valid(self.record.to_dict()))
        self.record.rid = 'garbage'
        self.assertFalse(self.targetClass.valid(self.record.to_dict()))

    @pytest.mark.unit
    def test_magics(self):
        self.assertIsInstance(repr(self.record), basestring)

    @pytest.mark.unit
    def test_export_to_dict(self):
        self.record.uid = self.uid
        record_dict = self.record.to_dict()
        self.assertIsInstance(record_dict, dict)
        json.dumps(record_dict)

    @pytest.mark.unit
    def test_rate_affects_cost(self):
        self.record.rate = 0.1
        self.record.time = 25
        cost1 = self.record.cost()
        self.record.rate = 0.2
        cost2 = self.record.cost()
        self.assertNotEqual(cost1, cost2)

    @pytest.mark.unit
    def test_base_affects_cost(self):
        self.record.rate = 0.1
        self.record.time = 25
        self.record.base = 0.5
        cost1 = self.record.cost()
        self.record.base = 1.0
        cost2 = self.record.cost()
        self.assertNotEqual(cost1, cost2)

    @pytest.mark.db
    def test_dont_save_invalid(self):
        self.record.uid = 'garbage'
        self.assertRaises(ValueError, self.record.save)

    @pytest.mark.db
    def test_save_valid(self):
        self.assertEqual(self.persistent.count(table='charges'), 0)
        self.record.uid = self.uid
        self.record.save()
        self.assertEqual(self.persistent.count(table='charges'), 1)
        self.record.save()
        self.assertEqual(self.persistent.count(table='charges'), 1)
        new_record = self.newRecord()
        new_record.uid = self.uid
        new_record.save()
        self.assertEqual(self.persistent.count(table='charges'), 2)
        self.record.save()
        self.assertEqual(self.persistent.count(table='charges'), 2)
        records = self.persistent.read(index=ObjectId(self.record.rid),
                                       result=[], table='charges')
        self.assertEqual(len(records), 1)
        self.assertNotIn('rid', records[0])
        self.assertIn('uid', records[0])

    @pytest.mark.db
    def test_auto_save(self):
        self.record.uid = self.uid
        self.assertIsNone(self.record.rid)
        self.assertEqual(self.persistent.count(table='charges'), 0)
        with self.record:
            # Complex operation!
            2+2
        self.assertEqual(self.persistent.count(table='charges'), 1)
        self.assertIsNotNone(self.record.rid)

    @pytest.mark.db
    def test_generate_from_context(self):
        self.record.uid = self.uid
        self.assertEqual(self.record.start, 0)
        self.assertEqual(self.record.time, 0)
        with self.record:
            # Complex operation!
            2+2
            # NOTE: This should be enough to get recorded with no slowness
            time.sleep(1e-3)
        self.assertGreater(self.record.start, 0)
        self.assertGreater(self.record.time, 0)
