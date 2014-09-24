import unittest

from bson.objectid import ObjectId
import pytest

import config.test_config
from common.engine.billing import ChargeRecord
from common.wrappers import database
from ModelingMachine.engine.billing import SecureResourceChargeRecord
from ModelingMachine.engine.worker_request import WorkerRequest

from tests.common.abstract_test_billing import AbstractTestCharges


def complex_operation():
    return 2+2


class TestSecureCharges(AbstractTestCharges, unittest.TestCase):
    targetClass = SecureResourceChargeRecord

    def setUp(self):
        self.uid = ObjectId()
        self.pid = ObjectId()
        raw_request = {'uid': self.uid, 'pid': self.pid, 'command': 'ping',
                       'instance_id': None}
        self.request = WorkerRequest(raw_request)
        self.request.instance_id = raw_request['instance_id']
        self.record = self.newRecord()

        self.persistent = database.new('persistent')
        self.persistent.destroy(table='charges')

    def newRecord(self):
            return SecureResourceChargeRecord(self.request,
                                              complex_operation)

    @pytest.mark.unit
    def test_invalid_alone(self):
        self.assertFalse(self.targetClass.valid({}))
