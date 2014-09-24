####################################################################
#
#       Request tracker tests
#
#       Author: Peter Prettenhofer
#
#       Copyright (C) 2013 DataRobot Inc.
####################################################################
import time
import unittest

from datetime import datetime

from common.wrappers import DatabaseRequestLogger
from common.wrappers import track_request


class TrackRequestTest(unittest.TestCase):

    def test_database_request_logger(self):
        """Smoke test database request logger. """

        t0 = datetime.utcnow()
        t1 = datetime.utcnow()

        class MockDB(object):
            def create(this, values=None, table=None):
                assert values['start_time'] == t0
                assert values['end_time'] == t1
                assert values['pid'] == 1
                assert values['dataset_id'] == 'foobar'
                assert values['command'] == 'EDA'

        drl = DatabaseRequestLogger(MockDB())

        request = {'pid': 1, 'dataset_id': 'foobar',
                   'command': 'EDA'}
        drl('func_name', request, t0, t1)

    def test_track_request(self):
        """Test the request tracker decorator. """
        dummy_request = {'pid': 1, 'dataset_id': 'foobar',
                         'command': 'EDA'}

        def mock_handler(func_name, request, start_time, end_time):
            self.assertEqual(func_name, 'request_handler')
            self.assertEqual(request, dummy_request)
            self.assertGreaterEqual((end_time - start_time).total_seconds(), 0.1)

        @track_request(mock_handler)
        def request_handler(self, request):
            # so some hard work
            time.sleep(0.1)

        # now lets call it
        request_handler(None, dummy_request)

    def test_track_unvalid_request(self):
        """Request w/o dataset_id or pid should raise ValueError. """

        class MockDB(object):
            def create(this, values=None, table=None):
                assert False

        drl = DatabaseRequestLogger(MockDB())

        t0 = datetime.utcnow()
        t1 = datetime.utcnow()
        request = {'command': 'EDA'}

        self.assertRaises(ValueError, drl, 'func_name', request, t0, t1)
