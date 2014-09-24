####################################################################
#
#       Modeling Machine Progress API Tests
#
#       Author: David Lapointe
#
#       Copyright (C) 2013 DataRobot Inc.
####################################################################

import unittest
from mock import patch, Mock
import zmq
from common.engine.progress import Progress, ProgressSink, ProgressState

class ZmqFakeSocket(object):
    def __init__(self, socktype):
        self.opts = {}
        self.messages = []
    def connect(self, host):
        pass
    def setsockopt(self, opt, *args):
        self.opts[opt] = args
    def send_multipart(self, message):
        self.messages.append(str(message))

class ProgressTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.progress_context_patch = patch('common.engine.progress.zmq')
        self.progress_context_mock = self.progress_context_patch.start()
        self.progress_context_mock.Context.return_value.socket = ZmqFakeSocket

    @classmethod
    def tearDownClass(self):
        self.progress_context_patch.stop()
    def setUp(self):
        self.progressInst = Progress(self.progress_context_mock.Context())
        self.progressSinkInst = ProgressSink()
    def test_types(self):
        self.assertIsInstance(self.progressInst, Progress)
        self.assertIsInstance(self.progressSinkInst, ProgressSink)
    def test_setids(self):
        #Neither of these should raise an Exception
        self.progressInst.set_ids("asdf", "blah")
        self.progressSinkInst.set_ids("asdf", "blah")
    def test_setpid(self):
        self.progressInst.set_ids("asdf")
        self.progressSinkInst.set_ids("asdf")
    def test_setprogress(self):
        self.progressInst.set_progress(message="asdf")
        self.progressSinkInst.set_progress(message="asdf")
    def test_sent_messages(self):
        def msglen(cnt):
            return self.assertEqual(len(self.progressInst.publisher.messages),
                cnt)
        msglen(0)
        self.progressInst.set_progress(message="asdf")
        msglen(1)
        self.progressInst.set_progress(message="blah")
        msglen(2)