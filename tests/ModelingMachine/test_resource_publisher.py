import unittest
from mock import patch, Mock
import common.resource_publisher
import zmq

class ZmqFakeSocket(object):
    def __init__(self, socktype):
        self.opts = {}
        self.messages = []
        self.recv_calls = 0
    def connect(self, host):
        pass
    def bind(self, address):
        pass
    def setsockopt(self, opt, *args):
        self.opts[opt] = args
    def send_multipart(self, message):
        self.messages.append(str(message))
    def recv_multipart(self, *args):
        self.recv_calls += 1
        if self.recv_calls % 2 == 0:
            raise zmq.ZMQError
        #client,empty,id1,id2,id3,command
        return ['1', '', 'pid', 'qid', 'id', str(self.recv_calls), '1']

class fake_sleep(object):
    def __init__(self, max_calls=0):
        self.calls = 0
        self.max_calls = max_calls
    def sleep(self, seconds):
        self.calls += 1
        if self.calls > self.max_calls:
            raise Exception('sleep')

class ResourcePublisherTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.zmq_patch = patch('common.resource_publisher.zmq')
        self.zmq_mock = self.zmq_patch.start()
        self.zmq_mock.ZMQError = zmq.ZMQError
        self.zmq_mock.Context.return_value.socket = ZmqFakeSocket

        self.psutil_patch = patch('common.resource_publisher.psutil')
        self.psutil_mock = self.psutil_patch.start()

        self.time_patch = patch('common.resource_publisher.time')
        self.time_mock = self.time_patch.start()

        self.gevent_patch = patch('common.resource_publisher.gevent')
        self.gevent_mock = self.gevent_patch.start()

        self.statsd_patch = patch('common.resource_publisher.drstatsd')
        self.statsd_mock = self.statsd_patch.start()

    @classmethod
    def tearDownClass(self):
        self.zmq_patch.stop()
        self.psutil_patch.stop()
        self.time_patch.stop()
        self.statsd_patch.stop()

    def setUp(self):
        #needed to exit the infinite resource publisher loop
        self.time_mock.sleep = fake_sleep().sleep

    def test_resource_publish(self):
        #allow sleep to be called once before exiting
        self.time_mock.sleep = fake_sleep(1).sleep
        self.gevent_mock.sleep.side_effect = Exception()
        self.psutil_mock.virtual_memory.return_value.percent = 100
        rp = common.resource_publisher.ResourceMonitor("resource_ipc")
        rp.add_process('1')
        rp.start_publishing('1', '1', '1')
        self.assertRaises(Exception, rp.run)
