import unittest
from mock import patch, Mock, call
import zmq
import config.test_config as config
import ModelingMachine.broker as broker

class ZmqFakeSocket(object):
    def __init__(self, socktype):
        self.opts = {}
        self.messages = []
    def bind(self, addr):
        pass
    def connect(self, host):
        pass
    def setsockopt(self, opt, *args):
        self.opts[opt] = args
    def send_multipart(self, message):
        self.messages.append(str(message))

class BrokerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        #broker.db = broker.redis.Redis(host=config.db_config['tempstore']['host'], port=config.db_config['tempstore']['port'])
        self.broker_context_patch = patch('ModelingMachine.broker.zmq.Context')
        self.broker_context_mock = self.broker_context_patch.start()
        self.broker_context_mock.return_value.socket = ZmqFakeSocket
        self.broker_poller_patch = patch('ModelingMachine.broker.zmq.Poller')
        self.broker_poller_mock = self.broker_poller_patch.start()

        self.broker_get_workers_patch = patch('ModelingMachine.broker.Broker.get_workers')
        self.broker_get_workers_mock = self.broker_get_workers_patch.start()

        self.broker_get_new_worker_id_patch = patch('ModelingMachine.broker.Broker.get_new_worker_id')
        self.broker_get_new_worker_id_mock = self.broker_get_new_worker_id_patch.start()
        self.broker_get_new_worker_id_mock.return_value = "2"

        self.broker_set_worker_online_patch = patch('ModelingMachine.broker.Broker.set_worker_online')
        self.broker_set_worker_online_mock = self.broker_set_worker_online_patch.start()

        self.broker_set_worker_offline_patch = patch('ModelingMachine.broker.Broker.set_worker_offline')
        self.broker_set_worker_offline_mock = self.broker_set_worker_offline_patch.start()

        self.worker_list = ["1"]

    @classmethod
    def tearDownClass(self):
        self.broker_context_patch.stop()
        self.broker_poller_patch.stop()
        self.broker_get_workers_patch.stop()
        self.broker_get_new_worker_id_patch.stop()
        self.broker_set_worker_online_patch.stop()
        self.broker_set_worker_offline_patch.stop()

    def setUp(self):
        self.broker_get_workers_mock.return_value = self.worker_list
        self.broker = broker.Broker(workers_db_key = 'test:workers')
        self.broker.workers.values()[0].address = "1"

    def test_init(self):
        self.assertItemsEqual(self.broker.workers.keys(), self.worker_list)

    def test_bind(self):
        self.assertIsNone(self.broker.bind("client_addr", "worker_addr"))

    def test_send_heartbeat(self):
        self.broker.send_heartbeat()
        self.assertEqual(self.broker.worker_socket.messages[0], str(["1", '', broker.Protocol.HEARTBEAT]))

    def test_purge(self):
        worker = self.broker.workers.values()[0]
        worker.expires = 0
        self.broker.purge()
        self.assertFalse(self.broker.workers)

    def test_parse_msg(self):
        msg = ["addr",'',"command","body"]
        self.assertItemsEqual(self.broker.parse_msg(msg), {"address": "addr", "command": "command", "body": ["body"]})

    def test_send_to_worker(self):
        worker = self.broker.workers.values()[0]

        self.broker.send_to_worker(worker, broker.Protocol.REQUEST, "hi")
        self.assertEqual(self.broker.worker_socket.messages[0], str([worker.address, '', broker.Protocol.REQUEST, "hi"]))

    def test_get_service(self):
        self.broker.get_service("service1")
        self.assertEqual(self.broker.services.values()[0].name, "service1")

    def test_get_worker_by_service(self):
        self.assertFalse(self.broker.get_worker_by_service("service_name"))
        self.broker.get_service("service_name")
        self.assertFalse(self.broker.get_worker_by_service("service_name"))

    def test_add_worker_services(self):
        worker = self.broker.workers.values()[0]
        self.broker.add_worker_services(worker, ["service1"])
        self.assertEqual(self.broker.services.values()[0].name, "service1")

    def test_process_client_msg(self):
        #invalid body
        msg = {"address": "addr", "command": "command", "body": []}
        self.assertFalse(self.broker.process_client_msg(msg))

        #invalid body
        msg = {"address": "addr", "command": "command", "body": [""]}
        self.assertFalse(self.broker.process_client_msg(msg))

        #valid body, service not available
        msg = {"address": "addr", "command": "command", "body": ["","service1"]}
        self.assertFalse(self.broker.process_client_msg(msg))

        worker = self.broker.workers.values()[0]
        #add a worker to the service, the queued request will be sent to it
        self.broker.add_worker_services(worker, ["service1"])
        #add a worker to the service again
        self.broker.add_worker_services(worker, ["service1"])

        #send another request to the service
        self.broker.worker_socket.messages = []
        msg = {"address": "addr", "command": "command", "body": ["","service1"]}
        self.assertTrue(self.broker.process_client_msg(msg))

        #check that the request was sent to the worker
        request = [worker.address, '', broker.Protocol.REQUEST] + [msg["address"], ''] + msg["body"]
        self.assertEqual(self.broker.worker_socket.messages[0], str(request))

        #send shutdown
        self.broker.worker_socket.messages = []
        msg = {"address": "addr", "command": broker.Protocol.SHUTDOWN, "body": ["1",""]}
        self.assertTrue(self.broker.process_client_msg(msg))
        request = [worker.address, '', broker.Protocol.SHUTDOWN]
        self.assertEqual(self.broker.worker_socket.messages[0], str(request))

    def test_proccess_client_ping(self):
        address = 'my_address'
        msg = {"address": address, "command": broker.Protocol.INTERNAL, "body": ["1", broker.Protocol.PING]}
        self.assertTrue(self.broker.process_client_msg(msg))
        self.assertEqual(self.broker.client_socket.messages[0], str([address, '', broker.Protocol.PONG]))

    def test_proccess_client_reset(self):
        address = 'my_address'
        msg = {"address": address, "command": broker.Protocol.INTERNAL, "body": ["1", broker.Protocol.RESET]}

        with patch.object(self.broker, 'remove_workers'):

            self.assertTrue(self.broker.process_client_msg(msg))

            self.assertFalse(self.broker.workers)
            self.assertFalse(self.broker.services)


    def test_process_client(self):
        #service not available
        msg = ["addr",'',"command","","service"]
        self.assertTrue(self.broker.process_client(msg))
        self.assertEqual(self.broker.client_socket.messages[0], str(["addr", '', 'false']))

        #worker doesn't exist
        msg = ["addr",'',"command","worker_id","service"]
        self.assertTrue(self.broker.process_client(msg))
        self.assertEqual(self.broker.client_socket.messages[1], str(["addr", '', 'false']))

        #send to specific worker
        msg = ["addr",'',"command","1","service"]
        self.assertTrue(self.broker.process_client(msg))
        self.assertEqual(self.broker.client_socket.messages[2], str(["addr", '', 'true']))
        self.assertEqual(self.broker.worker_socket.messages[0], str(["1", "", broker.Protocol.REQUEST, "addr", '', 'service']))

    def test_delete_worker(self):
        self.broker.delete_worker("1")
        self.assertFalse(self.broker.workers)

    def test_request_status(self):
        worker = self.broker.workers.values()[0]
        self.broker.request_status(worker)
        self.assertEqual(self.broker.worker_socket.messages[0], str(["1", "", broker.Protocol.STATUS]))

    def test_process_worker_assigns_id(self):
        #missing worker id
        msg = ["addr",'',"command"]
        self.assertTrue(self.broker.process_worker(msg))

    def test_process_worker(self):
        #disconnect the worker
        msg = ["addr",'',broker.Protocol.DISCONNECT,"1"]
        self.assertTrue(self.broker.process_worker(msg))
        self.assertFalse(self.broker.workers)

        #empty worker id
        msg = ["addr",'',"command", ""]
        self.assertTrue(self.broker.process_worker(msg))
        self.assertEqual(self.broker.worker_socket.messages[0], str(["addr", "", broker.Protocol.INITIALIZE, "2"]))

        #valid message, unknown command
        msg = ["addr",'',"command","worker_id"]
        self.assertTrue(self.broker.process_worker(msg))

        #valid message, status
        msg = ["addr",'',broker.Protocol.STATUS,"worker_id",'["service1"]']
        self.assertTrue(self.broker.process_worker(msg))
        self.assertEqual(self.broker.services.values()[0].workers[0].worker_id, "worker_id")

    def test_process_worker_shutdown(self):
        with patch('ModelingMachine.broker.Broker.remove_worker_services') as mock_rws:
            msg = ["addr",'',broker.Protocol.SHUTDOWN,"1"]
            self.assertTrue(self.broker.process_worker(msg))
            self.assertTrue(mock_rws.called)

    def test_remove_worker_services(self):
        worker = self.broker.workers.values()[0]
        self.broker.add_worker_services(worker, ["service1"])

        self.broker.remove_worker_services(worker)
        self.assertFalse(self.broker.services.values()[0].workers)

    def test_send_to_worker_id(self):
        self.broker.send_to_worker_id("1", broker.Protocol.HEARTBEAT)
        self.assertEqual(self.broker.worker_socket.messages[0], str(["1", "", broker.Protocol.HEARTBEAT]))

    def test_broadcast(self):
        service = 'manager'
        address = 'address'
        worker_list = [('1', 'address_1'), ('2', 'address_2'), ('3', 'address_3')]

        calls = []
        self.broker.workers = {}
        for worker_id, address in worker_list:
            self.broker.workers[worker_id] = address
            calls.append(call(address, broker.Protocol.REQUEST, ['', '', service]))

        self.assertTrue(len(self.broker.workers), len(worker_list))

        with patch.object(self.broker, 'send_to_worker') as mock_send_to_worker:

            msg = {'address': '', 'command': 'command', 'body': ['', service]}

            result = self.broker.process_client_msg(msg)

            self.assertTrue(result)

            mock_send_to_worker.assert_has_calls(calls, any_order = True)

if __name__ == '__main__':
    unittest.main()
