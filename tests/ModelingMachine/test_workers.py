import unittest
from mock import patch, Mock, DEFAULT
import zmq
import json
import config.test_config as config
import common.broker.workers as workers
import time
from collections import OrderedDict
from common.broker.workers import WorkerShutdown
from ModelingMachine.engine.vertex_factory import VertexCache, ModelCache
from config.engine import EngConfig


class ZmqFakeSocket(object):
    def __init__(self, socktype):
        self.opts = {}
        self.messages = []
        self.recv_message = ['', workers.Protocol.REQUEST, 'client_address', '', 'service_name', '"request"']
    def bind(self, addr):
        pass
    def connect(self, host):
        pass
    def close(self):
        pass
    def setsockopt(self, opt, *args):
        self.opts[opt] = args
    def send_multipart(self, message):
        self.messages.append(str(message))
    def recv_multipart(self):
        return self.recv_message

class WorkersTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.workers_context_patch = patch('common.broker.workers.zmq.Context')
        self.workers_context_mock = self.workers_context_patch.start()
        self.workers_context_mock.return_value.socket = ZmqFakeSocket
        self.workers_poller_patch = patch('common.broker.workers.zmq.Poller')
        self.workers_poller_mock = self.workers_poller_patch.start()

        self.workers_get_id_patch = patch('common.broker.workers.Workers.get_id')
        self.workers_get_id_mock = self.workers_get_id_patch.start()
        self.workers_get_id_mock.return_value = "1"

        self.workers_hb_patch = patch('common.broker.workers.Workers.send_heartbeats')
        self.workers_hb_mock = self.workers_hb_patch.start()

        self.workers_cp_patch = patch('common.broker.workers.Workers.check_pipes')
        self.workers_cp_mock = self.workers_cp_patch.start()

        self.workers_add_services_patch = patch('common.broker.workers.Workers.add_services')
        self.workers_add_services_mock = self.workers_add_services_patch.start()

        self.workers_register_patch = patch('common.broker.workers.Workers.register')
        self.workers_register_mock = self.workers_register_patch.start()

        self.workers_wp_patch = patch('common.broker.workers.WorkerProcess')
        self.workers_wp_mock = self.workers_wp_patch.start()
        self.workers_wp_mock.side_effect = Exception('WorkerProcess disabled')

    @classmethod
    def tearDownClass(self):
        self.workers_context_patch.stop()
        self.workers_poller_patch.stop()
        self.workers_get_id_patch.stop()
        self.workers_hb_patch.stop()
        self.workers_cp_patch.stop()
        self.workers_add_services_patch.stop()
        self.workers_register_patch.stop()
        self.workers_wp_patch.stop()

    def setUp(self):
        self.workers = workers.Workers(broker = None)
        self.workers.services = []
        self.workers.reconnect_to_broker()

    def test_send_services(self):
        self.workers.send_services()
        #message body "[]" is the empty json list of services
        self.assertEqual(self.workers.worker_socket.messages[-1], str(['', workers.Protocol.STATUS, "1", "[]"]))

    def test_process_msg(self):
        self.assertIsNone(self.workers.process_msg(workers.Protocol.HEARTBEAT, [""]))

        self.assertRaises(WorkerShutdown, self.workers.process_msg, workers.Protocol.SHUTDOWN, [""])
        self.assertEqual(self.workers.worker_socket.messages[-1], str(['', workers.Protocol.SHUTDOWN, "1"]))

        self.assertIsNone(self.workers.process_msg(workers.Protocol.STATUS, [""]))
        #message body "[]" is the empty json list of services
        self.assertEqual(self.workers.worker_socket.messages[-1], str(['', workers.Protocol.STATUS, "1", "[]"]))

        #this creates a new worker_socket
        self.assertIsNone(self.workers.process_msg(workers.Protocol.DISCONNECT, [""]))

        self.assertIsNone(self.workers.process_msg(workers.Protocol.INITIALIZE, [""]))
        self.assertEqual(self.workers.worker_socket.messages[-1], str(['', workers.Protocol.DISCONNECT, ""]))

        self.assertItemsEqual(self.workers.process_msg(workers.Protocol.REQUEST, ["client_address", "", "body"]), ["body"])

    @patch('common.broker.workers.FLIPPERS', autospec=True)
    def test_add_service(self, mock_flippers):
        mock_flippers.request_accounting = False
        self.workers.add_service("service_name")
        self.assertItemsEqual(self.workers.services, [{"name": "service_name", "request": None}])

    @patch('common.broker.workers.FLIPPERS', autospec=True)
    def test_add_service_with_flipper_on(self, mock_flippers):
        mock_flippers.request_accounting = True
        self.workers.add_service("service_name")
        self.assertItemsEqual(self.workers.services,
                [{"name": "service_name", "request": None, 'request_id': None}])

    @patch('common.broker.workers.FLIPPERS', autospec=True)
    def test_assign_request(self, mock_flippers):
        mock_flippers.request_accounting = False
        self.workers.add_service("service_name")
        self.assertTrue(self.workers.assign_request("service_name", {'qid': '1'}))
        self.assertItemsEqual(self.workers.services, [{"name": "service_name", "request": {'qid': '1'}}])

        #the one service is already occupied by a request
        self.assertFalse(self.workers.assign_request("service_name", {'qid': '1'}))
        self.assertItemsEqual(self.workers.services, [{"name": "service_name", "request": {'qid': '1'}}])

        #service 'manager' is always accepted
        self.assertTrue(self.workers.assign_request("manager", {'qid': '1'}))
        self.assertItemsEqual(self.workers.services, [{"name": "service_name", "request": {'qid': '1'}}])

    @patch('common.broker.workers.FLIPPERS', autospec=True)
    def test_clear_request(self, mock_flippers):
        mock_flippers.request_accounting = False
        self.workers.add_service("service_name")
        self.workers.assign_request("service_name", {'qid': '1'})
        self.workers.clear_request("service_name", {'qid': '1'})
        self.assertItemsEqual(self.workers.services, [{"name": "service_name", "request": None}])
        self.assertEqual(self.workers.worker_socket.messages[0], str(['', workers.Protocol.STATUS, "1", '["service_name"]']))
        self.assertTrue(self.workers.clear_request("manager", ""))

    @patch('common.broker.workers.FLIPPERS', autospec=True)
    def test_clear_request_with_flipper_on(self, mock_flippers):
        mock_flippers.request_accounting = True
        self.workers.add_service("service_name")
        self.workers.assign_request("service_name", {'qid': '1'})
        self.workers.clear_request("service_name", {'qid': '1'})
        self.assertItemsEqual(self.workers.services, [
            {"name": "service_name", "request": None, 'request_id': None}])
        self.assertEqual(self.workers.worker_socket.messages[0], str(['', workers.Protocol.STATUS, "1", '["service_name"]']))
        self.assertTrue(self.workers.clear_request("manager", ""))

    def test_cleanup_processes(self):
        with patch('common.broker.workers.sys', autospec=True) as mock_sys:
            wp = Mock()
            wp.is_alive.return_value = False
            wp.service = 'service1'
            wp.request = 'request1'
            wp2 = Mock()
            wp2.is_alive.return_value = True
            self.workers.worker_processes = [wp, wp2]
            with patch.object(self.workers, "clear_request") as mock_clear_request:
                self.workers.cleanup_processes()
                mock_clear_request.assert_called_once_with('service1', 'request1')

            self.assertItemsEqual(self.workers.worker_processes, [wp2])

    def test_poll_socket(self):
        self.workers.poller.poll.return_value = False
        self.assertIsNone(self.workers.poll_socket())

        self.workers.poller.poll.return_value = True
        self.assertEqual(self.workers.poll_socket(), self.workers.worker_socket.recv_message)

    def test_process_request(self):
        self.workers.poller.poll.return_value = True
        self.assertIsNone(self.workers.process_request())

    def test_process_manager_request_to_kill(self):
        service = 'manager'
        request =[service, '{"command": "kill"}']
        with patch.multiple(self.workers, wait_for_request = DEFAULT, kill_worker_by_request = DEFAULT, run_request=DEFAULT) as mocks:
            mocks['wait_for_request'].return_value = request

            result = self.workers.process_request()

            self.assertIsNone(result)
            mocks['kill_worker_by_request'].assert_called_once_with({'command': 'kill'})

    def test_process_manager_request_to_broadcast(self):
        service = 'manager'
        self.workers.add_service('service_name')
        self.workers.add_service('manager')

        request =[service, '{"command": "broadcast_command"}']
        with patch.multiple(self.workers, wait_for_request = DEFAULT, add_worker = DEFAULT) as mocks:
            mocks['wait_for_request'].return_value = request

            result = self.workers.process_request()

            self.assertIsNone(result)
            mocks['add_worker'].assert_called_once_with({'command': 'broadcast_command'}, None)

    def test_process_predict_request_with_cache(self):
        service = 'fit_single'

        self.workers.add_service('service_name')
        self.workers.add_service('fit_single')
        self.workers.model_cache.has_model_cache = True

        req = {'command':'predict_whatever', 'pid':'1234', 'blueprint_id':'1234', 'dataset_id':'1234', 'samplepct':'50', 'partitions':[[-1,-1]]}
        request =[service, json.dumps(req)]
        with patch.multiple(self.workers, wait_for_request = DEFAULT, add_worker = DEFAULT) as mocks:
            mocks['wait_for_request'].return_value = request
            self.workers.model_cache.get_cached_model = Mock()
            self.workers.model_cache.get_cached_model.return_value = 'test_cache'

            result = self.workers.process_request()

            self.assertIsNone(result)

            mocks['add_worker'].assert_called_once_with(req, 'test_cache')

    def test_process_predict_request_without_cache(self):
        service = 'fit_single'

        self.workers.add_service('service_name')
        self.workers.add_service('fit_single')

        self.workers.model_cache.has_model_cache = False

        req = {'command':'predict_whatever', 'pid':'1234', 'blueprint_id':'1234', 'dataset_id':'1234', 'samplepct':'50', 'partitions':[[-1,-1]]}
        request =[service, json.dumps(req)]
        with patch.multiple(self.workers, wait_for_request = DEFAULT, add_worker = DEFAULT) as mocks:
            mocks['wait_for_request'].return_value = request
            self.workers.model_cache.get_cached_model = Mock()
            self.workers.model_cache.get_cached_model.return_value = 'test_cache'

            result = self.workers.process_request()

            self.assertIsNone(result)

            mocks['add_worker'].assert_called_once_with(req, None)


    def test_get_cached_model(self):
        self.workers.model_cache.has_model_cache = 3

        req = {'command':'predict_whatever', 'pid':'1234', 'blueprint_id':'1234', 'dataset_id':'1234', 'samplepct':'50', 'partitions':[[-1,-1]]}

        #test get new model
        out = self.workers.model_cache.get_cached_model(req)
        self.assertIsInstance(out, VertexCache)
        self.assertEqual(OrderedDict(), self.workers.model_cache.cached_models)

        #test update
        self.workers.model_cache.update_cached_model(out,req)
        self.assertEqual(out, self.workers.model_cache.cached_models.values()[0])

        #test get existing model
        out2 = self.workers.model_cache.get_cached_model(req)
        self.assertEqual(out, out2)

    def test_shutdown(self):
        with patch.multiple(self.workers, try_run_once_at_shutdown=DEFAULT,
                            current_requests=DEFAULT) as mocks:
            self.workers.stop = True
            self.workers.stop_time = time.time()
            self.workers.worker_processes = [1]
            mocks['current_requests'].return_value = [{'pid': 'pid', 'uid': 'uid'}]
            self.assertFalse(self.workers.shutdown())

            self.workers.stop = False
            self.assertFalse(self.workers.shutdown())


if __name__ == '__main__':
    unittest.main()
