import unittest
import os
import json
import time

import config.test_config as config
import logging

import MMApp.app
import MMApp.upload_server
from MMApp.entities.project import ProjectService
from MMApp.entities.jobqueue import QueueService
from MMApp.entities.user import UserService
from MMApp.entities.admin_tools import AdminTools
from ModelingMachine.client import BrokerClientBase
from common.engine.progress import ProgressSink
from common.wrappers import database
from common.entities.job import DataRobotJob


class IntegrationTestBase(unittest.TestCase):
    ''' Base class for integration tests:
    - Sets the databases to point to test "instances"
    - Contains generic methods to create registered and guest users, login, upload files, select target variable, query the queue and models
    '''

    def __init__(self, *args, **kw):
        super(IntegrationTestBase, self).__init__(*args, **kw)
        self.model_types = ('Generalized Linear Model (Bernoulli Distribution)',)

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new("tempstore")
        self.persistent = database.new("persistent")

        self.persistent.db_connection.drop_database(self.persistent.dbname)
        self.clear_tempstore_except_workers()

        self.logger = logging.getLogger("datarobot.IntegrationTest")

        self.logger.debug('Redis port: %s' % self.tempstore.conn.info()['tcp_port'])
        self.logger.debug('Mongo database name: %s' % self.persistent.dbname)

        self.app = MMApp.app.app.test_client()
        self.upload_server = MMApp.upload_server.app.test_client()
        self.create_test_accounts()

    @classmethod
    def clear_tempstore_except_workers(self):
        workers = set(self.tempstore.conn.smembers('workers'))
        secure_workers = set(self.tempstore.conn.smembers('secure-workers'))
        ide_workers = set(self.tempstore.conn.smembers('ide-workers'))
        self.tempstore.conn.flushdb()
        if workers:
            self.tempstore.create(keyname='workers', values=workers)
        if secure_workers:
            self.tempstore.create(keyname='secure-workers', values=secure_workers)
        if ide_workers:
            self.tempstore.create(keyname='ide-workers', values=ide_workers)

    @classmethod
    def wait_for_workers(self, workers_db_key):
        """
            Poll for workers to wake up - wait up to 10 seconds
        """
        timeout = time.time() + 10
        while True:
            n_workers = self.tempstore.conn.scard(workers_db_key)
            self.logger.info('Got redis scard resp: %s', n_workers)
            if n_workers > 0:
                break

            if time.time() > timeout:
                raise Exception('Workers did not come up - please check syslog')

            time.sleep(1)

        self.logger.info('Workers successfully started')

    @classmethod
    def tearDownClass(self):
        self.persistent.db_connection.drop_database(self.persistent.dbname)
        self.clear_tempstore_except_workers()

    def ping_worker(self, client_broker, worker_name):
        expected_token = 'token_worker'
        self.client = BrokerClientBase(client_broker)
        self.client.ping(expected_token)

        admin_tools = AdminTools()

        timeout = time.time() + 10
        while time.time() < timeout:
            actual_token = admin_tools.read_pong(sender = worker_name)
            if actual_token:
                break
            time.sleep(1)

        self.assertEqual(expected_token, actual_token, 'Worker {} did not respond on time'.format(worker_name))

    def set_dataset(self, file_name, target_variable, metric=None):
        self.test_file = self.path_to_test_file(file_name)
        self.new_test_file =  self.test_file
        self.new_test_file_name = file_name
        self.target_variable = target_variable
        self.metric = metric

    @classmethod
    def path_to_test_file(cls, file_name):
        tests = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(tests, 'testdata', file_name)

    def create_project(self):
        self.login_successfully(self.app)
        return self.upload_file_and_select_response(self.app)

    @classmethod
    def create_test_accounts(self):
        user_service = UserService('registered_user@example.com')
        password = '4HQujem1A$E12$79'
        user_service.create_account(password)
        self.registered_user = user_service.get_user_info()
        self.registered_user['password'] = password

        self.guest_user = self.create_guest('guest@datarobot.com', 'super-secret')

    @classmethod
    def create_guest(self, username, password):
        guest_user = {'username': username, 'password': password, 'guest' : 1}
        uid = self.persistent.create(table='users', values=guest_user)
        guest_user['uid'] = str(uid)
        self.tempstore.create(keyname='user', index=username, values=guest_user)
        return guest_user

    def create_pid(self):
        response = self.app.post('/project')
        self.assertEqual(response.status_code, 200, response.data)
        session_token = json.loads(response.data)
        pid = str(session_token['pid'])
        self.assertIsNotNone(pid)
        metrics = str(session_token['metrics'])
        self.assertIsNotNone(metrics)
        return pid

    def upload_file(self, app, is_guest, from_url):
        '''@brief Helper function that uploads a file either as a guest or registered user from either a file or an url.
        @retval Returns the project id (pid) '''
        pid = self.create_pid()
        # We need to know the username in order to put it in the session.
        # We know the registered username but for guests we query the
        # server to get the username

        if is_guest:
            profile = self.get_profile()
            username = profile['username']
        else:
            username = self.registered_user['username']

        with self.upload_server.session_transaction() as _session:
            _session['user'] = username

        if from_url:
            # Upload a file from url as a guest
            data = {
                'pid' : pid,
                'url' : 'https://s3.amazonaws.com/datarobot_test/kickcars-sample-200.csv'
            }
            response = self.upload_server.post('/upload/url', content_type='application/json', data=json.dumps(data))
        else:
            response = self.post_file(self.test_file, pid, False)

        self.assertEqual(response.status_code, 200, response.data)
        response_data = json.loads(response.data)
        self.assertIn('sc', response_data)
        self.assertEqual(response_data['sc'], '1' if is_guest else '0')

        self.assertEqual(response_data['username'], username)

        with self.upload_server.session_transaction() as _session:
            self.assertEqual(username, _session['user'])

        self.assertIsNotNone(pid)
        self.wait_for_eda_init(app, pid)

        return pid

    def post_file(self, test_file, pid, is_prediction):
        f = open(self.test_file)
        data = dict(file = f)
        return self.upload_server.post('/upload/{0}{1}'
            .format(pid, '?is_prediction=1' if is_prediction else ''), data=data)

    def upload_new_file(self, app, pid, new_test_file=None):
        #pid = self.upload_file(app, is_guest = False, from_url = False)
        #Upload new data
        # Act

        if new_test_file is None:
            new_test_file = self.new_test_file
        response = self.post_file(new_test_file, pid, True)

        self.assertEqual(response.status_code, 200)

        # New data list
        response = app.get('/project/%s/newdata' % pid)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        # print json.dumps(response_data, indent=4, sort_keys=True)
        self.assertIsInstance(response_data, list)
        self.assertGreater(len(response_data), 1)

        new_dataset = [rd for rd in response_data if rd['originalName'] == self.new_test_file][0]
        self.assertIsInstance(new_dataset, dict)
        self.assertTrue('originalName' in new_dataset)
        self.assertTrue('dataset_id' in new_dataset)
        self.assertTrue('created' in new_dataset)
        self.assertTrue(new_dataset.get('newdata'))
        self.assertTrue('computed' in new_dataset)

        return new_dataset['dataset_id']

    def get_profile(self):
        response = self.app.get('/account/profile')
        self.assertEqual(response.status_code, 200)
        return json.loads(response.data)

    def upload_file_and_select_response(self, app):
        '''@brief Helper function that calls upload_file to submit a file and then executes a post on /aim in order to select a target variable
        @retval Returns the dataset id (pid) '''
        pid = self.upload_file(app, is_guest = False, from_url = False)
        target = self.target_variable
        mode = 1
        metric = (getattr(self, 'project_metric', None) or
                  config.EngConfig['DEFAULT_METRIC'])

        # Request univariates
        data = {'pid': pid, 'target':target, 'mode': mode,
                'metric': metric}
        response =  app.post('/aim', content_type='application/json', data = json.dumps(data))

        self.assertEqual(response.status_code, 200)
        return pid

    def get_q(self, pid, include_settings = False):
        response = self.app.get('/project/'+str(pid)+'/queue')
        self.assertEqual(response.status_code, 200, response.data)
        q_items = json.loads(response.data)
        if not include_settings:
            settings_index = next(i for (i,item) in enumerate(q_items) if item['qid'] == -1)
            settings_item = q_items.pop(settings_index) #Remove settings item
        return q_items

    def get_models(self, app, pid):
        response = app.get('/project/%s/models' % pid)
        self.assertEqual(response.status_code, 200)
        return json.loads(response.data)

    def get_model(self, pid, lid):
        response = self.app.get('/project/{}/models/{}'.format(pid, lid))
        self.assertEqual(response.status_code, 200)
        return json.loads(response.data)

    def create_model(self, pid, model):
        response = self.app.post('/project/{}/models'.format(pid),
            content_type='application/json', data=json.dumps(model))
        self.assertEqual(response.status_code, 200)
        return json.loads(response.data)

    def update_queue_settings(self, app, pid, parallel):
        data = {'workers' : parallel}
        response = app.put('/project/%s/queue/%s' % (pid,'-1'), content_type='application/json', data=json.dumps(data))

        response_data = json.loads(response.data)
        if parallel > 0: #is valid?
            self.assertEqual(response.status_code, 200)
            self.assertIn(response_data['message'], ["modified", "unmodified"])
        else:
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response_data['message'], 'Bad request')


    def login(self, username, password):
        data  = {'username':username, 'password':password}
        # See werkzeug.test.EnvironBuilder for other parameters to app.post
        return self.app.post('/account/login', content_type='application/json', data=json.dumps(data))

    def login_successfully(self, app):
        response = self.login(self.registered_user['username'], self.registered_user['password'])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/json')
        response_data = json.loads(response.data)
        self.assertEqual(response_data['message'], 'Log In Successful', response_data)
        self.assertEqual(response_data['uid'], self.registered_user['uid'])
        self.assertEqual(response_data['error'], '0')
        with app.session_transaction() as sess:
            self.assertTrue('user' in sess)

    def logout(self, app):
        response = app.get('/account/logout')
        self.assertEqual(response.status_code, 302)

    def wait_for_stage(self, app, pid, stages, timeout=20):
        timeout = time.time()+20

        def stage_matches(actual_stage):
            if not actual_stage:
                return False
            if isinstance(stages, list):
                for stage in stages:
                    if stage in actual_stage:
                        return True
            else:
                if stages in actual_stage:
                    return True

        stage_actual = None
        while not stage_matches(stage_actual):
            response = app.get('/project/%s/stage' % pid)
            data = json.loads(response.data)
            stage_actual = data.get('stage')
            if time.time() > timeout:
                self.fail('Stage did not enter: %s (Last:%s)' %
                          (str(stages), str(stage_actual)))
            time.sleep(1)

    def wait_for_eda_init(self, app, pid):
        timeout = time.time()+20

        while time.time() < timeout:
            response = app.get('/eda/%s' % pid)
            if response.status_code == 200:
                break

    def set_q_with_n_items(self, pid, n, app, model_types=None):
        '''
            Removes items from the queue until it has n items left.
            If model_types is set, it will ignore and leave those items in the queue
        '''
        if model_types is None:
            model_types = self.model_types

        q_items = self.get_q(pid)
        if not q_items:
            return

        # Make sure nothing has started
        not_started = [i for (i,item) in enumerate(q_items) if item['status'] == 'queue']
        in_progress = [i for (i,item) in enumerate(q_items) if item['status'] == 'inprogress']

        self.assertEqual(len(in_progress), 0)
        self.assertEqual(len(q_items), len(not_started))

        # Compare to the results returned by the web handler with the QueueService
        q = QueueService(pid, ProgressSink(), self.registered_user['uid'])
        service_q_items = q.get()
        service_not_started = [item for (i,item) in enumerate(service_q_items)
                               if item['status'] == 'queue']
        service_in_progress = [item for (i,item) in enumerate(service_q_items)
                               if item['status'] == 'inprogress']

        self.assertEqual(len(service_not_started), len(not_started))
        self.assertEqual(len(service_in_progress), len(in_progress))

        # Remove all using the q service directly, except for the first, this reduces time greatly
        # Leave 2 items on the q, 1 item won't work, this is a known issue
        service_not_started.reverse()
        leave = 0
        for q_item in service_not_started:
            if leave == n:  # n could be 0
                break
            if q_item['model_type'] in model_types:
                service_not_started.remove(q_item)
                leave += 1
            else:
                self.logger.info('removing queue item: %r', q_item['model_type'])

        self.assertEqual(leave, n, 'Not enough models were selected in set_q_with_n_items. ' +
            'model_types ({}) might not be valid for '
            'this dataset.'.format(model_types))

        for item in service_not_started:
            removed = q.remove(item['qid'])
            self.assertIsNotNone(removed)

        # Query the queue, we should have only one job on the queue
        q_items = q.get()
        # Make sure nothing has started
        not_started = [item for (i,item) in enumerate(q_items) if item['status'] == 'queue']
        in_progress = [item for (i,item) in enumerate(q_items) if item['status'] == 'inprogress']

        self.assertEqual(len(in_progress), 0)
        self.assertEqual(len(q_items), len(not_started)+1) #+1 for queue settings

        return not_started

    def set_q_with_simple_blueprint(self, pid, newsamplepct=None):
        not_started = self.set_q_with_n_items(pid, 1, self.app)
        sample_q_item = not_started[0]

        q_item = {}
        q_item.update(sample_q_item)
        q_item['blueprint'] = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GLMB'],'P']}
        if newsamplepct:
            q_item['samplepct'] = newsamplepct

        self.set_q_with_n_items(pid, 0, self.app)

        not_started = self.add_request_to_queue(pid, [q_item])
        new_simple_q_item = not_started[0]

        return new_simple_q_item

    def add_request_to_queue(self, pid, request, lid = None):

        q = QueueService(pid, ProgressSink(), self.registered_user['uid'])
        if isinstance(request,dict):
            request = [request]

        #FIXME: this is not how jobs are really added to queue: should use queue.put() instead
        for i in request:
            if 'lid' in i:
                i['_id'] = i.pop('lid')
            q.add(*DataRobotJob(i).to_joblist())

        # Make sure the queue got our request
        q_items = q.get()
        not_started = [item for (i,item) in enumerate(q_items) if item['status'] == 'queue']
        in_progress = [item for (i,item) in enumerate(q_items) if item['status'] == 'inprogress']

        self.assertTrue(not_started, 'The queue did not get our request (not-started queue is empty)')

        return not_started

    def service_queue(self, pid):
        response = self.app.get('/project/%s/service' % pid)
        self.assertEqual(response.status_code, 200)
        service_data = json.loads(response.data)
        self.assertGreaterEqual(service_data['count'], 1)

    def wait_for_q_item_to_complete(self, app, pid, qid, timeout, catch_error=True):
        '''Polls the app for qid to finish
        if qid is None it will wait until any job finishes and return the qid
        '''

        def find_q_item_in_db():
            project_service = ProjectService(pid, self.registered_user['uid'])
            models = project_service.get_leaderboard(done_only=True)
            if models:
                if qid is None:
                    db_qid = int(models[0]['qid'])
                    self.logger.debug('Returning first qid {} from database'.format(qid))
                    return db_qid
                elif filter(lambda m: m['qid'] == str(qid), models):
                    self.logger.debug('{} found in database'.format(qid))
                    return qid

        self.logger.debug('Waiting for %s q item to complete' % (qid if qid else 'first'))
        timeout = time.time() + timeout
        inprogress_cache = []

        queue = QueueService(pid, ProgressSink(), self.registered_user['uid'])

        while True:
            response = app.get('/project/%s/service' % pid)
            self.assertEqual(response.status_code, 200)
            q_items = queue.get()
            # This checks for the case where a fast server completes the q item before we even pull the queue
            if not q_items or (qid and filter(lambda q: q['qid'] == str(qid), q_items)):
                db_qid = find_q_item_in_db()
                if db_qid:
                    return int(db_qid)

            inprogress = []
            waiting = False
            for item in q_items:
                if qid is None:
                    if item['status'] == "inprogress":
                        inprogress.append(item["qid"])
                else:
                    if int(item['qid']) == int(qid):
                        if item['status'] == 'error':
                            if catch_error:
                                return qid
                            else:
                                self.fail('Queue item errored out')
                        waiting = True
            if qid is not None and not waiting:
                return qid
            for itemqid in inprogress_cache:
                if itemqid not in inprogress: # This qid completed
                    if not qid: # Any qid is fine
                        return itemqid
            inprogress_cache = inprogress

            self.assertLess(time.time(), timeout, 'timed out. qid: {}'.format(qid))
            time.sleep(2)

    def get_predictions(self, pid, lid, dataset_id):
        project_service = ProjectService(pid, self.registered_user['uid'])
        return project_service.get_predictions(lid, dataset_id)

    def execute_fit(self, pid):
        self.wait_for_stage(self.app, pid, ['post-aim', 'eda2', 'modeling'])

        new_q_item = self.set_q_with_simple_blueprint(pid)

        self.service_queue(pid)

        # Make sure our q item finishes
        self.wait_for_q_item_to_complete(self.app, pid = pid, qid = new_q_item['qid'], timeout = 30)

        # Get models
        leaderboard = self.get_models(self.app, pid)
        # print json.dumps(leaderboard, indent = 4, sort_keys = True)
        self.assertTrue(leaderboard, 'The server did not return any models')

        return new_q_item, leaderboard[0]
