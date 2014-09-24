###############################################################################
#
# Test MetaBlueprint on AWS
#
###############################################################################
import os
import sys
import requests
import json
import logging
import time
import gevent
import random

from urlparse import urljoin
from collections import namedtuple
from itertools import tee, izip, islice
from mbtest_database_service import MBTestDatabaseService
from common.utilities.fn_utils import retry

QueueStatus = namedtuple('QueueStatus', 'mode, n_queue_items')

nwise = lambda g, n=2: izip(*(islice(g, i, None) for i, g in enumerate(tee(g, n))))
RETRIES = 5
BACKOFF = 5


class Stage(object):
    PROJECT_CREATED = 'PROJECT_CREATED'
    FILE_UPLOADED = 'FILE_UPLOADED'
    TARGET_SELECTED = 'TARGET_SELECTED'
    TESTING = 'TESTING'


class Status(object):
    IN_PROCESS = 'IN_PROCESS'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'


def log_failure(tries_remaining, exception, delay, fn_name):
    logging.error('Caught exception in {} (retrying with {} attempts remaining): "{}"'.format(fn_name, tries_remaining -1 , exception))


class MBTest(object):
    """Run a metablueprint test using a file, target, and metric.

    This script will make all necessary steps to run a dataset on the DR app:
    1. user signup
    2. dataset upload
    3. run aim
    4. poll queue and call next_steps until autopilot complete.

    Assumes that app server is available on ``host``.

    """
    def __init__(self, host='localhost', sleep_period=1, max_periods=240,
                 enable_test_case_updates=False, credentials=None):
        self.host = host
        self.base_url = 'http://'+host
        self.headers = {'content-type': 'application/json'}
        self.cookies = None
        self.sleep_period = sleep_period
        self.max_periods = max_periods
        self.enable_test_case_updates = enable_test_case_updates
        self.credentials = credentials
        self.last_log_update = 0

    @retry(RETRIES, backoff = BACKOFF, hook = log_failure)
    def login(self):
        '''Login with the credentials provided and save the cookie for future requests
        '''
        username, password = self.credentials
        data  = {'username':username, 'password':password}
        url = urljoin(self.base_url, '/account/login')
        r = requests.post(url, headers=self.headers, data=json.dumps(data), auth=('user','pass'))

        if r.status_code != 200:
            raise RuntimeError('Could not login with account {} {}. Error: {}'.format(username, password, r.text))

        self.cookies = r.cookies

    def create_account(self):
        '''Create a random user account and log in, used by local tests
        '''
        username = ''.join(random.choice('0123456789QWERTYUIOPASDFGHJKLZXCVBNM')
                           for i in range(16))+'@TEST.COM'
        self.username = username
        password = 'pass1234'
        url = urljoin(self.base_url, 'account/signup')
        data = {'username':username,'password':password}
        r = requests.post(url, data=json.dumps(data), headers=self.headers, auth=("user","pass"))
        if r.status_code != 200:
            raise RuntimeError('Could not create acount, is PUBLIC mode enabled?.\n\nUrl: {}\n\Error:{}'.format(url, r.text))

        self.cookies = r.cookies
        logging.info('Test Server URL: \n\n    %s \n'%self.base_url)
        logging.info('Test Account Created:\n\n   USERNAME = %s \n   PASSWORD = %s \n\n' %
                     (username,password))

    def select_target(self, pid, target, metric, options=None, mode=1):
        """Trigger AIM request.

        We set ``target``, ``metric`` and optional ``options`` which allow us
        to use advanced options (partitioning, recommenders, ...).
        """
        self.wait_for_stage(pid, 'aim', sleep_period = self.sleep_period)
        self._select_target(pid, target, metric, options)
        self.update_test_case({'stage': Stage.TARGET_SELECTED})
        self.wait_for_stage(pid, 'modeling', sleep_period = self.sleep_period)
        self.update_test_case({'stage': Stage.TESTING})

    def create_project(self, worker_options):
        url = urljoin(self.base_url, '/project')

        if self.cookies:
            r = requests.post(url, cookies=self.cookies, auth=("user","pass"),
                data = json.dumps(worker_options), headers = self.headers)
        else:
            r = requests.post(url, auth=("user","pass"), data = json.dumps(worker_options),
                headers = self.headers)

        if r.status_code == 200:
            pid = r.json().get('pid')
            if pid:
                logging.info('\n\n   PID - {} = {}\n\n'.format(self.ds_log_id, pid))
                return pid

        raise RuntimeError("Could not create project: " + r.text)

    @retry(RETRIES, backoff = BACKOFF, hook = log_failure)
    def upload_file(self, dataset_path, pid):
        """uploads a file to start a new project and return the pid"""
        url = urljoin(self.base_url, '/upload/{0}'.format(pid))
        data = {'file': open(dataset_path, 'rb')}
        if self.cookies:
            r = requests.post(url, files=data, cookies=self.cookies, auth=("user","pass"))
        else:
            r = requests.post(url, files=data, auth=("user","pass"))
            self.cookies = {'session':r.cookies.get('session')}

        self.check_response(pid, r)
        self.dataset_path = dataset_path
        logging.info('File Upload Successful', extra = {'pid': pid})
        return pid

    def _update_aim_request(self, data, options=None):
        if options:
            data.update(options)

    @retry(RETRIES, backoff = BACKOFF, hook = log_failure)
    def _select_target(self, pid, target, metric, options=None, mode=1):
        """selects the target variable and mode for a new project """
        url = urljoin(self.base_url, 'aim')
        data = {'pid':pid, 'target':target, 'mode':mode, 'metric': metric}
        self._update_aim_request(data, options)
        r = requests.post(url, data=json.dumps(data), headers=self.headers, cookies=self.cookies,
                          auth=("user","pass"))
        self.check_response(pid, r)

    @retry(RETRIES, backoff = BACKOFF, hook = log_failure)
    def wait_for_stage(self, pid, stage, wait_minutes=15, sleep_period=1):
        url = urljoin(self.base_url, 'project/'+pid+'/stage')
        timeout = time.time() + (60 * wait_minutes)
        while True:
            if time.time() > timeout:
                raise RuntimeError('Waiting for stage {} timed out after {} retries of {} minutes: {}'.format(
                    stage, RETRIES, wait_minutes, self.ds_log_id))

            r = requests.get(url, cookies=self.cookies, auth=("user","pass"))
            self.check_response(pid, r)
            stage_response = r.json().get('stage')
            logging.info('Stage {}:{}'.format(stage_response, self.ds_log_id))
            if stage_response  == stage:
                break
            gevent.sleep(sleep_period)

    def check_response(self, pid, r):
        gevent.sleep(0)
        if r.status_code != 200:
            try:
                msg = r.json()
            except ValueError:
                msg = {}

            if r.status_code == 400 and msg.get('status') == 401:
                logging.warn('Session expired for {}. Logging in again'.format(pid, r.text))
                self.login()
                return
            raise RuntimeError('Unexpected response for {}: {}\n\n{}'.format(pid, r.url, r.text))

    @retry(RETRIES, backoff = BACKOFF, hook = log_failure)
    def service_queue(self, pid):
        url = urljoin(self.base_url, 'project/' + pid + '/service')
        r = requests.get(url, cookies=self.cookies, auth=('user','pass'))
        self.check_response(pid, r)
        tasks_started = r.json().get('count')
        if tasks_started:
            logging.info('Servicing queue started {} tasks: {}'.format(tasks_started, self.ds_log_id))

    @retry(RETRIES, backoff = BACKOFF, hook = log_failure)
    def get_queue(self, pid):
        url = urljoin(self.base_url, 'project/'+pid+'/queue')
        r = requests.get(url, cookies=self.cookies, auth=("user","pass"))
        self.check_response(pid, r)
        # queue contains ``n_queue_items`` + 1 items - first item is the queue settings
        queue = r.json()
        if len(queue) == 0:
            raise RuntimeError('Expected queue settings ')
        return queue

    def poll_queue(self, pid, sleep_period=1):
        """Generator that polls the queue and returns the number of items.

        Sleeps for ``sleep_period`` seconds after each request.

        Yields
        ------
        queue_status : QueueStatus
            A status obj that gives the mode and number of items in the queue.
            The number of items comprises inprogress and queued items.
        """
        while True:
            self.service_queue(pid)
            queue = self.get_queue(pid)
            # get queue status
            queue_settings = queue.pop(0)
            mode = queue_settings.get('mode', None)
            # filter out errors
            n_queue_items = sum(1 for item in queue if item['status'] in ('inprogress', 'queue'))

            if time.time() > self.last_log_update + 180:
                self.last_log_update = time.time()
                self.update_test_case({'stage': Stage.TESTING, 'n_queue_items': n_queue_items})

            yield QueueStatus(mode, n_queue_items)

            gevent.sleep(sleep_period)

    def prime_queue(self, pid):
        """Blocks until the queue contains items.

        If blocks longer than ``max_periods`` it returns False.

        Returns
        -------
        primed : bool
            True if contains items, False otherwise.
        """
        primed = False
        queue_status = self.poll_queue(pid, sleep_period=self.sleep_period)
        for i in range(self.max_periods):
            qs = next(queue_status)
            if qs.n_queue_items > 0:
                primed = True
                break
        return primed

    @retry(RETRIES, backoff = BACKOFF, hook = log_failure)
    def next_steps(self, pid):
        """Calls next_steps to run the next iteration of the autopilot.

        It then tries to prime the queue -- if that fails we assume that
        the autopliot is complete.

        Returns
        -------
        queue_primed : bool
            True if we added new items to the queue else False.
        """
        url = urljoin(self.base_url, 'project/'+pid+'/next_steps')
        r = requests.get(url, cookies=self.cookies, auth=("user","pass"))
        self.check_response(pid, r)
        return self.prime_queue(pid)

    def run_queue(self, pid):
        """Polls queue and calls next_steps if empty.

        If queue empty for longer than 5 consectuive seconds then call next steps.
        The test is complete if there are no items inprogress and queued anymore.
        """

        # Poll queue - when empty for 5 consecutive seconds, call next_steps
        queue_status = self.poll_queue(pid, sleep_period=self.sleep_period)
        for qs_seq in nwise(queue_status, n=5):
            if all(qs.n_queue_items == 0 for qs in qs_seq):
                if not self.next_steps(pid):
                    break

    def execute(self, pid, target, metric, options=None):
        try:
            self.select_target(pid, target, metric, options)
            logging.info('priming queue: {}'.format(self.ds_log_id))
            self.prime_queue(pid)
            logging.info('queue primed: {}'.format(self.ds_log_id))
            logging.info('running queue: {}'.format(self.ds_log_id))
            self.run_queue(pid)
            self.update_test_case({'status': Status.COMPLETED, 'test_complete_time': 'TIMESTAMP'})
        except Exception as ex:
            self.update_test_case({'status': Status.FAILED, 'test_complete_time': 'TIMESTAMP',
                'error_msg': 'Failed running test: {}'.format(ex)})
            raise

    def update_test_case(self, data):
        if self.enable_test_case_updates:
            MBTestDatabaseService().update_case(self.test_case_id, data)
            logging.info('DATASET {}: {}'.format(self.ds_log_id, data))
        else:
            logging.warn('Updates disabled. DATASET {}: {}'.format(self.ds_log_id, data))

    def create_test_case(self, dataset, target, metric, test_run_id=None, dataset_index=None):
        msg = 'Creating test case record for RID {} with dataset {} metric {}'.format(
            test_run_id, dataset, metric)
        logging.info(msg)

        if self.enable_test_case_updates:
            mbtest_service = MBTestDatabaseService()
            self.test_case_id = mbtest_service.create_case(test_run_id, dataset, dataset_index,
                                                           target, metric)
            logging.info('\n\n   CID = {}\n   DATASET = [{}] {} \n\n'.format(self.test_case_id,
                                                                             dataset_index, dataset))
            return self.test_case_id
        else:
            logging.warn('Test case updates disabled. {}'.format(msg))

    def create_project_and_upload(self, dataset_path, worker_options = None,
        test_run_id = None, dataset_index = None, login = True):
        try:
            if login:
                self.login()
            pid = self.create_project(worker_options)
            self.update_test_case({'stage': Stage.PROJECT_CREATED, 'status': Status.IN_PROCESS,
                'pid': pid, 'worker_type': self.get_aws_instance_name(worker_options)})

            self.upload_file(dataset_path, pid)
            self.update_test_case({'stage': Stage.FILE_UPLOADED})
            return pid
        except Exception as ex:
            self.update_test_case({'status': Status.FAILED, 'error_msg': 'Failed starting test {}'.format(ex),
                'test_complete_time': 'TIMESTAMP'})
            raise

    def get_aws_instance_name(self, worker_options):
        size = worker_options and worker_options['worker_options']['size']
        #FIXME: Hard-coded values are no bueno
        return  'm2.4xlarge' if size == '>30' else 'm3.2xlarge'

    def create_account_and_execute(self, dataset_path, target, metric):
        self.create_account()
        self.test_case_id = None
        self.ds_log_id = os.path.basename(dataset_path)
        pid = self.create_project_and_upload(dataset_path, login=False)
        self.execute(pid, target, metric)


def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    requests_logger = logging.getLogger("requests")
    requests_logger.setLevel(logging.WARN)
    formatter = logging.Formatter("%(asctime)-15s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)


if __name__ == "__main__":
    configure_logger()
    if len(sys.argv) != 4:
        logging.error('Usage: {} dataset_path target metric'.format(sys.argv[0]))
        sys.exit(1)
    test = MBTest()
    test.create_account_and_execute(*sys.argv[1:])
