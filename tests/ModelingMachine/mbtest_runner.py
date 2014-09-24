import sys
import os
import errno
import gevent
from gevent.pool import Group
import argparse
from config.engine import EngConfig
import logging
import csv
import yaml
from StringIO import StringIO

from bson import ObjectId

from MMApp.entities.user import UserService, UserModel, USER_TABLE_NAME
from common.storage import S3Storage
from tests.ModelingMachine.mbtest_rd import MBTest, configure_logger
from mbtest_database_service import MBTestDatabaseService

from mbtest_data_csv2yaml import mbtest_data_csv_to_yaml_format


def create_temp_local_storage(local_storage, test_run_id):
    local_storage = os.path.join(local_storage, str(test_run_id))

    if not os.path.isdir(local_storage):
        try:
            logging.info('Creating path for temp local storage: {}'.format(local_storage))
            os.makedirs(local_storage)
        except OSError as error:
            logging.error('Could not create dir for local storage {} {}: {}'.format(
                local_storage, test_run_id, error))
            # Ignore race condition where another process may have created
            # this directory already
            if error.errno != errno.EEXIST:
                raise
    return local_storage


def create_test_account(test_run_id):
    from common.wrappers import database

    persistent = database.new('persistent')
    username = get_username_from_run_id(test_run_id)

    persistent.create(table=USER_TABLE_NAME, values={
        'activated' : 0,
        'invitecode' : 'not-used',
        'needs_approval' : 0,
        'username' : username,
        'max_workers' : 500
    })
    return username, username

def get_user(username):
    user_service = UserService(username)
    return user_service.find_user()

def get_username_from_run_id(test_run_id):
    return 'mb-test-{}@datarobot.com'.format(test_run_id)

def create_test_run(metablueprint, commit, user, run_number, dataset_count):
    if metablueprint is None:
        logging.info('Missing METABLUEPRINT. Creating random test run ID')
        return ObjectId()

    env_vars = (metablueprint, commit, user, run_number, dataset_count)
    logging.info('Creating test run ID using variables: {}'.format(env_vars))
    return MBTestDatabaseService().create_run(*env_vars)

def save_datasets(test_run_id,  datasets):
    MBTestDatabaseService().update_run(test_run_id, {'datasets': datasets})

def mark_as_complete(test_run_id):
    MBTestDatabaseService().update_run(test_run_id, {'end_time': 'TIMESTAMP'})

def load_datasets_from_file(dataset_filepath):
    """Load a mbtest data file in YAML format and return a seq of dicts.

    Returns
    -------
    documents : seq of dict
        A sequence of dataset dicts.
    """
    logging.info('Loading datasets from {}'.format(dataset_filepath))
    with open(dataset_filepath) as f:
        return yaml.load(f)


def load_datasets_from_database(test_run_id):
    """Load mbtest data file from DB given a ``test_run_id``.

    Returns
    -------
    documents : seq of dict
        A sequence of dataset dicts.
    """
    logging.info('Loading datasets from database for: {}'.format(test_run_id))
    mbtest_run =  MBTestDatabaseService().get_run(test_run_id)
    datasets = mbtest_run.get('datasets')
    logging.info('type(Documents): %r', type(datasets))
    logging.info('Documents: %s', str(datasets)[:200])

    if isinstance(datasets, basestring):
        documents = yaml.load(datasets)
        # documents should be seq of dicts -- if str then legacy format
        if isinstance(documents, basestring):
            logging.info('Loading datasets from legacy CSV format')
            reader = csv.reader(StringIO(datasets), delimiter=';')
            documents = mbtest_data_csv_to_yaml_format(reader)
    else:
        pass  # datasets is seq of dict

    return documents


def signup(username, password):
    user_service = UserService(username)
    user = UserModel(username = username, password = password)
    user_service._signup(user)

def has_dataset_started(test_run_id, dataset_index, filename):
    test_case = MBTestDatabaseService().get_case_by_dataset_index(test_run_id, dataset_index)
    return test_case and test_case.get('dataset') == filename

def runner(test_run_id, datasets, local_storage, credentials):
    failures = 0
    group = Group()
    for dataset_index, dataset in enumerate(datasets):
        try:
            g = run_mb_test(dataset_index, dataset)
            if g:
                group.add(g)
        except Exception:
            failures += 1
            logging.error('MBTest FAILED for [{}] {}'.format(dataset_index, dataset),
                exc_info = True)

    if len(group):
        group.join()

    mark_as_complete(test_run_id)

    return failures

def run_mb_test(dataset_index, dataset):
    filename = dataset.pop('dataset_name')
    target = dataset.pop('target')
    metric = dataset.pop('metric')
    # worker size is optional; either empty or '>30'
    worker_size = dataset.pop('worker_size', '')
    # everyting remaining in ``dataset`` is considered an advanced option and is passed
    # to the MBtest.execute funtion

    test = MBTest(enable_test_case_updates=True, sleep_period=30, max_periods=8,
                  credentials=credentials)
    test.ds_log_id = '[{}] {}'.format(dataset_index, filename)

    if has_dataset_started(test_run_id, dataset_index, filename):
        logging.info('Skipping dataset: {}'.format(test.ds_log_id))
        return

    logging.info('Working on dataset: {}'.format(test.ds_log_id))

    test.create_test_case(filename, target, metric, test_run_id, dataset_index)

    local_dataset_path = os.path.join(local_storage, filename)

    s3 = S3Storage(bucket='datarobot_data_science')
    s3.prefix = 'test_data/'
    s3.get(name = filename, local_filename=local_dataset_path)

    pid = test.create_project_and_upload(local_dataset_path, test_run_id=test_run_id,
        worker_options = {'worker_options' : {
            'size' : worker_size,
            'service_id': test_run_id,
            'worker_count': 20}
        }, dataset_index = dataset_index)

    test.ds_log_id = test.ds_log_id + ' - PID: ' + pid

    if os.path.isfile(local_dataset_path):
        os.remove(local_dataset_path)

    return gevent.spawn(test.execute, pid, target, metric, dataset)


if __name__ == '__main__':
    configure_logger()

    parser = argparse.ArgumentParser(description='Metablueprint test')
    parser.add_argument('--dataset_filepath')
    parser.add_argument('--commit', default=EngConfig.get('REV'))
    parser.add_argument('--metablueprint', default=EngConfig['metablueprint_classname'])
    parser.add_argument('--run_number', default="1")
    parser.add_argument('--test_run_id')
    parser.add_argument('--username', default="DataRobot")
    parser.add_argument('--password')
    parser.add_argument('--local_storage', default=EngConfig['DATA_DIR'])
    args = parser.parse_args()
    print(args)

    if args.test_run_id:
        datasets = load_datasets_from_database(args.test_run_id)
        test_run_id = args.test_run_id
    else:
        # This used only for local runs in dev computers
        if not args.dataset_filepath:
            raise ValueError('dataset_filepath is required when test_run_id is not specified')

        datasets = load_datasets_from_file(args.dataset_filepath)
        test_run_id = create_test_run(args.metablueprint, args.commit, args.username, args.run_number, len(datasets))
        test_run_id = str(test_run_id)
        save_datasets(test_run_id, datasets)

    logging.info('\n\n   {} DATASETS   RID: {}\n'.format(len(datasets), test_run_id))

    # Create new credentials or reuse an existing account to add more datasets/projects
    if args.username is None or args.password is None:
        username = get_username_from_run_id(test_run_id)
        if get_user(args.username):
            logging.info('User account already exists (skipping account creation)')
            credentials = (username, username)
        else:
            logging.info('No credentials passed-in. Creating test account now')
            credentials = create_test_account(test_run_id)
            signup(*credentials)
    else:
        credentials = (args.username, args.password)
        username = args.username

    user = get_user(username)
    logging.info('Credentials:\n\n   UID = {}\n\n   USERNAME = {}\n   PASSWORD = {}\n\n'.format(user.uid, *credentials))

    local_storage = create_temp_local_storage(args.local_storage, test_run_id)

    rc = runner(test_run_id, datasets, local_storage, credentials)
    sys.exit(rc)
