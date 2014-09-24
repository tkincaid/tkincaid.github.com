import sys
import datetime
import time
import random
import yaml

from bson import ObjectId
from common.wrappers import database

class MBTestDatabaseService(object):

    # Run types
    PARALLELE = 0
    SEQUENCIAL = 1

    # Environments
    AWS = 0
    AZURE = 1

    def __init__(self, rid = None, cid = None, persistent=None):
        """
        Meta Blueprint Database service

        @param rid test run id
        @param cid test case id

        """
        self.persistent = database.new('persistent') if persistent is None else persistent

        ObjectId(rid) # None is Acceptable
        ObjectId(cid)
        self.rid = database.ObjectId(rid) if rid else None
        self.cid = database.ObjectId(cid) if cid else None

    def get_run(self, run_id):
        return self.persistent.read({'_id': ObjectId(run_id)},table='mbtest_run', result={})

    def create_run(self, blueprint_version, commit, user, run_number,
                   dataset_count, awsdatarobot_commit="master", run_type=PARALLELE, environ=AWS,
                   datasets=None):
        """
        Create record in mongo for test run

        @param blueprint_version
        @param commit DataRobot commit id
        @param user user who run the test
        @param run_num test run number as specified by the user
        @param dataset_count number of dataset to be tested
        @param run_type paralelle or sequencial
        @param environ environment, AWS or AZURE

        Returns:
            test run id
        """

        # version_pattern = re.compile('\.mb([\.\w]+)\.Metablueprint')
        # match = version_pattern.search(blueprint_version)
        # if match:
        #    blueprint_version = match.groups(0)[0]

        now = int(datetime.datetime.now().strftime('%s'))
        rid = self.persistent.create(table='mbtest_run',
                                     values = {'environ' : environ,
                                               'blueprint_version': blueprint_version,
                                               'datarobot_commit': commit,
                                               'awsdatarobot_commit': awsdatarobot_commit,
                                               'user': user,
                                               'run_num': run_number,
                                               'dataset_count': dataset_count,
                                               'run_type': run_type,
                                               'start_time': now,
                                               'last_modified': now,
                                               'datasets': yaml.dump(datasets)})
        return rid

    def update_run(self, rid, data, update_time=None):
        """

        @param rid test run id
        @param data dict of keys/values to update
        @param update_time optional last modified date time to set. default to current date time

        """
        ObjectId(rid)

        if update_time is None:
            update_time = int(datetime.datetime.now().strftime('%s'))

        data['last_modified'] = update_time

        for k,v in data.iteritems():
            if v == 'TIMESTAMP':
                data[k] = update_time

        self.persistent.update(condition={'_id': ObjectId(rid)}, table='mbtest_run', values=data)

    def update_instance_ids(self, rid, data):
        """
        Set/update instance_ids in mbtest_run without wiping them.
        """
        if not data:
            return

        if not isinstance(data,list):
            raise ValueError('Instance ids must be list. Recieved data of Type %s: %s'%(type(data), data))

        self.persistent.conn['mbtest_run'].update(
                {'_id': ObjectId(rid)},
                {'$addToSet': {'instance_ids': {'$each': data} } },
                True)

    def set_run_complete(self,rid, complete_time=None):
        """

        @param rid run id
        @param complete_time option complete time, if not specified use current date time

        """

        ObjectId(rid)

        if complete_time:
            now = complete_time
        else:
            now = int(datetime.datetime.now().strftime('%s'))

        self.update_run(rid=rid, data={'finish_time': now})

    def create_case(self, rid, dataset, dataset_index, target, metric):
        """
        Create test case for test specified test run

        @param rid Test run ID to link test case to
        @param dataset Dataset name being use for the test case
        @param dataset_index The zero-based position of the dataset in the file
        @param target Name of target being used for modeling
        @param metric Name of metric being used for modeling

        - TIMING:
            start_time
            last_modified

        Returns:
            test case id
        """

        ObjectId(rid)

        now = int(datetime.datetime.now().strftime('%s'))
        cid = self.persistent.create(table='mbtest_case', values = {
            'rid': rid, 'dataset': dataset, 'dataset_index': dataset_index, 'target': target, 'metric': metric,
            'test_start_time': now, 'last_modified': now})
        return cid

    def get_case_by_dataset_index(self, rid, dataset_index):
        return self.persistent.read(table='mbtest_case', condition = {
            'rid': rid, 'dataset_index': dataset_index}, result = {})

    def update_case(self, cid, data, update_time=None):
        """

        @param rid test run id
        @param data dict of keys/values to update
        @param update_time optional last modified date time to set. default to current date time

        - TIMING:
            last_modified
            launch_start_time
            launch_complete_time
            provision_start_time
            provision_complete_time
            deploy_start_time
            deploy_complete_time
            test_start_time
            test_complete_time

        - STATUS:
            'running'
            'failed'
            'completed'

        - STAGE:
            launching
            provisioning
            deploying
            testing
            relaunching
            reprovisioning
            redeploying
            retesting
            completed
        - ERROR_MSG:
            Brief description of error

        pid
        model info: count, runtime, etc.....
        """

        ObjectId(cid)

        if update_time is None:
            update_time = int(datetime.datetime.now().strftime('%s'))

        data['last_modified'] = update_time

        for k,v in data.iteritems():
            if v == 'TIMESTAMP':
                data[k] = update_time

        self.persistent.update(condition={'_id': ObjectId(cid)}, table='mbtest_case', values=data)

    def set_case_pid(self, cid, pid):
        """

        @param cid case id
        @param pid project id

        """

        ObjectId(cid)
        ObjectId(pid)

        now = int(datetime.datetime.now().strftime('%s'))
        self.update_case(cid=cid, data={'pid': pid})

    def set_case_complete(self, cid, complete_time=None):
        """
        @param cid case id
        @param complete_time optional complete date time, default to current date time if not set

        """

        ObjectId(cid)

        if complete_time:
            now = complete_time
        else:
            now = int(datetime.datetime.now().strftime('%s'))
        self.update_case(cid=cid, data={'test_complete_time': now, 'stage': 'completed', 'status': 'completed'})

    def get_ram_usage_by_dataset(self):
        """The dataset_bp_stats collection has the average max ram usage for a recent time period
        for combinations of datasets and blueprints

        Returns a dictionary keyed by dataset name with the max of the average max ram usage across all blueprints
        that have been used with the dataset
        """
        agg = [{'$group': {'_id': '$dataset', 'ram': {'$push': '$stats.max_RAM'}}}]
        #agg = [{'$group': {'_id': '$dataset', 'ram': {'$max': '$stats.max_RAM'}}}]
        query = self.persistent.conn.dataset_bp_stats.aggregate(agg)
        result = query.get('result')
        if query.get('ok') and result:
            datasets = dict([[str(r['_id']), r['ram']] for r in result])
            return datasets

    def sort_datasets_by_time(self, dataset_names):
        since = time.time() - 2419200 #4 weeks
        agg = [{'$match': {'test_complete_time': {'$gt': since},
                           'test_start_time': {'$gt': since},
                           'dataset': {'$exists': True}}
               },
               {'$project': {'dataset': 1,
                             'test_time': {'$subtract': ["$test_complete_time", "$test_start_time"]}}
               },
               {'$group': {'_id': '$dataset', 'avg_time': {'$avg': '$test_time'}}
               }]

        query = self.persistent.conn.mbtest_case.aggregate(agg)
        result = query.get('result')
        if query.get('ok') and result:
            dslist = dict([[str(d['_id']),d['avg_time']] for d in result])

            scores = dslist.values()
            avg_score = sum(scores) * 1.0 / len(scores)

            #randomize a bit because the time measurement used isn't appropriate
            for d in dslist:
                dslist[d] += random.uniform(0, avg_score)

            sorted_datasets = sorted(dataset_names, key=lambda x: dslist.get(x, 0))
            return sorted_datasets

        return dataset_names


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser()
    parser.disable_interspersed_args()


    parser.add_option('--create-test-run', dest="create_run", action="store_true",
        default=False, help="Create a test run")

    parser.add_option('--blueprint-version', dest="blueprint_version", type="string",
        action="store", help="DataRobot meta blueprint version")

    parser.add_option('--commit', dest="commit", type="string",
        action="store", help="DataRobot commit id")

    parser.add_option('--user', dest="test_user", type="string",
        action="store", help="Test user")

    parser.add_option('--run-type', dest="run_type", type="int",
        action="store", default=0, help="Test run type, 0 for paralle or 1 for sequencial, default to parallele")

    parser.add_option('--environment', dest="environment", type="int",
        action="store", default=0, help="Test environment, 0 for AWS or 1 for AZURE; default to AWS if not specified")

    parser.add_option('--run-number', dest="run_number", type="int",
        action="store", help="Test run number")

    parser.add_option('--dataset-count', dest="dataset_count", type="int",
        action="store", help="Number of datasets in the run")

    parser.add_option('--awsdatarobot-commit', dest="awsdatarobot_commit", type="string",
        action="store", help="awsdatarobot commit")


    parser.add_option('--create-test-case', dest="create_case", action="store_true",
        default=False, help="Create a test case")

    parser.add_option('--test-run-id', dest="test_rid", type="string",
        action="store", help="Test run Id")

    parser.add_option('--dataset', dest="dataset", type="string",
        action="store", help="Dataset file")

    parser.add_option('--target', dest="target", type="string",
        action="store", help="Test target")

    parser.add_option('--metric', dest="metric", type="string",
        action="store", help="Test metric")

    parser.add_option('--memory-size', dest="memory_size", type="int",
        action="store", default=30, help="Requiremed memory size in GB, default to 30GB if not specified")


    parser.add_option('--update-run', dest="update_run", action="store_true",
        default=False, help="Update a test run")

    parser.add_option('--update-case', dest="update_case", action="store_true",
        default=False, help="Update a test case")

    parser.add_option('--test-case-id', dest="test_cid", type="string",
        action="store", help="Test case id")

    parser.add_option('--update-data', dest="update_data", type="string",
        action="store", help="Key/value pair dict to update")


    parser.add_option('--set-run-complete', dest="run_complete", type="string",
        action="store", help="Set the specified test run id to set completed")

    parser.add_option('--set-case-complete', dest="case_complete", type="string",
        action="store", help="Set the specified test case id to set completed")

    parser.add_option('--set-case-pid', dest="case_pid", type="string",
        action="store", help="Set project id for a test case")


    if not sys.argv[1:]:
        parser.print_help()
        sys.exit(2)


    (options, args) = parser.parse_args()
    sys.argv[:] = args


    if options.create_run:
        if not options.blueprint_version \
            or not options.commit \
            or not options.test_user \
            or not options.run_number \
            or not options.dataset_count \
            or not options.awsdatarobot_commit:

            print "ERROR: Create a test run requires: blueprint-version, commit, user, run-number and dataset-count, awsdatarobot_commit"
            sys.exit(1)
        else:
            if options.awsdatarobot_commit:
                rid = MBTestDatabaseService().create_run(blueprint_version=options.blueprint_version,
                    commit=options.commit, user=options.test_user, run_number=options.run_number,
                    dataset_count=options.dataset_count, awsdatarobot_commit=options.awsdatarobot_commit,
                    run_type=options.run_type, environ=options.environment)
            else:
                rid = MBTestDatabaseService().create_run(blueprint_version=options.blueprint_version,
                    commit=options.commit, user=options.test_user, run_number=options.run_number,
                    dataset_count=options.dataset_count, run_type=options.run_type, environ=options.environment)
            print rid
            sys.exit(0)

    if options.create_case:
        if not options.test_rid \
            or not options.dataset \
            or not options.target \
            or not options.metric:

            print "ERROR: Create a test case requires test run id, dataset, target, metric, and optional memory size"
            sys.exit(1)
        else:
            cid = MBTestDatabaseService().create_case(rid=options.test_rid, dataset=options.dataset,
                target=options.target, metric=options.metric, memory_size=options.memory_size)
            print cid
            sys.exit(0)

    if options.update_run:
        if not options.update_data or not options.test_rid:
            print "ERROR: Update a test run requires both test run id and update data"
            sys.exit(1)
        else:
            data = eval(options.update_data)
            MBTestDatabaseService().update_run(rid=options.test_rid, data=data)
            sys.exit(0)

    if options.update_case:
        if not options.update_data or not options.test_cid:
            print "ERROR: Update a test case requires both test case id and update data"
            sys.exit(1)
        else:
            data = eval(options.update_data)
            MBTestDatabaseService().update_case(cid=options.test_cid, data=data)
            sys.exit(0)

    if options.run_complete:
        MBTestDatabaseService().set_run_complete(rid=options.run_complete)
        sys.exit(0)

    if options.case_complete:
        MBTestDatabaseService().set_case_complete(cid=options.case_complete)
        sys.exit(0)

    if options.case_pid:
        if not options.test_cid:
            print "ERROR: Set project id for a test case requires test case id"
            sys.exit(1)
        else:
            MBTestDatabaseService().set_case_pid(cid=options.test_cid, pid=options.case_pid)
            sys.exit(0)
