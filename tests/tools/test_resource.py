from tools.task_model import collect_historical_data, create_training_data, create_models
from tools.dataset_stats import collect_dataset_info
import tools.dataset_stats
from mock import Mock, patch
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
import unittest
import cPickle
import os

TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../testdata')

# some test data
test_bp = {'1':(['NUM'],['NI'],'T'),'2':(['1'],['GBM'],'P')}
test_pid = '123'
test_leaderboard = [{
    'blueprint': test_bp,
    'pid': test_pid,
    'samplepct': 64,
    'task_info': {
        'reps=1': [[{
            'task_name': 'GBM',
            'fit max RAM': 1000000,
            'fit CPU time': 10,
            'fit CPU pct': 100,
        }],[{
            'task_name': 'NI',
            'fit max RAM': 2000000,
            'fit CPU time': 20,
            'fit CPU pct': 200,
        }]]
    }}
]
test_dataset_info = [
    {
     'dataset':'test.csv',
     'stats': {
        'rows':100,
     }
    }
]

# fake mongo DB collections
class FakeLeaderboard(object):
    def find(self,cond,fields):
        return test_leaderboard

class FakeDatasetInfo(object):
    def find(self):
        return test_dataset_info

class FakeStats(object):
    def __init__(self):
        self.conds = []
        self.values = []
    def update(self,cond,value,upsert):
        self.conds.append(cond)
        self.values.append(value)


class TestTaskModel(unittest.TestCase):
    def test_historical_data(self):
        ''' check collect_historical_data() works correctly '''

        # set up test DB
        stats = FakeStats()
        source_db = {'leaderboard': FakeLeaderboard()}
        dest_db = {'dataset_bp_stats': stats}
        project_info = {test_pid:'test.csv'}

        # summarize data
        collect_historical_data(source_db,dest_db,project_info)

        # check DB update was called with correct keys
        self.assertEqual(stats.conds,[{'dataset':'test.csv','blueprint':test_bp}])
        # check calculated stats are correct
        self.assertEqual(stats.values,[
                {'dataset':'test.csv',
                 'blueprint':test_bp,
                 'stats': {
                    'max_RAM':2000000,
                    'total_CPU':30,
                    'max_cores':200}
        }])

    def test_training_data(self):
        ''' check training data for models is collected correctly '''

        # fake DB objects
        source_db = {'leaderboard': FakeLeaderboard(), 'dataset_info': FakeDatasetInfo()}
        project_info = {test_pid:'test.csv'}

        # output dicts
        target_mr = defaultdict(list)
        target_tc = defaultdict(list)
        target_ac = defaultdict(list)
        pids = defaultdict(list)
        train = defaultdict(lambda:defaultdict(list))

        # get training data
        create_training_data(source_db, target_mr, target_tc, target_ac, pids, train, project_info)

        # check output is correct
        self.assertEqual(target_mr,{'NI': [2000000], 'GBM': [1000000]})
        self.assertEqual(target_tc,{'NI': [20], 'GBM': [10]})
        self.assertEqual(target_ac,{'NI': [200], 'GBM': [100]})
        self.assertEqual(pids, {'NI': ['123'], 'GBM': ['123']})
        self.assertEqual(train, {'NI': {'index': [2], 'rows': [64.0], 'interactions': [0], 'trees': [0], 'input_count': [1], 'grid_count': [0]},
                                 'GBM': {'index': [1], 'rows': [64.0], 'interactions': [0], 'trees': [0], 'input_count': [1], 'grid_count': [0]},
        })

    def test_create_models(self):
        ''' check that models are created from training data '''

        # test training data
        target_mr = {'NI': [2000000], 'GBM': [1000000]}
        target_tc = {'NI': [20], 'GBM': [10]}
        target_ac = {'NI': [200], 'GBM': [100]}
        pids = {'NI': ['123'], 'GBM': ['123']}
        train = {'NI': {'index': [2], 'rows': [64.0], 'interactions': [0], 'trees': [0], 'input_count': [1], 'grid_count': [0]},
                                 'GBM': {'index': [1], 'rows': [64.0], 'interactions': [0], 'trees': [0], 'input_count': [1], 'grid_count': [0]}}
        # fake output DB
        output = FakeStats()
        dest_db = {'task_coefficients': output}

        # create models from training data
        create_models(dest_db, target_mr, target_tc, target_ac, pids, train)

        # check output
        self.assertEqual(output.conds, [{'task': 'NI'}, {'task': 'GBM'}])
        self.assertEqual(len(output.values),2)
        self.assertEqual(set(output.values[0].keys()),set(['task','gbm']))
        self.assertEqual(set(output.values[0]['gbm'].keys()),set(['max_RAM','total_CPU','max_cores']))
        model = cPickle.loads(output.values[0]['gbm']['max_RAM'])
        self.assertIsInstance(model,GradientBoostingRegressor)

    @patch('tools.dataset_stats.get_s3_dataset_list')
    @patch('tools.dataset_stats.get_seen_datasets')
    def test_collect_dataset_info(self,seendatasets,s3datasets):
        ''' check dataset stat collection '''

        # select test data set
        s3datasets.return_value = ['credit-sample-200.csv']
        tools.dataset_stats.LOCAL_DATA_DIR = TESTDATA_DIR

        # mock some objects
        seendatasets.return_value = []
        persistent = Mock()

        # collect data set stats
        collect_dataset_info(persistent,Mock())

        # check results
        self.assertEqual(persistent.update.call_args[1],
                         {'table': 'dataset_info',
                          'values': {
                             'stats': {
                                 'rows': 200, 'cols': 12, 'cat': 0, 'num': 12, 'txt': 0, 'max_card': 0},
                             'dataset': 'credit-sample-200.csv'}
                         })

if __name__ == '__main__':
    unittest.main()
