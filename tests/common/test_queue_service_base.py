import unittest
import pytest
from bson.objectid import ObjectId
from mock import Mock, patch

from config.test_config import db_config as config
from common.wrappers import database

from common.services.queue_service_base import QueueServiceBase as QueueService
from common.services.project import ProjectServiceBase as ProjectService
import common

@pytest.mark.integration
@patch.object(common.services.project.ProjectServiceBase, 'assert_has_permission')
@patch.object(common.services.eda.EdaService, 'assert_has_permission')
@patch.object(common.services.eda.EdaService, 'get_all_metrics')
class TestQueueClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new("tempstore", host=config['tempstore']['host'],
                port=config['tempstore']['port'])
        self.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

    @classmethod
    def tearDownClass(self):
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='leaderboard')
        self.persistent.destroy(table='metadata')
        self.persistent.destroy(table='prediction_tabulation')

    def test_qsb_blend(self, *args, **kwargs):
        uid = str(ObjectId('5359d6cb8bd88f5cddefd3a8'))
        pid = self.persistent.create({'partition': {'reps': 5, 'holdout_pct': 20}}, table='project')
        ds = str(ObjectId)
        lid = self.persistent.create({'pid': pid, 'blueprint': {'1': (['NUM'], 'RFC', 'P')},
            'dataset_id': ds, 'samplepct': 64, 'test': {}, 'partition_stats': (0, -1)}, table='leaderboard')


        fake_progress = Mock()
        fake_progress.set_ids.return_value = True
        q = QueueService(str(pid), fake_progress)
        blend_args = {'blender_items': [str(lid)],
                      'blender_args': ['logitx'],
                      'blender_family': 'binomial',
                      'blender_method': 'AVG'}
        out = q.blend(blend_args)
        # The following keys are required:
        for key in ['samplepct', 'blueprint_id', 'max_reps', 'dataset_id', 'max_folds']:
            self.assertIn(key, out)

    def test_add_doesnt_just_put_a_question_mark(self, *args):
        pid = self.persistent.create(
            {'partition': {
                'reps': 5, 'holdout_pct': 20, 'total_size': 200}},
            table='project')

        ds = str(ObjectId('5223deadbeefdeadbeef0000'))
        job1 = {
            'pid': str(pid),
            'blueprint': {'1': (['NUM'], ['RFC'], 'P')},
            'blueprint_id': '67f1a3381e9779579946482227f4341c',
            'dataset_id': str(ds),
            'samplepct': 64,
            'bp': 1,
        }
        repository_job = {  # Has no BP
            'pid': str(pid),
            'blueprint': {'1': (['NUM'], ['SVC'], 'P')},
            'blueprint_id': '9bb6c522ef22738c417bf450978856c6',
            'dataset_id': str(ds),
            'samplepct': 64,
        }

        fake_progress = Mock()
        fake_progress.set_ids.return_value = True
        q = QueueService(str(pid), fake_progress)

        q.add(job1)
        q.add(repository_job)

        the_queue = self.tempstore.read(
            keyname='queue',
            index=str(pid),
            result=[]
        )

        self.assertEqual(the_queue[1]['bp'], 2)


if __name__ == '__main__':
    unittest.main()
