import unittest
import os
import json

from integration_test_base import IntegrationTestBase
from tools import replay

class TestReplayModule(IntegrationTestBase):
    '''Tests that we are able to re-run a local job by specifying lid and
    partition.  This is important for quick debugging of failed jobs

    I am very embarrassed that it takes over twenty seconds to run these
    tests - perhaps we can set Jenkins to run it every 4 hours instead of
    every pull request
    '''

    @classmethod
    def setUpClass(cls):
        super(TestReplayModule, cls).setUpClass()
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')


    def test_replay(self):
        '''Run one model, get its lid, and use it to test replay'''
        pid = self.upload_file_and_select_response(self.app)

        self.wait_for_stage(self.app, pid, ['post-aim', 'eda2', 'modeling'])
        self.set_q_with_n_items(pid, 1, self.app)

        # Get things started
        response = self.app.get('/project/%s/service' % pid)
        self.assertEqual(response.status_code, 200)
        service_data = json.loads(response.data)
        self.assertGreaterEqual(service_data["count"], 1)

        # Only one in-process
        q_items = self.get_q(pid)
        not_started = [item for (i,item) in enumerate(q_items) if item['status'] == 'queue']
        in_progress = [item for (i,item) in enumerate(q_items) if item['status'] == 'inprogress']

        self.assertGreaterEqual(len(in_progress), 1)

        # Make sure one (any) q item finishes
        id_to_predict  = self.wait_for_q_item_to_complete(self.app, pid = pid, qid = None, timeout = 30)

        # Get models
        leaderboard = self.get_models(self.app, pid)
        # print json.dumps(leaderboard, indent = 4, sort_keys = True)
        leaderboard_item = leaderboard[0]
        lid = leaderboard_item['_id']

        # ACT
        x = replay.rerun(lid, 0)

        # ASSERT
        self.assertIn('report', x)
        self.assertIn('predictions', x)

    def test_rerun(self):
        '''Run one model, get its lid, and use it to test replay'''
        pid = self.upload_file_and_select_response(self.app)

        self.wait_for_stage(self.app, pid, ['post-aim', 'eda2', 'modeling'])
        self.set_q_with_n_items(pid, 1, self.app)

        # Get things started
        response = self.app.get('/project/%s/service' % pid)
        self.assertEqual(response.status_code, 200)
        service_data = json.loads(response.data)
        self.assertGreaterEqual(service_data["count"], 1)

        # Only one in-process
        q_items = self.get_q(pid)
        not_started = [item for (i,item) in enumerate(q_items) if item['status'] == 'queue']
        in_progress = [item for (i,item) in enumerate(q_items) if item['status'] == 'inprogress']

        self.assertGreaterEqual(len(in_progress), 1)

        # Make sure one (any) q item finishes
        id_to_predict  = self.wait_for_q_item_to_complete(self.app, pid = pid, qid = None, timeout = 30)

        # Get models
        leaderboard = self.get_models(self.app, pid)
        # print json.dumps(leaderboard, indent = 4, sort_keys = True)
        leaderboard_item = leaderboard[0]
        lid = leaderboard_item['_id']

        x = replay.rerun(lid, 0)
        x = replay.rerun(lid, 1)
        x = replay.rerun(lid, 2)
        x = replay.rerun(lid, 3)
        x = replay.rerun(lid, 4)

        # ACT
        x = replay.run_5_cv(lid)

        # ASSERT
        self.assertIn('report', x)
        self.assertIn('predictions', x)


