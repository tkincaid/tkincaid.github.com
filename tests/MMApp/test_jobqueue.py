############################################################################
#
#       unit test for QueueService
#
#       Author: TC
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import unittest
import time
import json
from mock import patch, DEFAULT, Mock
import pytest
import cPickle as pickle

from bson import ObjectId

from test_base import TestBase
from MMApp.entities.jobqueue import QueueService
from common.services.queue_service_base import QueueCounter, Queue, InProgress, QueueSettings, QueueException
from common.engine.progress import ProgressState
from common.entities.job import DataRobotJob

DEFAULT_LOCK_VALUE = 1

class JobQueueTestCase(TestBase):

    def setUp(self):
        self.project_id = '51f015fd07632960eebc5a4b'
        self.q_item = {"blueprint": {}, "require": [-1], 'bp':1}
        self.q = QueueService(self.project_id, Mock())
        self.pid = str(ObjectId())
        self.blueprint_id = 'e27d0e503e6324730d773a2e2e4dded1'
        self.dataset_id = '53a9bd64637aba7942799231'
        self.model_signature = {'pid': self.pid, 'blueprint_id': self.blueprint_id, 'dataset_id': self.dataset_id, 'samplepct' : 64}
        self.q_item = {'bp':1}
        self.q_item.update(self.model_signature)

        self.q.get_worker_options = lambda: {}

        self.addTestData()

    def tearDown(self):
        if self.q and self.q.redlock:
            self.q.redlock.unlock()

        self.redis_conn.flushall()

    def addTestData(self):
        self.q.tempstore.create(keyname='queue_settings',
                                index=self.project_id,
                                values={'parallel':2, 'mode':1})

    def assertQueueMatch(self, items, expected):
        self.assertEqual(len(items), len(expected))
        for i, item in enumerate(items):
            self.assertContains(item, expected[i])

    @pytest.mark.db
    def test_add(self):
        self.q.progress = Mock()
        self.q.progress.set_progress = Mock()
        q_item = {'blueprint':'asdf'}
        q_item.update(self.q_item)
        expected = {'qid': 1, 'blueprint':'asdf'}
        expected.update(self.q_item)
        expected = [expected]
        self.assertQueueMatch(self.q.add(q_item), expected)

        self.assertTrue(self.q.progress.set_progress.called)

    def test_add_worker_job(self):
            output = self.q.add_worker_job("1", "pid", "qid")
            self.assertEqual(output, 1)
            jobs = self.q.get_worker_jobs("1")
            self.assertEqual(jobs, set(['pid:qid']))

    @pytest.mark.db
    def test_remove_worker_job(self):
            output = self.q.add_worker_job("1", "pid", "qid")
            self.assertEqual(output, 1)
            jobs = self.q.get_worker_jobs("1")
            self.assertEqual(jobs, set(['pid:qid']))
            output = self.q.remove_worker_job("1", "pid", "qid")
            self.assertEqual(output, 1)
            jobs = self.q.get_worker_jobs("1")
            self.assertEqual(jobs, set([]))

    @pytest.mark.db
    def test_get_worker_jobs(self):
            output = self.q.add_worker_job("1", "pid", "qid")
            self.assertEqual(output, 1)
            jobs = self.q.get_worker_jobs("1")
            self.assertEqual(jobs, set(['pid:qid']))
            jobs = self.q.get_worker_jobs("2")
            self.assertEqual(jobs, set([]))

    @pytest.mark.unit
    def test_parse_worker_job(self):
            output = self.q.parse_worker_job("pid:qid")
            self.assertEqual(output, ("pid","qid"))

            self.assertRaises(IndexError, self.q.parse_worker_job, "invalid_job")

    @pytest.mark.unit
    def test_remove_from_queue(self):
        with patch.multiple(self.q, _remove_if_in_queue = DEFAULT, remove_from_persistent = DEFAULT) as mocks:

            item = {"qid" : 1, "pid": "52812ee5a6844e30fc4adcac" , "lid" : "1", "model_type": "Gradient Boosted Trees"}

            # Arrange
            mocks['_remove_if_in_queue'].return_value = item

            # Act
            output = self.q.remove(item['qid'])

            # Assert
            self.assertItemsEqual(item, output)
            mocks['remove_from_persistent'].assert_called_once_with(item)

    @pytest.mark.unit
    def test_remove_from_inprogress(self):
        with patch.multiple(self.q, _remove_if_in_queue = DEFAULT, remove_from_persistent = DEFAULT) as mocks:

            item = {"qid" : 1, "pid": "52812ee5a6844e30fc4adcac"}

            # Arrange
            mocks['_remove_if_in_queue'].return_value = None

            with patch.multiple(self.q.inprogress, remove = DEFAULT, hmget = DEFAULT) as inprogress_mocks:

                inprogress_mocks['hmget'].return_value = {1: '{"qid" : 1, "pid": "52812ee5a6844e30fc4adcac"}'}

                # Act
                output = self.q.remove(item['qid'])

                # Assert
                self.assertItemsEqual(item,output)
                mocks['remove_from_persistent'].assert_called_once_with(item)

    @pytest.mark.unit
    def test_remove_from_errors(self):
        with patch.multiple(self.q, _remove_if_in_queue = DEFAULT, remove_from_persistent = DEFAULT) as mocks:

            item = {"qid" : 1, "pid": "52812ee5a6844e30fc4adcac" , "lid" : "52812fd8a6844e327e4adcac", "model_type": "Gradient Boosted Trees"}

            # Arrange
            mocks['_remove_if_in_queue'].return_value = None

            with patch.multiple(self.q.inprogress, remove = DEFAULT, hmget = DEFAULT) as inprogress_mocks:

                inprogress_mocks['hmget'].return_value = None

                with patch.object(self.q.queue_error, 'hdel') as q_errors_hdel_mock:

                    # Act
                    output = self.q.remove(item['qid'])

                    # Assert
                    self.assertIsNone(output)
                    q_errors_hdel_mock.assert_called_once_with(item['qid'])
                    self.assertFalse(mocks['remove_from_persistent'].called)

    @pytest.mark.unit
    def test_remove_from_persistent(self):
        from MMApp.entities.project import ProjectService

        existingItem = {"qid" : 1, "pid": "52812ee5a6844e30fc4adcac" , "lid" : "52812fd8a6844e327e4adcac", "model_type": "Gradient Boosted Trees" }
        newItem = {"qid" : 1, "pid": "52812ee5a6844e30fc4adcac" , "lid" : "new", "model_type": "Gradient Boosted Trees"}
        update_vals = {'s': 0}

        # test remove_from_persistent when lb 'lid' key is 'new'
        with patch.multiple(ProjectService,
                get_leaderboard_item    = DEFAULT,
                delete_leaderboard_item = DEFAULT,
                save_leaderboard_item = DEFAULT ) as mock_new:
            # make sure a lid of 'new' returns false
            self.assertFalse( self.q.remove_from_persistent(newItem) )


        # test remove_from_persistent when lb 's' key is 0
        with patch.multiple(ProjectService,
                get_leaderboard_item    = DEFAULT,
                delete_leaderboard_item = DEFAULT,
                save_leaderboard_item = DEFAULT ) as mock_zero:
            output = self.q.remove_from_persistent(existingItem)
            mock_zero['delete_leaderboard_item'].assert_called_once_with(existingItem['lid'])
            self.assertTrue(output)

        # test remove_from_persistent when lb 's' key is 1
        existingItem.update({'s':1})
        with patch.multiple(ProjectService,
                get_leaderboard_item    = DEFAULT,
                delete_leaderboard_item = DEFAULT,
                save_leaderboard_item = DEFAULT ) as mock_one:
            output = self.q.remove_from_persistent(existingItem)
            mock_one['save_leaderboard_item'].assert_called_once_with(existingItem['lid'],update_vals)
            self.assertTrue(output)

    @pytest.mark.db
    def test_get(self):
        q_item = self.q_item
        self.q.add(q_item)

        #simulate putting something in the errors list
        self.q.tempstore.update(values={'1': json.dumps({})}, **self.q.errorskw)

        expected = [{'qid': -1, 'status': 'settings', 'workers': 2 , 'mode':1},
            {'qid': 1, 'status': 'queue', 'bp' : 1},
            {'status': 'error'}]

        q = self.q.get()

        self.assertQueueMatch(q, expected)

    @pytest.mark.db
    def test_inprogress_add(self):
        q_item = self.q_item
        self.q.add(q_item)
        self.q.set_autopilot({'workers':1})
        item = self.q.next()

        request = self.q.tempstore.read(index=str(item['qid']), **self.q.inprogress.requestskw)
        self.assertEqual(request, self.q.inprogress.REQUEST_STATUS_OPEN)

    @pytest.mark.db
    def test_inprogress_remove(self):
        q_item = self.q_item
        self.q.add(q_item)
        self.q.set_autopilot({'workers':1})
        item = self.q.next()

        request = self.q.tempstore.read(index=str(item['qid']), **self.q.inprogress.requestskw)
        self.assertEqual(request, self.q.inprogress.REQUEST_STATUS_OPEN)

        self.q.inprogress.remove(str(item['qid']))

        inprog = self.q.tempstore.read(**self.q.inprogress.kwindex)
        self.assertEqual(inprog, "")
        request = self.q.tempstore.read(index=str(item['qid']), **self.q.inprogress.requestskw)
        self.assertEqual(request, "")

    @pytest.mark.db
    def test_next(self):
        q_item = self.q_item
        q_item['max_reps'] = 1
        self.q.add(q_item)
        original_item = self.q.next()

        expected = [{'qid': -1, 'status': 'settings', 'workers': 2, 'mode':1},
            {'qid': 1, 'status': 'inprogress','bp': 1 }]

        q = self.q.get()

        self.assertQueueMatch(q, expected)

        self.q.tempstore.destroy(index=str(original_item['qid']), **self.q.inprogress.requestskw)
        item = self.q.next()
        self.assertItemsEqual(item, original_item)

        #test inprogress greater than parallel
        self.q.set_autopilot({'workers':0})
        self.q.add(q_item)
        self.assertEqual(self.q.next(), None)

    @pytest.mark.db
    def test_next_does_not_send_duplicates(self):
        """
        Test the QueueService does not send the same request multiple times even when the time is different from Redis (time will be mocked)
        """
        q_item = self.q_item
        q_item.update(self.model_signature)
        self.q.add(q_item)

        # Mock time mismatch: Web server and redis are 1 hour apart
        with patch('MMApp.entities.jobqueue.time') as mock_time:
            mock_time.time.return_value = time.time() - 3600
            original_item = self.q.next()
            self.assertIsNotNone(original_item)
            for i in xrange(20):
                original_item = self.q.next()
                self.assertIsNone(original_item)

    @pytest.mark.db
    def test_inprogress_retry_expired_request(self):
        q_item = {}
        q_item.update(self.q_item)
        q_item['max_reps'] = 1
        self.q.add(q_item)
        original_item = self.q.next()

        #don't wait for the timeout
        self.q.tempstore.destroy(index=str(original_item['qid']), **self.q.inprogress.requestskw)

        item = self.q.inprogress.retry_expired_request()
        self.assertItemsEqual(original_item, item)

        request = self.q.tempstore.read(index=str(original_item['qid']), **self.q.inprogress.requestskw)
        self.assertEqual(request, self.q.inprogress.REQUEST_STATUS_OPEN)

        #add a predict item and check that it's the next item returned
        #to confirm that the outstanding request isn't resubmitted
        self.q_item['pid'] = str(ObjectId())
        q_item = {'predict': 1, 'scoring_dataset_id' : '53f34eb8cb093677008bee9e'}
        q_item.update(self.q_item)
        self.q.add(q_item)
        item = self.q.next()

        expected = {'qid':2, 'predict': 1, 'scoring_dataset_id' : '53f34eb8cb093677008bee9e'}
        expected.update(self.q_item)
        self.assertQueueMatch([item], [expected])

    @pytest.mark.unit
    def test_parallel(self):
        self.q.set_autopilot({'workers':3})

        print self.q.settings
        expected = 3
        self.assertEqual(self.q.parallel, expected)

        self.q.set_autopilot({'pause':True})

        print self.q.settings
        expected = 0
        self.assertEqual(self.q.parallel, expected)

        self.q.set_autopilot({'workers':4})

        print self.q.settings
        expected = 0
        self.assertEqual(self.q.parallel, expected)

        self.q.set_autopilot({'pause':False})

        print self.q.settings
        expected = 4
        self.assertEqual(self.q.parallel, expected)

    @pytest.mark.db
    def test_onhold(self):
        #add two items to the queue
        #the second item requires the first
        q_item = self.q_item
        q_item['max_reps'] = 1
        self.q.add(q_item)

        q_item = self.q_item
        q_item.update({'require': ['1'], 'pid' : str(ObjectId())})
        self.q.add(q_item)

        self.q.set_autopilot({'workers':2, 'mode':1})

        #try to add both items to in progress
        self.q.next()
        self.q.next()

        #the second item should still have status 'queue'
        #and be in the onhold hash
        expected = [{'qid': -1, 'status': 'settings', 'workers': 2, 'mode':1},
            {'qid': 1, 'status': 'inprogress', 'bp': 1},
            {'qid': 2, 'status': 'queue', 'require': ['1'], 'bp': 1}]

        q = self.q.get()
        self.assertQueueMatch(q, expected)

        #check that the JSON string in requirements is correct
        expected = '[[2, ["1"]]]'

        reqs = self.q.tempstore.read(**self.q.requirementskw)

        self.assertEqual(reqs, expected)

        #remove item with id 1 from the queue
        #and try to add the item 2 again
        self.q.remove("1")
        self.q.next()

        #item two should now have status 'inprogress'
        expected = [{'qid': -1, 'status': 'settings', 'workers': 2, 'mode':1},
            {'qid': 2, 'status': 'inprogress', 'require': ['1'], 'bp': 1}]

        q = self.q.get()
        self.assertQueueMatch(q, expected)

    @pytest.mark.db
    def test_requirements(self):
        q_item = self.q_item
        q_item['max_reps'] = 1
        self.q.add(q_item)

        q_item = self.q_item
        q_item.update({'require': [-1], 'blueprint_id':str(ObjectId())})
        self.q.add(q_item)

        self.q.set_autopilot({'workers':1, 'mode':1})

        item = self.q.next()
        expected = {"qid": 1, "bp": 1}
        self.assertQueueMatch([item], [expected])

        self.q.remove("1")

        item = self.q.next()
        expected = {"qid": 2, "require": ['1'], "bp": 1}
        self.assertQueueMatch([item], [expected])

        expected = [{'qid': -1, 'status': 'settings', 'workers': 1,'mode':1},
            {'qid': 2, 'status': 'inprogress', 'require': ['1'], 'bp' : 1}]

        q = self.q.get()
        self.assertQueueMatch(q, expected)

        #check that prior requirements are maintained in the database
        q_item['blueprint_id'] = str(ObjectId())
        self.q.add(q_item)
        self.q.set_autopilot({'workers':2})

        self.q.next()

        expected = '[[3, ["2"]]]'
        reqs = self.q.tempstore.read(**self.q.requirementskw)

        self.assertEqual(reqs, expected)

        #set this item dependent on the queue item still in progress
        q_item = self.q_item
        q_item.update({'require': [-2], 'blueprint_id': str(ObjectId())})
        self.q.add(q_item)
        self.q.set_autopilot({'workers':3})
        self.q.next()

        expected = '[[3, ["2"]], [4, ["2"]]]'
        reqs = self.q.tempstore.read(**self.q.requirementskw)

        self.assertEqual(reqs, expected)

    @pytest.mark.db
    def test_delete(self):
        self.assertIsNone(self.q.delete())

    @pytest.mark.db
    def test_save(self):
        self.assertIsNone(self.q.save())

    @pytest.mark.db
    def test_resources(self):
        expected = {"cpu": {}, "mem": {}}

        res = self.q.resources()

        self.assertEqual(res, expected)

    @pytest.mark.db
    def test_queue_counter(self):
        qc = QueueCounter("1")

        self.assertEqual(str(qc), "1")
        self.assertEqual(qc.get(), '')
        self.assertEqual(qc.delete(), 0)
        self.assertEqual(qc.incr(), 1)
        self.assertEqual(qc.get(), "1")
        self.assertEqual(qc.delete(), True)

    @pytest.mark.db
    def test_queue(self):
        q = Queue("1")

        self.assertEqual(str(q), "1")
        self.assertEqual(q.get(), [])
        self.assertEqual(q.push(1), 1)
        self.assertEqual(q.get(), [1])
        self.assertEqual(q.delete(), True)

    @pytest.mark.db
    def test_inprogress(self):
        p = InProgress("1")

        self.assertEqual(str(p), "1")
        self.assertEqual(p.hmset({"test": "1"}), True)
        self.assertEqual(p.hvals(), ["1"])
        self.assertEqual(p.delete(), True)

    @pytest.mark.db
    def test_queue_settings(self):
        qs = QueueSettings("1")

        self.assertEqual(str(qs),"1")
        self.assertEqual(qs.hmset({"test": "1"}), True)
        self.assertEqual(qs.hget("test"), "1")
        self.assertEqual(qs.hmget(), {'test':'1'})
        self.assertEqual(qs.hvals(), ["1"])
        self.assertEqual(qs.hkeys(), ["test"])
        self.assertEqual(qs.hdel("test"), True)
        self.assertEqual(qs.delete(), False)
        qs.hmset({"test": "1"})
        self.assertEqual(qs.delete(), True)

    @pytest.mark.db
    def test_set_as_complete(self):
        qid = '12345'
        lid = str(ObjectId())

        report = {'key1' : 'value1'}

        # None before we get started
        items = self.q.inprogress.hmget(qid)
        self.assertFalse(items, 'Shouldn\'t be anything in here yet')

        # Arrange: Add to the in-progress q and check it was saved
        self.q.inprogress.hmset({qid : json.dumps(report)})
        items = self.q.inprogress.hmget(qid)[qid]
        self.assertEqual(items, json.dumps(report))

        # Act: Mark as complete
        self.q.set_as_complete(qid, lid)

        # Assert: Verify the in-progress value was removed and check for the complete value
        items = self.q.inprogress.hmget(qid)
        self.assertFalse(items, 'Shouldn\'t be anything in here anymore')

        completed_qid = self.q.tempstore.read(result=[],
                                              limit=(-1,-1),
                                              remove=True,
                                              **self.q.completedkw)

        self.assertEqual(completed_qid, qid)

    @pytest.mark.unit
    def test_set_as_started(self):
        qid = '543210'

        with patch.object(self.q, 'progress') as mock_progress:
            # Act: Mark as started
            self.q.set_as_started(qid)

            mock_progress.set_ids.assert_called_once_with(self.q.pid, qid)
            mock_progress.set_progress.assert_called_once_with(state=ProgressState.STARTING, percent=0, message="Building")

    @pytest.mark.db
    def test_set_error(self):
        qid = '12345'
        lid = '54321'

        report = {'key1' : 'value1'}

        # None before we get started
        items = self.q.inprogress.hmget(qid)
        self.assertFalse(items, 'Shouldn\'t be anything in here yet')

        # Arrange: Add to the in-progress q and check it was saved
        self.q.inprogress.hmset({qid : json.dumps(report)})
        items = self.q.inprogress.hmget(qid)[qid]
        self.assertEqual(items, json.dumps(report))

        # Act
        error = 'Ugly error'
        message = {'lid': lid, 'error': error}
        self.q.set_error(qid, lid, message)

        # Assert: Set with errir, verified the in-progress value was removed, check for the error and complete values
        items = self.q.inprogress.hmget(qid)
        self.assertFalse(items, 'Shouldn\'t be anything in here anymore')

        completed_qid = self.q.tempstore.read(result=[],
                                              limit=(-1,-1),
                                              remove=True,
                                              **self.q.completedkw)
        self.assertEqual(completed_qid, qid)

        report_with_error = self.q.tempstore.read(remove=True,
                                                result={},
                                                limit=(-1,-1),
                                                **self.q.errorskw)
        report_with_error = json.loads(report_with_error[qid])

        self.assertTrue('error_log' in report_with_error)
        self.assertDictContainsSubset(report, report_with_error)
        self.assertEqual(report_with_error['error_log'], error)

    @pytest.mark.unit
    @patch('MMApp.entities.jobqueue.DatasetService')
    def test_set_error_resets_prediction_status(self, MockDatasetService):

        item = {
            'prediction' : 1,
            'scoring_dataset_id' : 123
        }
        with patch.object(self.q, 'inprogress') as mock_in_progress:
            mock_in_progress.hmget.return_value = item

        dataset_service = MockDatasetService.return_value
        self.assertFalse(dataset_service.remove_lid_from_computing.called)

    @pytest.mark.db
    def test_set_error_qid(self):
        """Regression test for int qids.

        We had a case when the qid in inprogress where strs (usual) but
        the parameter to set_error was an int.
        """
        qid = 12345
        lid = '54321'

        report = {'key1' : 'value1'}

        # None before we get started
        items = self.q.inprogress.hmget(qid)
        self.assertFalse(items, 'Shouldn\'t be anything in here yet')

        # Arrange: Add to the in-progress q and check it was saved
        self.q.inprogress.hmset({str(qid): json.dumps(report)})
        # Act
        error = 'Ugly error'
        message = {'lid': lid, 'error': error}
        with patch.object(self.q, 'tempstore') as mock_tempstore:
            self.q.set_error(qid, lid, message)
            self.assertTrue(mock_tempstore.commit.called)

    @pytest.mark.unit
    def test_set_error_records_the_deserialized_error_message(self):
        qid = 25
        lid = '53e2797b637aba4d28261786'

        expected_error = 'BOOM!'

        message = {'error' : pickle.dumps(Exception(expected_error))}
        actual_error = self.q.set_error(qid, lid, message)

        self.assertEqual(expected_error, actual_error)

    @pytest.mark.unit
    def test_validate_user_model_required_keys(self):
        user_model_request = {
            "model_type": "user model 1",
        }
        # Act & Assert
        self.assertRaisesRegexp(QueueException, 'Invalid request', self.q.validate_user_model, user_model_request)

    @pytest.mark.unit
    def test_validate_user_model_target(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        user_model_request = {
            "key": "1",
            "model_type": "user model 1",
            "modelfit": "function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  gbm.fit(datasub,response,n.trees=500, interaction.depth=10,shrinkage=0.1,bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n",
            "modelpredict": "function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  predict.gbm(model,datasub,n.trees=500,type=\"response\");\n}\n",
            "pid": pid,
            "uid": uid
        }

        project = {
            "_id" : "5277b75c637aba14406f7480",
            "active" : 1,
            "created" : 1383856656.47,
            "default_dataset_id" : "e6d2f674-2f8e-477b-9084-895357e9fcf5",
            "holdout_pct" : 20,
            "holdout_unlocked" : False,
            "originalName" : "",
            "partition" : {
                "folds" : 5,
                "holdout_pct": 20,
                "reps" : 5
            },
            "stage" : "modeling",
            "target" : {
                "type" : "Binary",
                "name" : "IsBadBuy",
                "size" : 157
            },
            "target_options" : {
                "positive_class" : None,
                "name" : "IsBadBuy",
                "missing_maps_to" : None
            },
            "uid" : "5277b75c637aba1117b334ef"
        }


        with patch('MMApp.entities.jobqueue.ProjectService') as MockProjectService:
            project_service = MockProjectService.return_value

            # Arrange: Valid request
            project_service.get.return_value = project
            # Act
            output = self.q.validate_user_model(user_model_request)
            # Assert
            self.assertTrue(output)

            # Arrange: Invalid request - No target
            project.pop('target')
            project_service.get.return_value = project
            # Act & Assert
            self.assertRaisesRegexp(QueueException, 'The target variable has not been set', self.q.validate_user_model, user_model_request)


    @pytest.mark.unit
    def test_validate_user_model_holdout_locked_sample_gt_training(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        user_model_request = {
            'samplepct': 96,
            "key": "1",
            "modeltype":"R",
            "model_type": "RStudio model 1",
            "modelfit": "function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  gbm.fit(datasub,response,n.trees=500, interaction.depth=10,shrinkage=0.1,bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n",
            "modelpredict": "function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  predict.gbm(model,datasub,n.trees=500,type=\"response\");\n}\n",
            "pid": pid,
            "uid": uid
        }

        project = {
            "_id" : "5277b75c637aba14406f7480",
            "active" : 1,
            "created" : 1383856656.47,
            "default_dataset_id" : "e6d2f674-2f8e-477b-9084-895357e9fcf5",
            "holdout_pct" : 20,
            "holdout_unlocked" : False,
            "originalName" : "",
            "partition" : {
                "folds" : 5,
                "holdout_pct": 20,
                "reps" : 5
            },
            "stage" : "modeling",
            "target" : {
                "type" : "Binary",
                "name" : "IsBadBuy",
                "size" : 157
            },
            "target_options" : {
                "positive_class" : None,
                "name" : "IsBadBuy",
                "missing_maps_to" : None
            },
            "uid" : "5277b75c637aba1117b334ef"
        }


        with patch('MMApp.entities.jobqueue.ProjectService') as MockProjectService:
            project_service = MockProjectService.return_value

            project_service.get.return_value = project

            # Act & Assert
            self.assertRaisesRegexp(QueueException, 'Invalid request: when holdout is locked', self.q.validate_user_model, user_model_request)

    @pytest.mark.db
    def test_queue_len(self):
        progress_sink = Mock()
        queue = QueueService(self.project_id, progress_sink)
        n_items = 5
        for i in xrange(n_items):
            queue.push(str(i))

        self.assertEqual(queue.len(), n_items)

    @pytest.mark.db
    def test_inprogress_id_by_dict(self):
        progress_sink = Mock()
        queue = QueueService(self.project_id, progress_sink)
        new_item = {'qid':'1','some':'blah','fake':'foo','job':'bar'}
        queue.inprogress.add(new_item)
        items = queue.inprogress.hmget()
        self.assertIn('1', items)

    @pytest.mark.db
    def test_parallelcv(self):
        item = {'max_reps':5,'blueprint':'test', 'lid':'test', 'partition_stats':{str((0,-1)):'test'}}
        item.update(self.q_item)
        job = DataRobotJob(item)
        joblist = job.to_joblist()
        self.assertIsInstance(joblist,list)
        self.assertEqual(len(joblist),4)
        check = set()
        for j in joblist:
            check.add(j['partitions'][0][0])
            self.assertEqual(j.get('s'),1)
        self.assertEqual(check,set(range(1,5)))
        self.q.add(*joblist)
        q = self.q.get()
        self.assertEqual(len(q),5)
        for i in q:
            self.assertTrue(i.get('qid')==-1 or i.get('blueprint')=='test')
            self.assertTrue(i.get('qid')==-1 or i.get('s')==1)

class TestVerifyRequestIsUnique(unittest.TestCase):

    def setUp(self):
        self.patchers = []
        project_service_patch = patch('common.services.queue_service_base.ProjectService')
        MockProjectService = project_service_patch.start()
        self.patchers.append(project_service_patch)

        self.uid = ObjectId()
        self.pid = ObjectId()
        self.lid = ObjectId()

        self.project_service = MockProjectService.return_value
        self.queue = QueueService(self.pid, Mock(), uid = self.uid)
        self.addCleanup(self.stopPatching)

    def stopPatching(self):
        super(TestVerifyRequestIsUnique, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    @patch('common.services.queue_service_base.find_model_in_queue')
    def test_put_finds_duplicate_in_queue(self, mock_find_model_in_queue):
        self.project_service.read_leaderboard_item.return_value = None
        mock_find_model_in_queue.return_value = True

        request = {
            'pid' : self.pid,
            'blueprint_id': 'e27d0e503e6324730d773a2e2e4dded1',
            'bp': 4,
            'dataset_id': '53a9bd64637aba7942799231',
            'max_reps': 1,
            'model_type': 'RandomForest Classifier (Gini)',
            'samplepct': 59,
        }

        with self.assertRaises(QueueException):
            self.queue.verify_request_is_unique(request)

    def test_same_signature_and_lid(self):
        item = {
            '_id' : self.lid,
            'pid' : self.pid,
            'blueprint_id': 'e27d0e503e6324730d773a2e2e4dded1',
            'dataset_id': '53a9bd64637aba7942799231',
            'samplepct': 64
        }

        self.project_service.read_leaderboard_item.return_value = item

        request = {}
        request.update(item)
        request['lid'] = item['_id']

        # Same id is cool, it may be updating partitions
        is_unique = self.queue.verify_request_is_unique(request)
        self.assertTrue(is_unique)

    def test_same_signature_different_lids(self):
        item = {
            '_id' : self.lid,
            'pid' : self.pid,
            'blueprint_id': 'e27d0e503e6324730d773a2e2e4dded1',
            'dataset_id': '53a9bd64637aba7942799231',
            'samplepct': 64
        }

        self.project_service.read_leaderboard_item.return_value = item

        request = {}
        request.update(item)
        request['_id'] = ObjectId()

        with self.assertRaises(QueueException):
            self.queue.verify_request_is_unique(request)


    def test_predictions_are_ok(self):
        request = {
            '_id' : self.lid,
            'pid' : self.pid,
            'blueprint_id': 'e27d0e503e6324730d773a2e2e4dded1',
            'dataset_id': '53a9bd64637aba7942799231',
            'predict': 1,
            'scoring_dataset_id': ObjectId(),
            'samplepct': 64
        }

        lb_item = self.queue.verify_request_is_unique(request)
        self.assertIsNone(lb_item)

    def test_5cv_are_ok(self):
        bp_id  = '1b9beceb16bac13d9f117c4325dad80f'
        dataset_id = '53aafa0e637aba06f7edb400'
        lid = ObjectId('53a9bd64637aba9923179427')
        lb = {
            '_id' : lid,
            'blueprint_id' : bp_id,
            'dataset_id' : dataset_id,
            'pid' : ObjectId(),
            'qid' : 29,
            'runs' : 1,
            'samplepct' : 30
        }

        self.project_service.read_leaderboard_item.return_value = lb

        request = [
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':5, 'dataset_id':dataset_id, 'new_lid':False },
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':4, 'dataset_id':dataset_id, 'new_lid':False },
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':3, 'dataset_id':dataset_id, 'new_lid':False },
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':2, 'dataset_id':dataset_id, 'new_lid':False },
        ]

        for r in request:
            result = self.queue.verify_request_is_unique(r)
            self.assertIsNotNone(result)

class TestGetUniqueJobList(unittest.TestCase):
    def setUp(self):
        self.pid = ObjectId()
        self.uid = ObjectId()
        self.lid = ObjectId()
        self.dataset_id = ObjectId()
        self.blueprint_id = '08f15f00ed64dfd34fbae67b442228e5'
        self.queue = QueueService(self.pid, Mock(), uid = self.uid)

    def test_single_new_request_is_okay(self):
        j1 = {
            '_id': self.lid,
            'pid': self.pid,
            'blueprint_id': self.blueprint_id,
            'dataset_id': self.dataset_id,
            'samplepct': 64
        }
        joblist = [j1]

        with patch.object(self.queue, 'verify_request_is_unique', return_value = None):
            unique_job_list = self.queue.get_unique_job_list(joblist)

        self.assertEqual(unique_job_list, joblist)

    def test_single_existing_lid_is_okay(self):
        j1 = {
            'lid': self.lid,
            'pid': self.pid,
            'blueprint_id': self.blueprint_id,
            'dataset_id': self.dataset_id,
            'samplepct': 64
        }

        joblist = [j1]

        lb = {}
        lb.update(j1)

        with patch.object(self.queue, 'verify_request_is_unique', return_value = lb):
            unique_job_list = self.queue.get_unique_job_list(joblist)

        self.assertEqual(unique_job_list, joblist)

    def test_multiple_new_requests_chooses_one(self):
        j1 = {
            'lid': self.lid,
            'pid': self.pid,
            'blueprint_id': self.blueprint_id,
            'dataset_id': self.dataset_id,
            'samplepct': 64
        }

        j2 = {}
        j2.update(j1)
        j2['lid'] = ObjectId()

        joblist = [j1, j2]

        with patch.object(self.queue, 'verify_request_is_unique', return_value = None):
            unique_job_list = self.queue.get_unique_job_list(joblist)

        self.assertEqual(len(unique_job_list), 1)
        self.assertEqual(unique_job_list[0], j1)

    def test_multiple_existing_requests_none_dropped(self):
        j1 = {
            'lid': self.lid,
            'pid': self.pid,
            'blueprint_id': self.blueprint_id,
            'dataset_id': self.dataset_id,
            'samplepct': 64
        }

        j2 = {}
        j2.update(j1)
        j2['lid'] = ObjectId()

        joblist = [j1, j2]

        with patch.object(self.queue, 'verify_request_is_unique') as mock_verify:
            mock_verify.side_effect = lambda x: x

            unique_job_list = self.queue.get_unique_job_list(joblist)

        # Verify order
        self.assertEqual(unique_job_list[0], j1)
        self.assertEqual(unique_job_list[1], j2)


    def test_mixed_new_and_existing_chooses_only_existing_lids(self):
        j1 = {
            'lid': self.lid,
            'pid': self.pid,
            'blueprint_id': self.blueprint_id,
            'dataset_id': self.dataset_id,
            'samplepct': 64
        }

        j2 = {}
        j2.update(j1)
        j2['lid'] = ObjectId()

        j3 = {}
        j3.update(j1)
        j3['lid'] = ObjectId()

        j4 = {}
        j4.update(j1)
        j4['lid'] = ObjectId()

        j5 = {}
        j5.update(j1)
        j5['lid'] = ObjectId()

        joblist = [j1, j2, j3, j4, j5]

        with patch.object(self.queue, 'verify_request_is_unique') as mock_verify:
            mock_verify.side_effect = [j1, None, j3, None, j5]

            unique_job_list = self.queue.get_unique_job_list(joblist)

        self.assertEqual(len(unique_job_list), 3)

        # Verify order
        self.assertEqual(unique_job_list[0], j1)
        self.assertEqual(unique_job_list[1], j3)
        self.assertEqual(unique_job_list[2], j5)

    def test_full_5cv_for_new_sample_size_in_one_single_request(self):
        bp_id  = '1b9beceb16bac13d9f117c4325dad80f'
        dataset_id = '53aafa0e637aba06f7edb400'
        lid = ObjectId('53a9bd64637aba9923179427')

        job_list = [
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':5, 'dataset_id':dataset_id, 'new_lid':False },
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':4, 'dataset_id':dataset_id, 'new_lid':False },
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':3, 'dataset_id':dataset_id, 'new_lid':False },
            {'lid': str(lid), 'samplepct':30, 'uid': self.uid, 'blueprint_id': bp_id, 'pid': self.pid, 'runs':2, 'dataset_id':dataset_id, 'new_lid':False },
        ]

        with patch.object(self.queue, 'verify_request_is_unique', return_value = None):
            unique_list = self.queue.get_unique_job_list(job_list)

        sorted_unique_list = sorted(unique_list, key = lambda x: x['runs'])
        sorted_job_list = sorted(unique_list, key = lambda x: x['runs'])

        self.assertEqual(sorted_unique_list, sorted_job_list)

    @pytest.mark.db
    def test_add_releases_lock_on_failure(self):
        with patch.multiple(self.queue, _add = DEFAULT, get_unique_job_list = DEFAULT) as mocks:
            mocks['_add'].side_effect = Exception('BOOM!')

            with self.assertRaises(Exception):
                self.queue.add([])

            result = self.queue.redlock.lock()
            self.assertTrue(result)

    @pytest.mark.db
    def test_two_process_can_lock_a_project_each_simultaneously(self):
        q = QueueService(ObjectId(), Mock(), uid = self.uid)
        lock_acquired = q.redlock.lock()
        self.assertTrue(lock_acquired)

        q = QueueService(ObjectId(), Mock(), uid = self.uid)

        lock_acquired = q.redlock.lock()
        self.assertTrue(lock_acquired)

    @pytest.mark.db
    def test_unique_job_list_checks_in_progress_jobs(self):

        j1 = {
            'lid': str(self.lid),
            'pid': str(self.pid),
            'blueprint_id': self.blueprint_id,
            'dataset_id': str(self.dataset_id),
            'samplepct': 64,
            'qid': 1,
            'max_reps': 1
        }

        j2 = {
            'lid': str(self.lid),
            'pid': str(self.pid),
            'blueprint_id': self.blueprint_id,
            'dataset_id': str(self.dataset_id),
            'samplepct': 25,
            'qid': 2,
            'max_reps': 1
        }

        job_list = [j1, j2]
        self.queue.inprogress.add(j1)

        result =  self.queue.get_unique_job_list(job_list)
        self.assertTrue(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], j2)


if __name__ == '__main__':
    unittest.main()
