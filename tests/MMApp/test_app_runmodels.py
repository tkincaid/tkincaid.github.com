####################################################################
#
#       Test for MMApp Run Models / User Tasks \
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc 2013
#
####################################################################

import copy
import sys
import os
import unittest
import json
import logging
from mock import Mock, patch, DEFAULT

from MMApp.api import app as api_app
from MMApp.app import app as mmapp_app
from MMApp.entities.user_tasks import UserTasks, TaskAccessError
from MMApp.entities.user import UserService
from MMApp.entities.roles import RoleProvider
from MMApp.entities.project import ProjectService
from MMApp.entities.permissions import Roles, Permissions
from MMApp.entities.jobqueue import QueueService
from MMApp.entities.ide import IdeService, IdeSetupStatus

from config.test_config import db_config
from common.wrappers import database
from common.wrappers.database import ObjectId
from common.engine.progress import Progress, ProgressSink
from common.api.api_client import APIClient
import common.entities.blueprint as blueprint_module
from common.entities.job import ModelRequest, BlenderRequest

from tests.safeworkercode.usermodel_test_base import UserModelTestBase

class UserTasksTest(UserModelTestBase):

    @classmethod
    def setUpClass(self):
        sys.path.append('..')
        self.tempstore = database.new('tempstore')
        self.persistent = database.new('persistent')
        #TODO: update tests to use fake dynamic IDE keys instead of the master api key
        self.api_auth_header = {'web-api-key': api_app.web_api_key}

    @classmethod
    def tearDownClass(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table='model_access')
        self.persistent.destroy(table='model_code')
        self.persistent.destroy(table='users')
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='user')

    def setUp(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table='model_access')
        self.persistent.destroy(table='model_code')
        self.persistent.destroy(table='users')
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='user')
        self.persistent.destroy(table='leaderboard')
        self.pid = ObjectId(None)
        self.dataset_id = ObjectId(None)

    def fake_data(self,uid=None):
        """ make fake user tasks
        """
        if uid is None:
            uid = self.persistent.create({'username':'testuser1'}, table='users')
        x = ' a b \n \n c d '
        data = {'modelfit':x,'modelpredict':x,'name':'modelx','uid':uid,'type':'modeling', 'modeltype':'R'}
        x2 = 'asdf'
        data2 = {'modelfit':x2,'modelpredict':x2,'name':'modelx','uid':uid,'type':'modeling', 'modeltype':'R'}
        return uid,x,data,x2,data2

    def set_fixture(self):
        username = 'a@asdf.com'
        userservice = UserService(username)
        userservice.create_account('asdfasdf')
        uid,x,data,x2,data2 = self.fake_data(userservice.uid)
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        rc = usertasks.store_model(data)
        rc = usertasks.store_model(data2)
        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']
        return username,uid,task_id

    def make_request(self,pid,uid,usertasks=[],include_bp=False):
        """ batch request including some datarobot blueprints and optionally some user tasks """
        bp1 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GLMB'],'P']}
        bp2 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['RFI nt=10;ls=5'],'P']}
        args = {'samplepct':50, 'dataset_id':str(self.dataset_id), 'pid':str(pid), 'uid':str(uid), 'max_reps':1, 'total_size':200, 'bp': 'x'}
        out = []
        menu = {}
        for bp in [bp1,bp2]:
            bpid = blueprint_module.blueprint_id(bp)
            i = {'blueprint_id':bpid}
            i.update(args)
            if include_bp:
                i['blueprint'] = bp
            out.append(i)
            menu[bpid] = {'blueprint':bp}

        for i in usertasks:
            j = {'selectedCode':{'version_id':str(i)}}
            j.update(args)
            out.append(j)

        self.persistent.update(values={'menu':menu}, condition={'pid':pid}, table='metablueprint')
        return out

    def test_modelrequest_class_eq_operator(self):
        pid = str(ObjectId(None))
        uid = str(ObjectId(None))
        request = self.make_request(pid,uid)
        a,b = (ModelRequest(i,pid,uid, {'reps':5,'holdout_pct':20}) for i in request[:2])
        self.assertEqual(a in [a], True)
        self.assertEqual(a==a, True)
        self.assertEqual(b==b, True)
        self.assertEqual(a==b, False)
        request = self.make_request(pid,uid)
        for i in request:
            i['max_reps']='all'
        c,d = (ModelRequest(i,pid,uid, {'reps':5,'holdout_pct':20}) for i in request[:2])
        self.assertEqual(a==c, False)
        self.assertEqual(b==d, False)


    def test_queue_service_put(self):
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()
        pid = self.persistent.create({'target':'asdfasdfasdf', 'partition': {'holdout_pct':20,'reps':5} }, table='project')
        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']
        version_id = tasks[0]['code'][0]['version_id']

        #test a request for new models (nothing in leaderboard yet)
        # expect new_lid==True in these requests
        request = self.make_request(pid,uid,usertasks=[version_id])

        queue = QueueService(pid, Mock(), uid)
        queue.put(request)

        q = queue.get()
        counter = 0
        for qitem in q:
            if qitem['status'] == 'settings':
                continue
            self.assertEqual(qitem['new_lid'],True)
            self.assertGreaterEqual(set(qitem.keys()),set(['blueprint_id','blueprint','samplepct','dataset_id','partitions','pid','uid']))
            qitem.pop('lid')
            qitem['pid'] = ObjectId(qitem['pid'])
            qitem['partition_stats'] = dict((str(tuple(p)),'test') for p in qitem['partitions']) #partition_stats is required for next test to work
            self.assertEqual(qitem['s'],0)
            self.persistent.create(qitem, table='leaderboard')
            counter+=1
        self.assertEqual(counter,len(request))

        #empty the queue
        queue.delete()
        q = queue.get()
        self.assertEqual(len(q), 1) #empty queue (settings only)

        #test a request that match existing models in leaderboard64
        # expect new_lid==False in these requests
        request2 = []
        for r in request:
            r['max_reps']='all'
            request2.append(r)

        queue.put(request2)

        q = queue.get()
        check_lids = set()
        counter = 0
        for qitem in q:
            if qitem['status'] == 'settings':
                continue
            print qitem
            self.assertEqual(qitem['new_lid'],False)
            try:
                ObjectId(qitem['lid'])
            except:
                self.fail('expected a lid to be a valid ObjectId')
            self.assertEqual(qitem['s'],1)

            self.assertGreaterEqual(set(qitem.keys()),set(['blueprint_id','blueprint','samplepct','dataset_id','partitions','pid','uid']))
            check_lids.add( (qitem['blueprint_id'],qitem['lid']) )
            counter += 1

        self.assertEqual(counter, 4*len(request)) #4 partitions per request run as separate jobs
        #assert one-to-one mappging of request-item (blueprint_id) to lid
        self.assertEqual(len(check_lids),len(request))


    def test_app_runmodels(self):
        """
        1. create user tasks
        2. create some fake leaderboard items
        3. test user model
        4. test datarobot model
        5. test blender
        6. test model that is in leaderboard
        """
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()
        pid = self.persistent.create({'target':'asdfasdfasdf', 'partition': {'holdout_pct':20,'reps':5} }, table='project')
        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']
        version_id = tasks[0]['code'][0]['version_id']

        request = self.make_request(pid,uid,usertasks=[version_id])

        print request

        url = '/project/%s/models'%pid
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.post(url, content_type='application/json', data = json.dumps(request))
            self.assertEqual(response.status_code, 200)

        queue = QueueService(pid, ProgressSink(), uid)
        q = queue.get()
        self.assertEqual(len(q),4)

    def make_fake_leaderboard(self, pid, uid):
        request = self.make_request(pid,uid,include_bp=True)
        for n,i in enumerate(request):
            i['pid'] = ObjectId(pid)
            i['uid'] = ObjectId(uid)
            i['bp'] = str(n+1)
            i['model_type'] = 'test model %s'%n
            i['features'] = ['asdf']
            i['icons'] = [0]
            i['partition_stats'] = {str((0,-1)):'test'}
            i['test'] = {'LogLoss':[0.5], 'MAD':[0.12], 'metrics':['LogLoss', 'MAD']}
            i['max_folds'] = 0
            lid = self.persistent.create(i,table='leaderboard')
            self.persistent.update(values={'lid': str(lid)}, condition={'_id': lid}, table='leaderboard')

    @patch('MMApp.entities.project.ProjectService.get_valid_metrics')
    def test_run_blender(self, fake_metrics):
        """
        test new blender request sent thru /blend

        assumes, blender payload is creted in models-controller.js as follows:
          var blendModels = [];
          angular.forEach($scope.modelService.selectedModels, function (m) {
              blendModels.push(m.lid);
          });
          var payload = {'blender_method':blender,'blender_items':blendModels};
        """
        fake_metrics.return_value = ['LogLoss']

        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()
        dummy_projecty = {'target': {'name' : 'TargetName', 'type' : 'Binary'}, 'metric' : 'LogLoss', 'partition': {'holdout_pct':20,'reps':5} }
        pid = self.persistent.create(dummy_projecty, table='project')
        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        #fake some leaderboard models
        self.make_fake_leaderboard(pid,uid)

        #get leaderboard thru project service
        project = ProjectService(str(pid),str(uid))
        leaderboard = project.get_leaderboard(UI_censoring=True)

        blender_items = []
        for i in leaderboard:
            blender_items.append(str(i['_id']))

        #make a blender request from the UI:
        payload = {'blender_items':blender_items, 'blender_method':'GLM'}

        url = '/project/%s/blend'%pid
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.post(url, content_type='application/json', data = json.dumps(payload))
            self.assertEqual(response.status_code, 200)

        queue = QueueService(pid, ProgressSink(), uid)
        q = queue.get()
        # Expect a single model, the blender request
        n=0
        first2jobs = []
        for i in q:
            if i['status'] == 'settings':
                continue
            self.assertEqual(i['status'],'queue')
            if n<2:
                i.pop('status')
                first2jobs.append(i)
            self.assertTrue(i['lid'] not in blender_items)
            self.assertEqual(i['model_type'], 'GLM Blender')
            self.assertEqual(i['new_lid'], True)
            self.assertEqual(i['partitions'], [[0, -1]])
            n+=1
        self.assertEqual(n,1)

        old_keys = [u'blueprint', u'lid', u'samplepct', u'uid', u'blueprint_id', u'total_size', u'qid',
                u'require', u'icons', u'pid', u'max_reps', 'status', u's', u'bp', u'features', u'model_type',
                u'dataset_id', u'new_lid', u'max_folds', u'blender']
        #TODO: remove unecesary keys. Test /blend with BlenderRequest

        #test queue processing:
        queue.set_autopilot({'workers':2, 'mode':0})
        with patch('MMApp.entities.jobqueue.SecureBrokerClient') as mock:
            count = queue.start_new_tasks(2)
            self.assertEqual(count, 1)
            self.assertEqual(mock.call_count, 1)
        q = queue.get()
        n=0
        for i in q:
            if i['status'] == 'settings':
                continue
            if n<2:
                self.assertEqual(i['status'], 'inprogress')
                #check that queue did not change items when moving to "in progress"
                i.pop('status')
                self.assertEqual(i, first2jobs[n])
            else:
                self.assertEqual(i['status'], 'queue')
            n+=1

    @patch('MMApp.entities.project.ProjectService.get_valid_metrics')
    def test_BlenderRequest_submits_correct_number_partitions_5CV_not_run(self, fake_metrics):
        """
        new blender request:
        request = {'blender_request':{'items','method'}, 'samplepct', 'dataset_id'}
        """
        fake_metrics.return_value = ['LogLoss']

        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()
        pid = self.persistent.create({'target': {'name': 'TargetName'}}, table='project')
        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        #fake some leaderboard models
        self.make_fake_leaderboard(pid,uid)

        #get leaderboard thru project service
        project = ProjectService(str(pid),str(uid))
        leaderboard = project.get_leaderboard(UI_censoring=True)

        blender_items = []
        for i in leaderboard:
            blender_items.append(str(i['_id']))

        #make a blender request from the UI:
        payload = {'blender_items':blender_items, 'blender_method':'GLM'}

        req = {'blender_request': payload}
        request = BlenderRequest(req, str(pid), str(uid), {'holdout_pct':20,'reps':5})

        expected_keys = set(['samplepct', 'features', 'icons', 'partitions', 'pid', 'model_type',
                'blender_request', 'max_folds', 'blender', 'uid'])
        self.assertEqual(set(request.keys()), expected_keys)
        self.assertEqual(request['blender'] , {}) #not yet set

        #call to get input values from leaderboard and verify the request gets filled in
        request.get_inputs_from_leaderboard(project)

        expected_keys |= set(['blueprint', 'blueprint_id', 'bp', 'dataset_id'])
        self.assertEqual(set(request.keys()), expected_keys)

        #check that the blender blueprint was composed correctly
        input_id = request.input_id()
        expected_bp = {'1':[[input_id],['GLMB '],'P']}
        self.assertEqual(request['blueprint']['1'][0], expected_bp['1'][0])
        self.assertEqual(request['blueprint']['1'][1], expected_bp['1'][1])
        self.assertEqual(request['blueprint']['1'][2], expected_bp['1'][2])

        queue = QueueService(pid, Mock(), uid)
        #check joblist
        out = request.to_joblist( queue.get() )

        n=0
        for i in out:
            self.assertEqual(i['model_type'], 'GLM Blender')
            self.assertEqual(i['partitions'], [[0, -1]])
            n+=1
        self.assertEqual(n, 1)



    @patch('MMApp.entities.project.ProjectService.get_valid_metrics')
    def test_BlenderRequest_submits_correct_blender_CVjobs_in_q(self, fake_metrics):
        """
        new blender request:
        request = {'blender_request':{'items','method'}, 'samplepct', 'dataset_id'}
        """
        fake_metrics.return_value = ['LogLoss']

        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()
        pid = self.persistent.create({'target': {'name': 'TargetName'},'partition': {'reps':5, 'holdout_pct':20}}, table='project')
        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        #fake some leaderboard models
        self.make_fake_leaderboard(pid,uid)


        #get leaderboard thru project service
        project = ProjectService(str(pid),str(uid))
        leaderboard = project.get_leaderboard(UI_censoring=True)

        # Put the other cvs of the leaderboardjobs in the queue
        items = []
        for item in leaderboard:
            new_item = copy.deepcopy(item)
            lid = str(item['_id'])
            new_item['partitions'] =[[i, -1] for i in range(5)]
            new_item['pid'] = str(pid)
            new_item['lid'] = lid
            del new_item['_id']
            del new_item['partition_stats']
            items.append(new_item)
        # Get the lids for the blender items
        blender_items = []
        for i in leaderboard:
            blender_items.append(str(i['_id']))

        #make a blender request from the UI:
        payload = {'blender_items':blender_items, 'blender_method':'GLM'}

        req = {'blender_request': payload}
        request = BlenderRequest(req, str(pid), str(uid), {'reps':5,'holdout_pct':20})

        expected_keys = set(['samplepct', 'features', 'icons', 'partitions', 'pid', 'model_type',
                'blender_request', 'max_folds', 'blender', 'uid'])
        self.assertEqual(set(request.keys()), expected_keys)
        self.assertEqual(request['blender'] , {}) #not yet set

        #call to get input values from leaderboard and verify the request gets filled in
        request.get_inputs_from_leaderboard(project)

        expected_keys |= set(['blueprint', 'blueprint_id', 'bp', 'dataset_id'])
        self.assertEqual(set(request.keys()), expected_keys)

        #check that the blender blueprint was composed correctly
        input_id = request.input_id()
        expected_bp = {'1':[[input_id],['GLMB '],'P']}
        self.assertEqual(request['blueprint']['1'][0], expected_bp['1'][0])
        self.assertEqual(request['blueprint']['1'][1], expected_bp['1'][1])
        self.assertEqual(request['blueprint']['1'][2], expected_bp['1'][2])

        queue = QueueService(pid, Mock(), uid)
        queue.put(items)
        #check joblist
        out = request.to_joblist( queue.get() )

        n=0
        for i in out:
            lid = blender_items[0] if False and n<4 else blender_items[1]
            if False and n<4:
                self.assertEqual(i['lid'], lid)
            else:
                self.assertEqual(i['model_type'], 'GLM Blender')
                self.assertEqual(i['partitions'], [[i, -1] for i in range(5)])
            n+=1
        self.assertEqual(n,1)


    def test_missing_partitions(self):
        lid = str(ObjectId(None))
        pid = str(ObjectId(None))
        uid = str(ObjectId(None))
        item = {'lid':lid, 'samplepct':50, 'partition_stats':{str((0,-1)):'test'}, 'test':{'LogLoss':[0.5]}}
        queue = [{'lid':lid, 'samplepct':50, 'partitions':[[1,-1]]}]
        request = BlenderRequest({'blender_request':{'items':[lid], 'method':'GLM'}}, pid, uid, {'reps':5,'holdout_pct':20})
        out = request.find_missing_partitions(item,queue)
        self.assertEqual(sorted(out,key=lambda x:x[0]), [[i,-1] for i in range(2,5)])













if __name__=='__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    unittest.main()
