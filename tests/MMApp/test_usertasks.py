# -*- coding: UTF-8 -*-# -*- coding: UTF-8 -*-
####################################################################
#
#       Test for MMApp User Tasks Service Class
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc 2013
#
####################################################################

import sys
import os
import unittest
import json
import logging
import string
import random

from MMApp.api import app as api_app
from MMApp.app import app as mmapp_app
from MMApp.entities.user_tasks import UserTasks, TaskAccessError, code_hash, code_trim, model_doc
from MMApp.entities.user import UserService
from MMApp.entities.roles import RoleProvider
from MMApp.entities.permissions import Roles, Permissions
from MMApp.entities.jobqueue import QueueService
from MMApp.entities.ide import IdeService, IdeSetupStatus

from config.test_config import db_config
from common.wrappers import database
from common.wrappers.database import ObjectId
from common.engine.progress import Progress, ProgressSink
from common.api.api_client import APIClient

from tests.safeworkercode.usermodel_test_base import UserModelTestBase

class UserTasksHelperMethodsTest(UserModelTestBase):
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

    def fake_data(self,uid=None):
        if uid is None:
            uid = self.persistent.create({'username':'testuser1'}, table='users')
        x = ' a b \n \n c d '
        data = {'modelfit':x,'modelpredict':x,'name':'modelx','uid':uid,'type':'modeling', 'modeltype':'R'}
        x2 = 'asdf'
        data2 = {'modelfit':x2,'modelpredict':x2,'name':'modelx','uid':uid,'type':'modeling', 'modeltype':'R'}
        return uid,x,data,x2,data2

    def test_code_trim(self):
        #NOTE: modifying the trim method can cause errors accessing stored data
        x = ' a b \n \n c d '
        answer = ' a b \n c d '
        out = code_trim(x)
        self.assertEqual(out, answer)

    def test_code_trim_with_unicode(self):
        x = u'Les playmobils sont des jouets indémodables, résistants et représentant plein de scènes de la vie;  Mes enfants adorent et y jouent très souvent!'
        out = code_trim(x)
        self.assertEqual(x, out)

    def test_code_hash(self):
        #NOTE: modifying the hashing method can cause errors accessing stored data
        uid,x,data,x2,data2 = self.fake_data()
        out = code_hash(data)
        answer = '6a19651e9e463d5eed1ae77a5a541e7463f902df'
        self.assertEqual(out, answer)


    def test_model_doc(self):
        uid,x,data,x2,data2 = self.fake_data()
        task_id = ObjectId(None)
        out = model_doc(task_id, data)
        self.assertIsInstance(out,dict)
        self.assertEqual(set(out.keys()),set(['task_id','name','modelfit','modelpredict','hash',
            'modeltype','classname','modelsource',
            'created','type']))
        self.assertEqual(out['task_id'], task_id)
        self.assertEqual(out['hash'], code_hash(data))


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

    def fake_data(self,uid=None):
        if uid is None:
            uid = self.persistent.create({'username':'testuser1'}, table='users')
        x = ' a b \n \n c d '
        data = {'modelfit':x,'modelpredict':x,'name':'modelx','uid':uid,'type':'modeling', 'modeltype':'R'}
        x2 = 'asdf'
        data2 = {'modelfit':x2,'modelpredict':x2,'name':'modelx','uid':uid,'type':'modeling', 'modeltype':'R'}
        return uid,x,data,x2,data2

    def create_random_task(self, uid):
        x = ''.join(random.sample(list(string.letters),20))
        return x, {'modelfit':x, 'modelpredict':x, 'name':x, 'uid':uid, 'type':'modeling', 'modeltype':'R'}



    def test_store_model(self):
        uid,x,data,x2,data2 = self.fake_data()

        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)

        #case 1: new model

        rc = usertasks.store_model(data)

        #self.assertEqual(rc, 100)

        model_access = list(self.persistent.conn['model_access'].find())
        model_code = list(self.persistent.conn['model_code'].find())

        self.assertEqual(len(model_access),1)
        self.assertEqual(len(model_code),1)
        self.assertEqual(model_code[0]['modelfit'],x)
        self.assertEqual(model_code[0]['modelpredict'],x)

        #case 2: new version of same model

        rc = usertasks.store_model(data2)

        #self.assertEqual(rc, 101)

        model_access2 = list(self.persistent.conn['model_access'].find())
        model_code2 = list(self.persistent.conn['model_code'].find())

        self.assertEqual(model_access, model_access2)
        self.assertEqual(len(model_code2),2)
        self.assertEqual(model_code2[0]['task_id'], model_code2[1]['task_id'])
        self.assertEqual(model_code2[0]['name'], model_code2[1]['name'])
        self.assertEqual(model_code2[0]['modelfit'], x)
        self.assertEqual(model_code2[1]['modelfit'], x2)

        #case 3: existing model version

        rc = usertasks.store_model(data2)

        #self.assertEqual(rc, 0)

        model_access3 = list(self.persistent.conn['model_access'].find())
        model_code3 = list(self.persistent.conn['model_code'].find())

        self.assertEqual(model_access2, model_access3)
        self.assertEqual(model_code2, model_code3)

    def test_get(self):
        uid,x,data,x2,data2 = self.fake_data()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        rc = usertasks.store_model(data)
        rc = usertasks.store_model(data2)

        out = usertasks.get(uid)
        self.assertIsInstance(out,list)
        self.assertEqual(set(out[0].keys()),set(['code','name','type','mytask','task_id','description','model_type','selectedCode']))
        self.assertEqual(len(out[0]['code']),2)
        self.assertEqual(set(out[0]['code'][0].keys()),set(['created','modelfit','modelpredict','modelsource','modeltype','classname','version_id']))

        ids =[]
        for i in out[0]['code']:
            ids.append(str(i['version_id']))

        out = usertasks.get_tasks_by_ids(uid, ids)
        self.assertEqual(set(out.keys()),set(ids))
        for key,value in out.items():
            self.assertEqual(set(value.keys()),set(['modelfit','modelpredict','modelsource','modeltype','classname']))
            for i in value.values():
                type(i) in [str,unicode]

        self.persistent.destroy(table='model_access')
        with self.assertRaises(TaskAccessError):
            out = usertasks.get_tasks_by_ids(uid, ids)

        self.persistent.destroy(table='model_code')
        with self.assertRaises(TaskAccessError):
            out = usertasks.get_tasks_by_ids(uid, ids)

    def test_get_many(self):
        """ test for the bug where not all tasks were reported in the UI """
        uid,x,data,x2,data2 = self.fake_data()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)

        number_of_tasks = 40
        for i in range(number_of_tasks): #create 40 random tasks
            x,data = self.create_random_task(uid)
            rc = usertasks.store_model(data)

        out = usertasks.get(uid)
        self.assertEqual(len(out), number_of_tasks)


    def test_share_add(self):
        uid,x,data,x2,data2 = self.fake_data()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        rc = usertasks.store_model(data)
        rc = usertasks.store_model(data2)
        uid2 = self.persistent.create({'username':'testuser2'}, table='users')
        uid3 = self.persistent.create({'username':'testuser3'}, table='users')
        uid4 = self.persistent.create({'username':'testuser4'}, table='users')
        data['uid'] = uid4
        rc = usertasks.store_model(data)
        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']

        #case 1: successful add
        rc = usertasks.share_add(uid, task_id, uid2, 'testuser2')
        self.assertEqual(rc,100)

        model_access = list(self.persistent.conn['model_access'].find())
        self.assertEqual(len(model_access),3)

        #case 2: not authorized to add
        rc = usertasks.share_add(uid3, task_id, uid4, 'testuser4')
        self.assertEqual(rc,200)

        model_access = list(self.persistent.conn['model_access'].find())
        self.assertEqual(len(model_access),3)

        #case 3: nothing to add
        rc = usertasks.share_add(uid, task_id, uid2, 'testuser2')
        self.assertEqual(rc,102)

        model_access = list(self.persistent.conn['model_access'].find())
        self.assertEqual(len(model_access),3)

        #case 4: add_uid already has a model by this name
        rc = usertasks.share_add(uid, task_id, uid4, 'testuser4')
        self.assertEqual(rc,101)

        model_access = list(self.persistent.conn['model_access'].find())
        self.assertEqual(len(model_access),4)

    def test_share_rm(self):
        uid,x,data,x2,data2 = self.fake_data()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        rc = usertasks.store_model(data)
        rc = usertasks.store_model(data2)
        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']

        model_access = list(self.persistent.conn['model_access'].find())

        uid2 = self.persistent.create({'username':'testuser2'}, table='users')
        rc = usertasks.share_add(uid, task_id, uid2, 'testuser2')

        model_access2 = list(self.persistent.conn['model_access'].find())

        #case 1: cannot remove author
        rc = usertasks.share_rm(uid2, task_id, uid)
        self.assertEqual(rc, 201)

        model_access3 = list(self.persistent.conn['model_access'].find())
        self.assertEqual(model_access3, model_access2)

        #case 2: successful removal
        rc = usertasks.share_rm(uid, task_id, uid2)
        self.assertEqual(rc, 100)

        model_access4 = list(self.persistent.conn['model_access'].find())
        self.assertEqual(model_access4, model_access)

    def test_share_list(self):
        uid,x,data,x2,data2 = self.fake_data()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        rc = usertasks.store_model(data)
        rc = usertasks.store_model(data2)
        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']
        uid2 = self.persistent.create({'username':'testuser2'}, table='users')
        rc = usertasks.share_add(uid, task_id, uid2, 'testuser2')

        model_access = list(self.persistent.conn['model_access'].find())

        out = usertasks.share_list(uid, task_id)
        self.assertEqual(len(out),2)
        self.assertEqual(set(out[0].keys()), set(['uid','username','author']))

    def test_share(self):
        userservice = UserService('a@asdf.com')
        userservice.create_account('asdfasdf')
        uid,x,data,x2,data2 = self.fake_data(userservice.uid)
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        rc = usertasks.store_model(data)
        rc = usertasks.store_model(data2)
        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']
        userservice = UserService('b@asdf.com')
        userservice.create_account('asdfasdf')

        model_access = list(self.persistent.conn['model_access'].find())

        #case 1: request to add a user that doesn't exist
        request = {'task_id':str(task_id), 'action':'add', 'username':'nonexisting-user'}
        out = usertasks.share(uid, request)
        self.assertEqual(out, {'username': 'nonexisting-user', 'message': 'User not found', 'error': 1})

        model_access2 = list(self.persistent.conn['model_access'].find())
        self.assertEqual(model_access, model_access2)

        #case 2: request to add a valid user
        request = {'task_id':str(task_id), 'action':'add', 'username':'b@asdf.com'}
        out = usertasks.share(uid, request)
        self.assertEqual(out, {'username': 'b@asdf.com', 'message': 'OK', 'error': 0})

        model_access3 = list(self.persistent.conn['model_access'].find())
        self.assertEqual(len(model_access3), 2)

        #case 3: request to remove a valid user
        request = {'task_id':str(task_id), 'action':'remove', 'username':'b@asdf.com'}
        out = usertasks.share(uid, request)
        self.assertEqual(out, {'username': 'b@asdf.com', 'message': 'OK', 'error': 0})

        model_access4 = list(self.persistent.conn['model_access'].find())
        self.assertEqual(model_access4, model_access)

    def test_store_from_ide(self):
        username = 'a@asdf.com'
        userservice = UserService(username)
        userservice.create_account('asdfasdf')
        uid = userservice.uid
        pid = self.persistent.create({'target':'asdfasdfasdf', 'default_dataset_id':'asdf',
            'partition':{'holdout_pct':20,'reps':5}}, table='project')
        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        modelfit = """function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n
                            gbm.fit(datasub,response,n.trees=500, interaction.depth=10,shrinkage=0.1,
                            bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n"""

        modelpredict = """function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n
                            predict.gbm(model,datasub,n.trees=500,type=\"response\");\n}\n"""

        user_model_request = {
            "key": "1",
            "model_type": "user model 1",
            "modelfit": modelfit,
            "modelpredict": modelpredict,
            "uid": str(uid),
            "pid": str(pid)
        }

        with api_app.test_client() as api_client:
            response = api_client.post('queue', content_type='application/json',
                                       headers=self.api_auth_header,
                                       data=json.dumps(user_model_request))
            self.assertEqual(response.status_code, 200)
            api_data = json.loads(response.data)
            self.assertEqual(api_data['message'], 'OK')

            model_access = list(self.persistent.conn['model_access'].find())
            model_code = list(self.persistent.conn['model_code'].find())
            print model_code

            self.assertEqual(len(model_access),1)
            self.assertEqual(model_access[0]['name'],'user model 1')
            self.assertEqual(model_access[0]['uid'],ObjectId(uid))
            self.assertEqual(model_access[0]['username'],username)
            self.assertEqual(len(model_code),1)
            self.assertEqual(model_code[0]['modelfit'],modelfit)
            self.assertEqual(model_code[0]['modelpredict'],modelpredict)

    def test_store_python_model_from_ide_works(self):
        userservice = UserService('a@asdf.com')
        userservice.create_account('asdfasdf')
        uid = userservice.uid
        pid = self.persistent.create({'target':'asdfasdfasdf', 'default_dataset_id':'asdf',
            'partition':{'holdout_pct':20,'reps':5}}, table='project')

        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        user_model_request = self.generate_one_py_request()
        user_model_request['uid'] = str(uid)
        user_model_request['pid'] = str(pid)

        with api_app.test_client() as api_client:
            response = api_client.post('queue', content_type='application/json',
                                        headers=self.api_auth_header,
                                        data = json.dumps(user_model_request))
            self.assertEqual(response.status_code, 200)
            api_data = json.loads(response.data)
            self.assertEqual(api_data['message'], 'OK')

            model_access = list(self.persistent.conn['model_access'].find())
            model_code = list(self.persistent.conn['model_code'].find())

            self.assertEqual(len(model_access),1)
            self.assertEqual(model_access[0]['name'],'user model 1')
            self.assertEqual(model_access[0]['uid'],ObjectId(uid))
            self.assertEqual(len(model_code),1)
            self.assertEqual(model_code[0]['modelsource'],
                             user_model_request['modelsource'])

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

    def test_update_description(self):
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()

        rc = usertasks.update_description(uid, task_id, 'my task description')

        model_code = list(self.persistent.conn['model_code'].find())
        self.assertEqual(sum(i['name']=='modelx' for i in model_code),2)
        self.assertEqual(all(i.get('description') for i in model_code), True)

    def test_rename(self):
        username,uid,task_id = self.set_fixture()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username2 = 'b@asdf.com'
        userservice = UserService(username2)
        userservice.create_account('asdfasdf')
        uid2 = userservice.uid
        rc = usertasks.share_add(ObjectId(uid), task_id, ObjectId(uid2), username2)

        rc = usertasks.rename(uid, task_id, 'test new task name')

        model_access = list(self.persistent.conn['model_access'].find())
        model_code = list(self.persistent.conn['model_code'].find())

        self.assertEqual(all(i['name']=='test new task name' for i in model_access),True)
        self.assertEqual(all(i['name']=='test new task name' for i in model_code),True)

    def test_remove_task(self):
        username,uid,task_id = self.set_fixture()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username2 = 'b@asdf.com'
        userservice = UserService(username2)
        userservice.create_account('asdfasdf')
        uid2 = userservice.uid
        model_access = list(self.persistent.conn['model_access'].find())
        model_code = list(self.persistent.conn['model_code'].find())

        rc = usertasks.share_add(ObjectId(uid), task_id, ObjectId(uid2), username2)
        model_access2 = list(self.persistent.conn['model_access'].find())
        self.assertEqual(len(model_access2),2)

        rc = usertasks.remove_task(uid2,task_id)
        self.assertEqual(rc,101)
        model_access3 = list(self.persistent.conn['model_access'].find())
        model_code3 = list(self.persistent.conn['model_code'].find())
        self.assertEqual(model_access3, model_access)
        self.assertEqual(model_code3, model_code)

        rc = usertasks.remove_task(uid,task_id)
        self.assertEqual(rc,100)
        model_access4 = list(self.persistent.conn['model_access'].find())
        model_code4 = list(self.persistent.conn['model_code'].find())
        self.assertEqual(model_access4,[])
        self.assertEqual(model_code4,[])

    def test_copy_task(self):
        username,uid,task_id = self.set_fixture()
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)

        tasks = usertasks.get(uid)
        version_id = tasks[0]['code'][0]['version_id']
        task_id = tasks[0]['task_id']

        rc = usertasks.copy(uid, task_id, version_id, 'my new task')

        model_access = list(self.persistent.conn['model_access'].find())
        model_code = list(self.persistent.conn['model_code'].find())

        self.assertEqual(len(model_access),2)
        self.assertEqual(model_access[-1]['name'],'my new task')
        self.assertNotEqual(model_access[0]['task_id'],model_access[-1]['task_id'])

        self.assertEqual(model_code[-1]['name'],'my new task')
        self.assertNotEqual(model_code[0]['task_id'],model_code[-1]['task_id'])
        self.assertEqual(model_code[0]['hash'],model_code[-1]['hash'])

    def test_app(self):
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()
        username2 = 'b@asdf.com'
        userservice = UserService(username2)
        userservice.create_account('asdfasdf')

        model_access1 = list(self.persistent.conn['model_access'].find())

        request = {'action':'add','username':username2}
        url = '/task/%s/share'%task_id
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.post(url, content_type='application/json', data = json.dumps(request))
            self.assertEqual(response.status_code, 200)

        model_access = list(self.persistent.conn['model_access'].find())
        self.assertEqual(len(model_access),2)
        self.assertEqual(len(set(i['task_id'] for i in model_access)),1)
        self.assertEqual(set(i['username'] for i in model_access),set([username,username2]))

        url = '/task/%s/share'%task_id
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.get(url)
            self.assertEqual(response.status_code, 200)

        request = {'action':'remove','username':username2}
        url = '/task/%s/share'%task_id
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.post(url, content_type='application/json', data = json.dumps(request))
            self.assertEqual(response.status_code, 200)

        model_access2 = list(self.persistent.conn['model_access'].find())
        self.assertEqual(model_access1,model_access2)

        request = {'description':'my task description'}
        url = '/task/%s'%str(task_id)
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.put(url, content_type='application/json', data = json.dumps(request))
            self.assertEqual(response.status_code, 200)

        model_code = list(self.persistent.conn['model_code'].find())
        self.assertEqual(all(i.get('description')=='my task description' for i in model_code), True)

        url = '/task'
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.get(url)
            self.assertEqual(response.status_code, 200)

        request = {'new_name':'new task name'}
        url = '/task/%s'%str(task_id)
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.put(url, content_type='application/json', data = json.dumps(request))
            self.assertEqual(response.status_code, 200)

        model_code = list(self.persistent.conn['model_code'].find())
        self.assertEqual(all(i.get('name')=='new task name' for i in model_code), True)

        tasks = usertasks.get(uid)
        version_id = tasks[0]['code'][0]['version_id']
        task_id = tasks[0]['task_id']

        request = {'version_id':str(version_id),'newname':'copied task name'}
        url = '/task/%s/copy'%str(task_id)
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.post(url, content_type='application/json', data = json.dumps(request))
            self.assertEqual(response.status_code, 200)

        model_code2 = list(self.persistent.conn['model_code'].find())
        self.assertEqual(len(model_code2),len(model_code)+1)
        self.assertIn('copied task name',set(i['name'] for i in model_code2))

        url = '/task/%s'%str(task_id)
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.delete(url)
            self.assertEqual(response.status_code, 200)

        model_code3 = list(self.persistent.conn['model_code'].find())
        self.assertTrue(all(i['name']=='copied task name' for i in model_code3))

        task_id = model_code3[0]['task_id']
        url = '/task/%s'%str(task_id)
        with mmapp_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = username
            response = client.delete(url)
            self.assertEqual(response.status_code, 200)

        model_code4 = list(self.persistent.conn['model_code'].find())
        self.assertEqual(model_code4,[])



    def test_api(self):
        """
        test api calls that use UserTask
        """
        usertasks = UserTasks(persistent=self.persistent, tempstore=self.tempstore)
        username,uid,task_id = self.set_fixture()
        pid = self.persistent.create({'target':'asdfasdfasdf', 'default_dataset_id':'asdf',
            'partition':{'holdout_pct':20,'reps':5}}, table='project')
        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        tasks = usertasks.get(uid)
        task_id = tasks[0]['task_id']
        version_id = tasks[0]['code'][0]['version_id']

        url = '/task_code/%s/%s/%s'%(uid,task_id,version_id)
        with api_app.test_client() as client:
            response = client.get(url, headers=self.api_auth_header)
            self.assertEqual(response.status_code, 200)

        task_code = json.loads(response.data)

        self.assertEqual(set(task_code.keys()), set(['modelfit','modelpredict','modelsource']))






    def test_retrieve_R_model_from_ide(self):
        userservice = UserService('a@asdf.com')
        userservice.create_account('asdfasdf')
        uid = userservice.uid
        pid = self.persistent.create({'target':'asdfasdfasdf', 'default_dataset_id':'asdf',
            'partition':{'holdout_pct':20,'reps':5}}, table='project')

        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])

        modelfit_v1 = """function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n
                            gbm.fit(datasub,response,n.trees=500, interaction.depth=10,shrinkage=0.1,
                            bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n"""

        modelpredict_v1 = """function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n
                            predict.gbm(model,datasub,n.trees=500,type=\"response\");\n}\n"""

        modelfit_v2 = """function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n
                            gbm.fit(datasub,response,n.trees=300, interaction.depth=10,shrinkage=0.1,
                            bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n"""

        modelpredict_v2 = """function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n
                            predict.gbm(model,datasub,n.trees=300,type=\"response\");\n}\n"""

        user_model_request = {
            "key": "1",
            "model_type": "user model 1",
            "modelfit": modelfit_v1,
            "modelpredict": modelpredict_v1,
            "uid": str(uid),
            "pid": str(pid)
        }

        with api_app.test_client() as api_client:
            # store model v1
            response = api_client.post('queue', content_type='application/json',
                                       headers=self.api_auth_header, data=json.dumps(user_model_request))
            self.assertEqual(response.status_code, 200)
            # store model v2
            user_model_request['modelfit'] = modelfit_v2
            user_model_request['modelpredict'] = modelpredict_v2
            response = api_client.post('queue', content_type='application/json',
                                       headers=self.api_auth_header, data=json.dumps(user_model_request))
            self.assertEqual(response.status_code, 200)
            # get task by name only
            response = api_client.get('task/%s/user%%20model%%201?pid=%s&key=1' % (str(uid), str(pid)),
                                       headers=self.api_auth_header)
            self.assertEqual(response.status_code, 200)
            # should return last stored model
            self.assertIn(modelfit_v2,response.data)
            self.assertIn(modelpredict_v2,response.data)
            # get task by name - version 1
            response = api_client.get('task/%s/user%%20model%%201/v1?pid=%s&key=1' % (str(uid), str(pid)),
                                       headers=self.api_auth_header)
            self.assertEqual(response.status_code, 200)
            # should return model v1
            self.assertIn(modelfit_v1,response.data)
            self.assertIn(modelpredict_v1,response.data)
            # get task by name - version 1
            response = api_client.get('task/%s/user%%20model%%201/2?pid=%s&key=1' % (str(uid), str(pid)),
                                       headers=self.api_auth_header)
            self.assertEqual(response.status_code, 200)
            # should return model v2
            self.assertIn(modelfit_v2,response.data)
            self.assertIn(modelpredict_v2,response.data)
            # handle invalid name
            response = api_client.get('task/%s/sdflhwerfhsdfsdfsdfca/2?pid=%s&key=1' % (str(uid), str(pid)),
                                       headers=self.api_auth_header)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Invalid task name",response.data)
            # handle invalid version
            response = api_client.get('task/%s/user%%20model%%201/3?pid=%s&key=1' % (str(uid), str(pid)),
                                       headers=self.api_auth_header)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Invalid version",response.data)

            model_access = list(self.persistent.conn['model_access'].find())
            model_code = list(self.persistent.conn['model_code'].find().sort('created'))

            self.assertEqual(len(model_access),1)
            self.assertEqual(model_access[0]['name'],'user model 1')
            self.assertEqual(model_access[0]['uid'],ObjectId(uid))
            self.assertEqual(len(model_code),2)
            self.assertEqual(model_code[0]['modelfit'], modelfit_v1)
            self.assertEqual(model_code[1]['modelfit'], modelfit_v2)











if __name__=='__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    unittest.main()
