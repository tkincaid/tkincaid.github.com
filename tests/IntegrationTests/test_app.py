############################################################################
#
#       integration test for MMApp.app
#
#       Author: Ulises Reyes
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import unittest
from mock import patch, Mock
from bson.objectid import ObjectId
import json
import os
import time
import urlparse

from tests.IntegrationTests.integration_test_base import IntegrationTestBase
import MMApp.app
from MMApp.entities.user import UserService
from MMApp.utilities.web_response import *

class IntegrationTestApp(IntegrationTestBase):
    ''' Performs integration testing by testing app.py as a black box.
        In other words, some features are chained but the responses of the submitted requests are verified without further checks on other components such as the databases or file system
    '''

    @classmethod
    def setUpClass(self):
        super(IntegrationTestApp, self).setUpClass()
        IntegrationTestApp.pid = None

    def setUp(self):
        super(IntegrationTestApp, self).setUp()
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')
        if not IntegrationTestApp.pid:
            IntegrationTestApp.pid = self.create_project()
            self.logout(self.app)

        self.pid = IntegrationTestApp.pid

    def tearDown(self):
        self.logout(self.app)

    def test_logout(self):
        self.login_successfully(self.app)
        self.logout(self.app)

        user_service = UserService(self.registered_user['username'])
        user_info = user_service.get_user_info()
        self.assertFalse(user_info)

    def test_front(self):
        # with statement necessary to provide correct context
        # and access to session variable
        with self.app as c:
            # no user session set
            response = c.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'text/html; charset=utf-8')
            self.assertTrue(len(response.data) != 0)

            # Invalid user set in session
            with c.session_transaction() as sess:
                sess['user'] = 'testNoOne'
            response = c.get('/')

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'text/html; charset=utf-8')
            self.assertTrue(len(response.data) != 0)
            with c.session_transaction() as sess:
                self.assertTrue('user' not in sess)

            # Valid user set in session
            self.login_successfully(c)
            with c.session_transaction() as sess:
                sess['user'] = self.registered_user['username']

            response = c.get('/')

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'text/html; charset=utf-8')
            self.assertTrue(len(response.data) != 0)
            with c.session_transaction() as sess:
                self.assertTrue(sess['user'] == self.registered_user['username'])


    @patch('MMApp.app.is_private_mode')
    def test_upload_url_guest_user(self, mock_is_private_mode):
        mock_is_private_mode.return_value = False
        ''' Uploads a file from a URL and creates a guest account if there is not a username in the session
        '''
        with self.app as c:
            pid = self.upload_file(c, is_guest = True, from_url = True)

            response = c.get('/listprojects')
            response_data = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response_data, list)
            self.assertGreater(len(response_data), 0)
            project = response_data[0]
            criteria = ['_id', 'active', 'created', 'originalName', 'project_name']
            self.assertTrue(all(k in project for k in criteria), 'Response keys %s did not include the expected keys: %s' % (project.keys(), criteria))

            response = c.get('/project/%s' % pid)
            dataset = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(dataset, dict)
            criteria = ['_id', 'project_name', 'eda_labels1', 'eda_labels0', 'filename', 'active', 'columns', 'stage']
            self.assertTrue(all(k in dataset for k in criteria), 'Response keys %s did not include the expected keys: %s' % (dataset.keys(), criteria))

    def test_get_eda(self):
        ''' Initiates requests for univariates by calling aim. Next, it polls eda and univariate results
        '''
        with self.app as c:
            self.login_successfully(c)
            response = c.get('/eda/%s' % self.pid)
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.data)
            self.assertTrue('data' in response_data, 'eda key could not be found in the response')

    def test_queue_delete_nonexistent_qid(self):
        with self.app as c:
            self.login_successfully(c)
            qid = 9999
            response = c.delete('/project/{}/queue/{}'.format(self.pid, qid))
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.data)
            self.assertEqual(response_data['message'], 'OK')

    def test_queue(self):
        with self.app as c:
            self.login_successfully(c)
            pid = self.pid
            self.wait_for_stage(self.app, pid, 'modeling')
            q_items = self.get_q(pid, include_settings = True)

            self.assertIsInstance(q_items, list, 'The response is not a list')
            #There is a special item that contains settings
            self.assertGreater(next(i for (i,d) in enumerate(q_items) if d['status'].lower() == 'settings'), -1, 'The list did not include a settings item')

            #The rest are queue items which meet certain criteria, i.e., contain specific keys

            settings_item = q_items.pop(0)
            total_q_items = len(q_items)
            criteria = ['qid', 'pid', 'status', 'pid']
            matching_items = [i for (i,d) in enumerate(q_items) if all(k in d for k in criteria)]
            self.assertEqual( len(matching_items), total_q_items, 'Not all queue items have the following keys %s' % criteria)

            #Delete Queue items - 1st on the list
            q_id = q_items[total_q_items/2]['qid']
            response = c.delete('/project/'+str(pid)+'/queue/%s' % q_id)
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.data)
            self.assertEqual(response_data['message'], 'OK')

            #We should have less queue items now: new_total = total - deleted
            deleted_items = 1
            expected_new_total = total_q_items - deleted_items

            q_items = self.get_q(pid)
            new_total = len(q_items) #do not count the settings object
            self.assertEqual(new_total, expected_new_total, 'The new total q item count is greater than the expected one, the delete function may have failed')

    @patch('MMApp.app.RoleProvider')
    def test_queue_settings(self, MockRoleProvider):
        with self.app as c:
            self.login_successfully(c)
            pid = self.pid
            self.update_queue_settings(c, pid, 4)
            self.update_queue_settings(c, pid, -2)
            #TODO: Validation must be added to prevent high numbers
            self.update_queue_settings(c, pid, 1000)
            self.update_queue_settings(c, pid, 4)

    def test_account_signup(self):
        # Change runmode to PUBLIC
        with patch.dict(MMApp.app.app.config, {'runmode': 0 }, clear = False):
            #signup brand-new user
            data = dict(username = 'brand_new_account@example.com', password = 'super-secret')
            response = self.app.post('/account/signup', content_type='application/json', data=json.dumps(data))
            response_data = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response_data['error'], '0')
            self.assertGreater(response_data['message'].find('Sign Up Successful'), -1, response_data['message'])

            #Signup with existing account
            response = self.app.post('/account/signup', content_type='application/json', data=json.dumps(data))
            response_data = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response_data['error'], '1')
            self.assertGreater(response_data['message'].find('username is already taken'), -1, response_data['message'])

    def test_guest_account_signup(self):
        #signup guest user
        with self.app as c:
            with c.session_transaction() as sess:
                old_username = 'old_guest_account_name'
                sess['user'] = old_username

            # Change runmode to PUBLIC
            with patch.dict(MMApp.app.app.config, {'runmode': 0 }, clear = False):
                # Old guest name does not exist
                data = dict(username = 'new_account@example.com', password = 'super-secret')
                response = c.post('/account/signup', content_type='application/json', data=json.dumps(data))
                response_data = json.loads(response.data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response_data['error'], '1')
                self.assertIn('Guest not found', response_data['message'], response_data['message'])


                # guest name exist
                self.create_guest(old_username, 'super-secret')
                data = dict(username = 'new_account@example.com', password = 'super-secret')
                response = c.post('/account/signup', content_type='application/json', data=json.dumps(data))
                response_data = json.loads(response.data)
                self.assertEqual(response.status_code, 200)
                self.assertGreater(response_data['message'].find('Sign Up Successful'), -1, response_data['message'])
                self.assertEqual(response_data['error'], '0')


    def test_account_login(self):
        with self.app as c:
            # Invalid username
            response = self.login('notauser', '12345')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')
            self.assertDictEqual(json.loads(response.data), {"message": "Incorrect username/password", "error": "1"})
            with c.session_transaction() as sess:
                self.assertTrue('user' not in sess)

            # Wrong password
            response = self.login(self.registered_user['username'], 'wrongpassword')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')
            self.assertDictEqual(json.loads(response.data), {"message": "Incorrect username/password", "error": "1"})
            with c.session_transaction() as sess:
                self.assertTrue('user' not in sess)

            # Valid login
            self.login_successfully(c)

            # Already logged in
            response = self.login(self.registered_user['username'], 'password1')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')
            data = json.loads(response.data)
            self.assertItemsEqual(data, {
                'message': 'Already logged in',
                'error': '0',
                'uid': self.registered_user['uid']
            })
            with c.session_transaction() as sess:
                self.assertTrue('user' in sess)

    def test_account_user(self):
        # not testing for sc (status_code) 1  = Logged in as guest user
        with self.app as c:
            # statusCode (status_code) -1 = Not logged in
            response = c.get('/account/profile')
            user = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')
            self.assertEqual(user['statusCode'], '-1')

            with c.session_transaction() as sess:
                sess['user'] = 'testNoOne'
            response = c.get('/account/profile')

            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.content_type, 'application/json')
            self.assertDictEqual(json.loads(response.data), {"message": "bad request"})

            # statusCode (status_code) 0  = Logged in as registered user
            with c.session_transaction() as sess:
                sess['user'] = self.registered_user['username']
            response = c.get('/account/profile')
            user = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')
            self.assertEqual(user['username'], self.registered_user['username'])
            self.assertEqual(user['statusCode'], '0')

    def test_create_job_does_not_create_duplicate_lb_records(self):
        uid = self.registered_user['uid']
        pid = self.pid

        self.login_successfully(self.app)
        self.wait_for_stage(self.app, pid, 'modeling')

        q_items = self.get_q(pid)
        original_count = len(q_items)
        q_item = q_items.pop()

        model = {
            'samplepct': 52,
            'uid': uid,
            'blueprint_id': q_item['blueprint_id'],
            'dataset_id': q_item['dataset_id'],
            'icons': [1],
            'max_reps': 1,
            'bp': 51
        }

        models = [model]

        self.create_model(pid, models)

        q_items = self.get_q(pid)
        new_count = len(q_items)

        self.assertEqual(original_count, new_count - 1)

        q_item_added = q_items.pop()
        self.assertEqual(q_item_added['blueprint_id'], model['blueprint_id'])

        # Attempt to create duplicate
        self.create_model(pid, models)

        q_items = self.get_q(pid)
        new_count = len(q_items)

        self.assertEqual(original_count, new_count - 1)

    @patch('MMApp.app.ProjectService', autospec = True)
    @patch('MMApp.app.FLIPPERS', autospec = True)
    @patch('MMApp.app.NotificationService', autospec = True)
    def test_invite_new_user_to_project(self, MockNotificationService, MockFlippers, MockProjectService):
        pid = str(ObjectId())

        MockFlippers.enable_user_invite = True
        project_service = MockProjectService.return_value
        project_service.get_project_info_for_user.return_value = {}

        with patch.object(MMApp.app, 'get_user_info', return_value = {'uid':str(ObjectId), 'username' : 'project-owner@datarobot.com'}):
            payload = {'email' : 'does-not-exist@datarobot.com'}
            response = self.app.post('/project/{}/team'.format(pid),
                content_type='application/json', data=json.dumps(payload))

            response_data = json.loads(response.data)
            self.assertIn('link', response_data)
            self.assertIn('status', response_data)

            self.assertEqual(response_data['status'], 'OK')
            link = response_data['link']
            self.assertEqual(response.status_code, 200)

            path = urlparse.urlparse(link).path
            #Make sure it works!
            response = self.app.get(path, follow_redirects = True)

            # Test we got the join.html page
            self.assertEqual(response.status_code, 200, 'The join link did not work: {0} ({1})'.format(link, path))
            self.assertEqual(response.content_type, 'text/html; charset=utf-8')

    @patch('MMApp.app.FLIPPERS', autospec = True)
    @patch('MMApp.app.NotificationService', autospec = True)
    def test_approval_workflow(self, MockNotificationService, MockFlippers):
        MockFlippers.enable_user_invite = True

        self.login_successfully(self.app)

        # Invite new user
        email = 'new-user@email.com'
        payload = {'email' : email}
        response = self.app.post('/project/{}/team'.format(self.pid),
            content_type='application/json', data=json.dumps(payload))

        notification_service_instance =  MockNotificationService.return_value
        # Verify user was notified
        self.assertEqual(notification_service_instance.invite_new_user_to_project.call_count, 1)

        self.assertEqual(response.status_code, 200, response.data)
        response_data = json.loads(response.data)
        link = response_data['link']
        i = link.find('code=') + 5
        invite_code = link[i: link.find('&')]

        self.logout(self.app)

        # New user Joins
        payload = {
            'email' : email,
            'password' : email,
            'passwordConfirmation' : email,
            'firstName' : 'user',
            'lastName' : 'does-not-exist',
            'code' : invite_code,
            'tos': True
        }


        response = self.app.get('/join?email={0}&code={1}'.format(email, invite_code), follow_redirects = True)
        self.assertEqual(response.status_code, 200, response.data)

        # Verify user will be redirected to the "Registration Received screen"
        response = self.app.post('/join', data= json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200, response.data)
        response_data = json.loads(response.data)
        self.assertIn('pending', response_data['status'], response_data)

        # Verify notifications were sent: user got confirmation email and datarobot got approval request
        self.assertEqual(notification_service_instance.registration_received.call_count, 1)
        self.assertEqual(notification_service_instance.registration_needs_approval.call_count, 1)

        # Can't log in
        response = self.login(email, email)
        self.assertEqual(response.status_code, 200, response.data)
        response_data = json.loads(response.data)
        self.assertIn('error', response_data)
        self.assertEqual(response_data['error'], '1', 'Error was expected')
        self.assertIn('has not been approved', response_data['message'])

        # Approve account (first get the new user's uid)
        self.login_successfully(self.app)
        approve_link = '/manage/account/approve/{0}/{1}'.format(email, invite_code)
        response = self.app.get(approve_link)
        self.assertEqual(response.status_code, 200, response.data)
        self.logout(self.app)

        # Verify user is notified
        self.assertTrue(notification_service_instance.registration_approved.called)

        # Verify user can login
        response = self.login(email, email)
        self.assertEqual(response.status_code, 200, response.data)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['error'], '0', 'New user should be able to login: ' + response_data['message'])

    @patch('MMApp.entities.user.mandrill.Mandrill', autospec = True)
    def test_change_password_workflow(self, MockMandrill):
        # Don't send email during this test
        client = MockMandrill.return_value
        client.messages = Mock()
        client.messages.send.return_value = [{}]

        # Create account
        username = 'i-forgot-my-passsword@datarobot.com'
        password = '4HQujem1A$E12$79'
        user_service = UserService(username)
        user_service.create_account(password)

        # Should be case-insensitive
        data = {'username': username.upper()}

        response = self.app.post('/account/request/password/reset',
            content_type = 'application/json', data = json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn('success', response.data)

        # We should get an email with a reset key
        # For simplicity, we'll get it from the DB
        user = UserService(username).find_user()
        self.assertIsNotNone(user.reset_key)

        # Clicking the email takes us to the right page

        response = self.app.get('/account/request/password/reset?code={}'.format(user.reset_key),
            follow_redirects = True)
        self.assertEqual(response.status_code, 200)

        # Now change password
        new_password = 'SpicyChicken69'

        data = {
            'password' : new_password,
            'passwordConfirmation' : new_password,
            'reset_key' : user.reset_key
        }
        response = self.app.post('/account/password/reset',
            content_type = 'application/json', data = json.dumps(data))
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        username = response_data.get('profile', {}).get('username')
        self.assertEqual(username, username)

        # Login with new password
        response = self.login(username, new_password)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('Already logged in', response_data['message'])

        self.logout(self.app)

        response = self.login(username, new_password)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('Log In Successful', response_data['message'])

    def test_clone_project(self):
        self.login_successfully(self.app)

        new_pid = self.create_pid()

        data = {
            'newProjectId': new_pid
        }

        self.app.post('/project/{}'.format(self.pid),
            content_type = 'application/json', data = json.dumps(data))

        self.wait_for_stage(self.app, new_pid, 'aim')

if __name__ == '__main__':
    unittest.main()

