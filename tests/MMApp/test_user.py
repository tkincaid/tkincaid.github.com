# -*- coding: UTF-8 -*-# -*- coding: UTF-8 -*-
############################################################################
#
#       unit test for UserService
#
#       Author: ??
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import unittest
import redis
import pymongo
import json
import uuid
import os
import sys

from MMApp.entities.permissions import Permissions, AccountRoles
from common.services.security import datarobot_crypt
from common.wrappers import database
from bson.objectid import ObjectId
from mock import patch, Mock

from test_base import TestBase
from MMApp.entities.db_conn import DBConnections
from MMApp.entities.user import UserService, UserException, UserModel, UnknownFlagError, USER_TABLE_NAME
from MMApp.entities.user import UnknownFlagError
from pymongo.errors import InvalidId
from config.engine import EngConfig

import pytest

class UserTestBase(TestBase):

    def create_test_account(self):
        precalculated_hash = '$pbkdf2-sha512$7950$DyFkbI1RipFybs25956TMg$tQauovU5vItyzxd32N3P3GUKGxIp8VaRru0/Gv1I9iEWffdSHqwHGx0GjZrtueh44yfWIQOWD7OK8XFoszWuQQ' #secret12345

        #Registered user
        self.redis_conn.hmset('user:test_user@example.com', {'guest':0.0, 'max_workers' : 10.0})
        self.existing_user_password = 'secret12345'
        self.existing_user = {
            'username': 'test_user@example.com',
            'first_name' : 'FirstName',
            'last_name' : 'LastName',
            'password': precalculated_hash,
            'invitecode': '12345678',
            'linked_account': 'me@social-network.com',
            'account_roles': [AccountRoles.USER],
            'activated': 1 }
        self.get_collection('users').insert(self.existing_user)

        #Whitelist user, not registered (no password)
        self.redis_conn.hmset('user:test_new_user@example.com', {'guest':0})
        self.get_collection('users').insert({"username": "test_new_user@example.com", "invitecode": "12345678", "activated": 0 })

        self.redis_conn.hmset('user:test_user_guest', {'guest':1})
        self.get_collection('users').insert({"username": "test_user_guest", 'guest':1})

class TestUserService(UserTestBase):

    def setUp(self):
        # Password for sampleUser is Testing123!
        self.sample_user = {u'_id': ObjectId("5283a67fdced6bc05c16884d"),
                           u'activated': 1,
                           u'password': u'$pbkdf2-sha512$7954$tRZCiFGqNYaQMqaUEsK4lw$rovvE5et47cmhvOQh0eOOYqi3tEWPRziErhw6XbghWhOfZ9Se3C9ooVIE/mv160giA.iikhqDsBD3srftTVflA',
                           u'username': u'testuser@example.com'}
        self.service = UserService()
        self.create_test_account()

    def tearDown(self):
        self.remove_test_account()
        self.redis_conn.flushall()

    @pytest.mark.unit
    def test_valid_email(self):
        valid = [
            "example@example.com",
            "example@example.co.uk",
            "example+data@example.com",
            "a@example.com",

        ]
        invalid = [
            "example@example", #don't allow local domains
            "@example.com",
            "example.com",
            ""
        ]
        for email in valid:
            self.assertTrue(UserService.valid_email(email), email)
        for email in invalid:
            self.assertFalse(UserService.valid_email(email), email)

    @pytest.mark.unit
    def test_validate_invite(self):
        username = 'user@datarobot.com'
        service = UserService(username)

        self.assertRaisesRegexp(UserException, 'required',
                service.validate_invite, None)

        invite_code = 'code'

        with patch.object(service, 'find_user', return_value = False):
            self.assertRaisesRegexp(UserException, 'User not found',
                service.validate_invite, invite_code)

        user = UserModel(invite_code = invite_code, activated = 1)
        with patch.object(service, 'find_user', return_value = user):
            self.assertRaisesRegexp(UserException, 'has registered',
                service.validate_invite, invite_code)

        user = UserModel(invite_code = 'wrong')
        with patch.object(service, 'find_user', return_value = user):
            self.assertRaisesRegexp(UserException, 'does not match',
                service.validate_invite, invite_code)

    @pytest.mark.unit
    def test_valid_username(self):
        valid = [
            "Bob",
            "Alice",
            "MyAwesomeRobot",
            "My-Awesome-Robot",
            "My.Awesome.Robot",
            "a9879e09fc4-7b09d4"
        ]
        invalid = [
            "\x00QZ\x01",
            "",
            "AB",
            "My Awesome Robot",
            "Awesome\nBot",
            "a" * 100,
            "-",
            "<script>alert('hello')</script>",
        ]
        for name in valid:
            self.assertTrue(UserService.valid_username(name), name)
        for name in invalid:
            self.assertFalse(UserService.valid_username(name), name)

    @pytest.mark.unit
    def test_name(self):
        # TODO: More cases
        valid = [
            "Bob Smith",
        ]
        invalid = [
            "Alice",
            "",
            "-",
        ]
        for name in valid:
            self.assertTrue(UserService.valid_name(name))
        for name in invalid:
            self.assertFalse(UserService.valid_name(name))

    @pytest.mark.unit
    def test_valid_password(self):
        self.assertFalse(UserService.valid_password("1234567"))
        self.assertTrue(UserService.valid_password("12345678"))
        self.assertTrue(UserService.valid_password("x" * 255))
        self.assertFalse(UserService.valid_password("x" * 256))

    @pytest.mark.unit
    def test_clean_name(self):
        self.assertEqual(UserService.clean_name("abc123!@#$%^alksajl23"), 'abc123_alksajl23')
        self.assertEqual(UserService.clean_name("abcd"), "abcd")
        self.assertEqual(UserService.clean_name('"alskd*Y&(*&xna2134")'), '_alskd_Y_xna2134_')

    def remove_test_account(self):
        self.get_collection('users').drop()

    @pytest.mark.db
    def test_non_ascii_user_info(self):
        non_ascii_name = 'ออักษรไทยᓀ    ᓀ   ᓂ   ᓂ   ᓄ   ᓄ ᓇ ᓇ社會科學院語學研究𝀲 𝀳   𝀴 𝀵 𝀶   𝀷'
        user_service = UserService(self.existing_user['username'])
        user_service.save(UserModel(first_name = non_ascii_name))

        user = user_service.find_user()
        # full_name should not assume ascii
        self.assertIsNotNone(user.full_name)

    @pytest.mark.db
    def test_login(self):
        self.service.username = None
        self.assertFalse(self.service.login(None))

        self.service.username = 'does-not-exist'
        self.assertFalse(self.service.login(None), self.service.username)

        self.service.username = 'test_user@example.com'
        self.assertFalse(self.service.login('wrong-password'), self.service.username)

        self.service.username = 'test_user@example.com'
        self.assertTrue(self.service.login(self.existing_user_password), self.service.get_error() + ': ' + self.service.username)

    @pytest.mark.db
    def test_login_is_case_insensitive(self):
        # Case insensitive in mongo queries
        self.service.username = 'test_USER@example.com'
        self.assertTrue(self.service.login(self.existing_user_password), self.service.get_error())

        # Make sure calls to redis are case insensitive too
        user_info = self.service.get_user_info()
        self.assertIn('uid', user_info)

    @pytest.mark.unit
    def test_login_does_not_allow_inactive_users(self):
        username = self.existing_user['username']
        user_service = UserService(username)

        with patch.object(user_service, 'persistent') as mock_db:
            mock_db.read.return_value = [self.existing_user]
            self.assertTrue(user_service.login(self.existing_user_password))


            self.existing_user['activated'] = 0
            self.assertFalse(user_service.login(self.existing_user_password))

            self.existing_user.pop('activated')
            self.assertFalse(user_service.login(self.existing_user_password))

    @pytest.mark.db
    def test_signup_removes_invite_code_and_returns_uid(self):
        username = 'no-invite-code@datarobot.com'
        user = {'username' : username, 'password' : 'does-not-matter',
            UserService.RANDOM_KEY_INVITE_CODE: 'does-not-matter'}

        user_service = UserService(username)

        db_user = user_service.persistent.create(table=USER_TABLE_NAME,
            values = user, result=[])

        db_user = user_service.persistent.read(table=USER_TABLE_NAME,
            condition={'username': username}, result=[])

        user = UserModel(**user)
        with patch.object(user_service, 'validate_signup', return_value = True):
            result = user_service._signup(user)
            self.assertTrue(result)

        query = user_service.persistent.read(table=USER_TABLE_NAME,
            condition={'username': user.username}, result=[])

        db_user = query[0]

        self.assertTrue(db_user)
        self.assertTrue(user_service.uid)
        self.assertNotIn(UserService.RANDOM_KEY_INVITE_CODE, db_user)

    @pytest.mark.db
    def test_signup_bad_credentials(self):
        self.service.username = ''
        self.assertFalse(self.service.create_account('some-password'))
        self.assertEqual(self.service.get_error(), 'Invalid email')

        self.service.username = 'test_user_2@example.com'
        self.assertFalse(self.service.create_account(None))
        self.assertEqual(self.service.get_error(), 'Password must have between 8 and 255 characters and no spaces')

    @pytest.mark.db
    def test_signup_ok_credentials(self):
        self.service.username = 'new_datarobot_account@example.com'
        username = self.service.create_account('ok_password')
        self.assertIsNotNone(username)
        self.assertIsNotNone(self.service.uid)

    @pytest.mark.db
    def test_validate_signup(self):
        #Invalid email
        self.service.username = 'invalid email'
        self.assertFalse(self.service.validate_signup('ok_password'))
        self.assertTrue('Invalid email' in self.service.error)


        #Invalid password
        self.service.username = 'test@test.com'
        self.assertFalse(self.service.validate_signup('invalid password'))
        self.assertTrue('Password must have' in self.service.error)

    @pytest.mark.unit
    def test_find_user(self):
        self.service.username = 'test@test.com'

        with patch.object(self.service, 'persistent') as mock_db:
            mock_db.read.return_value = []

            self.assertFalse(self.service.find_user())

        with patch.object(self.service, 'persistent') as mock_db:
            mock_db.read.return_value = [{'username':'test@test.com'}]

            self.assertTrue(self.service.find_user())

    @pytest.mark.db
    def test_find_user_is_case_insensitive(self):
        account = self.existing_user['username']
        service = UserService(account.upper())
        self.assertTrue(service.find_user())

    @pytest.mark.db
    def test_signup_invite(self):
        # Invalid: no username
        self.service.username = ''
        self.assertFalse(self.service.signup_invite(UserModel(password='testing123', invite_code='12345678')))
        self.assertEqual(self.service.get_error(), 'Invalid invite')

        # Invalid: username not in whitelist
        self.service.error = None
        self.service.username = 'unknown_user@example.com'
        self.assertFalse(self.service.signup_invite(UserModel(password='testing123', invite_code='12345678')))
        self.assertEqual(self.service.get_error(), 'Invalid invite')

        # Valid create_account
        self.service.error = None
        self.service.username = 'test_new_user@example.com'
        self.assertTrue(self.service.signup_invite(UserModel(password='testing123', invite_code='12345678')), self.service.get_error())
        self.assertEqual(self.service.get_error(), None)

    @pytest.mark.db
    def test_unregistered_login(self):
        """
        Test scenario where a user is invited and goes straight to the login page without registering
        """
        # Whitelist user, not registered (no password)
        new_username = 'test_new_user@example.com'
        self.service.username = new_username
        self.assertFalse(self.service.login('any-password'))

    @pytest.mark.db
    def test_signup_invalid_invite(self):
        # Valid create_account
        self.service.error = None
        self.service.username = 'test_new_user@example.com'
        new_user = UserModel(password='testing123', invite_code='12345678')
        result = self.service.signup_invite(new_user)
        error = self.service.get_error()
        self.assertTrue(result, error)
        self.assertEqual(error, None)

        # User already taken: username in whitelist but account has been activated
        new_user = UserModel(password='testing123', invite_code='12345678')
        result = self.service.signup_invite(new_user)
        error = self.service.get_error()
        self.assertFalse(result, error)
        self.assertEqual(error, 'Invalid invite')

    @pytest.mark.unit
    def test_create_api_token(self):
        with patch('base64.urlsafe_b64encode') as mock_base64encode:
            with patch.object(self.service, 'persistent') as mock_db:
                test_api_token = 'RKgUYc-U5BIy_TsxZdUgJJ1KwltXG-tP'
                mock_base64encode.return_value = test_api_token
                self.assertEqual(self.service.create_api_token(), test_api_token)

    @pytest.mark.unit
    def test_get_api_token(self):
        with patch.object(self.service, 'find_user') as mock_find_user:
            # Expects api_token to not exist
            mock_find_user.return_value = UserModel()
            self.assertTrue(self.service.get_api_token() is None)

            # api_token exists
            test_api_token = u'W-FJYAWZMEUheCw3XiZJvWjra4AJxrmO'
            mock_find_user.return_value = UserModel(api_token = test_api_token)
            api_token = self.service.get_api_token()
            self.assertEqual(test_api_token, api_token)

    @pytest.mark.unit
    def test_add_new_reset_key(self):
        """Tests that a reset_key is added when it does not exist"""
        with patch.object(self.service, 'persistent') as mock_db:
            # Password for sampleUser is Testing123!

            mock_db.read.return_value = self.sample_user
            reset_key = self.service.add_reset_key()
            self.assertTrue(len(reset_key) == 32, "length of reset_key: %d" % (len(reset_key)))
            self.assertTrue(mock_db.update.called)
            self.assertTrue(mock_db.update.call_count == 1,
                            "call count = %d" % (mock_db.update.call_count))

    @pytest.mark.unit
    def test_existing_reset_key(self):
        """Tests that the same reset_key is returned if it already exists"""
        with patch.object(self.service, 'persistent') as mock_db:
            test_reset_key = u'AciIvldLjn9otixEE0CbF1Qi25_PaQHC'
            self.sample_user[u'reset_key'] = test_reset_key

            mock_db.read.return_value = self.sample_user
            reset_key = self.service.add_reset_key()
            self.assertTrue(len(reset_key) == 32, "length of reset_key: %d" % (len(reset_key)))
            self.assertTrue(not mock_db.update.called)
            self.assertTrue(reset_key == test_reset_key, "reset_key = %s" % (reset_key))

    @pytest.mark.unit
    def test_remove_reset_key_success(self):
        with patch.object(self.service, 'persistent') as mock_db:
            test_reset_key = u'AciIvldLjn9otixEE0CbF1Qi25_PaQHC'
            self.sample_user[u'reset_key'] = test_reset_key

            self.service._remove_reset_key(test_reset_key)
            self.assertTrue(mock_db.conn.users.update.called)
            self.assertEqual(mock_db.conn.users.update.call_count, 1)

    @pytest.mark.unit
    def test_reset_key_exists(self):
        with patch.object(self.service, 'persistent') as mock_db:
            # Case 1: reset key does not exist
            test_reset_key = u'AciIvldLjn9otixEE0CbF1Qi25_PaQHC'
            mock_db.read.return_value = []
            return_value = self.service.reset_key_exists(test_reset_key)
            self.assertFalse(return_value)

            # Case 2: reset key exists
            self.sample_user['reset_key'] = test_reset_key
            mock_db.read.return_value = list(self.sample_user)
            return_value = self.service.reset_key_exists(test_reset_key)
            self.assertTrue(return_value)

    @pytest.mark.unit
    def test_remove_reset_key_exception(self):
        def side_effect(*args):
            raise Exception

        with patch.object(self.service, 'persistent') as mock_db:
            test_reset_key = u'AciIFKJSDKJHGFJHSUY1178asSADj23C'
            mock_db.conn.users.update.side_effect = side_effect
            self.assertRaisesRegexp(UserException,
                                    'Error removing password reset key',
                                    self.service._remove_reset_key, test_reset_key)

    @pytest.mark.unit
    def test_change_password_success(self):
        with patch.object(self.service, 'persistent') as mock_db:
            with patch('common.services.security.datarobot_crypt.encrypt') as mock_datarobot_crypt:
                self.service._change_password(self.sample_user[u'username'], 'supersecret')
                # new_password_hash = supersecret
                new_password_hash = '$pbkdf2-sha512$7340$ei.l9P7/v3fuHcOY0xpjLA$0rbYobijG/FXrE90pRRGn1EN4V0GN7YXkn94HfLpwBS6M5Uu2difjPDKjo6C/8n5vIYRdh9GAVQOiOu4yOg7Fg'
                self.assertTrue(mock_datarobot_crypt.called)
                self.assertEqual(mock_datarobot_crypt.call_count, 1)
                self.assertTrue(mock_db.update.called)
                self.assertEqual(mock_db.update.call_count, 1)

    @pytest.mark.unit
    def test_change_password_invalid(self):
        with patch.object(self.service, 'valid_password') as mock_valid_password:
            mock_valid_password.return_value = False
            self.assertRaisesRegexp(UserException,
                    'Invalid password',
                    self.service._change_password, self.sample_user[u'username'], 'abc')

    @pytest.mark.unit
    def test_change_password_exception(self):
        def side_effect(*args):
            raise Exception

        with patch.object(self.service, 'persistent') as mock_db:
            with patch('common.services.security.datarobot_crypt.encrypt') as mock_datarobot_crypt:
                mock_db.update.side_effect = side_effect
                self.assertRaisesRegexp(UserException,
                                        'Error changing password',
                                        self.service._change_password, self.sample_user[u'username'], 'supersecret')

    @pytest.mark.unit
    def test_send_forgot_password_invalid_email(self):
        # Email does not exist in database
        with patch.object(self.service, 'persistent') as mock_db:
            mock_db.read.return_value = []
            return_value = self.service.send_forgot_password_email('http://localhost/account/password_reset')
            self.assertEqual(self.service.error, 'Email does not exist')
            self.assertFalse(return_value)

    @pytest.mark.unit
    def test_send_forgot_password_mandrill_error(self):
        with patch.object(self.service, 'persistent') as mock_db:
            with patch('mandrill.Mandrill') as mock_mandrill:
                with patch.object(self.service, 'add_reset_key') as mock_add_reset_key:
                    # Case 1: status = rejected
                    send_result = [{u'_id': u'48964c48246f4ec5bf0e01188f029159',
                                    u'email': u'support@datarobot.com',
                                    u'reject_reason': None,
                                    u'status': u'rejected'}]
                    mock_db.read.return_value = self.sample_user
                    mock_add_reset_key.return_value = True
                    mock_mandrill.return_value.messages.send.return_value = send_result

                    return_value = self.service.send_forgot_password_email('http://localhost/account/password_reset')
                    self.assertEqual(self.service.error, 'Error sending password reset email')
                    self.assertFalse(return_value)

                    # Case 2: status = invalid
                    send_result[0]['status'] = u'invalid'
                    return_value = self.service.send_forgot_password_email('http://localhost/account/password_reset')
                    self.assertEqual(self.service.error, 'Error sending password reset email')
                    self.assertFalse(return_value)

    @pytest.mark.unit
    def test_send_forgot_password_success(self):
        with patch.object(self.service, 'persistent') as mock_db:
            with patch('mandrill.Mandrill') as mock_mandrill:
                with patch.object(self.service, 'add_reset_key') as mock_add_reset_key:
                    # Case 1: status = success
                    send_result = [{u'_id': u'48964c48246f4ec5bf0e01188f029159',
                                    u'email': u'support@datarobot.com',
                                    u'reject_reason': None,
                                    u'status': u'sent'}]
                    mock_db.read.return_value = self.sample_user
                    mock_add_reset_key.return_value = True
                    mock_mandrill.return_value.messages.send.return_value = send_result

                    return_value = self.service.send_forgot_password_email('http://localhost/account/password_reset')
                    self.assertTrue(return_value)

                    # Case 2: status = queued
                    send_result[0]['status'] = u'queued'
                    return_value = self.service.send_forgot_password_email('http://localhost/account/password_reset')
                    self.assertTrue(return_value)

                    # Case 3: status = scheduled
                    send_result[0]['status'] = u'scheduled'
                    return_value = self.service.send_forgot_password_email('http://localhost/account/password_reset')
                    self.assertTrue(return_value)

    @pytest.mark.db
    def test_create_account_username_taken(self):
        #Valid
        new_username = 'staff@datarobot.com'
        old_username = 'test_user_guest'
        self.service.username = new_username
        self.assertTrue(self.service.create_account('ok_password', old_username), self.service.get_error())

        # Username already taken
        self.assertFalse(self.service.create_account('ok_password', old_username ))
        self.assertEqual(self.service.get_error(), 'This username is already taken')

    @pytest.mark.db
    def test_signup_with_old_username(self):
        #will not work with non-guests
        new_username = 'new_user@example.com'
        old_username = 'test_user@example.com'
        self.service.username = new_username
        self.assertFalse(self.service.create_account('ok_password', old_username ))
        self.assertEqual(self.service.get_error(), 'Unexpected non-guest account')

        #Success
        new_username = 'new_user@example.com'
        old_username = 'test_user_guest'
        self.service.username = new_username
        username = self.service.create_account('ok_password', old_username)
        self.assertEqual(username, new_username)
        self.assertIsNotNone(self.service.uid)

    @pytest.mark.db
    def test_get_user_info(self):
        self.service.username = None
        self.assertIsNone(self.service.get_user_info())

        self.service.username = 'random_user'
        self.assertEqual(self.service.get_user_info(), {})

        self.service.username = 'test_user@example.com'
        self.service.user_info = {'k', 'v'}
        self.assertIsNotNone(self.service.get_user_info())


        self.service.username = 'test_user@example.com'
        self.service.user_info = None
        self.user_info = self.service.get_user_info()
        self.assertIsNotNone(self.user_info)
        self.assertEqual(self.user_info['guest'], 0)

    @pytest.mark.db
    def test_get_account(self):
        service = UserService(self.existing_user['username'])
        self.assertIsInstance(service.get_account(), dict)
        account = service.get_account()
        self.assertGreater(len(service.get_account()), 0)
        service.username = None
        self.assertFalse(service.get_account())

    @pytest.mark.db
    def test_get_account_describes_permissions(self):
        service = UserService(self.existing_user['username'])
        account = service.get_account()
        self.assertIn('permissions', account)
        permissions = account['permissions'].keys()
        self.assertTrue(permissions)
        for k in permissions:
            self.assertTrue(account['permissions'][k])


    @pytest.mark.db
    def test_get_account_doesnot_leak_password_nor_roles(self):
        service = UserService(self.existing_user['username'])
        account = service.get_account()
        self.assertNotIn('password', account)
        self.assertNotIn('account_roles', account)

    @pytest.mark.db
    def test_is_guest(self):
        self.service.username = 'test_user@example.com'
        self.assertFalse(self.service.is_guest())

        self.service.user_info = None

        self.service.username = 'random_user'
        self.assertTrue(self.service.is_guest())

    @pytest.mark.db
    def test_cleanup_user(self):
        #Registered user

        service = UserService(self.existing_user['username'])
        user = UserModel(**self.existing_user)
        service.set_user_info(user)

        user_info = service.get_user_info()

        self.assertEqual(user_info.get('username'), self.existing_user['username'], )

        service.cleanup()

        user_info = service.get_user_info()

        self.assertFalse(user_info)

    @pytest.mark.db
    def test_cleanup_guest(self):
        # Guest
        self.service.username = 'test_user_guest'
        key = 'user:' + self.service.username
        user_info = self.redis_conn.hgetall(key)

        #simulate create_account
        uid = self.get_collection('users').insert({'username': self.service.username})
        #Similate keys inserted by worker
        self.redis_conn.rpush('userkeys:' + str(uid), 1)

        #Make sure document and keys exist in MongoDB and Redis
        user =  self.get_collection('users').find_one({'_id' : ObjectId(uid)})
        self.assertIsNotNone(user)
        self.assertGreater(len(user), 1)

        user = self.redis_conn.hgetall('user:' + self.service.username)
        self.assertGreater(len(user), 0)


        user_info.update({'uid': str(uid)})
        self.service.user_info = user_info
        self.service.cleanup()

        #Make sure they are gone
        #Make sure document and keys exist in MongoDB and Redis
        user =  self.get_collection('users').find_one({'_id' : ObjectId(uid)})
        self.assertIsNone(user)

        user = self.redis_conn.hgetall('user:' + self.service.username)
        self.assertEqual(len(user), 0)

    @pytest.mark.db
    def test_create_guest(self):
        u_id = str(uuid.uuid4())

        self.service.create_guest(u_id)

        self.assertTrue(self.service.is_guest())
        self.assertEqual(self.service.username, u_id)
        self.assertIsNotNone(self.service.uid)

    @pytest.mark.db
    def test_active(self):

        #simulate create_account
        username = 'test_active'
        uid = self.get_collection('users').insert({'username': username})
        uid = str(uid)

        user_service = UserService(username)
        user_service.uid = uid
        self.assertFalse(user_service.is_active())

        user_service.set_active()
        self.assertTrue(user_service.is_active())

        user_service.set_inactive()
        self.assertFalse(user_service.is_active())

        # It's ok to call set_inactive multiple times
        user_service.set_inactive()
        user_service.set_inactive()

    @pytest.mark.db
    def test_online(self):
        uid = str(ObjectId())
        self.service.uid = uid
        self.service.set_online()
        online = self.redis_conn.hgetall('state:users')
        self.assertEqual(online, {uid: '-1'})
        self.service.set_online()
        online = self.redis_conn.hgetall('state:users')
        self.assertEqual(online, {uid: '-1'})
        self.service.set_offline()
        online = self.redis_conn.hgetall('state:users')
        self.assertTrue(online.has_key(uid))
        self.assertNotEqual(online, {uid: '-1'})
        uid2 = str(ObjectId())
        self.service.uid = uid2
        self.service.set_online()
        online = self.redis_conn.hgetall('state:users')
        self.assertEqual(len(online), 2)
        self.assertTrue(online.has_key(uid))
        self.assertTrue(online.has_key(uid2))
        self.assertNotEqual(online[uid], '-1')
        self.assertEqual(online[uid2], '-1')

    @pytest.mark.unit
    def test_get_user_projects(self):
        #Arrange
        username = 'test_user'
        uid = '313233343536373839303930'
        uservice = UserService(uid)
        uservice.uid = uid
        with patch.object(uservice, 'persistent') as mock_db:

            mock_db.read.return_value= [
                {'_id':'AFakePid1',
                 'uid':uid},
                {'_id':'AFakePid2',
                 'uid':uid}]

            #Act
            project_list = uservice.get_user_projects()
            self.assertGreater(len(project_list), 0)

            #Assert
            for proj in project_list:
                self.assertEqual(proj['uid'], uid)
                # This is kind of a silly test since it's such a simple function
                # but we'll roll with it since it at least will help us catch
                # if we ever decide to change DB wrappers or something.

    @pytest.mark.db
    def test_max_workers1(self):
        self.service.username = 'test_user@example.com'
        self.service.login(self.existing_user_password)
        user_info = self.service.get_user_info()
        #print '\n\n%s\n\n'%user_info
        self.assertEqual(user_info['max_workers'], EngConfig['MAX_WORKERS'])

    @pytest.mark.db
    def test_max_workers2(self):
        self.service.username = 'test_user@example.com'
        self.service.persistent.update(table='users',
                    condition={'username': self.service.username},
                    values={'max_workers':1})
        self.service.login(self.existing_user_password)
        user_info = self.service.get_user_info()
        #print '\n\n%s\n\n'%user_info
        self.assertEqual(user_info['max_workers'],1)

    def test_create_invite(self):
        uservice = UserService()
        user_key = ObjectId()
        with patch.object(uservice.persistent, 'create', return_value = user_key):
            uid, invite_code = uservice.create_invite(self.existing_user['username'])

            self.assertEqual(uid, user_key)
            self.assertIsNotNone(invite_code)

    def test_create_invite_does_not_invite_same_user(self):
        account_name = self.existing_user['username']
        service = UserService(account_name)

        with patch.object(service, 'find_user', return_value = [UserModel()]):
            self.assertRaisesRegexp(UserException, 'User already exists', service.create_invite, account_name)

    @pytest.mark.db
    def test_create_invite_db(self):
        account_name = 'invited_user@datarobot.com'
        service = UserService(account_name)
        uid, invite_code = service.create_invite(self.existing_user['username'])

        service = UserService(account_name)
        user = service.find_user()

        self.assertIsNotNone(user)
        self.assertEqual(uid, user.uid)
        self.assertEqual(invite_code, user.invite_code)
        self.assertTrue(user.needs_approval)

    @pytest.mark.db
    def test_needs_approval(self):
        account_name = 'invited_user@datarobot.com'
        service = UserService(account_name)
        uid, invite_code = service.create_invite(self.existing_user['username'])

        self.assertTrue(service.needs_approval())

    @pytest.mark.db
    def test_approve_account_approved(self):
        account_name = 'invited_user@datarobot.com'
        service = UserService(account_name)
        uid, invite_code = service.create_invite(self.existing_user['username'])

        result = service.approve_account(invite_code, True)
        self.assertTrue(result)

        user = service.find_user()
        self.assertTrue(user)
        self.assertFalse(user.needs_approval)

    @pytest.mark.db
    def test_get_inviter(self):
        account_name = 'invited_user@datarobot.com'
        service = UserService(account_name)
        uid, invite_code = service.create_invite(self.existing_user['username'])

        inviter = service.get_inviter()
        self.assertIn(self.existing_user['first_name'], inviter)
        self.assertIn(self.existing_user['last_name'], inviter)

    @pytest.mark.unit
    def test_hash_username(self):
        user = UserModel(username = 'test1@example.com')
        hash1 = user.username_hash

        self.assertEqual(len(hash1), 32)

        user = UserModel(username = 'test2@example.com')
        hash2 = user.username_hash
        self.assertNotEqual(hash1, hash2)

        user = UserModel(username = 'TEST1@example.com')
        hash2 = user.username_hash
        self.assertEqual(hash1, hash2)

        user = UserModel(username = 'TEsT1@exAmplE.com')
        hash2 = user.username_hash
        self.assertEqual(hash1, hash2)

        user = UserModel(username = '  TEsT1@exAmplE.com  ')
        hash2 = user.username_hash
        self.assertEqual(hash1, hash2)

    @pytest.mark.db
    def test_save_existing_user(self):
        username = self.existing_user['username']
        service = UserService(username)

        existing_user = UserModel(**self.existing_user)
        existing_user.first_name = 'silly-name'
        existing_user.last_name = 'silly-last-name'
        existing_user.company = 'new-company'
        existing_user.industry = 'new-industry'
        existing_user.username = 'should not change'

        service.save(existing_user)
        db_user = service.find_user()

        self.assertEqual(existing_user.first_name, db_user.first_name)
        self.assertNotEqual(db_user.username, existing_user.username)

    @pytest.mark.unit
    def test_to_dict_includes_account_permissions_if_not_empty(self):
        existing_user = UserModel(**self.existing_user)
        serialized = existing_user.to_dict()
        self.assertNotIn('account_permissions', serialized)

        existing_user.flags.GRAYBOX_ENABLED = True
        serialized = existing_user.to_dict()
        self.assertIn('account_permissions', serialized)

    @pytest.mark.db
    def test_change_password_invalid_old_password(self):
        username = self.existing_user['username']
        service = UserService(username)

        self.assertRaisesRegexp(UserException,
            'Invalid password',
            service.change_password,
            'invalid',
            'new-cool-password')

    @pytest.mark.db
    def test_change_password_and_login(self):
        username = self.existing_user['username']
        old_password = self.existing_user_password

        # Old password works
        service = UserService(username)
        is_logged_in = service.login(old_password)
        self.assertTrue(is_logged_in)

        # New does not
        new_password = 'DataRobot2014'
        is_logged_in = service.login(new_password)
        self.assertFalse(is_logged_in)

        # Change it
        service.change_password(old_password, new_password)

        # New password works
        is_logged_in = service.login(new_password)
        self.assertTrue(is_logged_in)

        # Old does not
        is_logged_in = service.login(old_password)
        self.assertFalse(is_logged_in)

    @pytest.mark.db
    def test_change_password_timestamps_on_success(self):
        new_password = 'DataRobot2014'
        username = self.existing_user['username']
        old_password = self.existing_user_password

        service = UserService(username)
        service.change_password(old_password, new_password)

        user = service.find_user()
        self.assertIsNotNone(user.password_changed_on)


    @pytest.mark.db
    def test_find_by_linked_account_finds_regular_usernames(self):
        username = self.existing_user['username']
        service = UserService(username)

        user = service.find_by_linked_account()
        self.assertTrue(user)
        self.assertEqual(user.username, username)

    @pytest.mark.db
    def test_find_by_linked_account(self):
        username = self.existing_user['linked_account']
        service = UserService(username)

        user = service.find_by_linked_account()
        self.assertTrue(user)
        self.assertEqual(user.linked_account, username)


@pytest.mark.db
class TestUserFlags(UserTestBase):

    def setUp(self):
        self.create_test_account()

    def test_user_flags_default_to_false(self):
        user = UserModel(**self.existing_user)
        self.assertFalse(user.flags.GRAYBOX_ENABLED)

    def test_set_unknown_flag_raises_error(self):
        user = UserModel(**self.existing_user)
        with self.assertRaises(UnknownFlagError):
            user.flags.set_flag('some_crazy_flag', True)

    @pytest.mark.db
    def test_manually_setting_some_unknown_flag_does_not_persist(self):
        username = self.existing_user['username']
        service = UserService(username)

        user = UserModel(**self.existing_user)
        user.flags.some_crazy_flag = True

        service.save(user)
        db_user = service.find_user()
        with self.assertRaises(AttributeError):
            res = db_user.flags.some_crazy_flag

    @pytest.mark.db
    def test_only_saves_flags_with_value(self):
        db_user = self.save_user_with_flag()
        flags = db_user.flags.to_dict()

        self.assertNotIn(Permissions.GRAYBOX_ENABLED, flags)

    def save_user_with_flag(self):
        username = self.existing_user['username']
        service = UserService(username)

        user = UserModel(**self.existing_user)
        user.flags.GRAYBOX_ENABLED = True

        service.save(user)

        return service.find_user()

    @pytest.mark.db
    def test_serialize_and_unserialize(self):
        db_user = self.save_user_with_flag()
        self.assertTrue(db_user.flags.GRAYBOX_ENABLED)



if __name__ == '__main__':
    unittest.main()
