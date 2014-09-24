import unittest

from mock import patch

from common.services.notification import NotificationService
from MMApp.entities.user import UserModel

class TestNotificationService(unittest.TestCase):

    def setUp(self):
        super(TestNotificationService, self).setUp()
        self.addCleanup(self.stopPatching)
        self.patchers = []
        mandril_patcher = patch('mandrill.Mandrill')
        self.mandril_mock = mandril_patcher.start()
        self.patchers.append(mandril_patcher)

    def stopPatching(self):
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def test_invite_new_user_to_project(self):
        notifications = NotificationService()
        link = 'http://www.our_site.com/path/to/join'
        email = 'does-not-exist@datarobot.com'
        project_name = 'project-title'
        sender = 'project-owner@datarobot.com'
        invite_data = {'link' : link, 'email': email, 'project_name' : project_name, 'sender' : sender}



        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'sent'}]
        self.assertEqual(notifications.invite_new_user_to_project(invite_data), True)

        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'rejected'}]
        self.assertEqual(notifications.invite_new_user_to_project(invite_data), False)

        self.mandril_mock.return_value.messages.send.side_effect = Exception('BOOM!')
        self.assertEqual(notifications.invite_new_user_to_project(invite_data), False)

    def test_new_join_invite(self):
        notifications = NotificationService()
        link = 'http://www.our_site.com/path/to/join'
        approve_link = 'http://www.our_site.com/path/to/join'
        deny_link = 'http://www.our_site.com/path/to/join'
        email = 'does-not-exist@datarobot.com'
        project_name = 'project-title'
        sender = 'Sender\'s Name'
        invite_data = {'link' : link, 'email': email, 'project_name' : project_name, 'sender' : sender, 'approve_link': approve_link, 'deny_link':deny_link}
        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'sent'}]
        self.assertEqual(notifications.new_join_invite(invite_data), True)

    def test_invite_user_to_project(self):
        notifications = NotificationService()
        link = 'http://www.our_site.com/project'
        email = 'does-not-exist@datarobot.com'
        project_name = 'project-title'
        sender = 'project-owner@datarobot.com'
        invite_data = {'link' : link, 'email': email, 'project_name' : project_name, 'sender' : sender}


        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'sent'}]
        self.assertEqual(notifications.invite_user_to_project(invite_data), True)

        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'rejected'}]
        self.assertEqual(notifications.invite_user_to_project(invite_data), False)

        self.mandril_mock.return_value.messages.send.side_effect = Exception('BOOM!')
        self.assertEqual(notifications.invite_user_to_project(invite_data), False)

    def test_registration_needs_approval(self):
        notifications = NotificationService()
        user_info = {'username' : 'does-not-exist@email.com'}
        approve_link = 'http://www.our_site.com/approve'
        deny_link = 'http://www.our_site.com/deny'

        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'sent'}]
        approval_info = {
            'new_user' : user_info['username'],
            'inviter' : 'Existing User',
            'approve_link' : approve_link,
            'deny_link': deny_link
        }
        self.assertEqual(notifications.registration_needs_approval(approval_info), True)


    def test_registration_received(self):
        notifications = NotificationService()
        user_info = {'username' : 'does-not-exist@email.com'}
        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'sent'}]
        self.assertEqual(notifications.registration_received(user_info), True)

    def test_registration_approved(self):
        notifications = NotificationService()
        user_info = UserModel(username='does-not-exist@email.com')
        link = 'http://www.our_site.com/login'
        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'sent'}]
        self.assertEqual(notifications.registration_approved(user_info, link), True)

    def test_registration_denied(self):
        notifications = NotificationService()
        user_info = {'username' : 'does-not-exist@email.com'}
        self.mandril_mock.return_value.messages.send.return_value = [{'status': 'sent'}]
        self.assertEqual(notifications.registration_denied(user_info), True)

    def test_should_ignore_with_valid_receiver(self):
        email = 'user@my-domain.com'
        message = {
            'to': [{'email': email}],
        }

        notifications = NotificationService()
        result = notifications.should_ignore(message)
        self.assertFalse(result)

    def test_should_ignore_with_ui_test_account(self):
        email = 'ui_test_2@datarobot.com'
        message = {
            'to': [{'email': email}],
        }

        notifications = NotificationService()
        result = notifications.should_ignore(message)
        self.assertTrue(result)

    def test_send_email_may_ignore_messages(self):
        notifications = NotificationService()
        with patch.object(notifications, 'should_ignore', return_value = True):
            self.mandril_mock.return_value.messages.send.side_effect = Exception('BOOM!')
            result = notifications.send_email({})
            self.assertTrue(result)

    def test_send_email_may_not_ignore_messages(self):
        notifications = NotificationService()
        with patch.object(notifications, 'should_ignore', return_value = True):
            result = notifications.send_email({})
            self.assertTrue(result)
