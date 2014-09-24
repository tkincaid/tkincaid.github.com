import unittest
import json
import time
import requests

import pytest

from integration_test_base import IntegrationTestBase
import MMApp.app
import MMApp.api
from MMApp.entities.ide import IdeSetupStatus, IdeService
from safeworkercode.docker_client import DockerClient


class TestRStudio(IntegrationTestBase):
    ''' Integration test for RStudio. Every component in the application is started, including nginx in order to route api calls.
    (Web API calls are do not use Flask's test client anymore)

    **Requirements:**
        - The app must not be running otherwise the proceesses mentioned above will not be able to listen to their address:port.        -
        - RStudio server must be up and running
        - Redis (port for testing)  & MongoDB must be up and running
        - For now, worker requires sudo sudo access to setup RStudio and user worker requires sudo to create LXC containers.
          You will be prompted for your sudo password a few seconds after the test has started unless you execute the script:

          run_integration_tests.sh

    **This test touches the MMApp, Web API, MongoDB, Redis, Rstudio and the Majordomo components:**
        - Broker
        - Worker
        - IDE Broker
        - IDE Worker
        - User Broker
        - User Worker

    **Future Work:**
        - Configure tests to listen to different address:port than the app. This is a matter of overriding the config file at runtime
        - Avoid sudo access, this will require actually changing user_worker and worker code.
    '''

    @classmethod
    def setUpClass(self):
        super(TestRStudio, self).setUpClass()
        self.api  = MMApp.api.app.test_client()
        TestRStudio.pid = None

    def setUp(self):
        super(TestRStudio, self).setUp()
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')
        if not TestRStudio.pid:
            TestRStudio.pid = self.create_project()
            TestRStudio.uid = self.registered_user['uid']
            self.logout(self.app)

        self.pid = TestRStudio.pid


    def test_ide_with_guest(self):
        with self.app.session_transaction() as _session:
            _session['user'] = self.guest_user['username']
        pid = '313233343536373839303930'

        #Repeated requests yield no results
        for i in range(5):
            response = self.app.get('/project/%s/ide_status' % pid)
            self.assertEqual(response.status_code, 200)
            rstudio_status = json.loads(response.data)

            criteria = ['status']
            self.assertTrue(all(k in rstudio_status and rstudio_status[k] for k in criteria), 'Response keys %s did not include the expected keys: %s' % (rstudio_status.keys(), criteria))

            self.assertEqual(rstudio_status['status'], IdeSetupStatus.NOT_STARTED)

    #FIXME: new redis keys are being wiped out by integration test base
    @pytest.mark.skip
    def test_ide_setup_with_registered_user(self):
        # Assert image does not exist
        docker_client = DockerClient()
        pid = self.pid

        self.assertTrue(self.containers_do_not_exist(docker_client, pid))
        try:
            #Start RStudio
            status = self.start_rstudio(pid)
            self.assertEqual(status, IdeSetupStatus.COMPLETED)

            self.assert_rstudio_started(pid)

            self.assert_ide_terminates_on_project_delete(pid)
        finally:
            #Logout: Remove container
            self.app.get('/account/logout')
            self.wait_for_logout_complete(pid)

    def assert_rstudio_started(self, pid):
        docker_client = DockerClient()
        #Make sure the ide status still shows completed if requested again
        response = self.app.get('/project/%s/ide_status' % pid)
        rstudio_status = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(rstudio_status['status'].lower(), IdeSetupStatus.COMPLETED.lower())

        self.assertTrue(self.wait_for_reachable(rstudio_status), 'IDE not reachable: {0}'.format(rstudio_status))

        #Assert container and image exist
        self.assertTrue(self.containers_exist(docker_client, pid))

    def assert_ide_terminates_on_project_delete(self, pid):
        # Delete project
        docker_client = DockerClient()
        response = self.app.delete('/project/' + pid)
        self.assertEqual(response.status_code, 200)

        #Assert rstudio was terminated (image and container removed)
        timeout = time.time() + 30
        while True:
            if self.containers_do_not_exist(docker_client, pid):
                break

            self.assertTrue(time.time() < timeout, 'The containers were not removed in time')

            time.sleep(1)

    @pytest.mark.skip
    def test_rapid_ide_requests(self):
        pid = self.pid
        try:
            response = self.app.get('/project/%s/ide_status' % pid)
            rstudio_status = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            #Wait for RStudio setup to complete
            status = self.start_rstudio(pid)
            self.assertEqual(status, IdeSetupStatus.COMPLETED)

            #Make sure the ide status still shows completed if requested again
            response = self.app.get('/project/%s/ide_status' % pid)
            rstudio_status = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(rstudio_status['status'].lower(), IdeSetupStatus.COMPLETED.lower())

            response = self.app.get('/account/logout')
            self.assertEqual(response.status_code, 302)

            self.login_successfully(self.app)

            response = self.app.get('/project/%s/ide_status' % pid)
            rstudio_status = json.loads(response.data)
            self.assertEqual(response.status_code, 200)

            response = self.app.get('/account/logout')
            self.assertEqual(response.status_code, 302)

            self.login_successfully(self.app)

            status = self.start_rstudio(pid)
            self.assertEqual(status, IdeSetupStatus.COMPLETED)
        finally:
            #Logout: Remove container
            response = self.app.get('/account/logout')
            #TODO: clean up storage
            self.wait_for_logout_complete(pid)

    def containers_exist(self, docker_client, pid):
        if docker_client.find_container_by_name(self.get_rstudio_container_name(pid), True) \
           and docker_client.find_container_by_name(self.get_ipython_container_name(pid), True):
           return True
        return False

    def containers_do_not_exist(self, docker_client, pid):
        if docker_client.find_container_by_name(self.get_rstudio_container_name(pid), True) \
           or docker_client.find_container_by_name(self.get_ipython_container_name(pid), True):
           return False
        return True

    def get_rstudio_container_name(self, pid):
        return 'ide-{0}-{1}-rstudio'.format(self.registered_user['uid'], pid)

    def get_ipython_container_name(self, pid):
        return 'ide-{0}-{1}-ipython'.format(self.registered_user['uid'], pid)

    def wait_for_logout_complete(self, pid):
        docker_client = DockerClient()
        timeout = time.time() + 15
        while True:
            #TODO: check that the file system is cleaned up
            if self.containers_do_not_exist(docker_client, pid):
                break

            if time.time() > timeout:
                raise Exception('Image not removed')

            time.sleep(1)

    def wait_for_reachable(self, status):
        timeout = time.time() + 5
        while time.time() < timeout:

            ide_service = IdeService(self.uid, self.pid)
            ide_status = ide_service.get_status()
            if ide_service.ide_server_is_running(ide_status):
                return True

            time.sleep(1)

        return False

    def start_rstudio(self, pid):
        timeout = time.time() + 15
        while True:
            response = self.app.get('/project/%s/ide_status' % pid)
            rstudio_status = json.loads(response.data)
            self.assertEqual(response.status_code, 200)

            criteria = ['status']
            self.assertTrue(all(k in rstudio_status for k in criteria), 'Response keys %s did not include the expected keys: %s' % (rstudio_status.keys(), criteria))

            print rstudio_status['status']
            if rstudio_status['status'] == IdeSetupStatus.COMPLETED:
                criteria = ['status', 'rstudio_location', 'python_location', 'username', 'password']
                self.assertTrue(all(k in rstudio_status for k in criteria), 'Response keys %s did not include the expected keys: %s' % (rstudio_status.keys(), criteria))
                location = rstudio_status['rstudio_location']
                self.assertEqual(location.count('_'), 3, location)
                self.assertEqual(location.count('_'), 3, rstudio_status['python_location'])
                return rstudio_status['status']

            if time.time() > timeout:
                raise Exception('Time out, Rstudio was not ready: {0}'.format(rstudio_status['status']))

            time.sleep(1)


if __name__ == '__main__':
    unittest.main()
