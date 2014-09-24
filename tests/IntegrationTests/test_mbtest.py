import os
import mock
import base64
import json
import unittest
from mock import patch

from tests.IntegrationTests.integration_test_base import IntegrationTestBase
import tests.ModelingMachine.mbtest_rd
import MMApp.app


def convert_kw(kw):
    """Convert requests arguments to flask arguments.

    1. auth argument to authorization header.
    2. files to data.
    3. remove cookies (don't need them anyways.)
    """
    if 'auth' in kw:
        headers = kw.get('headers', {})
        username, password = kw.pop('auth')
        headers['Authorization'] = ('Basic ' + base64.b64encode(username +
                                    ":" + password))
        kw['headers'] = headers
    if 'files' in kw:
        kw['data'] = kw.pop('files')
    if 'cookies' in kw:
        kw.pop('cookies')
    return kw


class ResponseWrapper(object):
    """Wrapping a flask.Response so that it looks like a requests.Response. """

    def __init__(self, wrapee, url):
        self.wrapee = wrapee
        self.status_code = wrapee.status_code
        self.url = url

    def json(self):
        return json.loads(self.wrapee.data or '{}')

    @property
    def text(self):
        return self.wrapee.data

    @property
    def cookies(self):
        return {}


class MBTestTest(IntegrationTestBase):

    @mock.patch('tests.ModelingMachine.mbtest_rd.requests')
    def test_smoke(self, mock_requests):
        """Runs mbtest on kickcars-sample-200.

        This needs to mock the requests library and use the Flask TestClient instead.
        """
        filepath = os.path.join(os.path.split(__file__)[0], '..', 'testdata',
                                'kickcars-sample-200.csv')
        target, metric = 'IsBadBuy', 'LogLoss'

        # evil mocking of requests.get and post
        def post(*args, **kw):
            args = list(args)
            url = args.pop(0)
            url = url.split('http://')[1]
            kw = convert_kw(kw)

            # The upload functionality now lives in a dedicated server.
            # This separation is transparent for browsers since we redirect through nginx.
            # However, during testing we must explicitly create 2 flask apps that point
            # to each server
            if '/upload' in url:
                with self.upload_server.session_transaction() as session:
                    session['user'] = test.username
                return ResponseWrapper(self.upload_server.post(url, *args, **kw), url)

            # Mock runmode in order to always set PUBLIC mode during testing
            with patch.dict(MMApp.app.app.config, {'runmode': 0 }, clear = False):
                return ResponseWrapper(self.app.post(url, *args, **kw), url)

        def get(*args, **kw):
            args = list(args)
            url = args.pop(0)
            url = url.split('http://')[1]
            kw = convert_kw(kw)
            return ResponseWrapper(self.app.get(url, *args, **kw), url)

        mock_requests.post = post
        mock_requests.get = get

        # now run the tests
        test = tests.ModelingMachine.mbtest_rd.MBTest(host='', sleep_period=1, max_periods=10)
        test.create_account_and_execute(filepath, target, metric)
        n_lb_entries = self.persistent.count(table='leaderboard')
        # Random upper bound on the number of LB entries
        #  this checks for a regression error we had with dublicate entries
        #  if we run indeed more than 35 BPs please update.
        self.assertLess(n_lb_entries, 35)
        self.assertGreater(n_lb_entries, 0)
        # TODO check if LB has 5-fold cv results
        #self.get_collection('leaderboard')

if __name__ == '__main__':
    unittest.main()
