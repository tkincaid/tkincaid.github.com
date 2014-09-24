import os
import mock
import unittest
import requests
import json

from collections import Counter

import tests.ModelingMachine.mbtest_rd


OK_REQUEST = requests.Response()
OK_REQUEST.status_code = 200


def make_json_response(obj, status_code=200):
    r = requests.Response()
    r._content = json.dumps(obj)
    r.status_code = status_code
    return r


def post_side_effect(url, *args, **kw):
    if url.endswith('project'):
        r = make_json_response({'pid': 'mbtest-foobar'})
    elif url.endswith('upload/mbtest-foobar'):
        r = make_json_response({'pid': 'mbtest-foobar'})
    elif url.endswith('aim'):
        r = make_json_response({})
    elif url.endswith('signup'):
        r = make_json_response({'error': '0'})
    else:
        raise ValueError('No POST sideeffect for url "%r"' % url)
    return r


def get_side_effect(url, *args, **kw):
    if url.endswith('service'):
        r = make_json_response({})
    elif url.endswith('queue'):
        # error should be filtered out
        r = make_json_response([{'mode': 0}, {'status': 'error'}])
    elif url.endswith('next_steps'):
        r = make_json_response({})
    else:
        raise ValueError('No GET sideeffect for url "%r"' % url)
    return r


class MBTestMockTest(unittest.TestCase):
    """Mocked test for MBTest script. """

    @mock.patch('tests.ModelingMachine.mbtest_rd.MBTest.wait_for_stage')
    @mock.patch('tests.ModelingMachine.mbtest_rd.requests')
    def test_smoke(self, mock_requests, mock_wait_for_stage):
        """This is really just a smoke test. """
        mock_requests.post.side_effect = post_side_effect
        mock_requests.get.side_effect = get_side_effect
        filepath = os.path.join(os.path.split(__file__)[0], '..', 'testdata',
                                'kickcars-sample-200.csv')
        target, metric = 'isBadBuy', 'LogLoss'

        test = tests.ModelingMachine.mbtest_rd.MBTest(sleep_period=0.001, max_periods=10)
        test.create_account_and_execute(filepath, target, metric)

        called_urls = {call[0][0] for call in mock_requests.post.call_args_list}
        expected_urls = {'http://localhost/account/signup',
                         'http://localhost/project',
                         'http://localhost/upload/mbtest-foobar',
                         'http://localhost/aim'}
        self.assertSetEqual(called_urls, expected_urls)

        call_counts = Counter(call[0][0] for call in mock_requests.get.call_args_list)
        print call_counts
        self.assertEqual(call_counts['http://localhost/project/mbtest-foobar/service'], 25)
        self.assertEqual(call_counts['http://localhost/project/mbtest-foobar/next_steps'], 1)
        self.assertEqual(call_counts['http://localhost/project/mbtest-foobar/queue'], 25)
