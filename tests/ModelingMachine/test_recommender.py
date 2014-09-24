import unittest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

from mock import MagicMock
from mock import patch
from collections import OrderedDict

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.response import Response
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.recommender import PrecomputedUserTopkBlueprintInterpreter
from ModelingMachine.engine.recommender import UserTopkNotSupported
from ModelingMachine.engine.tasks.recsys.base import PrecomputedUserTopk
from ModelingMachine.engine.tasks.recsys.base import sizeof_topk
from ModelingMachine.engine.tasks.recsys.sgdsvd import SGDSVDRecommender
from ModelingMachine.engine.tasks.recsys.dummy import MostPopularItemsRecommender
from ModelingMachine.engine.tasks.recsys.neighbors import ItemKNNRecommender
from ModelingMachine.engine.tasks.cfconverter import CFConverter


class Request(dict):
    """Dummy class for mock requests. """
    blueprint = tuple()


class PrecomputeTopkBase(object):
    """Base test class for different Task classes that support pre-computed topk. """

    Task = None
    task_args = ''
    n_items = 100
    n_users = 1000
    n_samples = 10000
    task = None

    def create_cf_syn_data(self, n_samples=1000, n_users=500, n_items=10, reps=1, rows=None,
                           categoricals=False):
        """Creates a synthetic recommender dataset using only user-item id features. """
        rs = np.random.RandomState(13)
        user_ids = rs.randint(0, n_users, size=n_samples)
        if categoricals:
            user_ids = map(lambda ui: 'UID-%d' % ui, user_ids)
        item_ids = rs.randint(0, n_items, size=n_samples)
        if categoricals:
            item_ids = map(lambda ii: 'IID-%d' % ii, item_ids)

        X = pd.DataFrame(data=OrderedDict([('user_id', user_ids),
                                           ('item_id', item_ids)]))
        # set special columns to adhere blueprint_interpreter.BuildData.dataframe
        X.special_columns = {'I': 'item_id', 'U': 'user_id'}
        Y = Response.from_array(rs.randint(1, 5, size=n_samples))
        Z = Partition(size=X.shape[0], folds=5, reps=reps, total_size=X.shape[0])
        Z.set(max_reps=reps, max_folds=0)
        X = Container(dataframe=X)
        return X, Y, Z

    def _train_task(self):
        """Initialize task attribute -- dont want to fit task for each test case. """
        if not self.task:
            X, Y, Z = self.create_cf_syn_data(n_samples=self.n_samples,
                                              n_users=self.n_users, n_items=self.n_items,
                                              categoricals=True)
            Z = Z.set(partitions=[(-1, -1)])

            cfc = CFConverter()
            cfc.fit(X, Y, Z)
            c = cfc.transform(X, Y, Z)

            task = self.Task(self.task_args)
            task.fit(c, Y, Z)
            self.task = task
            self.df = X.dataframe

    def test_organic_topk_smoke(self):
        """PrecomputedUserTopkBlueprintInterpreter smoke test. """
        self._train_task()

        topk_req = MagicMock()
        topk_req.user_id = ['UID-1']
        topk_req.known_items = False
        topk_req.n_items = 20
        topk_req.threshold = None
        request = Request(scoring_data=topk_req)

        vertex = MagicMock()
        vertex.steps = [[self.task]]
        vertex_data = {'vertex_object': vertex}
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        model.vertex_factory = MagicMock()
        model.vertex_factory.get.return_value = vertex_data
        out = model.predict()
        self.assertEqual(len(out), len(topk_req.user_id))
        self.assertEqual(out[0][0], topk_req.user_id[0])

        # check if all output item ids are in training set
        item_set = set(np.unique(self.df['item_id']))
        items = out[0][1]
        for eid in items:
            self.assertTrue(eid in item_set)

    def test_organic_topk_coldstart(self):
        """PrecomputedUserTopkBlueprintInterpreter test cold starts. """
        self._train_task()

        topk_req = MagicMock()
        topk_req.user_id = ['Foobar']
        topk_req.known_items = False
        topk_req.n_items = 20
        topk_req.threshold = None
        request = Request(scoring_data=topk_req)

        vertex = MagicMock()
        vertex.steps = [[self.task]]
        vertex_data = {'vertex_object': vertex}
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        model.vertex_factory = MagicMock()
        model.vertex_factory.get.return_value = vertex_data
        out = model.predict()
        self.assertEqual(len(out), len(topk_req.user_id))
        self.assertEqual(out[0][0], topk_req.user_id[0])
        topk = self.task.topk_[(-1, -1)]
        desired_items = map(lambda iid: topk.item_inv_map[iid], topk.coldstart_items[:topk_req.n_items])
        np.testing.assert_array_equal(out[0][1], desired_items)
        np.testing.assert_array_equal(out[0][2], topk.coldstart_ratings[:topk_req.n_items])

        # check if all output item ids are in training set
        item_set = set(np.unique(self.df['item_id']))
        items = out[0][1]
        for eid in items:
            self.assertTrue(eid in item_set)


class TestPrecomputeTopkSGDSVD(unittest.TestCase, PrecomputeTopkBase):

    Task = SGDSVDRecommender
    task_args = 'c=10;a=0.01;e=0.01;mi=10'


class TestPrecomputeTopkItemMean(unittest.TestCase, PrecomputeTopkBase):

    Task = MostPopularItemsRecommender


class TestPrecomputeTopkItemMean(unittest.TestCase, PrecomputeTopkBase):
    """ItemKNNRecommender not yet support pre-computed topk. """

    Task = ItemKNNRecommender

    def test_organic_topk_smoke(self):
        with self.assertRaises(UserTopkNotSupported):
            super(TestPrecomputeTopkItemMean, self).test_organic_topk_smoke()

    def test_organic_topk_coldstart(self):
        with self.assertRaises(UserTopkNotSupported):
            super(TestPrecomputeTopkItemMean, self).test_organic_topk_coldstart()


class TestPrecomputedUserTopkBlueprintInterpreter(unittest.TestCase):
    """White box test for PrecomputedUserTopkBlueprintInterpreter.predict

    Inject a PrecomputedUserTopk object and test if filtering works correctly.
    """

    def setUp(self):
        # the precomputed topk object we return from the mock
        items = np.array([[0, 1, 2], [1, 2, 0]])
        ratings = np.array([[4.3, 2.3, 0.1], [2.3, 1.2, -0.4]])
        user_map = {'UID-1': 0, 'UID-2': 1}
        item_inv_map = {i: 'IID-%d' % i for i in range(3)}
        known_items = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool))
        coldstart_items = np.array([0, 1, 2])
        coldstart_ratings = np.array([3.0, 3.0, 3.0])
        topk = PrecomputedUserTopk(items, ratings, user_map, item_inv_map,
                                   known_items, coldstart_items, coldstart_ratings)
        print(sizeof_topk(topk))
        self.topk = topk

    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_smoke(self, mock_task_func):
        """Smoke test for PrecomputedUserTopkBlueprintInterpreter. """
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        # smoke test - get topk for UID-1
        topk_req = MagicMock()
        topk_req.user_id = ['UID-1']
        topk_req.known_items = True
        topk_req.n_items = 20
        topk_req.threshold = None
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-1', ['IID-0', 'IID-1', 'IID-2'],
                                (4.3, 2.3, 0.1))])

    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_threshold(self, mock_task_func):
        """Filter by threshold (filter out item IID-2)"""
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        topk_req = MagicMock()
        topk_req.user_id = ['UID-1']
        topk_req.known_items = True
        topk_req.n_items = 20
        topk_req.threshold = 1.0
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-1', ['IID-0', 'IID-1'],
                                (4.3, 2.3))])

    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_known_items(self, mock_task_func):
        """Filter by known items"""
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        topk_req = MagicMock()
        topk_req.user_id = ['UID-1']
        topk_req.known_items = False
        topk_req.n_items = 20
        topk_req.threshold = None
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-1', ['IID-2'],
                                (0.1, ))])

    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_threshold_known_items(self, mock_task_func):
        """Filter by threshold & known items - corner case with 0 items returned"""
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        topk_req = MagicMock()
        topk_req.user_id = ['UID-1']
        topk_req.known_items = False
        topk_req.n_items = 20
        topk_req.threshold = 1.0
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-1', [], tuple())])

    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_multiple(self, mock_task_func):
        """Retrieve multiple user ids. """
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        topk_req = MagicMock()
        topk_req.user_id = ['UID-1', 'UID-2']
        topk_req.known_items = True
        topk_req.n_items = 20
        topk_req.threshold = None
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-1', ['IID-0', 'IID-1', 'IID-2'],
                                (4.3, 2.3, 0.1)),
                                ('UID-2', ['IID-1', 'IID-2', 'IID-0'],
                                (2.3, 1.2, -0.4))])


    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_multiple_coldstart(self, mock_task_func):
        """Retrieve multiple user ids. """
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        topk_req = MagicMock()
        topk_req.user_id = ['UID-1', 'UID-666']
        topk_req.known_items = True
        topk_req.n_items = 20
        topk_req.threshold = None
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-1', ['IID-0', 'IID-1', 'IID-2'],
                                (4.3, 2.3, 0.1)),
                                ('UID-666', ['IID-0', 'IID-1', 'IID-2'],
                                (3.0, 3.0, 3.0))])

    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_coldstart_known(self, mock_task_func):
        """Known does not apply to coldstarts. """
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        topk_req = MagicMock()
        topk_req.user_id = ['UID-666']
        topk_req.known_items = False
        topk_req.n_items = 20
        topk_req.threshold = 3.0
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-666', ['IID-0', 'IID-1', 'IID-2'],
                               (3.0, 3.0, 3.0))])

    @patch('ModelingMachine.engine.recommender.predictor_task_from_blueprint_iter')
    def test_coldstart_threshold(self, mock_task_func):
        """Threshold applies to coldstart. """
        mock_task = MagicMock()
        mock_task.topk_ = {(-1, -1): self.topk}
        mock_task_func.return_value = mock_task

        topk_req = MagicMock()
        topk_req.user_id = ['UID-666']
        topk_req.known_items = False
        topk_req.n_items = 20
        topk_req.threshold = 3.1
        request = Request(scoring_data=topk_req)
        model = PrecomputedUserTopkBlueprintInterpreter(MagicMock(), MagicMock(),
                                                        request=request)
        out = model.predict()
        self.assertEqual(out, [('UID-666', [], tuple())])
