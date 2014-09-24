import unittest
from mock import Mock, patch
from common.services.resource import ResourceServiceGBM

class TestResourceService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_bp = {"1": [["NUM"],["GS" ],"T" ], "2": [["1"],["LR1 p=1"],"P"]}
        cls.test_dataset = "forrest-output.csv"

    def test_seen_dataset_bp(self):
        ''' get historical data for dataset & bp '''
        rs = ResourceServiceGBM()
        with patch.object(rs,'persistent') as mock_db:
            mock_db.read.return_value = {'stats': {'test':1} }
            res = rs.estimate_resources(self.test_bp, self.test_dataset)
            self.assertEqual(res,{'test':1})

    def test_unseen_dataset_bp(self):
        ''' predict resources for dataset & bp with no historical info '''
        rs = ResourceServiceGBM()
        # make fake predictions
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_model.predict.side_effect = [[1],[2],[3],[4],[5],[6]]
        with patch.object(rs,'persistent') as mock_db:
            with patch('common.services.resource.cPickle.loads') as p_mock:
                p_mock.return_value = mock_model
                # fake model DB
                mock_db.read.side_effect = [
                        {},
                        {'stats':{'x':1}},
                        {'gbm':{'max_RAM':'x'}},
                        {'gbm':{'total_CPU':'x'}},
                        {'gbm':{'max_cores':'x'}},
                        {'gbm':{'max_RAM':'x'}},
                        {'gbm':{'total_CPU':'x'}},
                        {'gbm':{'max_cores':'x'}},
                ]
                res = rs.estimate_resources(self.test_bp, self.test_dataset, predict = True)
                self.assertEqual(res,{'total_CPU': 7, 'max_cores': 6, 'max_RAM': 4})
