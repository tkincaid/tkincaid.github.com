from ModelingMachine.metablueprint.gxavmb_v8_4 import Metablueprint
import pandas
import unittest
from mock import patch
from common.wrappers.database import ObjectId

class TestCustomMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = pandas.DataFrame({'a':[12,3,4,5,6],'b':['a','b','c','d','e'],'y':[3,2,4,2,5]})
        cls.pid = ObjectId('52ae1fa6fc46e73a8dda005c')
        cls.mb = Metablueprint(cls.pid, None)
        cls.mb._dataset_id = '52ae1fa6fc46e73a8dda005c'
        cls.mb._project = { 'target': { 'name': 'y', 'size': 1234, 'type': 'Binary' } }

    def test_custom_meta(self):
        with patch.object(self.mb.persistent,'update') as um:
            out = self.mb.addMetadata(self.test_data)
            self.mb.persistent.update.assert_any_call(
                condition={'pid':self.pid,'_id':ObjectId(self.mb._dataset_id)},
                values={'pct_min_y': 0.4},
                table='metadata')

    def test_mb_blender(self):
        with patch.object(self.mb,'top_models') as top_models:
            top_models.return_value = [
                {'blueprint_id': '123' ,'reference_model':False, 'samplesize': 1234, 'max_reps': 5 },
                {'blueprint_id': '456' ,'reference_model':False, 'samplesize': 1234, 'max_reps': 5 }
            ]
            self.mb._leaderboard=[
                {'_id':'abc123', 'blueprint_id': '123' ,'partition_stats': [1,2,3,4,5], 'test': {'Gini': [0.86,0.86]}, 'blueprint': {'1': [['NUM'],['NI','GLMB'],'P']}},
                {'_id':'def345', 'blueprint_id': '456' ,'partition_stats': [1,2,3,4,5], 'test': {'Gini': [0.84,0.84]}, 'blueprint': {'1': [['NUM'],['NI','RFC'],'P']}}
            ]
            with patch.object(self.mb.q,'blend') as q_blend:
                self.mb.next_steps()
                args = q_blend.call_args[0][0]
                self.assertEqual(len(args['blender_items']),2)
                self.assertEqual(args['blender_family'],'binomial')
                self.assertIn('blender_method',args)

if __name__ == '__main__':
    unittest.main()
