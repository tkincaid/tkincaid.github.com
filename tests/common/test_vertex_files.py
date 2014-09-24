import unittest
import pytest
from mock import Mock, patch
from bson.objectid import ObjectId

from common.engine.vertex_files import delete_models, get_vertex_filenames, vertex_id, vertex_filename


class TestVertexFiles(unittest.TestCase):

    def setUp(self):
        self.addCleanup(self.stopPatching)

        self.patchers = []
        self.patchers.append(patch("common.services.project.ProjectServiceBase"))
        self.patchers.append(patch("common.services.queue_service_base.QueueServiceBase"))
        self.patchers.append(patch("common.engine.vertex_files.FileTransaction"))

        self.MockProjectService = self.patchers[0].start()
        self.MockQueueService = self.patchers[1].start()
        self.MockFileTransaction = self.patchers[2].start()

        self.pid = ObjectId()
        self.dataset_id = 'asdf'

        self.bp1 = {}
        self.bp1['1'] = (['NUM'],['NI'],'T')
        self.bp1['2'] = (['1'],['GLMB'],'P')

        self.bp2 = {}
        self.bp2['1'] = (['NUM'],['NI'],'T')
        self.bp2['2'] = (['CAT'],['DM'],'T')
        self.bp2['3'] = (['1','2'],['GLMB'],'P')

        self.bp3 = {}
        self.bp3['1'] = (['NUM'],['NI'],'T')
        self.bp3['2'] = (['1'],['RFC nt=10;ls=5'],'P')

    def stopPatching(self):
        super(TestVertexFiles, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def fakeModel(self, bp):
        return {'blueprint':bp, 'partitions':[[0,-1]], 'dataset_id':self.dataset_id,
                'pid':str(self.pid), 'max_folds':0, 'blender':{}, 'qid':1, 'samplepct':64}

    def fakeBlender(self,*models):
        out = self.fakeModel({'1':[['1234567890'],['GAMG'],'P']})
        out['partitions'] = [[i,-1] for i in range(5)]
        out['blender']['inputs'] = [{'dataset_id':model['dataset_id'],
            'samplepct':model['samplepct'], 'blueprint': model['blueprint'],
            'blender': model['blender']} for model in models]
        return out

    def fakeLeaderboard(self):
        out = []
        for bp in [self.bp1, self.bp3]:
            item = self.fakeModel(bp)
            item['_id'] = ObjectId()
            out.append(item)
        return out

    def test_delete_models(self):
        leaderboard = self.fakeLeaderboard()
        queue = [{'status':'settings'}, self.fakeModel(self.bp2)]

        self.MockProjectService.return_value.read_leaderboard.return_value = leaderboard
        self.MockProjectService.return_value.delete_leaderboard_items.return_value = True
        self.MockProjectService.return_value.delete_predictions.return_value = True
        self.MockQueueService.return_value.get.return_value = queue
        self.MockFileTransaction.return_value.exists.return_value = True
        self.MockFileTransaction.return_value.delete.return_value = True

        model_ids = [str(i['_id']) for i in leaderboard]

        out = delete_models(model_ids, self.MockProjectService(), self.MockQueueService())

        self.assertEqual(len(get_vertex_filenames(leaderboard)), 3)
        self.assertEqual(out, 2)

        self.assertEqual(self.MockProjectService.return_value.read_leaderboard.call_count, 1)
        self.assertEqual(self.MockQueueService.return_value.get.call_count, 1)
        self.assertEqual(self.MockProjectService.return_value.delete_leaderboard_items.call_count, 1)
        self.assertEqual(self.MockProjectService.return_value.delete_predictions.call_count, len(model_ids))
        self.assertEqual(self.MockFileTransaction.return_value.exists.call_count, out)
        self.assertEqual(self.MockFileTransaction.return_value.delete.call_count, out)

    def test_vertex_id(self):
        blueprint = {'1':[['NUM'], ['NI'], 'T'], '2':[['1'], ['GLMB'], 'P']}
        v= {'blueprint':blueprint, 'vertex_index':2, 'samplepct':50, 'dataset_id':'asdf'}
        out = vertex_id(v)
        expected = '2b57aa977647d0a49b4297e285777516b9195a49'
        self.assertEqual(out, expected)

        build_id = str(ObjectId(None))
        v= {'blueprint':blueprint, 'vertex_index':2, 'samplepct':50, 'dataset_id':'asdf', 'build_id':build_id}
        out = vertex_id(v)
        expected = '2b57aa977647d0a49b4297e285777516b9195a49|'+build_id
        self.assertEqual(out, expected)

    def test_vertex_filename(self):
        blueprint = {'1':[['NUM'], ['NI'], 'T'], '2':[['1'], ['GLMB'], 'P']}
        v= {'blueprint':blueprint, 'vertex_index':2, 'samplepct':50, 'dataset_id':'asdf', 'pid':'testpid'}
        p= (-1,-1)
        vid='2b57aa977647d0a49b4297e285777516b9195a49'
        e='projects/testpid/vertex/testpid-912ec803b2ce49e4a541068d495ab570-'+vid+'-0-0'
        out = vertex_filename(v, p)
        self.assertEqual(out, e)

        build_id = str(ObjectId(None))
        v= {'blueprint':blueprint, 'vertex_index':2, 'samplepct':50, 'dataset_id':'asdf', 'build_id':build_id, 'pid':'testpid'}
        out = vertex_filename(v, p, build_id=False)
        self.assertEqual(out, e)

        e='projects/testpid/vertex/testpid-912ec803b2ce49e4a541068d495ab570-'+vid+'-0-0-'+build_id
        out = vertex_filename(v, p)
        self.assertEqual(out, e)

if __name__ == '__main__':
    unittest.main()


