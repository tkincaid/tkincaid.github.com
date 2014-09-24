import unittest
import os
from mock import Mock, patch
from bson.objectid import ObjectId
from ModelingMachine.engine.monitor import FakeMonitor

from ModelingMachine.engine.mocks import RequestData
from ModelingMachine.engine.blueprint_interpreter import BlueprintInterpreter
from ModelingMachine.engine.worker_request import WorkerRequest, VertexDefinition
from ModelingMachine.engine.vertex_factory import VertexFactory, VertexNotStoredError, vertex_id, VertexCache, ModelCache
from common.engine.vertex_files import get_vertex_filenames
from ModelingMachine.engine.vertex import Vertex
import ModelingMachine.engine.user_vertex
from config.engine import EngConfig
from tests.ModelingMachine.blueprint_interpreter_test_helper import BlueprintInterpreterTestHelper
import ModelingMachine

from tests.IntegrationTests.storage_test_base import StorageTestBase

class fakeVertex(Vertex):
    def __init__(self,*args,**kwargs):
        self.fit_count = 0
        self.act_count = 0
        super(fakeVertex,self).__init__(*args,**kwargs)
        self.__class__.__name__ = "Vertex"
    def _act(self,*args,**kwargs):
        self.act_count += 1
        return super(fakeVertex,self)._act(*args,**kwargs)
    def _fit_and_act(self,*args,**kwargs):
        self.fit_count += 1
        return super(fakeVertex,self)._fit_and_act(*args,**kwargs)
    def update(self,load):
        self.fit_count=load['fit_count']
        self.act_count=load['act_count']
        return super(fakeVertex,self).update(load)
    def dump(self,*args,**kwargs):
        load = super(fakeVertex,self).dump(*args,**kwargs)
        load['fit_count'] = self.fit_count
        load['act_count'] = self.act_count
        return load
    def save(self):
        """ save vertex data to files on disk """
        out = {}
        for partition in self.fit_parts:
            load = self.dump(partition)
            out[partition] = self._dump_pickle(load)
        return out


class VertexFactoryTest(StorageTestBase):
    @classmethod
    def setUpClass(cls):
        super(VertexFactoryTest, cls).setUpClass()
        cls.vertex_monitor_patch = patch('ModelingMachine.engine.vertex.Monitor', FakeMonitor)
        cls.vertex_monitor_mock = cls.vertex_monitor_patch.start()

        cls.vf = VertexFactory()
        cls.bp1 = {}
        cls.bp1['1'] = (['NUM'],['NI'],'T')
        cls.bp1['2'] = (['1'],['GLMB'],'P')

        cls.bp2 = {}
        cls.bp2['1'] = (['NUM'],['NI'],'T')
        cls.bp2['2'] = (['CAT'],['DM'],'T')
        cls.bp2['3'] = (['1','2'],['GLMB'],'P')

        cls.bp3 = {}
        cls.bp3['1'] = (['NUM'],['NI'],'T')
        cls.bp3['2'] = (['1'],['RFC nt=10;ls=5'],'P')
        cls.requirements = [
            {'blueprint': {'1': [['1234'], ['STK'], 'T'], '2': [['1'], ['GAM'], 'P']}, 'dataset_id': '1234', 'samplesize': 100, 'vertex_index': '2'},
            {'blueprint': {'1': [['1234'], ['STK'], 'T'], '2': [['1'], ['GAM'], 'P']}, 'dataset_id': '1234', 'samplesize': 100, 'vertex_index': '1'},
            {'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB'], 'P')}, 'dataset_id': '1234', 'samplesize': 100, 'vertex_index': '2'},
            {'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB'], 'P')}, 'dataset_id': '1234', 'samplesize': 100, 'vertex_index': '1'},
            {'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['RFC nt=10;ls=5'], 'P')}, 'dataset_id': '1234', 'samplesize': 100, 'vertex_index': '2'},
            {'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['RFC nt=10;ls=5'], 'P')}, 'dataset_id': '1234', 'samplesize': 100, 'vertex_index': '1'}
        ]

        cls.bp_helper = BlueprintInterpreterTestHelper(
            BlueprintInterpreter,
            WorkerRequest,
            RequestData,
            VertexFactory
        )

    @classmethod
    def tearDownClass(cls):
        super(VertexFactoryTest, cls).tearDownClass()
        cls.vertex_monitor_patch.stop()

    def setUp(self):
        self.pid = ObjectId()
        self.dataset_id = str(ObjectId())
        self.test_directory, self.datasets = self.create_test_files()

    def tearDown(self):
        self.clear_test_dir()

    def create_request(self):
        req_dict = self.bp_helper.create_blender_request(None, None, self.bp1,self.bp3)
        request = WorkerRequest(req_dict)
        vertex_factory = VertexFactory()
        return request, vertex_factory

    def test_get_vertex_filenames(self):
        pid = ObjectId()
        dataset_id = str(ObjectId())

        models = []
        models += [self.bp_helper.create_request(pid, dataset_id, self.bp1)]

        filenames = get_vertex_filenames(models)
        self.assertEqual(len(filenames), 2)

        models += [self.bp_helper.create_request(pid, dataset_id, self.bp2)]
        #+2 new vertices
        filenames = get_vertex_filenames(models)
        self.assertEqual(len(filenames), 4)

        models += [self.bp_helper.create_request(pid, dataset_id, self.bp3)]
        #+1 new vertices
        filenames = get_vertex_filenames(models)
        self.assertEqual(len(filenames), 5)

        models += [self.bp_helper.create_blender_request(pid, dataset_id, self.bp1,self.bp3)]
        #+1 new vertices, all 5 partitions
        filenames = get_vertex_filenames(models)
        self.assertEqual(len(filenames), 6*5)



    def test_required_vertices(self):
        request, vertex_factory = self.create_request()

        for vertex_definition in vertex_factory._required_vertices(request):
            #print vertex_definition
            self.assertIsInstance(vertex_definition, VertexDefinition)

        self.assertEqual( len(list(vertex_factory._required_vertices(request))), 6 )

    def test_get(self):
        request, vertex_factory = self.create_request()

        for vertex_definition in request.blueprint:
            vertex = vertex_factory.get(vertex_definition)
            self.assertIsInstance(vertex, dict)
            print vertex
            self.assertEqual(Vertex(**vertex).task_list, vertex_definition.tasks)

    def test_add_and_load(self):
        request, vertex_factory = self.create_request()
        vertex_factory.load(request)
        self.assertEqual(vertex_factory.cache, {})

        seen = set()
        for i,vertex_definition in enumerate(vertex_factory._required_vertices(request)):
            vid = vertex_id(vertex_definition)
            #repeated vertices will break this test
            if vid in seen:
                continue
            seen.add(vid)
            vertex = Vertex( vertex_definition.tasks, id=vid )
            vertex.fit_parts = set(vertex_definition.partitions)
            rc = vertex_factory.add(vertex_definition, vertex.save())
            self.assertGreater(len(vertex_factory), 0)
            cache_keys = vertex_factory.cache.keys()

        vertex_factory.clear_cache()
        self.assertEqual(vertex_factory.cache, {})

        vertex_factory.load(request)
        self.assertEqual(set(vertex_factory.cache.keys()), set(cache_keys))

    def test_execution_with_vertex_factory(self):
        blueprints = [self.bp1, self.bp3]
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        result = self.bp_helper.execute_blueprints(blueprints, request_data)

        request, vertex_factory = self.create_request()
        self.assertEqual(vertex_factory.cache, {})
        request['command'] = 'predict'
        request['scoring_dataset_id'] = 'asdf'

        bi = BlueprintInterpreter(vertex_factory, request_data, request)
        out = bi._build(request,subset='main')
        for p in out:
            self.assertEqual( out(**p).mean(), result['output'][3](**p).mean() )


    def test_get_from_cache(self):
        request, vertex_factory = self.create_request()

        for vertex_definition in request.blueprint:
            with patch('ModelingMachine.engine.vertex_factory.time.sleep') as mock:
                with self.assertRaises(VertexNotStoredError):
                    vertex_factory.get_from_cache(vertex_definition, strict_cache=True)
                    self.assertEqual(mock.call_count,3)

    @patch('ModelingMachine.engine.task_executor.Vertex',fakeVertex)
    def broken_test_get_cached_vertex(self):
        #####################################################################################
        # THIS TEST IS NO LONGER POSSIBLE BECAUSE VERTEX FILES CANNOT BE OVERWRITTEN
        # THIS TEST USED TO EXPLOIT A BUG IN THE APP
        ######################################################################################
        vertex_factory = VertexFactory()
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        # empty cache
        folder = EngConfig['DATA_DIR']
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path) and 'tmp' in os.path.basename(file_path):
                os.unlink(file_path)

        #
        # fit BP 1: NI & GLM
        # should not load anything from cache
        #
        req_dict = {
            'uid': '1',
            'blueprint':self.bp1,
            'samplepct':64,
            'dataset_id':self.bp_helper.dataset_id,
            'pid':self.bp_helper.pid,
            'uid':self.bp_helper.oid3,
            'qid':1,
            'max_folds':1,
            'command':'fit',
            'partitions':[(0,-1)]}
        request = WorkerRequest(req_dict)
        # vertex_factory should create empty vertices for first BP
        cached_vertex1 = vertex_factory.get(list(request.blueprint)[0])
        cached_vertex2 = vertex_factory.get(list(request.blueprint)[1])
        self.assertEqual(cached_vertex1['stored_files'],{})
        self.assertEqual(cached_vertex2['stored_files'],{})
        # run first BP
        bi = BlueprintInterpreter(vertex_factory, request_data, request)
        out = bi._build(request,subset='main')
        # check that fit was called once per vertex but not act
        cached_vertex1 = vertex_factory.get(list(request.blueprint)[0])
        cached_vertex2 = vertex_factory.get(list(request.blueprint)[1])
        self.assertEqual(len(vertex_factory.cache.keys()),2) # cache should have 2 objects
        cached_vertex1 = fakeVertex(**cached_vertex1)
        cached_vertex2 = fakeVertex(**cached_vertex2)
        self.assertEqual(cached_vertex1.fit_count,1)
        self.assertEqual(cached_vertex1.act_count,0)
        self.assertEqual(cached_vertex2.fit_count,1)
        self.assertEqual(cached_vertex2.act_count,0)

        #
        # fit BP 3: NI & RF
        # vertex 1 should be loaded from cache and vertex 2 should be fit
        #
        req_dict['blueprint'] = self.bp3
        request = WorkerRequest(req_dict)
        # NI vertex should have partition 0 fit and RF vertex should be empty
        cached_vertex1 = vertex_factory.get(list(request.blueprint)[0])
        cached_vertex2 = vertex_factory.get(list(request.blueprint)[1])
        cached_vertex1 = fakeVertex(**cached_vertex1)
        cached_vertex2 = fakeVertex(**cached_vertex2)
        self.assertEqual(cached_vertex1.fit_parts,set([(0, -1)]))
        self.assertEqual(cached_vertex2.fit_parts,set([]))
        # run second BP
        bi = BlueprintInterpreter(vertex_factory, request_data, request)
        out = bi._build(request,subset='main')
        # make sure act gets called for NI but not RF
        cached_vertex1 = vertex_factory.get(list(request.blueprint)[0])
        cached_vertex2 = vertex_factory.get(list(request.blueprint)[1])
        cached_vertex1 = fakeVertex(**cached_vertex1)
        cached_vertex2 = fakeVertex(**cached_vertex2)
        self.assertEqual(len(vertex_factory.cache.keys()),3) # cache should have 3 objects
        self.assertEqual(cached_vertex1.fit_count,2) # fit is always called, even for cached objects
        self.assertEqual(cached_vertex1.act_count,1)
        self.assertEqual(cached_vertex2.fit_count,1)
        self.assertEqual(cached_vertex2.act_count,0)

        #
        # re-fit BP 3: NI & RF with one unfit partition
        # vertex 1 & 2 should be loaded from cache and both should be fit
        #
        req_dict['partitions'] = [(0,-1),(1,-1)]
        request = WorkerRequest(req_dict)
        # both NI & RF should have partition 0 fit
        cached_vertex1 = vertex_factory.get(list(request.blueprint)[0])
        cached_vertex2 = vertex_factory.get(list(request.blueprint)[1])
        cached_vertex1 = fakeVertex(**cached_vertex1)
        cached_vertex2 = fakeVertex(**cached_vertex2)
        self.assertEqual(cached_vertex1.fit_parts,set([(0, -1)]))
        self.assertEqual(cached_vertex2.fit_parts,set([(0, -1)]))
        # re-run second BP on different partitions
        bi = BlueprintInterpreter(vertex_factory, request_data, request)
        out = bi._build(request,subset='main')
        # make sure act is not called
        cached_vertex1 = vertex_factory.get(list(request.blueprint)[0])
        cached_vertex2 = vertex_factory.get(list(request.blueprint)[1])
        cached_vertex1 = fakeVertex(**cached_vertex1)
        cached_vertex2 = fakeVertex(**cached_vertex2)
        self.assertEqual(len(vertex_factory.cache.keys()),3) # cache should still have 3 objects
        self.assertEqual(cached_vertex1.fit_count,3)
        self.assertEqual(cached_vertex1.act_count,1)
        self.assertEqual(cached_vertex2.fit_count,2)
        self.assertEqual(cached_vertex2.act_count,0)


class ModelCacheMixinTest(unittest.TestCase):

    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE':2})
    def test_model_cache_update(self):
        """
        test that the update function correctly adds new models to the cache
        AND the cache size (# of models) is limited to the value of 'WORKER_MODEL_CACHE'
        """
        mc = ModelCache()
        request = {'pid':'1234','samplepct':50, 'dataset_id':'1234', 'partitions':[[-1,-1]]}
        for bp in range(1,10):
            request['blueprint_id'] = bp
            mc.update_cached_model(bp, request)
            self.assertEqual( len(mc.cached_models), min(2,bp) )
            if bp>2:
                self.assertTrue(bp-2 not in mc.cached_models.values())

    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE':2})
    def test_model_cache_get(self):
        mc = ModelCache()
        request = {'pid':'1234','samplepct':50, 'dataset_id':'1234', 'partitions':[[-1,-1]]}

        #model in cache
        request['blueprint_id'] = 1
        mc.update_cached_model('test value', request)
        out = mc.get_cached_model(request)
        self.assertEqual(out, 'test value')

        #model Not in cache
        request['blueprint_id'] = 2
        out = mc.get_cached_model(request) #expect to return new instance of VertexCache
        self.assertIsInstance(out, VertexCache)
        self.assertEqual(out.cache, {})

    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE':2})
    def test_model_cache_remove(self):
        mc = ModelCache()
        request = {'pid':'1234','samplepct':50, 'dataset_id':'1234', 'partitions':[[-1,-1]]}

        for bp in range(1,3):
            request['blueprint_id'] = bp
            mc.update_cached_model(bp, request)
            out = mc.get_cached_model(request)
            self.assertEqual(out, bp)

        self.assertEqual(mc.cached_models.values(), [1,2])
        mc.remove_from_cache(request)

        self.assertEqual(mc.cached_models.values(), [1])

    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE':2})
    def test_model_cache_clear(self):
        mc = ModelCache()
        request = {'pid':'1234','samplepct':50, 'dataset_id':'1234', 'partitions':[[-1,-1]]}

        for bp in range(1,3):
            request['blueprint_id'] = bp
            mc.update_cached_model(bp, request)
            out = mc.get_cached_model(request)
            self.assertEqual(out, bp)

        self.assertEqual(mc.cached_models.values(), [1,2])

        mc.clear_cache()

        self.assertEqual(mc.cached_models.values(), [])



if __name__ == '__main__':
    unittest.main()
