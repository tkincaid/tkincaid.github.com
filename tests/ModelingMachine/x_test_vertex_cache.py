import unittest
from ModelingMachine.engine.vertex_cache import VertexCache, NotStoredError, BadRequestException
from ModelingMachine.engine.vertex import Vertex
from mock import patch

class VetexCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good_request = {
            'samplesize': 1000,
            'pid': 'asdsdf',
            'blueprint': {'1': [['NUM'], ['NI'], 'T'], '2': [['1'], ['GLM'], 'P'] },
            'dataset_id': '1234123',
            'partitions': [[1,-1],[2,-1]]
        }
        with patch("ModelingMachine.engine.vertex_cache.file_service.load_vertex") as mock_object:
            mock_object.return_value = Vertex(['NI'],'b')
            cls.vc = VertexCache(cls.good_request)

    def test_validate_request(self):
        bad_request = {}
        self.assertRaisesRegexp(BadRequestException,"Invalid request",VertexCache,bad_request)
        bad_request = {
            'samplesize': 1,
            'pid': 1,
            'blueprint': 1,
            'dataset_id': 1
        }
        self.assertRaisesRegexp(BadRequestException,"Either max_reps or paritions must be in request", VertexCache,bad_request)

        good_request = {
            'samplesize': 1000,
            'pid': 'asdsdf',
            'blueprint': {'1': [['NUM'],['NI','GLM'],'P'] },
            'dataset_id': '1234123',
            'max_reps': 1
        }
        with patch("ModelingMachine.engine.vertex_cache.file_service.load_vertex") as mock_object:
            mock_object.return_value = Vertex(['NI'],'b')
            vc = VertexCache(good_request)
            self.assertTrue(mock_object.called)
            self.assertEqual(len(vc.vertices.keys()),1)

        with patch("ModelingMachine.engine.vertex_cache.file_service.load_vertex") as mock_object:
            mock_object.return_value = Vertex(['NI'],'b')
            vc = VertexCache(self.good_request)
            self.assertTrue(mock_object.called)
            self.assertEqual(len(vc.vertices.keys()),4)

    def test_get_required_vertices(self):
        rv = self.vc._get_required_vertices('2', self.good_request)
        self.assertEqual(rv, [
            {'branch_id': '2', 'dataset_id': '1234123', 'samplesize': 1000},
            {'branch_id': '1', 'dataset_id': '1234123', 'samplesize': 1000}])

    def test_get_vertex_sample_hash(self):
        sh = self.vc._get_vertex_sample_hash('2', 123)
        self.assertEqual(sh, '99938a8e9b42afb63355475135826c2dec727e2c')

    def test_get_vertex_name(self):
        vn = self.vc._get_vertex_name('1234', '12345', {'r': 1, 'k': -1})
        self.assertEqual(vn, 'asdsdf-827ccb0eea8a706c4c34a16891f84e7b-1234-2-0')

    def test_get_vertex_id(self):
        vi = self.vc._get_vertex_id('2', '1234', {'r': 1, 'k': -1}, 1234)
        self.assertEqual(vi, 'asdsdf-81dc9bdb52d04dc20036dbd8313ed055-1f1e630b70a4c0b967eefc1c34e1a8e7973841c9-2-0')

    def test_get_vertex(self):
        vertex_definition = {'samplesize': 1234, 'branch_id': '1', 'partition': [{'r': 1, 'k': -1}], 'dataset_id': '12345'}
        with patch("ModelingMachine.engine.vertex_cache.file_service.load_vertex") as mock_object:
            mock_object.return_value = Vertex(['NI'],'b')
            v = self.vc.get_vertex(vertex_definition, False)
        self.assertEqual(v.steps[0][0].__class__.__name__, 'Numeric_impute')

if __name__ == '__main__':
    unittest.main()
