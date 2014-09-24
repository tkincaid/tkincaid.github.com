import unittest
import pytest
from mock import Mock, patch
from bson.objectid import ObjectId

from config.engine import EngConfig
from config.test_config import db_config as config
from common.wrappers import database


from common.services.project import delete_project


class TestDeleteProject(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new("tempstore", host=config['tempstore']['host'],
                port=config['tempstore']['port'])
        self.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

        self.tables = ['project','leaderboard','metadata','eda','eda_map','predictions','predict_parts','prediction_tabulation']
        self.pidkeys = ['bp_counter','qid_counter','inprogress','queue','queue_settings','onhold','completed','errors','wsready','stage']
        self.lidkeys = ['parallelcv']


    def setUp(self):
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

    @classmethod
    def tearDownClass(self):
        self.tempstore.conn.flushdb()
        for table in self.tables:
            self.persistent.destroy(table=table)

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

    def fakeLeaderboard(self, pid):
        out = []
        for bp in [self.bp1, self.bp3]:
            item = self.fakeModel(bp)
            item['_id'] = ObjectId()
            item['pid'] = pid
            out.append(item)
        return out

    def fakeProject(self, is_deleted):
        pid = ObjectId()

        leaderboard = self.fakeLeaderboard(pid)

        #mongo data
        for table in self.tables:
            if table=='project':
                self.persistent.create({'_id': pid, 'is_deleted':is_deleted}, table='project')
            elif table=='leaderboard':
                for i in leaderboard:
                    self.persistent.create(i, table='leaderboard')
            else:
                self.persistent.create({'pid': pid, 'test':'test'}, table=table)

        #redis data
        for key in self.pidkeys:
            self.tempstore.create('test', keyname=key, index=str(pid))

        for key in self.lidkeys:
            for lid in [i['_id'] for i in leaderboard]:
                self.tempstore.create('test', keyname=key, index=str(lid))

        return pid

    def project_exists(self, pid):
        check = []
        for table in self.tables:
            if table=='project':
                criteria={'_id':pid}
            else:
                criteria={'pid':pid}
            query = self.persistent.read(criteria, table=table, result={})
            check.append( query!={} )
        return all(check)

    def test_delete_project(self):
        pid1 = self.fakeProject(False)
        pid1_redis_keys = set(self.tempstore.conn.keys())

        pid2 = self.fakeProject(True)
        pid2_redis_keys = set(self.tempstore.conn.keys()) - pid1_redis_keys

        #make sure the set of keys in redis aren't empty
        self.assertGreater(len(pid1_redis_keys),0)
        self.assertGreater(len(pid2_redis_keys),0)

        #try to delete project 2
        with patch('common.services.project.delete_vertex_files') as mock:
            out = delete_project(pid2)
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(out, True)
        #keys for pid2 should have been removed, but keys for pid1 should not.
        self.assertEqual(set(self.tempstore.conn.keys()), pid1_redis_keys)
        self.assertEqual(self.project_exists(pid1), True)
        self.assertEqual(self.project_exists(pid2), False)

        #try to delete project 1
        with patch('common.services.project.delete_vertex_files') as mock:
            #for pid2: is_deleted=False, thus nothing should be deleted
            out = delete_project(pid1) 
            self.assertEqual(mock.call_count, 0)
            self.assertEqual(out, False)
        #nothing should have been deleted
        self.assertEqual(set(self.tempstore.conn.keys()), pid1_redis_keys)
        self.assertEqual(self.project_exists(pid1), True)

        #mark project 1 for deletion and retry
        self.persistent.update(values={'is_deleted':True}, condition={'_id':pid1}, table='project')
        with patch('common.services.project.delete_vertex_files') as mock:
            out = delete_project(pid1) 
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(out, True)
        #everything should have been deleted
        self.assertEqual(len(self.tempstore.conn.keys()), 0)
        self.assertEqual(self.project_exists(pid1), False)



if __name__ == '__main__':
    unittest.main()
        
        
