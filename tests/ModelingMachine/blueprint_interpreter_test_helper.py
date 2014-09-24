import os
import pandas
from bson.objectid import ObjectId

class BlueprintInterpreterTestHelper(object):

    def __init__(self, BlueprintInterpreter, WorkerRequest, RequestData, VertexFactory):
        self.BlueprintInterpreter = BlueprintInterpreter
        self.WorkerRequest = WorkerRequest
        self.RequestData = RequestData
        self.VertexFactory = VertexFactory
        self.oid1 = ObjectId()
        self.oid3 = ObjectId()
        self.pid = str(ObjectId())
        self.dataset_id = str(ObjectId())
        self.user_bp = {'1':[['ALL'],['USERTASK id=%s'%self.oid1],'P']}

    @property
    def user_task(self):
        modelfit = """function(response,data){model=function(data){prediction=rep(0.5,dim(data)[1]);return(prediction);};return(model);};"""
        modelpredict = """function(model,data){predictions=model(data);return(predictions);};"""
        out = {str(self.oid1): {'modelfit':modelfit, 'modelpredict':modelpredict, 'modelsource':'', 'modeltype':'R'}}
        return out

    def get_requestdata(self, data_dir, dataset):
            """ create a fake request data to go with request1 """
            nrows = 50
            filename = os.path.join(data_dir, dataset['filename'])
            target = dataset['target']
            ds = pandas.read_csv(filename)
            Y = ds.pop(target[0])
            y = {'main':Y.take(range(nrows)), 'holdout':Y.take(range(160,200))}
            out = self.RequestData(target[0],target[1],y)
            vts= []
            for col in ds.columns:
                vt = 'C' if ds[col].dtype=='object' else 'N'
                vts.append(vt)
            out.datasets[self.dataset_id] = {'main': ds.take(range(nrows)), 'holdout': ds.take(range(160,200)), 'vartypes': ''.join(vts)}
            out.usertasks = self.user_task
            out.partition['total_size'] = 50
            return out

    def create_request(self, pid=None, dataset_id=None, bp=None):
        if pid is None:
            pid = self.pid

        if dataset_id is None:
            dataset_id = self.dataset_id

        """ create a fake request """
        out = { 'blueprint': bp,
                'samplepct': 60,
                'partitions': [[0,-1]],
                'max_folds': 0,
                'blender': {},
                'command':'fit',
                'dataset_id': dataset_id,
                'pid': pid,
                'uid': str(self.oid3),
                'qid': 'testqid',
                'blueprint_id': 'testbp_id',
                }
        return out

    def create_blender_request(self, pid, dataset_id, bp1, bp2):
        if pid is None:
            pid = self.pid
        if dataset_id is None:
            dataset_id = self.dataset_id
        req = self.create_request(pid, dataset_id)
        req['partitions'] = [[i,-1] for i in range(5)]
        req['blueprint'] = {'1':[['1234567890'],['STK'],'T'],'2':[['1'],['GAMG'],'P']}
        req['blender'] = {}
        ins =  [{'dataset_id':dataset_id,'samplepct':100,'blueprint':bp1,'blender':{}}]
        ins += [{'dataset_id':dataset_id,'samplepct':100,'blueprint':bp2,'blender':{}}]
        req['blender']['inputs'] = ins
        return req

    def execute_blueprints(self, blueprints, request_data, pid=None, dataset_id=None):
        """
            Takes a list of blueprints and request datasets

            Returns the blueprint iterator (bi) object along with the output (output) of each build
        """
        bp1 = blueprints[0]
        bp2 = blueprints[1]

        result = {}

        # build a simple blueprint
        req = self.WorkerRequest(self.create_request(pid, dataset_id, bp1))
        vertex_factory = self.VertexFactory()
        bi = self.BlueprintInterpreter( vertex_factory, request_data, req)
        print bi
        out = bi._build(req)
        r = self.get_partition_results(out)
        print bi

        # predict on hold out data
        req2 = self.create_request(pid, dataset_id, bp1)
        req2 = self.WorkerRequest(req2)
        out2 = bi._build(req2, subset='holdout')
        r2 = self.get_partition_results(out2)

        self.asssert_build_output_differ(r, r2)

        #build a second blueprint
        req3 = self.WorkerRequest(self.create_request(pid, dataset_id, bp2))
        out3 = bi._build(req3)
        r3 = self.get_partition_results(out3)
        print bi

        #add more partitions
        req['partitions'] = req3['partitions'] = [[i,-1] for i in range(1,5)]
        out15 = bi._build(req)
        r15 = self.get_partition_results(out15)
        out35 = bi._build(req3)
        r35 = self.get_partition_results(out35)
        print bi

        #build a blender
        req4 = self.WorkerRequest(self.create_blender_request(pid, dataset_id, bp1, bp2))
        out4 = bi._build(req4)
        r4 = self.get_partition_results(out4)

        result['output'] = [out, out2, out3, out4]
        result['results'] = [r, r2, r3, r4]
        result['bi'] = bi

        return result

    def get_partition_results(self, out):
        partition_results = []
        for p in out:
            result = {'p': p, 'shape': out(**p).shape, 'mean': out(**p).mean()}
            print result['p'], result['shape'], result['mean']
            partition_results.append(result)
        return partition_results

    def asssert_build_output_differ(self, build_output1, build_output2):
        for i, p in enumerate(build_output1):
            assert build_output1[i]['mean'] !=  build_output2[i]['mean']
