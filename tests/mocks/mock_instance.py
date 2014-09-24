from bson import ObjectId
from copy import deepcopy
import json
import os

from MMApp.entities.instance import InstanceServiceContract, InstanceModel
from MMApp.entities.prediction_api_instance import ApiInstanceModel
from MMApp.entities.prediction_api_instance import SHARED_PREDICTION_API_KEYWORD

class MockInstanceService(InstanceServiceContract):

    def __init__(self, uid):
        self.uid = ObjectId(uid) if uid else None
        instance_file = os.path.join(os.path.dirname(__file__),'../testdata/fixtures/instances.json')

        with open(instance_file) as f:
            mock_instances = json.loads(f.read())
            self.IN_MEMORY_INSTANCES = [InstanceModel(**i) for i in mock_instances]


    def terminate(self, instance_id):
        return self.IN_MEMORY_INSTANCES[0]

    def stop(self, instance_id):
        return self.IN_MEMORY_INSTANCES[0]

    def start(self, instance_id):
        return self.IN_MEMORY_INSTANCES[0]

    def launch(self, instances):
        for i in instances:
            i._id = ObjectId()
        return instances

    def save(self, instance):
        instance._id  = ObjectId()
        self.IN_MEMORY_INSTANCES.append(instance)
        return instance._id

    def get(self, instance_id = None, include_all = False):
        if instance_id:
            instance_id = str(instance_id)
            for m in self.IN_MEMORY_INSTANCES:
                if m._id == instance_id:
                    return deepcopy(m)
        else:
            return deepcopy(self.IN_MEMORY_INSTANCES)

class MockApiInstanceService(MockInstanceService):

    def activate_model(self, instance_id, lid):
        pass

    def deactivate_model(self, instance_id, lid):
        pass

    def get_default_prediction_api_instance(self, ):
        return ApiInstanceModel.from_dict({
            '_id': SHARED_PREDICTION_API_KEYWORD,
            'is_dedicated' : 0,
            'status':8,
            'host_name':'mock-prediction-api.datarobot.com',
            'resource':'prediction',
            'type':'NA',
        })