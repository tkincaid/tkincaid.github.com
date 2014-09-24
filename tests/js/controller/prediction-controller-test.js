define(
  [
    'angular-mocks',
    'datarobot',
    'lodash',
    'js/controller/prediction-controller.min'
  ],
  function(angularMocks, datarobot, _, LbModel){

    describe('PredictionController', function(){

      beforeEach(module('datarobot'));

      var scope, _ProjectService, _DatasetService, mockBackend;
      beforeEach(inject(function($controller, $rootScope, ProjectService, DatasetService, $httpBackend){
        _ProjectService = ProjectService;
        _DatasetService = DatasetService;
        mockBackend = $httpBackend;
        scope = $rootScope.$new();
        $controller('PredictionController', { $scope: scope});
      }));

      afterEach(function() {
        mockBackend.verifyNoOutstandingExpectation();
        mockBackend.verifyNoOutstandingRequest();
       });

      describe('Prediction action button', function(){
        // Original datasets (Universe, Raw Features, Informative Features):
        //   When holdout is locked we always show a download button as the predictions are computed when the model runs.
        //   newdata = false

        it('Should show "Download Prediction" for the original dataset', function(){
          var modelId = 1;
          var dataset = { computed : [], newdata: false};
          _ProjectService.project = { holdout_unlocked : false};
          scope.model = {lid: modelId};
          var status = scope.getButtonStatus(dataset);
          expect(status.val).toEqual(scope.status.DOWNLOAD);
          expect(status.status).toMatch(/Download/);
        });

        // Additional uploaded datasets:
        //   always computes predictions on the full data - holdout_unlocked is irrelevant.
        //   newdata = true

        it('Should show "Compute Prediction" for a new dataset', function(){
          var modelId = 1;
          var dataset = { computed : [], newdata: true};
          scope.model = {lid: modelId};
          var status = scope.getButtonStatus(dataset);
          expect(status.val).toEqual(scope.status.COMPUTE);
          expect(status.status).toMatch(/Compute/);
        });

        it('Should show "Compute Prediction" for a dataset for which predictions have been marked as deleted', function(){
          var modelId = 1;
          var dataset = { computed : [modelId], newdata:true, deleted: [modelId]};
          _ProjectService.project = { holdout_unlocked : false};
          scope.model = {lid: modelId};
          var status = scope.getButtonStatus(dataset);
          expect(status.val).toEqual(scope.status.COMPUTE);
          expect(status.status).toMatch(/Compute/);
        });

        it('Should show "Download Prediction" for a new dataset when dataset.computed ~~ model.lid', function(){
          var modelId = 1;
          var dataset = { computed : [modelId], computing : [modelId], newdata: true};
          scope.model = {lid: modelId};
          var status = scope.getButtonStatus(dataset);
          expect(status.val).toEqual(scope.status.DOWNLOAD);
          expect(status.status).toMatch(/Download/);
        });

        it('Should show "Calculating" for a dataset that has been submitted for processing', function(){
          var modelId = 1;
          var dataset = {computed : [], newdata : true,  status: scope.status.CALCULATING};
          scope.model = {lid: modelId};
          var status = scope.getButtonStatus(dataset);
          expect(status.val).toEqual(scope.status.CALCULATING);
          expect(status.status).toMatch(/Calculating/);
        });

        it('Should show "Calculating" for a dataset that has been submitted for processing AFTER refreshing', function(){
          var modelId = 1;
          var dataset = { computed : [], computing : [modelId],  newdata: true};
          scope.model = {lid: modelId};
          var status = scope.getButtonStatus(dataset);
          expect(status.val).toEqual(scope.status.CALCULATING);
          expect(status.status).toMatch(/Calculating/);
        });

        it('Should show "Calculating" for a dataset for which predictions have been marked as deleted and are now in process', function(){
          var modelId = 1;
          var dataset = { computed : [], computing : [modelId],  deleted:[modelId], newdata: false};
          scope.model = {lid: modelId};
          var status = scope.getButtonStatus(dataset);
          expect(status.val).toEqual(scope.status.CALCULATING);
          expect(status.status).toMatch(/Calculating/);
        });
      });

      describe('On Prediction Done', function(){
        it('Should remove lid from deleted (computed)', function(){
          var lid = '53b59963a530d72ee17385da';

          var prediction = {dataset_id : '53b59737a530d72e637385da', deleted: [lid]};
          _DatasetService.predictionList = [prediction];

          var data = {
            message: "{\"prediction_done\": {\"dataset_id\": \"53b59737a530d72e637385da\", \"computed\": \"53b59963a530d72ee17385da\"}}"
          };

          scope.onPredictDone(data);

          expect(prediction.deleted).toEqual([]);
          expect(prediction.computed).toEqual([lid]);
        });

        it('Should add to computedh list', function(){
          var lid = '53b59963a530d72ee17385da';

          var prediction = {dataset_id : '53b59737a530d72e637385da', deleted: [lid]};
          _DatasetService.predictionList = [prediction];

          var data = {
            message: "{\"prediction_done\": {\"dataset_id\": \"53b59737a530d72e637385da\", \"computedh\": \"53b59963a530d72ee17385da\"}}"
          };

          scope.onPredictDone(data);

          expect(prediction.deleted).toEqual([]);
          expect(prediction.computedh).toEqual([lid]);
        });
      });

      describe('Deleting datasets', function(){
        it('Should remove the dataset from the in-memory copy', function(){
          _DatasetService.predictionList = [
            {dataset_id : 1},
            {dataset_id : 2},
          ];

          var pid = 123;
          _ProjectService.project = { pid : pid};

          var removeDatasetId = 1;

          mockBackend.expectDELETE('/project/' + pid + '/dataset/' + removeDatasetId).respond(200, '');

          scope.removeDataset(removeDatasetId);

          mockBackend.flush();

          expect(_DatasetService.predictionList.length).toEqual(1);
          expect(_DatasetService.predictionList[0].dataset_id).toEqual(2);
        });
      });
    });
  }
);