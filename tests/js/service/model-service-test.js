define(
  [
    'lodash',
    'angular-mocks',
    'datarobot',
    'js/service/model-service.min',
    'js/service/project-service.min',
    'js/service/user-service.min'
  ],
  function(_) {
    describe('ModelService', function() {

      beforeEach(module('datarobot'));

      var metrics =[
        {name: 'AUC', ascending: false},
        {name: 'Gini', ascending: false},
        {name: 'Gini Norm', ascending: false},
        {name: 'LogLoss', ascending: true},
        {name: 'MAD', ascending: true}
      ];

      var modelService, projectService, mockBackend;
      beforeEach(inject(function (ModelService, ProjectService, $httpBackend) {
        modelService = ModelService;
        projectService = ProjectService;
        mockBackend = $httpBackend;
      }));

      describe('Getting the leaderboard from the server', function() {

        it('should return saved models if they are available', function() {
          modelService.modelsData = [
            { lid: '52dd8522100d2b5ea4b62e37' }
          ];

          projectService.project = { pid: 'ABC123', filename: 'kickcars-training-sample.csv' };

          modelService.getLeaderboard(projectService.project).then(function(response) {
            expect(response).toBe(modelService.modelsData);
          });
        });

        it('should fetch and update models when local data (modelsData) is present', function() {

          projectService.project = { pid: 'ABC123', filename: 'kickcars-training-sample.csv' };

          // existing leaderboard with 2 rows
          modelService.modelsData = [
            { lid: '52dd8522100d2b5ea4b62e37', Label: 'Generalized Linear Model (Bernoulli Distribution) (1)' },
            { lid: '52dd8522100d2b5ea4b62e38', Label: 'Regularized Logistic Regression (L1) (2)'}
          ];

          // full fetch returns existing rows plus 2 more
          // fetch delivers a _id from mongo as the lid.
          var models = [
              { _id: '52dd8522100d2b5ea4b62e37', model_type:'Generalized Linear Model (Bernoulli Distribution)', bp:1 },
              { _id: '52dd8522100d2b5ea4b62e38', model_type:'Regularized Logistic Regression (L1)', bp:2},
              { _id: '52dd8522100d2b5ea4b62e39', model_type:'Regularized Logistic Regression (L2)', bp:3 },
              { _id: '52dd8522100d2b5ea4b62e40', model_type:'RandomForest Classifier (Gini)', bp:4 }
            ];

          // full returns existing rows plus 2 more
          // after the fetch is performed it filters the response through leaderboard-model.js

          // key : _id is converted to key : lid
          var assert = [
              { lid: '52dd8522100d2b5ea4b62e37', Label: 'Generalized Linear Model (Bernoulli Distribution) (1)', model_type:'Generalized Linear Model (Bernoulli Distribution)', bp:1},
              { lid: '52dd8522100d2b5ea4b62e38', Label: 'Regularized Logistic Regression (L1) (2)', model_type:'Regularized Logistic Regression (L1)', bp:2 },
              { lid: '52dd8522100d2b5ea4b62e39', Label: 'Regularized Logistic Regression (L2) (3)', model_type:'Regularized Logistic Regression (L2)', bp:3},
              { lid: '52dd8522100d2b5ea4b62e40', Label: 'RandomForest Classifier (Gini) (4)', model_type:'RandomForest Classifier (Gini)', bp:4}
            ];

          mockBackend
            .expectGET('/project/' + projectService.project.pid + '/models')
            .respond(models);

          modelService.getLeaderboard(projectService.project, true).then(function(response) {
            expect(response).toEqual(assert);
          });

          mockBackend.flush();
        });

        it('should create the metric list', function(){
          expect(_.size(modelService.metricList)).toEqual(0);

          modelService.setMetrics(metrics);

          expect(_.size(modelService.metricList)).toEqual(5);
        });

        it('should return the direction', function(){
          expect(modelService.isMetricSortedInAscendingOrder('does-not-exist')).toBe(null);

          modelService.setMetrics(metrics);

          expect(modelService.isMetricSortedInAscendingOrder('AUC')).toBe(false);
          expect(modelService.isMetricSortedInAscendingOrder('MAD')).toBe(true);
        });
      });

    });
  }
);