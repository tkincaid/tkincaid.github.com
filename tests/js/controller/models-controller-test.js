define(
  [
    'angular-mocks',
    'datarobot',
    'lodash',
    'js/model/leaderboard-model.min',
    'js/controller/models-controller.min'
  ],
  function(angularMocks, datarobot, _){

    describe('ModelsController', function(){

      beforeEach(module('datarobot'));

      var scope, _LbModel, _ModelService, _ProjectService, mockBackend;
      var modelsServerResponse = [
        {"grid_scores": {}, "best_parameters": {}, "qid": "1", "task_version": {"NI": "0.1", "GLMB": "0.1"}, "roc": "", "part_size": [["1", 122, 35]], "max_reps": 1, "samplesize": 157.0, "ec2": {"reps=1": {"spot_price": 0, "on_demand_price": 0, "availability_zone": "None", "instance_size": "None", "CPU_count": 4, "instance_type": "local", "workers": 2}}, "uid": "52df07e3637aba07aa1b9c15", "dataset_id": "52df07e3637aba5fec709cd2", "partition_stats": {"(0, -1)": {"train_size": 122, "test_size": 35, "time_real": "0.0083"}}, "features": ["Missing Values Imputed"], "resource_summary": {"reps=1": {"total_cpu_time": 0.05999999999999997, "total_noncached_clock_time": 0.06165289878845215, "cpu_usage": 0.9731902502407271, "bp_time": 0.17249798774719238, "cached": "false", "max_noncached_ram": 0, "bp_cost": 1.964560416009691e-05, "noncached_cost": 0, "rate": 0.013666666666666666, "cost": 7.021580139795939e-06, "total_noncached_cpu_time": 0.05999999999999997, "total_clock_time": 0.06165289878845215, "max_ram": 0, "noncached_cpu_usage": 0}}, "training_dataset_id": "52df07e3637aba5fec709cd2", "icons": [0], "total_size": 157, "parts": [["1", 23]], "test": {"LogLoss": [0.54582], "AUC": [0.70455], "Gini": [0.14026], "Gini Norm": [0.40909], "labels": ["(0,-1)"], "metrics": ["LogLoss", "AUC", "Ians Metric", "Gini", "Gini Norm", "Rate@Top10%", "Rate@Top5%"], "Rate@Top5%": [1.0], "Ians Metric": [0.23741], "Rate@Top10%": [1.0]}, "insights": "NA", "parts_label": ["partition", "NonZeroCoefficients"], "blueprint": {"1": [["NUM"], ["NI"], "T"], "2": [["1"], ["GLMB"], "P"]}, "hash": "5b6b8ad6e3613e1309139bb51b8d2aa2e401f277", "blueprint_id": "d4c06a5c23cf1d917019720bceba32c8", "pid": "52df07e3637aba5fec709cd1", "originalName": null, "lift": "", "bp": 1, "task_parameters": "{'NI': {'threshold': '50', 'fullset': 'True'}, 'GLMB': {'GridSearch: stratified': 'True', 'GridSearch: algorithm': 'Tom', 'GridSearch: max_iterations': '15', 'GridSearch: test_fraction': 'None', 'GridSearch: CV_folds': '5', 'p': '1.5', 'GridSearch: metric': 'None', 'distribution': 'Bernoulli', 'GridSearch: step': '10', 'GridSearch: random_state': '1234'}}", "task_cnt": 2, "max_folds": 0, "metablueprint": ["NewMetablueprint", "5.0"], "vertex_cache_hits": 0, "finish_time": {"reps=1": 1390348265.782807}, "vertices": {"1": {"output_shape": [[157, 24]]}, "2": {"output_shape": [[157, 1]]}}, "time_real": [["1", "0.0083"]], "s": 0, "task_info": {"reps=1": [[{"fit max RAM": 0, "fit CPU pct": 0, "fit CPU time": 0.010000000000000009, "transform max RAM": 0, "fit clock time": 0.010514020919799805, "fit avg RAM": 0, "cached": false, "ytrans": null, "fit total RAM": 12288917504, "transform avg RAM": 0, "transform clock time": 0.03032207489013672, "version": "0.1", "arguments": null, "task_name": "NI", "transform CPU time": 0.02999999999999997, "transform total RAM": 12288917504, "transform CPU pct": 0}], [{"fit max RAM": 0, "fit CPU pct": 0, "fit CPU time": 0.009999999999999981, "predict CPU pct": 0, "fit clock time": 0.01416778564453125, "predict max RAM": 0, "fit avg RAM": 0, "cached": false, "ytrans": null, "fit total RAM": 12288917504, "predict CPU time": 0.010000000000000009, "predict total RAM": 12288917504, "predict clock time": 0.006649017333984375, "version": "0.1", "predict avg RAM": 0, "arguments": null, "task_name": "GLMB"}]]}, "time": {"finish_time": {"reps=1": 1390348265.782815}, "total_time": {"reps=1": 0.17249798774719238}, "start_time": {"reps=1": 1390348265.610318}}, "model_type": "GLM - Bernoulli", "vertex_cnt": 2, "_id": "52df07e9637aba601a709cd0", "blend": 0, "reference_model": false, "extras": ""},
        {"grid_scores": {}, "best_parameters": {}, "qid": "3", "task_version": {"GLMB": "0.1", "GS": "0.1"}, "roc": "", "part_size": [["1", 122, 35]], "max_reps": 1, "samplesize": 157.0, "ec2": {"reps=1": {"spot_price": 0, "on_demand_price": 0, "availability_zone": "None", "instance_size": "None", "CPU_count": 4, "instance_type": "local", "workers": 2}}, "uid": "52df07e3637aba07aa1b9c15", "dataset_id": "52df07e3637aba5fec709cd2", "partition_stats": {"(0, -1)": {"train_size": 122, "test_size": 35, "time_real": "0.07955"}}, "features": ["Constant Splines"], "resource_summary": {"reps=1": {"total_cpu_time": 0.1, "total_noncached_clock_time": 0.22187376022338867, "cpu_usage": 0.4507067437777105, "bp_time": 0.45728397369384766, "cached": "false", "max_noncached_ram": 0, "bp_cost": 5.20795636706882e-05, "noncached_cost": 0, "rate": 0.013666666666666666, "cost": 2.5268956025441486e-05, "total_noncached_cpu_time": 0.1, "total_clock_time": 0.22187376022338867, "max_ram": 0, "noncached_cpu_usage": 0}}, "training_dataset_id": "52df07e3637aba5fec709cd2", "icons": [0], "total_size": 157, "parts": [["1", 85]], "test": {"LogLoss": [13.06148], "AUC": [0.375], "Gini": [-0.08312], "Gini Norm": [-0.24242], "labels": ["(0,-1)"], "metrics": ["LogLoss", "AUC", "Ians Metric", "Gini", "Gini Norm", "Rate@Top10%", "Rate@Top5%"], "Rate@Top5%": [0.6], "Ians Metric": [-0.06061], "Rate@Top10%": [0.6]}, "insights": "NA", "parts_label": ["partition", "NonZeroCoefficients"], "blueprint": {"1": [["NUM"], ["GS"], "T"], "2": [["1"], ["GLMB"], "P"]}, "hash": "7956ab85bc868d0cddbd0803ffee8bb62e782077", "blueprint_id": "10b9623bb54915ee8240a2ad18d0a727", "pid": "52df07e3637aba5fec709cd1", "originalName": null, "lift": "", "bp": 3, "task_parameters": "{'GLMB': {'GridSearch: stratified': 'True', 'GridSearch: algorithm': 'Tom', 'GridSearch: max_iterations': '15', 'GridSearch: test_fraction': 'None', 'GridSearch: CV_folds': '5', 'p': '1.5', 'GridSearch: metric': 'None', 'distribution': 'Bernoulli', 'GridSearch: step': '10', 'GridSearch: random_state': '1234'}, 'GS': {'splines_number': '5', 'threshold': '50', 'fullset': 'True'}}", "task_cnt": 2, "max_folds": 0, "metablueprint": ["NewMetablueprint", "5.0"], "vertex_cache_hits": 0, "finish_time": {"reps=1": 1390348267.260414}, "vertices": {"1": {"output_shape": [[157, 85]]}, "2": {"output_shape": [[157, 1]]}}, "time_real": [["1", "0.07955"]], "s": 0, "task_info": {"reps=1": [[{"fit max RAM": 0, "fit CPU pct": 0, "fit CPU time": 0.04000000000000001, "transform max RAM": 0, "fit clock time": 0.05651402473449707, "fit avg RAM": 0, "cached": false, "ytrans": null, "fit total RAM": 12288917504, "transform avg RAM": 0, "transform clock time": 0.02829289436340332, "version": "0.1", "arguments": null, "task_name": "GS", "transform CPU time": 0.01999999999999999, "transform total RAM": 12288917504, "transform CPU pct": 0}], [{"fit max RAM": 0, "fit CPU pct": 0, "fit CPU time": 0.03, "predict CPU pct": 0, "fit clock time": 0.12298583984375, "predict max RAM": 0, "fit avg RAM": 0, "cached": false, "ytrans": null, "fit total RAM": 12288917504, "predict CPU time": 0.010000000000000009, "predict total RAM": 12288917504, "predict clock time": 0.014081001281738281, "version": "0.1", "predict avg RAM": 0, "arguments": null, "task_name": "GLMB"}]]}, "time": {"finish_time": {"reps=1": 1390348267.260421}, "total_time": {"reps=1": 0.45728397369384766}, "start_time": {"reps=1": 1390348266.803139}}, "model_type": "GLM - Bernoulli", "vertex_cnt": 2, "_id": "52df07ea637aba6043709cd0", "blend": 0, "reference_model": false, "extras": ""},
        {"grid_scores": {}, "best_parameters": {}, "qid": "4", "task_version": {"SVMR": "0.1", "LS": "0.1"}, "roc": "", "part_size": [["1", 122, 35]], "max_reps": 1, "samplesize": 157.0, "ec2": {"reps=1": {"spot_price": 0, "on_demand_price": 0, "availability_zone": "None", "instance_size": "None", "CPU_count": 4, "instance_type": "local", "workers": 2}}, "uid": "52df07e3637aba07aa1b9c15", "dataset_id": "52df07e3637aba5fec709cd2", "partition_stats": {"(0, -1)": {"train_size": 122, "test_size": 35, "time_real": "0.03026"}}, "features": ["Linear Splines"], "resource_summary": {"reps=1": {"total_cpu_time": 0.12999999999999998, "total_noncached_clock_time": 0.16466403007507324, "cpu_usage": 0.7894863252206974, "bp_time": 0.30275893211364746, "cached": "false", "max_noncached_ram": 0, "bp_cost": 3.448087837960985e-05, "noncached_cost": 0, "rate": 0.013666666666666666, "cost": 1.8753403425216673e-05, "total_noncached_cpu_time": 0.12999999999999998, "total_clock_time": 0.16466403007507324, "max_ram": 0, "noncached_cpu_usage": 0}}, "training_dataset_id": "52df07e3637aba5fec709cd2", "icons": [0], "total_size": 157, "parts": [["1", 85]], "test": {"LogLoss": [8.03199], "AUC": [0.60985], "Gini": [0.07273], "Gini Norm": [0.21212], "labels": ["(0,-1)"], "metrics": ["LogLoss", "AUC", "Ians Metric", "Gini", "Gini Norm", "Rate@Top10%", "Rate@Top5%"], "Rate@Top5%": [0.33333], "Ians Metric": [0.12788], "Rate@Top10%": [0.25]}, "insights": "NA", "parts_label": ["partition", "NonZeroCoefficients"], "blueprint": {"1": [["NUM"], ["LS"], "T"], "2": [["1"], ["GLMB"], "P"]}, "hash": "f362964bcc1f8c5fcb5de4ab7a083233a64f77b9", "blueprint_id": "df1adb768de0bf5b440dda6bed63b234", "pid": "52df07e3637aba5fec709cd1", "originalName": null, "lift": "", "bp": 4, "task_parameters": "{'GLMB': {'GridSearch: stratified': 'True', 'GridSearch: algorithm': 'Tom', 'GridSearch: max_iterations': '15', 'GridSearch: test_fraction': 'None', 'GridSearch: CV_folds': '5', 'p': '1.5', 'GridSearch: metric': 'None', 'distribution': 'Bernoulli', 'GridSearch: step': '10', 'GridSearch: random_state': '1234'}, 'LS': {'splines_number': '5', 'threshold': '50', 'fullset': 'True'}}", "task_cnt": 2, "max_folds": 0, "metablueprint": ["NewMetablueprint", "5.0"], "vertex_cache_hits": 0, "finish_time": {"reps=1": 1390348267.884471}, "vertices": {"1": {"output_shape": [[157, 85]]}, "2": {"output_shape": [[157, 1]]}}, "time_real": [["1", "0.03026"]], "s": 0, "task_info": {"reps=1": [[{"fit max RAM": 0, "fit CPU pct": 0, "fit CPU time": 0.04000000000000001, "transform max RAM": 0, "fit clock time": 0.05914306640625, "fit avg RAM": 0, "cached": false, "ytrans": null, "fit total RAM": 12288917504, "transform avg RAM": 0, "transform clock time": 0.06054496765136719, "version": "0.1", "arguments": null, "task_name": "LS", "transform CPU time": 0.06, "transform total RAM": 12288917504, "transform CPU pct": 0}], [{"fit max RAM": 0, "fit CPU pct": 0, "fit CPU time": 0.02999999999999997, "predict CPU pct": 0, "fit clock time": 0.03933095932006836, "predict max RAM": 0, "fit avg RAM": 0, "cached": false, "ytrans": null, "fit total RAM": 12288917504, "predict CPU time": 0.0, "predict total RAM": 12288917504, "predict clock time": 0.005645036697387695, "version": "0.1", "predict avg RAM": 0, "arguments": null, "task_name": "GLMB"}]]}, "time": {"finish_time": {"reps=1": 1390348267.884478}, "total_time": {"reps=1": 0.30275893211364746}, "start_time": {"reps=1": 1390348267.58172}}, "model_type": "Support Vector Regressor (Radial Kernel)", "vertex_cnt": 2, "_id": "52df07eb637aba6068709cd0", "blend": 0, "reference_model": false, "extras": ""}
      ];

      beforeEach(inject(function($httpBackend, $controller, $rootScope, LbModel, ModelService, ProjectService){
        scope = $rootScope.$new();

        scope.safeApply = function(){};
        $controller('ModelsController', { $scope: scope});
        _LbModel = LbModel;
        _ModelService = ModelService;
        _ProjectService = ProjectService;
        mockBackend = $httpBackend;
      }));

      describe('Search', function(){
        it('Should work on model titles', function(){
          scope.modelService.modelsData = _.map(modelsServerResponse, function(m){ return new _LbModel(m); });
          scope.search('Bernoulli');
          expect(scope.filteredModels).not.toBeUndefined();
          expect(scope.filteredModels.length).toEqual(2);
        });

        it('Should work on model features', function(){
          scope.modelService.modelsData = _.map(modelsServerResponse, function(m){ return new _LbModel(m); });
          scope.search('Splines');
          expect(scope.filteredModels).not.toBeUndefined();
          expect(scope.filteredModels.length).toEqual(2);
        });

        it('Should be case insensitive', function(){
          scope.modelService.modelsData = _.map(modelsServerResponse, function(m){ return new _LbModel(m); });
          scope.search('linear splines');
          expect(scope.filteredModels).not.toBeUndefined();
          expect(scope.filteredModels.length).toEqual(1);
        });
      });

      describe('Remove models', function(){
        it('Should update the in-memory array and clear selected', function(){

          _ModelService.modelsData = [
            { lid: 1},
            { lid: 2},
            { lid: 3, modelSelected : true}
          ];

          _ProjectService.project = {pid: 1};
          mockBackend.expectPOST('/project/1/models/delete').respond(200,'');

          scope.removeModels(scope.getSelectedModels());

          expect(_ModelService.modelsData.length).toEqual(2);

          mockBackend.flush();

          expect(scope.getSelectedModels().length).toEqual(0);
        });

        afterEach(function(){
          mockBackend.verifyNoOutstandingExpectation();
          mockBackend.verifyNoOutstandingRequest();
        });

      });

      describe('Cancel models', function(){
        it('Should remove the the in-memory item if it is on its 1st run', function(){

          _ModelService.modelsData = [
            new _LbModel({ lid: '99'}),
            new _LbModel({ lid: '100'}),
          ];

          var socketData = {
            message : '{"lid": "100"}'
          };
          _ModelService.cancel(socketData);

          expect(_ModelService.modelsData.length).toEqual(1);
          expect(_ModelService.modelsData[0].lid).toEqual('99');
        });

        it('Should fetch the model if 5CV is in progress', function(){

          var pid = 123;
          _ModelService.pid = 123;
          var model = {
            lid: '100',
            test: {'metrics': ['AUC'], 'AUC': [0.80693] },
            partition_stats: {'(0, -1)': {'train_size': 1000, 'test_size': 1001, 'time_real': '0.02292'}, '(1, -1)': {'train_size': 1001, 'test_size': 1000, 'time_real': '0.02415'}}
          };
          _ModelService.modelsData = [
            new _LbModel({ lid: '99'}),
            new _LbModel(model)
          ];

          var lid = 100;
          var socketData = {
            message : '{"lid": "' + lid +'"}'
          };

          mockBackend.expectGET('/project/' + pid + '/models/' + lid)
          .respond(model);

          _ModelService.cancel(socketData);

          mockBackend.flush();

          expect(_ModelService.modelsData.length).toEqual(2);
        });
      });



// tests are very outdated with recent changes
/*
      describe('updateLeaderboardData', function(){

        it('Should add new elements on server fetch', function(){
          scope.models = [];
          scope.setDims = function(){ return false; };
          scope.currentMetric = 'Gini';
          var serverModels = _.cloneDeep(modelsServerResponse);
          var oLength = serverModels.length;

          scope.updateLeaderboardData(serverModels);

          expect(scope.models).not.toBeUndefined();

          expect(scope.models.length).toEqual(oLength);

          var newModels = [{'_id': '1'}, {'_id': '2'}];
          scope.updateLeaderboardData(newModels)

          var allModelsLength = serverModels.length + newModels.length;
          expect(scope.models.length).toEqual(allModelsLength);
        });

        it('Should update existing elements based on ID', function(){
          scope.currentMetric = 'Gini';
          scope.models = [];
          scope.setDims = function(){ return false; };
          scope.updateLeaderboardData(modelsServerResponse);

          var model1 = _.find(scope.models, { lid : '52df07e9637aba601a709cd0'});
          expect(model1).not.toBeUndefined();
          expect(model1.x).toBeUndefined();

          var updatedModel = [{'_id' : '52df07e9637aba601a709cd0', 'x': 'x'}];
          scope.updateLeaderboardData(updatedModel);
          expect(model1.x).not.toBeUndefined();
        });
      });
*/
    });
  }
);