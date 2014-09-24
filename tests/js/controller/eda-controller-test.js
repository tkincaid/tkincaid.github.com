define(
  [
    'angular-mocks',
    'datarobot',
    'lodash',
    'js/model/data-column.min',
    'DatasetServiceResponse',
    'js/controller/eda-controller.min'
  ],
  function(angularMocks, datarobot, _, DataColumn, DatasetServiceResponse){

    describe('EdaController', function(){

      beforeEach(module('datarobot'));

      // This is needed since the controller does not store this data in the scope, only in the service
      var scope, dsService, projectService;

      var serverResponse = DatasetServiceResponse.fetch();

      beforeEach(inject(function($controller, $rootScope, DatasetService, ProjectService){
        scope = $rootScope.$new();
        $controller('EdaController', { $scope: scope});
        dsService = DatasetService;
        projectService = ProjectService;
      }));


      describe('Sort', function(){
        it('Should sort high info first (low info bottom) -> raw index -> transform id', function(){
          projectService.project = {'target':{ 'name':'IsBadBuy'} };
          scope.filteredFeatures = _.map(serverResponse, function(data){return new DataColumn(data);});
          scope.orderFeatures(scope.filteredFeatures);

          var expectedOrder = ['IsBadBuy', 'Make', 'WarrantyCost', 'PRIMEUNIT'];
          for (var i = 0; i < expectedOrder.length; i++) {
            expect(scope.filteredFeatures[i].name).toEqual(expectedOrder[i]);
          }
        });

        it('No info columns should be sorted by info level (higher first)', function(){
          var customData = [
            {'name': '0', 'raw_variable_index': 75, 'transform_id': 0, 'profile' : {'info': 0.9}},
            {'name': '1', 'raw_variable_index': 3, 'transform_id': 0},
            {'name': '2', 'raw_variable_index': 3, 'transform_id': 1},
            {'name': '3', 'raw_variable_index': 3, 'transform_id': 2}
          ];
          projectService.project = {'target':{ 'name':'IsBadBuy'} };
          scope.filteredFeatures = _.map(customData, function(data){return new DataColumn(data);});
          scope.orderFeatures(scope.filteredFeatures);

          for (var i = 0; i < customData.length; i++) {
            expect(scope.filteredFeatures[i].name).toEqual(i.toString());
          }
        });

        it('Sort by info level including nested columns', function(){
          var customData = [
            {'name': '0', 'raw_variable_index': 1, 'transform_id': 0, 'profile' : {'info': 0.9}},
            {'name': '1', 'raw_variable_index': 100, 'transform_id': 0, 'profile' : {'info': 0.5}},
            {'name': '2', 'raw_variable_index': 50, 'transform_id': 0, 'profile' : {'info': 0.1}},
            {'name': '3', 'raw_variable_index': 2, 'transform_id': 0, 'profile' : {'info': 0.05}},
            {'name': '4', 'raw_variable_index': 2, 'transform_id': 2, 'profile' : {'info': 0.06}},
            {'name': '5', 'raw_variable_index': 2, 'transform_id': 1, 'profile' : {'info': 0.04}},
            {'name': '6', 'raw_variable_index': 3, 'transform_id': 0, 'profile' : {'info': 0.03}},
            {'name': '7', 'raw_variable_index': 3, 'transform_id': 1, 'profile' : {'info': 0.8}},
            {'name': '8', 'raw_variable_index': 3, 'transform_id': 2, 'profile' : {'info': 0.7}},
            {'name': '9', 'raw_variable_index': 75, 'transform_id': 0}

          ];
          projectService.project = {'target':{ 'name':'IsBadBuy'} };
          scope.filteredFeatures = _.map(customData, function(data){return new DataColumn(data);});
          scope.orderFeatures(scope.filteredFeatures);


          for (var i = 0; i < customData.length; i++) {
            expect(scope.filteredFeatures[i].name).toEqual(i.toString());
          }
        });

        it('Transformations are sorted below their parent', function(){
          var customData = [
            {'name': '0', 'raw_variable_index': 1, 'transform_id': 0, 'profile' : {'info': 0.6}},
            {'name': '1', 'raw_variable_index': 1, 'transform_id': 1, 'profile' : {'info': 0.7}},
            {'name': '2', 'raw_variable_index': 2, 'transform_id': 0, 'profile' : {'info': 0.5}},
            {'name': '3', 'raw_variable_index': 2, 'transform_id': 1, 'profile' : {'info': 0.1}}
          ];
          projectService.project = {'target':{ 'name':'IsBadBuy'} };
          scope.filteredFeatures = _.map(customData, function(data){return new DataColumn(data);});
          scope.orderFeatures(scope.filteredFeatures);

          for (var i = 0; i < customData.length; i++) {
            expect(scope.filteredFeatures[i].name).toEqual(i.toString());
          }
        });

        it('Text features are above low info variables', function(){
          var customData = [
            {'name': '0', 'raw_variable_index': 0, 'transform_id': 0, 'profile' : {'info': 0.7}},
            {'name': '1', 'raw_variable_index': 1, 'transform_id': 0, 'types': {'text':true}},
            {'name': '2', 'raw_variable_index': 2, 'transform_id': 0, 'profile' : {}},
            {'name': '3', 'raw_variable_index': 3, 'transform_id': 0}
          ];
          projectService.project = {'target':{ 'name':'IsBadBuy'} };
          scope.filteredFeatures = _.map(customData, function(data){return new DataColumn(data);});
          scope.orderFeatures(scope.filteredFeatures);

          for (var i = 0; i < customData.length; i++) {
            expect(scope.filteredFeatures[i].name).toEqual(i.toString());
          }
        });
      });

      describe('Permissions Validation', function(){
        it('should be false if project or permissions are not defined', function(){
          projectService.project = null;
          expect(scope.getPermissions('DOES_NOT_MATTER')).toBeFalsy();
        });

        it('should be false if project or permissions are not defined', function(){
          projectService.project = null;
          expect(scope.getPermissions('DOES_NOT_MATTER')).toBeFalsy();
        });

        it('should disable feature to save new feature lists if user does not have permissions', function(){
          projectService.project = null;
          expect(scope.isSaveNewFeatureDisabled()).toBeTruthy();
        });
      });


      describe('Feature lists', function(){
        it('should disable feature to save new feature lists if no features are selected', function(){

          scope.selectedFeatures = 0;

          projectService.project = {
            permissions:{
              CAN_EDIT_FEATURE_LISTS : false
            }
          };
          expect(scope.isSaveNewFeatureDisabled()).toBeTruthy();
        });

        it('should enable feature to save new feature lists if user has permissions and features are selected', function(){

          scope.selectedFeatures = 2;

          projectService.project = {
            permissions:{
              CAN_EDIT_FEATURE_LISTS : true
            }
          };
          expect(scope.isSaveNewFeatureDisabled()).toBeFalsy();
        });
      });
    });
});
