define(
  [
    'angular-mocks',
    'datarobot',
    'lodash',
    'js/model/queue-item.min',
    'js/controller/queue-controller.min'
  ],
  function(angularMocks, datarobot, _, QueueItem){

    describe('QueueController', function(){

      beforeEach(module('datarobot'));

      var scope;
      beforeEach(inject(function($controller, $rootScope, $q){
        scope = $rootScope.$new();
        $controller('QueueController', { $scope: scope});
      }));
    });
  }
);
