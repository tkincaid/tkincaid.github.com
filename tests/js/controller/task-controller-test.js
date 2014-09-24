define(
  [
    'angular-mocks',
    'datarobot',
        'js/model/task.min',
        'js/controller/task-controller.min',
        'js/service/task-service.min',
        'js/service/project-service.min',
        'js/service/dataset-service.min',
        'js/model/share.min',
        'js/directive/chart/blueprint-directive.min'
  ],
  function() {
    describe('TaskController', function(){

      beforeEach(module('datarobot'));

      var $scope, taskObj, task, taskService, mockBackend;
      beforeEach(inject(function($controller, $rootScope, $q, TaskService, Task, $httpBackend) {
        $scope = $rootScope.$new();
        task = new Task({ id: 'a1b2c3', name: 'User model 1', email: 'zelda@chestnuthill.com' });
        taskService = TaskService;
        $controller('TaskController', { $scope: $scope });
        mockBackend = $httpBackend;
      }));

      describe('$scope.setSelectedTask(task)', function() {
        it("should set a given task as selected if it isn't selected and add it to the array of selected tasks",
          function() {
            $scope.setSelectedTask(task);
            expect(task.selected).toBe(true);
            expect(taskService.selectedTasks.indexOf(task)).toNotBe(-1);
          }
        );
      });

      describe('$scope.shareTask(task)', function() {
        it("should create a share and add it to the task",
          function() {
            $scope.role = 'Admin';
            taskService.tasks.push(task);

            mockBackend
              .whenGET( '/listprojects')
              .respond([]);
            mockBackend
              .whenGET( '/task')
              .respond([]);
            mockBackend
              .expectGET('/account/profile')
              .respond({ username: 'took@peregrin.com' });

            mockBackend
              .expectPOST('/task/' + task.id + '/share')
              .respond({ error: 0 });

            $scope.shareTask(taskService.tasks[0]).then(function(sharedTask) {
              expect(typeof sharedTask.email).toBe('undefined');
              expect(sharedTask.shares.length).toBe(1);
              expect(sharedTask.id).toBe(task.id);
            });

            mockBackend.flush();
          }
        );
      });

    });
  }
);