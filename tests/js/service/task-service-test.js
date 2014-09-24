define(
  [
    'js/model/task.min',
    'angular-mocks',
    'datarobot',
    'js/service/task-service.min',
    'js/service/user-service.min',
    'js/service/project-service.min',
    'js/model/project.min'
  ],
  function() {
    describe('TaskService', function() {

      beforeEach(module('datarobot'));

      var userServiceMock = { username: 'username' };

      module(function($provide) {
        $provide.value('UserService', userServiceMock);
      });

      var taskService, mockBackend, task, projectService, project;
      beforeEach(inject(function (TaskService, $httpBackend, Task, ProjectService, Project) {
        taskService = TaskService;
        mockBackend = $httpBackend;
        task = Task;
        projectService = ProjectService;
        project = Project;
      }));

      describe('.get()', function() {
        it('should return saved tasks if they are available', function() {
          taskService.tasks = [
            { name: 'taco' },
            { name: 'burrito' }
          ];

          taskService.get().then(function(response) {
            expect(response).toBe(taskService.tasks);
          });
        });
      });

      describe('.get(true)', function() {
        it('should fetch tasks even if saved tasks are available', function() {
          taskService.tasks = [
            { name: 'taco' },
            { name: 'burrito' }
          ];

          var fakeResponse = [
            { name: 'salad' }
          ];

          projectService.project = new project({ filename: 'training-sample.csv' });

          var tasks = [
            new task({ name: 'salad' })
          ];

          mockBackend
            .expectGET('/task')
            .respond(fakeResponse);

          taskService.get(true).then(function(response) {
            expect(response[0].name).toEqual(tasks[0].name);
          });

          mockBackend.flush();
        });
      });

    });
  }
);