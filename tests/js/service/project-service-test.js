define(
  [
    'js/model/user.min',
    'lodash',
    'angular-mocks',
    'datarobot',
    'js/model/project.min',
    'js/service/project-service.min',
    'js/service/user-service.min'
  ],
  function(User, _) {
    describe('ProjectService', function() {

      beforeEach(module('datarobot'));

      var projectService, userService, mockBackend, project, _$timeout, cookies;

      beforeEach(inject(function (Project, ProjectService, UserService, $httpBackend, $timeout, $cookies) {
        project = Project;
        cookies = $cookies;
        projectService = ProjectService;
        userService = UserService;
        mockBackend = $httpBackend;
        _$timeout = $timeout;

        var projects = [
          new project({ filename: 'training-sample.csv', _id: 1, created: 1388984490})
        ];
        projectService.projects = projects;
        projectService.project = projects[0];

      }));

      afterEach(function(){
        mockBackend.verifyNoOutstandingExpectation();
        mockBackend.verifyNoOutstandingRequest();
      });

      it('should create a new project', function() {
        var serverResponse = {
          pid : 1,
          userid : 2,
          token : 'TOKEN'
        };

        mockBackend
          .expectPOST('/project')
          .respond(serverResponse);

        projectService.create().then(function(project) {
          expect(project).not.toBeUndefined();
          expect(project.pid).toEqual(serverResponse.pid);
          expect(project.session).not.toBeUndefined();
          // The server responds wtih sesion data (token), which is then
          // saved under th session property
          expect(project.session).toEqual(serverResponse);
        });

        mockBackend.flush();
      });

      it('should return cached projects if they are available', function() {
        projectService.project = new project({ filename: 'kickcars-training-sample.csv' });

        projectService.get().then(function(response) {
          expect(response).toBe(projectService.project);
        });
      });

      it('should update cached projects', function() {
        projectService.projects = [
          new project({ _id : 1, filename: 'kickcars-training-sample.csv' }),
          new project({ _id : 2, filename: 'my-cool-project.csv' })
        ];

        var serverMsg = { _id : 2, filename: 'project2' };

        projectService.update(serverMsg);
        expect(projectService.projects.length).toEqual(2);

        var project2 = _.find(projectService.projects, { pid : 2 });
        expect(project2.filename).toEqual('project2');

      });


      it('should get a session token from the server', function() {
        var pid = 1;
        var sessionResponse = {
          userid : 2,
          pid : 1,
          token : 'TOKEN',
          rev: 'edce33e'
        };

        mockBackend
          .expectGET('/project/'+ pid +'/session')
          .respond(sessionResponse);

        projectService.project = new project({_id:1});

        projectService.getSession(pid).then(function(session) {
          expect(session).toEqual(sessionResponse);
          expect(projectService.project.session).toEqual(sessionResponse);
        });

        mockBackend.flush();
      });

      it('should ignore the cache and fetch projects', function() {

        // someone just shared a project with us, we need to refresh
        // now we should have 2
        var serverResponse = [
          { filename: 'training-sample.csv', _id: 1, created: 1388984490},
          { filename: 'kickcars-200.csv', _id: 2, created: 1388984400 }
        ];

        userService.user =  new User({ username: 'ian' });

        mockBackend
          .expectGET('/listprojects')
          .respond(serverResponse);

        // projectService.project exists as 1, we pass true to refresh

        projectService.get(true).then(function(response) {
          // make sure that projectService.projects.length = 2 and not 1
          expect(projectService.projects.length).toEqual(2);
        });

        mockBackend.flush();
      });

      describe('Switching projects', function(){
        it('should switch to the last used project', function(){
          cookies.last_active = null; //Does not exist

          projectService.project = null;

          projectService.projects = [
            new project({ _id: 1 }),
            new project({ _id: 2 })
          ];

          projectService.switchActive();

          expect(projectService.project.pid).toEqual(2);
          expect(cookies.last_active).toEqual(projectService.project.pid);
        });


        it('should switch to the project saved in the cookies', function(){
          cookies.last_active = 1;

          projectService.projects = [
            new project({ _id: 1 }),
            new project({ _id: 2 })
          ];

          projectService.switchActive();

          expect(projectService.project.pid).toEqual(1);
          expect(cookies.last_active).toEqual(projectService.project.pid);
        });

        it('should switch to the project matching the passed-in pid', function(){
          cookies.last_active = 2;

          projectService.projects = [
            new project({ _id: 1 }),
            new project({ _id: 2 })
          ];

          projectService.switchActive();

          expect(projectService.project.pid).toEqual(2);
          expect(cookies.last_active).toEqual(projectService.project.pid);
        });

        it('should switch to the last used project when the cookie or passed-in pid is invalid', function(){
          //Invalid cookie

          cookies.last_active = 10;

          projectService.projects = [
            new project({ _id: 1 }),
            new project({ _id: 2 })
          ];

          projectService.switchActive();

          expect(projectService.project.pid).toEqual(2);
          expect(cookies.last_active).toEqual(projectService.project.pid);

          //Invalid passed-in pid

          cookies.last_active = null;

          projectService.projects = [
            new project({ _id: 1 }),
            new project({ _id: 2 })
          ];

          projectService.switchActive(100);

          expect(projectService.project.pid).toEqual(2);
          expect(cookies.last_active).toEqual(projectService.project.pid);
        });
      });

      describe('Deleting a project', function(){
        it('should switch to the next project when deleting a project', function(){
          spyOn(projectService, 'clear');
          mockBackend
            .expectDELETE('/project/1')
            .respond({});

          projectService.delete(1);

          mockBackend.flush();

          expect(projectService.clear).toHaveBeenCalled();
        });

        it('should switch to the next project when deleting a project', function(){
          var projects = [
            new project({ filename: 'test-1.csv', _id: 1, created: 1388984490}),
            new project({ filename: 'test-2.csv', _id: 2, created: 1388984490})
          ];

          projectService.projects = projects;
          projectService.switchActive(1);
          expect(projectService.project.pid).toEqual(1);

          projectService.clear(1);

          expect(projectService.projects.length).toEqual(1);
          expect(projectService.project.pid).toEqual(2);
        });

        it('should clear the last_active cookie', function(){
          var projects = [
            new project({ filename: 'test-1.csv', _id: 1, created: 1388984490})
          ];

          projectService.projects = projects;
          projectService.switchActive(1);
          projectService.clear(1);

          expect(cookies.last_active).toBe(null);
        });
      });
    });
  }
);