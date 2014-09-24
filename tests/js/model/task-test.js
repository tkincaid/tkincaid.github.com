define(
  [
    'js/model/task.min',
    'angular-mocks',
    'datarobot'
  ],
  function() {
    describe('Task', function() {

      beforeEach(module('datarobot'));

      var task, mockBackend;
      beforeEach(inject(function (Task, $httpBackend) {
        task = new Task({ id: '1a2b3c', description: 'Boosted trees' });
        mockBackend = $httpBackend;
      }));

      describe('.getShares()', function() {
        it('should share information if its available', function() {
          task.shares = [
            { username: 'Big Data' },
            { username: 'Real-time Data' }
          ];

          task.getShares().then(function(response) {
            expect(response).toBe(task.shares);
          });
        });
      });

      describe('.getShares(true)', function() {
        it('should fetch task information even if task information is already available', function() {
          task.shares = [
            { username: 'Big Data' },
            { username: 'Real-time Data' }
          ];

          var fakeResponse = [
            { username: 'chilifresh11' }
          ];

          mockBackend
            .expectGET('/task/' + task.id + '/share')
            .respond(fakeResponse);

          task.getShares(true).then(function(response) {
            expect(response[0].username).toEqual(task.shares[0].username);
          });

          mockBackend.flush();
        });
      });

      describe('.remove()', function() {
        it('should delete the task', function() {
          mockBackend
            .expectDELETE('/task/' + task.id)
            .respond('ok');

          task.remove().then(function(response) {
            expect(response.data).toEqual('ok');
          });

          mockBackend.flush();
        });
      });

      describe('.update()', function() {
        it('should update the task information', function() {
          task.description = 'Gradient boosted trees';

          mockBackend
            .expectPUT('/task/' + task.id, { description: task.description })
            .respond('ok');

          task.update().then(function(response) {
            expect(response).toEqual('ok');
          });

          mockBackend.flush();
        });
      });

    });
  }
);