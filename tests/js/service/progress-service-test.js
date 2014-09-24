define(
  [
    'angular-mocks',
    'datarobot',
    'js/service/progress-service.min'
  ],
  function() {
    describe('ProgressService', function() {

      beforeEach(module('datarobot'));

      var progressService, mockBackend;
      beforeEach(inject(function (ProgressService, $httpBackend) {
        progressService = ProgressService;
        mockBackend = $httpBackend;
      }));

      describe('.subscribe(socketEvent, callback)', function() {
        it('should add a callback to a list of callbacks to be executed when the ' +
           'event is triggered if the callback is not already there', function() {
          var callbackOne = function() {
            console.log("Callback one executed");
          };

          var callbackOneDuplicate = function() {
            console.log("Callback one executed");
          };

          var callbackTwo = function() {
            console.log("Callback two executed");
          };

          expect(progressService.getSubscribers(progressService.socketEvents.ProjectSync)).toBeUndefined();

          progressService.subscribe(progressService.socketEvents.ProjectSync, callbackOne);

          expect(progressService.getSubscribers(progressService.socketEvents.ProjectSync).length).toEqual(1);

          progressService.subscribe(progressService.socketEvents.ProjectSync, callbackOneDuplicate);

          expect(progressService.getSubscribers(progressService.socketEvents.ProjectSync).length).toEqual(1);

          progressService.subscribe(progressService.socketEvents.ProjectSync, callbackTwo);

          expect(progressService.getSubscribers(progressService.socketEvents.ProjectSync).length).toEqual(2);
        });
      });

      describe('Publish ', function() {
        it('should call all subscribers once', function() {

          var callback1ExecutionCount = 0;
          var callback2ExecutionCount = 0;

          var callback1 = function callback1(increment) {
            callback1ExecutionCount  += increment;
          };

          var callback2 = function callback2(increment) {
            callback2ExecutionCount  += increment;
          };

          for (var i = 0; i < 3; i++) {
            progressService.subscribe(progressService.socketEvents.ProjectSync, callback1);
            progressService.subscribe(progressService.socketEvents.ProjectSync, callback2);
          }

          expect(progressService.getSubscribers(progressService.socketEvents.ProjectSync).length).toEqual(2);

          progressService.publish(progressService.socketEvents.ProjectSync, 1);
          expect(callback1ExecutionCount).toEqual(1);
          expect(callback2ExecutionCount).toEqual(1);
        });
      });
    });
  }
);