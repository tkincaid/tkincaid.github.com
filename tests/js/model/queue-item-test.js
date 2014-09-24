define(
  [
    'angular-mocks',
    'datarobot',
    'js/model/queue-item.min'
  ],
  function(angularMocks, datarobot, QueueItem ) {
    describe('QueueItem', function() {

      beforeEach(module('datarobot'));

      describe('Status methods', function() {
        it('should isProgress return true when status equals isprogress', function() {

          item = {status:'inprogress'};
          qItem = new QueueItem(item);

          expect(item.isInProgress()).toBe(true);
          expect(item.isPending()).toBe(false);
          expect(item.hasError()).toBe(false);
        });

        it('should isPending return true when status equals queue', function() {
          item = {status:'queue'};
          qItem = new QueueItem(item);

          expect(item.isInProgress()).toBe(false);
          expect(item.isPending()).toBe(true);
          expect(item.hasError()).toBe(false);
        });

        it('should hasError return true when status equals error', function() {
          item = {status:'error'};
          qItem = new QueueItem(item);

          expect(item.isInProgress()).toBe(false);
          expect(item.isPending()).toBe(false);
          expect(item.hasError()).toBe(true);
        });
      });
    });
  }
);