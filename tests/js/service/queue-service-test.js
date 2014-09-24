define(
  [
    'angular-mocks',
    'datarobot',
    'lodash',
    'js/model/queue-item.min',
    'js/service/queue-service.min',
  ],
  function(angularMocks, datarobot, _, QueueItem){

    describe('QueueService', function(){

      beforeEach(module('datarobot'));

      var _QueueService;
      var queueServerResponse = [
        {"status": "settings", "qid": -1, "workers": 2, "mode": 1},
        {"qid": 1, "blueprint": {"1": [["NUM"], ["NI"], "T"], "2": [["1"], ["GLMB"], "P"]}, "lid": "new", "features": ["Missing Values Imputed"], "blueprint_id": "d4c06a5c23cf1d917019720bceba32c8", "icons": [0], "pid": "52d96134637aba3612827383", "max_reps": 1, "status": "queue", "samplesize": 1000, "bp": 1, "model_type": "GLM - Bernoulli", "max_folds": 0, "dataset_id": "52d96134637aba3612827384", "reference_model": false},
        {"qid": 2, "blueprint": {"1": [["NUM"], ["NI"], "T"], "2": [["1"], ["RFC t_a=2;t_n=1;t_f=0.15;e=0;ls=[1, 3, 5, 10];mf=[0.2, 0.3, 0.4, 0.5]"], "P"]}, "lid": "new", "features": ["Missing Values Imputed"], "blueprint_id": "473a851fbb96f8f8301eb70e016ae3c9", "icons": [1], "pid": "52d96134637aba3612827383", "max_reps": 1, "status": "queue", "samplesize": 1000, "bp": 2, "model_type": "Random Forest (scikit-learn)", "max_folds": 0, "dataset_id": "52d96134637aba3612827384", "reference_model": false},
        {"qid": 3, "blueprint": {"1": [["NUM"], ["GS"], "T"], "2": [["1"], ["GLMB"], "P"]}, "lid": "new", "features": ["Constant Splines"], "blueprint_id": "10b9623bb54915ee8240a2ad18d0a727", "icons": [0], "pid": "52d96134637aba3612827383", "max_reps": 1, "status": "queue", "samplesize": 1000, "bp": 3, "model_type": "GLM - Bernoulli", "max_folds": 0, "dataset_id": "52d96134637aba3612827384", "reference_model": false}
      ];


      beforeEach(inject(function(QueueService, $rootScope, $q){
        _QueueService = QueueService;
      }));

      describe('Queue status', function(){

        it('should determine whether there are in-progress items ', function(){

          _QueueService.queue = [
            new QueueItem({status:'other'}),
            new QueueItem({status:'inprogress'}),
            new QueueItem({status:'inprogress'})
          ];
          expect(_QueueService.getProgressQueue().length).toEqual(2);

          _QueueService.queue.pop();
          _QueueService.queue.pop();

          expect(_QueueService.getProgressQueue().length).toEqual(0);
        });

        it('should determine whether there are queued items ', function(){

          _QueueService.queue = [
            new QueueItem({status:'queue'}),
            new QueueItem({status:'other'})
          ];
          expect(_QueueService.getPendingQueue().length).toBe(1);

          _QueueService.queue = [
            new QueueItem({status:'other'})
          ];
          expect(_QueueService.getPendingQueue().length).toBe(0);
        });

        it('should determine whether there are items with error', function(){

          _QueueService.queue = [
            new QueueItem({status:'other'}),
            new QueueItem({status:'error'}),
            new QueueItem({status:'other'})
          ];
          expect(_QueueService.getErrorsQueue().length).toBe(1);

          _QueueService.queue = [
            new QueueItem({status:'other'})
          ];
          expect(_QueueService.getErrorsQueue().length).toBe(0);
        });

        it('should filter methods be idempotent', function(){

          _QueueService.queue = [
            new QueueItem({status:'other'}),
            new QueueItem({status:'inprogress'}),
            new QueueItem({status:'error'}),
            new QueueItem({status:'queue'}),
            new QueueItem({status:'inprogress'})
          ];

          expect(_QueueService.getErrorsQueue()).toEqual(_QueueService.getErrorsQueue());
          expect(_QueueService.getProgressQueue()).toEqual(_QueueService.getProgressQueue());
          expect(_QueueService.getPendingQueue()).toEqual(_QueueService.getPendingQueue());
        });

        it('should determine whether there are more workers available: in-progress < worker settings and pending queued items > 0', function(){

          _QueueService.autopilotSettings = {workers : 1} ;
          _QueueService.queue = [
            new QueueItem({status:'queue'}),
            new QueueItem({status:'error'})
          ];
          expect(_QueueService.hasWorkersAvailable()).toBe(true);

          _QueueService.autopilotSettings = {workers : 1} ;
          _QueueService.queue = [
            new QueueItem({status:'queue'}),
            new QueueItem({status:'inprogress'})
          ];
          expect(_QueueService.hasWorkersAvailable()).toBe(false);
        });

        it('should determine whether the queue is empty', function(){

          _QueueService.queue = null;
          expect(_QueueService.isQueueEmpty()).toBe(true);

          _QueueService.queue = [];
          expect(_QueueService.isQueueEmpty()).toBe(true);

          _QueueService.queue = [
            new QueueItem({status:'error'})
          ];
          expect(_QueueService.isQueueEmpty()).toBe(true);

          _QueueService.queue = [
            new QueueItem({status:'inprogress'})
          ];
          expect(_QueueService.isQueueEmpty()).toBe(false);

        });
      });
      describe('Process server response', function(){

        // No longer piggy backs

        // it('should pop worker settings from queue (server piggybacks these settings in the queue)', function(){

        //   var queue = _.cloneDeep(queueServerResponse);
        //   var oLength = queue.length;

        //   //worker settings was set
        //   var autopilotSettings = scope.popautopilotSettings(queue);
        //   expect(autopilotSettings).not.toBeUndefined();
        //   expect(autopilotSettings.status).toEqual('settings');

        //   // It was popped from the queue
        //   expect(queue.indexOf(autopilotSettings)).toEqual(-1);
        //   expect(queue.length).toEqual(oLength - 1);

        // });

        it('Should update the in-memory copy based on server fetch', function(){

          _QueueService.queue = [];
          var queue = _.cloneDeep(queueServerResponse);
          var oLength = queue.length;

          _QueueService.updateQueue(queue);

          expect(_QueueService.queue).not.toBeUndefined();
          expect(_QueueService.queue.length).toEqual(oLength - 1);

        });

        it('Should add individual items', function(){
          _QueueService.queue = [];
          var queueItem = {'qid': 4, 'blueprint': {}, 'lid': 'new'};
          _QueueService.updateQueue(queueItem);

          expect(_QueueService.queue.length).toEqual(1);
        });

        it('Should add multiple items', function(){
          _QueueService.queue = [];
          var queueItems = [
            {'qid': 4, 'blueprint': {}, 'lid': 'new'},
            {'qid': 5, 'blueprint': {}, 'lid': 'new'}
          ];
          _QueueService.updateQueue(queueItems);

          expect(_QueueService.queue.length).toEqual(2);
        });

        it('Should update individual items', function(){
          _QueueService.queue = [];
          var queueItem = {'qid': 1, 'blueprint': {}, 'lid': 'new'};
          _QueueService.updateQueue(queueItem);

          queueItem = {'qid': 1, 'blueprint': {}, 'lid': 'TEST'};
          _QueueService.updateQueue(queueItem);

          expect(_QueueService.queue.length).toEqual(1);
          expect(_QueueService.queue[0].lid).toEqual('TEST');
        });

        it('Should remove completed items', function(){
          _QueueService.queue = [{'qid': 1, 'blueprint': {}, 'lid': 'new'}];

          var serverData = [{'qid': 2, 'blueprint': {}, 'lid': 'new'}];
          _QueueService.clearCompleted(serverData);

          expect(_QueueService.queue.length).toEqual(0);

        });
      });

      describe('The queue', function(){
        it('Should remove an item from the queue by id', function(){
          _QueueService.queue = [{'qid': 1, 'blueprint': {}, 'lid': 'new'}];
          _QueueService.removeItem(1);
          expect(_QueueService.queue.length).toEqual(0);
        });

        it('Should set in progress status', function(){
          var qItem = new QueueItem({'qid': 1, 'blueprint': {}, 'lid': 'new'});
          _QueueService.queue = [qItem];

          _QueueService.setInProgress('1');

          expect(_QueueService.queue.length).toEqual(1);
          expect(_QueueService.queue[0].status).toEqual('inprogress');
        });

        it('Should set in error status', function(){
          var qItem = new QueueItem({'qid': 1, 'blueprint': {}, 'lid': 'new'});
          _QueueService.queue = [qItem];

          _QueueService.setError('1');

          expect(_QueueService.queue.length).toEqual(1);
          expect(_QueueService.queue[0].status).toEqual('error');
        });

        it('Should default to "pending" state', function(){
          var qItem = new QueueItem({'qid': 1, 'blueprint': {}, 'lid': 'new'});
          expect(qItem.isPending()).toBe(true);
        });
      });
    });
  }
);
