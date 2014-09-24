var QueueTests = function(){

  var self = this;

  this.autopilot = function(){
    var self = this;

    casper.test.begin('Queue: Pause Autopilot', 3, function(test) {
      casper.then(function(){
        queue.toggleQueuePaused();
      });

      casper.waitForSelector(queue.pausedIndicator, function(){
        test.assert(queue.isPaused(), 'Queue Paused');
      }, function timeout(){
        captureAndWarn('autopilot-no-pause.png', 'Autopilot did show the pause icon');
      });

      casper.then(function(){
        var itemsInProgress = queue.inProgressCount(),
          // itemsInQueue    = queue.inQueuedCount(),
          msg = 'Waiting for '+itemsInProgress+' Models to Finish';

        casper.echo(msg);
        casper.waitWhileVisible(queue.inProcess,function(){
          test.assertNot(casper.exists(queue.inProcess), 'Nothing is in-progress');

        },function timeout(){

          msg = 'Timeout ' + msg;
          capture('q-waiting-for-in-progress.png');
          test.fail(msg);

          // casper.echo('Test has timed out waiting for models to finish');
          // casper.echo('Assert if any models been added to inprogress that would indicate that pause has failed?');

          // var itemsAdded    = false,
          //     itemsQueued   = false,
          //     nowInProgress = queue.inProgressCount(),
          //     nowInQueue    = queue.inQueuedCount();

          // casper.then(function(){
          //   if (nowInProgress >= itemsInProgress){
          //     itemsAdded = true;
          //   }
          //   if (nowInQueue === itemsInQueue){
          //     itemsQueued = true;
          //   }
          // });

          // casper.then(function(){
          //   test.assert(itemsAdded,  'Current In-Progress Queue count ('+nowInProgress+') is less than the count ('+itemsInProgress+') at the start of the test.');
          //   test.assert(itemsQueued, 'Current Queue count ('+nowInQueue+') is equal to the count ('+itemsInQueue+') at the start of the test.');
          // });

        }, 120000);
      });

      casper.then(function(){
        test.assert(casper.exists(queue.queuedItems), 'Queued items are present');
      });

      casper.run(function() {
        test.done();
      });
    });

    casper.test.begin('Queue: Resuming Autopilot', 1, function(test) {
      casper.echo('Resuming '+queue.inQueuedCount()+' Models');
      casper.then(function(){
        queue.toggleQueuePaused();
      });
      casper.then(function(){
        casper.waitUntilVisible(queue.inProcess,function(){
          test.assert(casper.exists(queue.inProcess), queue.inProgressCount()+' Models are in-progress');
        }, function timeout(){
          var msg = 'pause';
          capture('queue-should-be-resumed.png');
          test.fail(msg);
        });
      });
      casper.run(function() {
        test.done();
      });
    });

    casper.test.begin('Queue: Clear Queued Items', 1, function(test) {
      casper.then(function (){
        queue.removeAll();
      });

      casper.then(function(){
        test.assertDoesntExist(queue.queuedItems, 'Nothing is queued');
      });

      casper.run(function() {
        test.done();
      });
    });

    casper.test.begin('Queue: Check For "Continue Autopilot" button', 1 , function(test) {

      var mode = queue.getAutoPilotMode();

      casper.then(function (){
        if (mode === queue.MODE.SEMI){
          casper.waitForSelector(queue.continueAutopilotButton, function(){
            test.assert( casper.visible(queue.continueAutopilotButton), '"Continue Autopilot" button should be visible');
          }, function timeout(){
            var msg = '"Continue Autopilot" button should exist... But it is NOT THERE';
            capture('no-autopilot.png', msg);
            test.fail(msg);
          });
        } else {
          casper.waitWhileVisible(queue.continueAutopilotButton, function(){
            test.assertNotVisible(queue.continueAutopilotButton, '"Continue Autopilot" button should not be visible');
          });
        }
      });

      casper.run(function() {
        test.done();
      });
    });
  };

  casper.test.begin('Queue: Initialization', 1, function(test) {
    var queuedMsg = 'The queue has queued items';
    casper.waitForSelector(queue.queuedItems, function(){
      test.assert(casper.exists(queue.queuedItems), queuedMsg);
    }, function timeout(){
      capture('queue-queued-items.png');
      test.fail(queuedMsg);
    });
    casper.run(function() { test.done(); });
  });

  casper.test.begin('Queue: Mode Settings', 7, function(test) {

    var currentMode;

    casper.then(function(){
      casper.click(queue.openAutopilotOptions);
      casper.waitForSelector(queue.autopilotOptionsMenu, function(){
        test.assertVisible(queue.autopilotOptionsMenu,'Autopilot options menu is open');
      });
    });

    casper.then(function(){
      casper.click(queue.autopilotOptionsMode);
      casper.waitForSelector(queue.autopilotOptionsModeMenu, function(){
        test.assertVisible(queue.autopilotOptionsModeMenu,'Autopilot mode menu is open');
      });
    });


    casper.then(function(){
      queue.setAutoPilotMode(1);
    });

    casper.wait(500, function() {
      currentMode = queue.getAutoPilotMode();
      test.assert(currentMode === queue.MODE.SEMI, 'AutoPilot mode is set to "Semi"');
    });

    casper.then(function(){
      queue.setAutoPilotMode(2);
    });

    casper.wait(500, function() {
      currentMode = queue.getAutoPilotMode();
      test.assert(currentMode === queue.MODE.OFF, 'AutoPilot mode is set to "Manual"');
    });

    casper.then(function(){
      queue.setAutoPilotMode(0);
    });

    casper.wait(500, function() {
      currentMode = queue.getAutoPilotMode();
      test.assert(currentMode === queue.MODE.FULL, 'AutoPilot mode is set to "Auto"');
    });

    casper.then(function(){
      queue.setAutoPilotMode(1);
      casper.wait(500, function() {
        currentMode = queue.getAutoPilotMode();
        test.assert(currentMode === queue.MODE.SEMI, 'AutoPilot mode is set back to "Semi"');
      });
    });

    casper.then(function(){
      casper.click(queue.openAutopilotOptions);
      casper.waitWhileSelector(queue.autopilotOptionsMenu, function(){
        test.assertNotVisible(queue.autopilotOptionsMenu,'Autopilot options menu is closed');
      });
    });

    casper.run(function() {
      test.done();
    });
  });

  casper.test.begin('Queue: Worker settings', 3, function(test) {

    var workerCount;

    casper.then(function(){
      workerCount = queue.getWorkerCount();
      test.assert(workerCount > 0, 'Worker count is greater than zero: ' + workerCount);
    });

    casper.waitFor(function(){
      queue.decreaseWorkers();
      var workerCount = queue.getWorkerCount();
      return workerCount === 0;
    });

    casper.then(function(){
      var workerCount = queue.getWorkerCount();
      test.assertEquals(workerCount, 0, 'Worker count was decreased to 0');
    });

    casper.waitFor(function(){
      queue.increaseWorkers();
      var workerCount = queue.getWorkerCount();
      return workerCount === queue.MAX_WORKERS;
    });

    casper.then(function(){
      var workerCount = queue.getWorkerCount();
      test.assertEquals(workerCount, queue.MAX_WORKERS, 'Workers increased to: ' + queue.MAX_WORKERS);
    });

    casper.run(function() {
      test.done();
    });
  });

  casper.test.begin('Queue: In process & Queued Items', 1, function(test) {

    var inProcesMsg = 'The queue has in-process items';
    casper.waitForSelector(queue.inProcess, function(){
      test.assert(casper.exists(queue.inProcess), inProcesMsg);
    }, function timeout(){
      capture('queue-in-process.png', inProcesMsg);
      test.fail(inProcesMsg);
    });

    casper.run(function() {
      test.done();
    });
  });

  casper.test.begin('Queue: Resource usage', 4, function(test) {

    casper.then(function(){
      queue.toggleResourcePanel();
    });

    casper.waitForSelector(queue.sidebarExtended, function(){
      test.assert(true, 'Sidebar extended');
    }, function timeout(){
      capture('sidebar.png');
      test.fail('Could not extend sidebar');
    }, 5000);

    casper.waitUntilVisible(queue.resourceUsageProcessingGraphs, function(){
      test.assert(true, 'Sidebar displays the aggregate CPU chart');
    }, function timeout(){
      capture('queue-aggregate-cpu.png');
      test.fail('No aggregate CPU graph displayed');
    }, 15000);

    casper.waitUntilVisible(queue.singleCpuChart, function(){
      test.assert(true, 'Sidebar displays individual CPU charts');
    }, function timeout(){
      capture('queue-single-cpu.png');
      test.fail('No invidual CPU graphs displayed');
    }, 15000);

    casper.waitUntilVisible(queue.resourceUsageMemoryGraphs, function(){
      test.assert(true, 'Memory graphs displayed');
    }, function timeout(){
      capture('queue-memory.png');
      test.fail('No memory graphs displayed');
    }, 15000);

    casper.then(function(){
      queue.toggleResourcePanel();
    });

    casper.run(function() {
      test.done();
    });
  });
};
module.exports = QueueTests;
