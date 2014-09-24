var TaskTests = function(){
  var randomIntFromInterval = function(min,max){
      return Math.floor(Math.random()*(max-min+1)+min);
  };

  this.checkDisplay = function() {
    casper.test.begin('Tasks', 3, function(test){
      main.selectTasksTab();
      var tasks = new Tasks();
      var expandedTaskId;
      casper.waitForSelector(tasks.view, function(){
        test.assert(tasks.isDisplayed(), 'Task View loaded');
        tasks.selectDataRobotTab();
      });

      casper.waitForSelector(tasks.tasks, function(){
        test.assert(tasks.areDataRobotTasksDisplayed(), 'DataRobot Tasks are displayed');
      });

      casper.then(function(){
        expandedTaskId = tasks.expandTask(0);
      });

      var drTaskMsg = 'DataRobot Task Blueprint displayed';
      casper.waitFor(function (){
        var blueprintIsDisplayed = tasks.isBlueprintDisplayed(expandedTaskId);
        if(blueprintIsDisplayed){
          test.assert(true, drTaskMsg);
          return true;
        }
      }, function then(){}, function timeout(){
        capture('dr-task-blueprint.png', drTaskMsg);
        test.fail(drTaskMsg);
      });
      casper.run(function() {
        test.done();
      });
    });
  };

  this.checkAll = function() {
    casper.test.begin('Tasks', 5, function(test){
      main.selectTasksTab();
      var tasks = new Tasks();
      var expandedTaskId;
      casper.waitForSelector(tasks.view, function(){
        test.assert(tasks.isDisplayed(), 'Task View loaded');
        tasks.selectDataRobotTab();
      });

      casper.waitForSelector(tasks.tasks, function(){
        test.assert(tasks.areDataRobotTasksDisplayed(), 'DataRobot Tasks are displayed');
      });

      casper.then(function(){
        expandedTaskId = tasks.expandTask(0);
      });

      var drTaskMsg = 'DataRobot Task Blueprint displayed';
      casper.waitFor(function (){
        var blueprintIsDisplayed = tasks.isBlueprintDisplayed(expandedTaskId);
        if(blueprintIsDisplayed){
          test.assert(true, drTaskMsg);
          return true;
        }
      }, function then(){}, function timeout(){
        capture('dr-task-blueprint.png', drTaskMsg);
        test.fail(drTaskMsg);
      });

      var taskName;
      var newSampleSize;
      casper.then(function(){
        newSampleSize = randomIntFromInterval(50, 63);
        newSampleSize = String(newSampleSize);
        taskName = tasks.getTaskTitle(expandedTaskId);
        test.assertTruthy(taskName, 'Task name retrieved: ' + taskName);
      });

      casper.then(function(){
        casper.echo('Running DataRobot task with sample size: ' + newSampleSize);
        tasks.run(expandedTaskId, newSampleSize);
      });

      //No sample size, this item may be in the in-process items
      casper.then(function(){
        casper.waitForSelector(queue.selector(taskName), function (){
          test.assert(true, 'Run DataRobot Task "' + taskName + '" (' + newSampleSize + ') in queue');
        }, function timeout(){
          captureAndWarn('run-task.png', 'Not found in queue');
          test.fail('Task failed');
        });
      });

      casper.run(function() {
        test.done();
      });
    });
  };
};

module.exports = TaskTests;
