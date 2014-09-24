var ProjectTests = require('./project-tests');
var EDATests = require('./eda-tests');
var QueueTests = require('queue-tests');
var LeaderboardTests = require('leaderboard-tests');
var TaskTests = require('tasks-tests');
var InsightTests = require('insights-tests');
var IdeTests = require('ide-tests');

var dataset = {
  url : 'https://s3.amazonaws.com/datarobot_test/kickcars-sample-200.csv',
  target: 'IsBadBuy'
};


var projectTests = new ProjectTests();
//Global variable set in login-tests
if(PROJECT_NAME){
  projectTests.select(PROJECT_NAME);
}
else{
  projectTests.create(dataset);
}

var edaTests = new EDATests();
edaTests.features();
edaTests.selectTarget(dataset.target);
edaTests.runModels();
var queueTests = new QueueTests();
var leaderboardTests = new LeaderboardTests();
leaderboardTests.basic();
var insightTests = new InsightTests();
queueTests.autopilot();
leaderboardTests.advanced();
// leaderboardTests.removeItems();
projectTests.holdout();
leaderboardTests.holdoutSorting();
var taskTests = new TaskTests();
taskTests.checkAll();
var ideTests = new IdeTests();
