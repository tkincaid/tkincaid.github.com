var ProjectTests = require('./project-tests');
var EDATests = require('./eda-tests');
var LeaderboardTests = require('leaderboard-tests');
var TaskTests = require('tasks-tests');
var InsightTests = require('insights-tests');


var projectTests = new ProjectTests();
//Global variable set in login-tests
for (var i = 0; i < PROJECT_NAMES.length; i++) {
  var projectName = PROJECT_NAMES[i];
  projectTests.select(projectName);
  var edaTests = new EDATests();
  edaTests.features();
  var leaderboardTests = new LeaderboardTests();
  leaderboardTests.just_top_model_ui();
  var insightTests = new InsightTests();
  var taskTests = new TaskTests();
  taskTests.checkDisplay();
}
