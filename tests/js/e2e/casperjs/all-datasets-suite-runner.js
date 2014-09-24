var ProjectTests = require('./project-tests');
var EDATests = require('./eda-tests');
var QueueTests = require('queue-tests');
var LeaderboardTests = require('leaderboard-tests');
var InsightTests = require('insights-tests');
var IdeTests = require('ide-tests');

for (var i = 0; i < DATASETS.length; i++) {
  var projectTests = new ProjectTests();
  var dataset = DATASETS[i];
  var navigateToNew = i > 0;
  projectTests.create(dataset, navigateToNew);
  var edaTests = new EDATests();
  edaTests.features();
}

