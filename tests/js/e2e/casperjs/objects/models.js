var _ = require('lodash');

var ModelsPage = function(x){

  this.allModels = '#leaderBoard .model-row';
  this.avgKickoffFromRow = '.col-score.kickoff';
  this.sampleSizeSelector = '.col-samplesize .fa-plus';
  this.sampleSizeForm = 'sample-size-form ';
  this.sampleSizeInput = this.sampleSizeForm + 'input';
  this.sampleSizeSubmit = this.sampleSizeForm + '.ui-submit-sample-size';
  this.batchForm = '#batchForm ';
  this.batchSampleSizeInput = this.batchForm + 'input.ui-batch-sample-size-input';
  this.batchSubmit =  this.batchForm + ' button.nbutton';
  this.featureListMenu = '.run-new-featurelist';
  this.modelTitle = '.model-name';
  this.blueprint = '.model-chart svg g.start';
  this.tabs = '.expanded-menu li';
  this.blueprintTab = this.tabs + '.ui-blueprint:not(.disabled)';
  this.modelInfoTab = this.tabs + '.ui-model-info:not(.disabled)';
  this.liftChartTab = this.tabs + '.ui-liftchart:not(.disabled)';
  this.liftTableTab = this.tabs + '.ui-lifttable:not(.disabled)';
  this.leaderboardTab = 'Leaderboard';
  this.learningCurveTab = 'Learning Curves';
  this.learningCurveGraph = '#learningCurve svg';
  this.metricMenuTrigger = '#metric-select .trigger';
  this.metricMenu = '.metric-selector';
  this.scores = '.ui-score';
  this.avgScores = '.ui-avg-score';
  this.avgScoresSort = '.ui-sorty-by-avg';
  this.holdoutSort = '.ui-sorty-by-holdout';
  this.holdoutScores = '.ui-holdout-score';
  this.predictTab = 'div.ui-deploy-predict';
  this.predictionActionButton = '.dr-row:nth-child(NUMBER) .predict';
  this.uploadDatasetButton = '.toggle-upload';
  this.uploadUrl = '#import_url';
  this.uploadButton = '#remoteurl_btn';
  this.newDataset = 'https://s3.amazonaws.com/datarobot_test/kickcars-sample-200.csv';
  this.predictions = '#deploy-predict .prediction-datasets .dr-row';
  this.featureListDropDown = '.feature-list-dropdown i';
  this.modelSelector = '.col-checkbox i';
  this.leaderboardMenuTrigger = '.ui-blend-dropdown';
  this.leaderboardMenu = '.leaderboard-options';
  this.runSelected = '.ui-run-selected-models';
  this.rowExpanded = '.expanded-row';

  var xPathFinishedModels = '//div[contains(@class, "model-row") and ' + //select the leaderboard items
    '*//*[contains(@class,"ui-score")]]'; // that have completed
  this.finishedModels = x(xPathFinishedModels);


  var xPathFindByTitle = '//div[contains(@class, "model-row") and ' + //select the leaderboard items
    '*//text()[contains(., "MODEL_TITLE")] and ' + // with this title
    '*//*[contains(@class,"ui-score")]]'; // that have completed

  var xPathFindEnabledMenuItem = '//li[not(@class="disabled")][text()[contains(., "MENU_ITEM")]]';

  this._getModel = function(lid){
    return '#' + lid + ' ';
  };

  this._getBlueprint = function(lid){
    return this._getModel(lid) + 'div.ui-blueprint svg';
  };

  this._getModelInfo = function(lid){
    return this._getModel(lid) + 'div.ui-modelinfo';
  };

  this._getLiftChart = function(lid){
    return this._getModel(lid) + 'div.ui-liftchart';
  };

  this._getLiftTable = function(lid){
    return this._getModel(lid) + 'div.ui-lifttable';
  };

  this._getPredict = function(lid){
    return this._getModel(lid) + this.predictTab;
  };

  this._getPredictionActionButton = function(lid, index){
    return this._getModel(lid) + this.predictionActionButton.replace('NUMBER', index + 1);
  };

  this._findPredictionActionButton = function(lid, index, text){
    var actionButton = this._getPredictionActionButton(lid, index);
    var label = casper.fetchText(actionButton);
    return label.indexOf(text) >= 0 ? actionButton : null;
  };

  this._getPredictions = function(lid, index){
    return this._getModel(lid) + this.predictions;
  };

  this._getUploadDataset = function(lid, index){
    return this._getModel(lid) + this.uploadDatasetButton;
  };

  this._getFeatureListDropDown = function(lid){
    return this._getModel(lid) + this.featureListDropDown;
  };

  this._getModelSelector = function(lid){
    return this._getModel(lid) + this.modelSelector;
  };

  this._getEnabledMenuItemSelector = function(blender){
    return x(xPathFindEnabledMenuItem.replace('MENU_ITEM', blender));
  };

  this.getLeaderboard = function(includInProcess){
    if(includInProcess){
      return casper.getElementsInfo(this.allModels);
    }
    return casper.getElementsInfo(x(xPathFinishedModels));
  };

  this.getTopModelId = function(){
    return this.getModelId(0);
  };

  this.getModelId = function(index){
    var leaderboardItems = this.getLeaderboard();
    if(leaderboardItems && leaderboardItems.length){
      return leaderboardItems[index].attributes.id;
    }
    else{
      captureAndWarn('get-model-id.png',
        'Could not retrieve model id. Leaderboard items: ' + leaderboardItems.length);
      return null;
    }

  };

  this.isRowExpanded = function(){
    return casper.exists(this.rowExpanded);
  };

  this.getModelIdByTitle = function(title){
    var xpath = xPathFindByTitle.replace('MODEL_TITLE', title);
    if(!casper.visible(x(xpath))){
      return;
    }

    var leaderboardItem = casper.getElementInfo(x(xpath));
    return leaderboardItem && leaderboardItem.attributes &&
      leaderboardItem.attributes.id;
  };

  this.submitSampleSize = function(lid, sampleSize){
    var model = this._getModel(lid);
    casper.click(model + this.sampleSizeSelector);
    var self = this;
    casper.waitForSelector(this.sampleSizeInput, function(){
      casper.sendKeys(self.sampleSizeInput, sampleSize, {reset: true});
      casper.click(self.sampleSizeSubmit);
    });
  };

  this.runAvgfromModelRow = function(lid){
    var model = this._getModel(lid);
    casper.click(model + this.avgKickoffFromRow);
  };

  this.downloadPrediction = function(lid, index){
    var actionButton = this._getPredictionActionButton(lid, index);

    casper.waitForSelector(actionButton, function(){
      casper.click(actionButton);
    }, function timeout(){
      captureAndWarn('lb-prediction-action-button.png',
        'Prediction action button did not load on time');
    });

    return 'predictions/' + lid.replace('lid-', '');
  };

  this.computePrediction = function(lid, index){
    casper.click(this._getPredictionActionButton(lid, index));
  };

  this.getModelTitle = function(lid){
    var fullModelTitle = casper.fetchText(this._getModel(lid) + this.modelTitle);

    // Remove bp
    var i = fullModelTitle.indexOf(' (');
    return fullModelTitle.substring(0, i).trim();
  };

  this.expandModel = function(lid){
    var model = this._getModel(lid);
    casper.click(model + this.modelTitle);
  };

  this.isBlueprintDisplayed = function(lid){
    return casper.visible(this._getBlueprint(lid));
  };

  this.isModelInfoDisplayed = function(lid){
    return casper.visible(this._getModelInfo(lid));
  };

  this.isLiftChartDisplayed = function(lid){
    casper.echo(this._getLiftChart(lid));
    return casper.visible(this._getLiftChart(lid));
  };

  this.isLiftTableDisplayed = function(lid){
    return casper.visible(this._getLiftTable(lid));
  };

  this.isPredictDisplayed = function(lid){
    return casper.visible(this._getPredict(lid));
  };

  this.selectPredictTab = function(lid){
    // Space in "Deploy " is necessary
    var self = this;
    //casper.clickLabel('Deploy ');
    casper.clickLabel('Predict');
    casper.waitFor(function(){
      return self.isPredictDisplayed(lid);
    });
  };

  this.selectModelInfoTab = function(){
    casper.click(this.modelInfoTab);
  };

  this.selectLiftChartTab = function(){
    casper.click(this.liftChartTab);
  };

  this.selectLiftTableTab = function(){
    casper.click(this.liftTableTab);
  };

  this.selectLearningCurveTab = function(){
    casper.clickLabel(this.learningCurveTab);
  };

  this.selectLeaderboardTab = function(){
    casper.clickLabel(this.leaderboardTab);
  };


  this.isLearningCurveGraphDisplayed = function(){
    return casper.visible(this.learningCurveGraph);
  };

  this.openMetricMenu = function(){
    casper.click(this.metricMenuTrigger);
  };

  this.setMetric = function(metric){
    casper.clickLabel(metric);
  };

  this.getMetric = function(){
    var label = casper.fetchText(this.metricMenuTrigger);
    var i  = label.indexOf(': ');
    return label.substring(i + 2).trim();
  };

  var _getScoreAmounts =  function(scoresSelector){
    var scores = casper.getElementsInfo(scoresSelector);
    scores = _.pluck(scores,'text');
    scores = _.map(scores,parseFloat);
    return _.filter(scores, function(score){
      return ! _.isNaN(score);
    });
  };

  this.getScores = function(){
    return _getScoreAmounts(this.scores);
  };

  this.getAvgScores = function(){
    return _getScoreAmounts(this.avgScores);
  };

  this.getHoldoutScores = function(){
    return _getScoreAmounts(this.holdoutScores);
  };

  this.addDataset = function(lid){
    casper.click(this._getUploadDataset(lid));
  };

  this.uploadNewDataset = function(url){
    casper.sendKeys(this.uploadUrl, url);
    casper.click(this.uploadButton);
  };

  this.getPredictions =  function(lid){
    return casper.getElementsInfo(this._getPredictions(lid));
  };

  this.isPredictionReady = function(lid, index){
    return this._findPredictionActionButton(lid, index, 'Download Prediction');
  };

  this.isReadyToComputePrediction = function(lid, index){
    return this._findPredictionActionButton(lid, index, 'Compute Prediction');
  };

  this.isPredictionInProcess = function(lid, index){
    return this._findPredictionActionButton(lid, index, 'Calculating');
  };

  this.openFeatureListMenu = function(lid){
    casper.click(this._getFeatureListDropDown(lid));
  };

  this.runFeatureList = function(featureList){
    casper.clickLabel(featureList);
  };

  this.selectModelsByIndex = function(modelIndexes){
    var models =this.getLeaderboard();
    var ids = [];
    for (var i = modelIndexes.length - 1; i >= 0; i--) {
      var id = models[i].attributes.id;
      casper.click(this._getModelSelector(id));
      ids.push(id);
    }
    return ids;
  };

  this.openRunBatchForm = function(){
    casper.click(this.runSelected);
  };

  this.runSelectedModels = function(newSampleSize){
    casper.sendKeys(this.batchSampleSizeInput, newSampleSize, {reset: true});
    casper.click(this.batchSubmit);
  };

  this.openLeaderboardMenu = function(){
    casper.click(this.leaderboardMenuTrigger);
  };

  this.blendSelectedModels = function(blender){
    var blenderSelector = this._getEnabledMenuItemSelector(blender);
    casper.waitForSelector(blenderSelector, function(){
      casper.click(blenderSelector);
    });
  };

  this.sortByAvgScore = function(){
    casper.click(this.avgScoresSort);
  };

  this.sortByHoldout = function(){
    casper.click(this.holdoutSort);
  };

  this.removeSelected =  function(){
    casper.clickLabel('Delete Selected Models');
  };

  this.modelsExist = function(selectedIds){
    var leaderboardItems = this.getLeaderboard();
    return _.some(leaderboardItems, function(l){
      return selectedIds.indexOf(l.attributes.id) >=0;
    });
  };
};

module.exports = ModelsPage;

