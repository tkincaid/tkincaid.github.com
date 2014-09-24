var Tasks = function(){

  this.view = '#task-view';
  this.dataRobotTab = 'DataRobot';
  this.tasks = '#task-container .dr-row';
  this.taskName = '.task-name';
  this.runButton = '.ui-task-run';
  this.preview = '.ui-task-preview';
  this.blueprint = '.task-blueprint svg';
  this.sampleSize = 'input';

  this.isDisplayed = function(){
    return casper.visible(this.view);
  };

  this._getTask = function(taskId){
    return '#' + taskId + ' ';
  };

  this._getRunButton = function(taskId){
    return this._getTask(taskId) + this.runButton;
  };

  this._getTaskName = function(taskId){
    return this._getTask(taskId) + this.taskName;
  };

  this._getPreview = function(taskId){
    return this._getTask(taskId) + this.previewButton;
  };

  this._getSampleSize = function(taskId){
    return this._getTask(taskId) + this.sampleSize;
  };

  this._getBlueprint = function(taskId){
    return this._getTask(taskId) + this.blueprint;
  };

  this.selectDataRobotTab = function(){
    casper.clickLabel(this.dataRobotTab);
  };

  this.areDataRobotTasksDisplayed = function(){
    return casper.visible(this.tasks);
  };

  this.get = function(){
    return casper.getElementsInfo(this.tasks);
  };

  this.getTaskTitle = function(taskId){
    return casper.fetchText(this._getTaskName(taskId));
  };

  this.run = function(taskId, newSampleSize){
    casper.click(this._getRunButton(taskId));
    var sampleSizeInput = this._getSampleSize(taskId);
    casper.waitForSelector(sampleSizeInput, function(){
      casper.sendKeys(sampleSizeInput, newSampleSize, {reset: true});
      casper.clickLabel('Submit');
    });
  };

  this.preview = function(taskId){
    return casper.click(this._getPreview(taskId));
  };

  this.expandTask = function(taskIndex){
    var tasks  = this.get();
    var task = tasks[taskIndex];
    var expandedTaskId = task.attributes.id;
    casper.click(this._getTaskName(expandedTaskId));
    return expandedTaskId;
  };

  this.isBlueprintDisplayed = function(taskId){
    return casper.visible(this._getBlueprint(taskId));
  };
};

module.exports = Tasks;
