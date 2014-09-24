var Queue = function(x){

  this.MAX_WORKERS = 4;
  this.inProcess = '#ui-queue-inprogress-container .InprogressItem';
  this.queuedItems = '#ui-queue-queued-container .QueueItem';
  this.resourceUsageProcessingGraphs = '.processing_usage svg';
  this.resourceUsageMemoryGraphs = '.memory_usage .memory_gradient_container';
  this.workerCount = '#parvalue';
  this.minusWorkers = '#parminus';
  this.moreWorkers = '#parplus';
  this.sidebar = '#toggle-sidebar .toggle-button';
  this.sidebarExtended = ' .fa-chevron-right';
  this.extendedSidebar = '#toggle-sidebar.ext-1';
  this.singleCpuChart = '.single_cpu_chart svg';
  this.remove = '.queue_remove';

  this.openAutopilotOptions = '.queue-options-button';
  this.autopilotOptionsMenu = '#pilot-options';
  this.autopilotOptionsMode = '.ui-autopilot-options-mode';
  this.autopilotOptionsModeMenu = '#autopilot-mode';


  this.continueAutopilotButton = '#ui-continue-autopilot-button';
  this.closeAutopilotPopup = 'div.dr-modal button:nth-child(2)';

  this.currentMode = '.ng-isolate-scope.selected';
  this.modeButtonFullAuto = '#ui-mode-auto';
  this.modeButtonSemiAuto = '#ui-mode-semi';
  this.modeButtonNoAuto   = '#ui-mode-none';

  this.queuePauseButton = '.queue-pause';
  this.pausedIndicator = this.queuePauseButton + ' .fa-play';

  this.MODE = {
    FULL : 'ui-mode-auto',
    SEMI : 'ui-mode-semi',
    OFF  : 'ui-mode-none'
  };


  this._getItemXpath = function(modelTitle, sampleSize, runNumber){
    var xpath = '*//*[contains(@title, "MODEL_TITLE")] '; //Filter by title

    if(sampleSize){
      xpath += 'and *//text()[contains(., "SAMPLE_SIZE%")]'; // and sample size
    }
    if(runNumber){
      xpath += 'and *//text()[contains(., "CV #RUN_NUMBER")]'; // cv run
    }
    xpath = xpath.replace('MODEL_TITLE', modelTitle)
      .replace('SAMPLE_SIZE', sampleSize)
      .replace('RUN_NUMBER', runNumber);

    return xpath;
  };


  this.selector = function(modelTitle, sampleSize, runNumber){
    var xpath = '//div[contains(@class, "QueueItem") and ' +
      this._getItemXpath(modelTitle, sampleSize, runNumber) + ']';

    casper.echo('Selector for "'+ modelTitle +'": ' + xpath);
    return x(xpath);
  };

  this.multipleSelector = function(selectedModels, sampleSize){
    var allXpath = '//div[';

    for (var i = selectedModels.length - 1; i >= 0; i--) {
      var modelTitle = selectedModels[i].title;
      allXpath += 'contains(@class, "QueueItem") and ' +
        this._getItemXpath(modelTitle, sampleSize);
      if(i > 0){
        allXpath += ' or ';
      }
      else{
        allXpath += ']';
      }
    }

    casper.echo('Selector for multiple queue items: ' + allXpath);
    return x(allXpath);
  };

  this.inProgressCount = function(){
    return casper.getElementsInfo(this.inProcess).length;
  };

  this.inQueuedCount = function(){
    return casper.exists(this.queuedItems) && casper.getElementsInfo(this.queuedItems).length;
  };

  this.getWorkerCount =  function(){
    return parseInt(casper.fetchText(this.workerCount));
  };

  this.setAutoPilotMode = function(mode){
    if (mode === 0){
      casper.click(this.modeButtonFullAuto);
    } else if (mode === 1){
      casper.click(this.modeButtonSemiAuto);
    } else if (mode === 2){
      casper.click(this.modeButtonNoAuto);
    }
  };

  this.getAutoPilotMode =  function(){
    return casper.getElementAttribute(this.currentMode,'id');
  };

  this.toggleQueuePaused = function(){
    casper.click(this.queuePauseButton);
  };

  this.decreaseWorkers =  function(){
    casper.click(this.minusWorkers);
  };

  this.increaseWorkers =  function(){
    casper.click(this.moreWorkers);
  };

  this.toggleResourcePanel = function(){
    casper.click(this.sidebar);
  };

  this.areSingleCpuChartsVisible = function(){
    return casper.visible(this.singleCpuChart);
  };

  this.removeQueued = function(){
    casper.click(this.queuedItems + ' ' + this.remove);
  };

  this.isAutopilotPopupVisible = function(){
    return casper.visible(this.autopilotPopup);
  };

  this.verifyAutopilot = function(){
    var self = this;
    casper.waitForSelector(this.autopilotButton, function(){
      try{
        casper.click(self.closeAutopilotPopup);
        return true;
      }
      catch(err){}
    }, function timeout(){
      captureAndWarn('no-autopilot.png', 'Autopilot did not pop up');
    }, 3000);
  };

  this.removeAll = function(){
    var self = this;
    var remainingInQueue = self.inQueuedCount();
    casper.then(function (){

      if(remainingInQueue){
        while(casper.visible(self.queuedItems)){
          self.removeQueued();
        }
      }

      casper.echo((remainingInQueue || 0) + ' queued items removed');
    });
  };

  this.isPaused = function(){
    return casper.visible(queue.pausedIndicator);
  };
};

module.exports = Queue;
