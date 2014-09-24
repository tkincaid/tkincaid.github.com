var Queue = function(){

  this.workerCount = by.css('.QueueSettings .parvalue');
  this.minusWorker = by.css('#parminus');
  this.processingModels = by.css('.top_monitor div.QueueItem');
  this.waitingModels = by.css('.queue_monitor div.QueueItem');
  this.resourceUsageProcessingGraphs = by.css('.processing_usage svg');
  this.resourceUsageMemoryGraphs = by.css('.memory_usage .memory_gradient_container');

  this.exists = function(fullModelTitle, sampleSize){
    // Remove bp
    var i = fullModelTitle.indexOf(' (');
    modelTitle = fullModelTitle.substring(0, i);


    var xpath = '//div[contains(@class, "QueueItem")]'; // Grab all queue items
    xpath += '//div[p[contains(@title,"'+ modelTitle +'")]'; //Filter by title
    xpath += ' and ';
    xpath += 'p[span[text()[contains(.,"'+ sampleSize +'%")]]]]'; // and sample size

    return element(by.xpath(xpath)).isPresent();
  };

  this.getWorkerCount =  function(){
    return element(this.workerCount).getText();
  };

  this.decreaseWorker =  function(){
    return element(this.minusWorker).click();
  };

  this.areCPUResourceGraphsPresent =  function(){
    return element(this.resourceUsageProcessingGraphs).isPresent();
  };

  this.areMemoryResourceGraphsPresent =  function(){
    return element(this.resourceUsageMemoryGraphs).isPresent();
  };

  this.areQueuedItemsDisplayed =  function(){
    return element(this.waitingModels).isDisplayed();
  };

  this.areInprocessItemsDisplayed =  function(){
    return element(this.processingModels).isDisplayed();
  };
};

module.exports = Queue;