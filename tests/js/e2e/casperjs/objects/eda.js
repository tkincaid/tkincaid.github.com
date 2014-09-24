var EDA = function(){

  this.view = '#eda-view';
  this.varTypes = 'div.col-vartype span:not(:empty)';
  this.graphContainer =  '.graph_container';
  this.targetVariable = '#target-feature-select';
  this.rankingMetric = '.metric-box .aim-button:nth-child(2)';
  this.runModelsButton = '.go.dogo';
  this.semiAutoButton = '.ui-mode-semi';
  this.targetHistogram = '.aim-column svg';
  this.selectedTarget = '.ui-target';
  this.edaRows = '.eda-row';
  this.edaRowsLoaded = '#eda-data-scroll .scroll-data .dr-row';
  this.returnToAim = '.return-to-aim';

  this.isDisplayed = function(){
    return casper.visible(this.view);
  };

  this.getEdaRows = function(){
    return casper.getElementsInfo(this.edaRows);
  };

  this.isTargetHistogramDisplayed = function(){
    return casper.visible(this.targetHistogram);
  };

  this.getVarTypes = function(){
    return casper.getElementsInfo(this.varTypes);
  };

  this.expandFeatureAt = function(featureIndex){
    return casper.click('#eda-data-scroll .scroll-data div:nth-child('+ (featureIndex + 1) +') .column-name');
  };

  this.isGraphDisplayed = function(){
    return casper.visible(this.graphContainer);
  };

  this.isTargetVariableDisplayed = function(){
    return casper.visible(this.targetVariable);
  };

  this.setTargetFeature = function(target){
    casper.sendKeys(this.targetVariable, target);
    casper.sendKeys(this.targetVariable, casper.page.event.key.Enter, {keepFocus: true});
  };

  this.setRankingMetric = function(){
    casper.click(this.rankingMetric);
  };

  this.selectSemiAutoModel = function(){
    casper.click(this.semiAutoButton);
  };

  // 6/1/14 Step no longer required but may come back
  // this.runModels = function(){
  //   casper.click(this.runModelsButton);
  // };

  this.isTargetSelected = function(){
    return casper.visible(this.selectedTarget);
  };
};

module.exports = EDA;