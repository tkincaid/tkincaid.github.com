var EDA = function(){

  this.edaView = by.id('eda-view');
  this.targetVariable = by.model('selections.targetFeatureName');
  this.feature = '#eda_data tr:nth-child(FEATURE_INDEX)>.column-name span';
  this.features = by.repeater('column in pages[currentPage]');
  this.edaSummary = by.css('td.vartype:not(:empty)');
  this.targetFeature = by.model('selections.targetFeatureName');
  this.rankingMetric = by.css('.metric-box .aim-button:nth-child(2)');
  this.runModelsButton = by.className('start-button');
  this.semiAutoButton = by.className('semi-automatically');


  this.isDisplayed = function(){
    return element(this.edaView).isDisplayed();
  };

  this.IsEdaSummaryPresent = function(){
    return element(this.edaSummary).isPresent();
  };

  this.isTargetVariableDisplayed = function(){
    return element(this.targetVariable).isPresent();
  };

  this.getVarTypes = function(){
    return browser.findElements(this.features.column('column.profile.type_label'));
  };

  this.expandFeature = function(index){
    return element(by.css(this.feature.replace('FEATURE_INDEX', index + 1))).click();
  };

  this.isGraphDisplayed = function(index){
    return element(this.features.row(index)).findElement(by.css('.graph_container')).isDisplayed();
  };

  this.setTargetFeature = function(target){
    element(this.targetFeature).sendKeys(target);
    element(this.targetFeature).sendKeys(protractor.Key.ENTER);
  };

  this.setRankingMetric = function(){
    element(this.rankingMetric).click();
    var button = element(this.runModelsButton);
    expect(button.isDisplayed()).toBe(true);
  };

  this.selectSemiAutoModel = function(){
    element(this.semiAutoButton).click();
  };

  this.runModels = function(){
    element(this.runModelsButton).click();
  };

};

module.exports = EDA;