var Models = function(){

  this.modelsView = by.css('#leaderBoard .model-row');
  this.leaderBoard = by.repeater('model in filteredModels');
  this.sampleSizeSelector = '.samplesize .fa-plus';
  this.sampleSizeInput = 'input';
  this.sampleSizeSubmit = '.sample-open .flatbutton';

  this.isDisplayed = function(){
    return element(this.modelsView).isDisplayed();
  };

  this.isPresent = function(){
    return element(this.modelsView).isPresent();
  };

  this.getLeaderboard = function(){
    return element.all(this.leaderBoard);
  };

  this.getModelAt = function(index){
    return element(this.leaderBoard.row(index));
  };

  this.submitSampleSize = function(lid, sampleSize){
    var modelSelector = 'div#' + lid + ' ';
    element(by.css(modelSelector + this.sampleSizeSelector)).click();
    var input = by.css(modelSelector + this.sampleSizeInput);
    element(input).clear();
    element(input).sendKeys(sampleSize);
    element(by.css(modelSelector + this.sampleSizeSubmit)).click();
  };
};

module.exports = Models;