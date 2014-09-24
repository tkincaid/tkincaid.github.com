var Main = function(){

  this.modelsTab = by.css('#nav li:nth-child(2) a');

  this.navigateToModels = function(){
    element(this.modelsTab).click();
    browser.waitForAngular();
  };
};

module.exports = Main;