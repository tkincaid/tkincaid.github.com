var NewProject = function(){

  this.uploadUrl = element(by.model('upload.url'));
  this.uploadButton = element(by.id('remoteurl_btn'));
  this.newProjectBanner = element(by.id('project_upload'));

  this.isPresent = function(){
    return this.newProjectBanner.isPresent();
  };

  this.isDisplayed = function(){
    return this.newProjectBanner.isDisplayed();
  };

  this.upload = function(url){
    this.uploadUrl.sendKeys(url);
    this.uploadButton.click();
  };
};

module.exports = NewProject;