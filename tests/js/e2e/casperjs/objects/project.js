var Project = function(){

  this.uploadUrl = '#import_url';
  this.uploadFile = '#browseFile';
  this.uploadFileForm = 'form#file-form';
  this.uploadFromURLButton = '#remoteurl_btn';
  this.newProjectBanner = '#project_upload';
  this.activeProject = '.dr-row.highlight ';
  this.lockButton = '.fa-lock';
  this.projectList = '#projects-list';
  this.holdoutPopup = '.modal-content';
  this.unlockButton = 'Unlock';
  this.unlockedIcon = '.fa-unlock';
  this.projectoptionstrigger = '.ui-project-options-dropdown';
  this.projectOptionsMenu = '#pop .project-options';
  this.projectUnlockMenuItem = this.projectOptionsMenu+' .ui-project-unlock-holdout';
  this.projectUnlockedMenuItem = this.projectOptionsMenu+' .ui-project-unlocked';
  this.projectsMenu = '.ui-projects';
  this.creaNewProjectLabel = 'Create New Project';
  this.currentProjectName = '.ui-current-project';

  this.isNewDisplayed = function(){
    return casper.visible(this.newProjectBanner);
  };

  this.uploadFromUrl = function(url){
    casper.sendKeys(this.uploadUrl, url);
    casper.click(this.uploadFromURLButton);
  };

  this.uploadFromFile = function(filename){
    casper.fill(this.uploadFileForm, {
        'file': filename
    }, true);
  };

  this._getHoldoutButton = function(){
    return this.activeProject  + this.lockButton;
  };

  this._projectOptionsDropDownTrigger = function(){
    return this.activeProject  + this.projectoptionstrigger;
  };

  this._getHoldoutUnlockedIcon = function(){
    return this.activeProject  + this.unlockedIcon;
  };

  this.unlockHoldout = function(){
    var self = this;
    casper.click(this._projectOptionsDropDownTrigger());
    casper.waitForSelector(this.projectOptionsMenu, function(){
      casper.click(self.projectUnlockMenuItem);
    });
    casper.waitForSelector(this.holdoutPopup, function(){
      casper.clickLabel(self.unlockButton);
    });
  };

  this.isDisplayed = function(){
    return casper.visible(this.projectList);
  };

  this.isUnlocked = function(){
    var self = this;
    casper.click(this._projectOptionsDropDownTrigger());
    casper.waitForSelector(this.projectOptionsMenu, function(){
      //casper.click(self.projectUnlockMenuItem);
    });
    return casper.visible(this.projectUnlockedMenuItem);
  };

  this.select = function(projectName){
    casper.click(this.projectsMenu);
    casper.clickLabel(projectName);
  };

  this.navigateToNew = function(){
    casper.click(this.projectsMenu);
    casper.clickLabel(this.creaNewProjectLabel);
  };

  this.getName = function(){
    return casper.fetchText(this.currentProjectName);
  };

};

module.exports = Project;