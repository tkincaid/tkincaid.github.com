var ProjectSuite = function(){

  var project = new Project();

  this.create = function(dataset, navigateToNew){
    casper.test.begin('Create new project', 1, function(test) {

      casper.then(function(){
        if(navigateToNew){
          project.navigateToNew();
        }
      });

      casper.waitForSelector(project.newProjectBanner, function() {
        test.assert(project.isNewDisplayed(), 'New project is displayed');
      }, function timeout(){
        capture('new-project-banner.png');
        test.skip(1, 'The new project screen did not load on time. URL redirecting...');
        main.forceNew();
      }, 5000);

      casper.waitForSelector(project.uploadFileForm, function() {

        if(dataset.url){
          casper.echo('Uploading from: ' + dataset.url);
          project.uploadFromUrl(dataset.url);
        }
        else if(dataset.filename){
          casper.echo('Uploading from: ' + dataset.filename);
          project.uploadFromFile(dataset.filename);
        }
        else{
          throw 'Dataset url or filename is required';
        }
      });

      casper.run(function() {
        test.done();
      });
    });
  };

  this.select = function(projectName){
    casper.test.begin('Select existing project', 1, function(test) {

      casper.waitForSelector(project.projectsMenu, function(){
        project.select(projectName);
      });

      casper.then(function(){

        var visibleName = projectName.slice(0,20);

        var pattern = new RegExp('^' + visibleName, 'gi');
        var currentProjectName = project.getName();
        test.assert(pattern.test(currentProjectName), 'Project selected: ' + projectName);
      });

      casper.then(function(){
        main.selectEDATab();
      });

      casper.run(function() {
        test.done();
      });
    });
  };


  this.holdout = function(){
    casper.test.begin('Unlock project', 1, function(test) {
      casper.then(function(){
        main.navigateToProjects();
      });

      // open dropdown menu
      casper.waitForSelector(project.projectList, function(){
        casper.click(project._projectOptionsDropDownTrigger());
      });

      // click unlock
      casper.waitUntilVisible(project.projectOptionsMenu,function(){
        casper.click(project.projectUnlockMenuItem);
      });

      // click unlock on the popup
      casper.waitUntilVisible(project.holdoutPopup,function(){
        casper.clickLabel(project.unlockButton);
      });

      // open dropdown menu again
      casper.waitForSelector(project.projectList, function(){
        casper.click(project._projectOptionsDropDownTrigger());
      });

      // Menu Option should Read 'Holdout is Unlocked'
      casper.waitUntilVisible(project.projectUnlockedMenuItem,function(){
        test.assert(true, ' Project Holdout Unlocked');
      }, function timeout(){
        capture('holdout-unlocked.png');
        test.fail('Could not unlock project');
      });

      casper.run(function() {
        test.done();
      });
    });
  };
};

module.exports = ProjectSuite;
