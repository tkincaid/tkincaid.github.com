var MainPage = function(baseUrl) {

  this.modelsTab = 'Models';
  this.EDATab = '#nav ul li:nth-child(1) a';
  this.rstudioTab = 'Studio';
  this.pythonTab = 'Python';
  this.tasksTab = 'Repository';
  this.insightsTab = 'Insights';
  this.projectsMenu = '.ui-projects';
  this.allProjectsMenuItem = 'Manage Projects';
  this.userMenu = '.ui-user-menu';
  this.logoutMenuItem = 'Logout';


  this.selectEDATab = function() {
    casper.click(this.EDATab);
  };

  this.selectModelsTab = function() {
    casper.clickLabel(this.modelsTab);
  };

  this.selectRStudioTab = function() {
    casper.clickLabel(this.rstudioTab);
  };

  this.selectIPythonTab = function() {
    casper.clickLabel(this.pythonTab);
  };

  this.selectTasksTab = function() {
    casper.clickLabel(this.tasksTab);
  };

  this.selectInsightsTab = function() {
    casper.clickLabel(this.insightsTab);
  };

  this.navigateToProjects = function() {
    casper.click(this.projectsMenu);
    casper.clickLabel(this.allProjectsMenuItem);
  };

  this.logout = function(){
    casper.click(this.userMenu);
    casper.clickLabel(this.logoutMenuItem);
  };

  this.hardLogout = function(){
    casper.thenOpen(baseUrl + '/account/logout');
  };

  this.forceNew = function(){
    casper.thenOpen(baseUrl + '/new');
  };
};

module.exports = MainPage;