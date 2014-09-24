var USERNAME = process.env.MMAPP_UI_TEST_USERNAME;
var PASSWORD = process.env.MMAPP_UI_TEST_PASSWORD;
var INVITE_CODE = process.env.MMAPP_UI_TEST_INVITE_CODE;

var HEADLESS = process.env.MMAPP_UI_TEST_DISABLE_GUI;
var UI_TEST_URL = process.env.MMAPP_UI_TEST_URL;
var SCREENSHOT_PATH = process.env.MMAPP_UI_SCREENSHOTS_PATH;

var Login = require('./objects/login');
var NewProject = require('./objects/new-project');
var EDA = require('./objects/eda');
var Models = require('./objects/models');
var Main = require('./objects/main');
var Queue = require('./objects/queue');
var Join = require('./objects/join');
var _ = require('lodash');

if(!UI_TEST_URL){
  UI_TEST_URL = 'http://user:pass@localhost';
}

describe('DataRobot', function() {

  describe('New Account', function(){
    it('should join', function() {
      var join = new Join();
      join.get(UI_TEST_URL, USERNAME, INVITE_CODE);
      join.signup(PASSWORD);
      join.logout();
    });

    it('should login', function() {
      var login = new Login();
      login.get(UI_TEST_URL);
      login.setUsername(USERNAME);
      login.setPassword(PASSWORD);
      login.login();
    });
  });

  describe('New Project View', function(){
    var newProject = new NewProject();

    it('should display', function() {
      browser.driver.wait(function(){
        return newProject.isPresent();
      }, 3000);

      expect(newProject.isDisplayed()).toBe(true);
    });

    it('should upload new file', function() {
      var fileName = 'https://s3.amazonaws.com/datarobot_test/kickcars-sample-200.csv';
      newProject.upload(fileName);
    });
  });

  describe('EDA View', function(){
    var eda = new EDA();

    it('should display', function() {

      browser.driver.wait(function(){
        return eda.IsEdaSummaryPresent();
      }, 10000);

      expect(eda.isDisplayed()).toBe(true);
    });

    it('should list features with var types', function() {
      eda.getVarTypes().then(function(varTypes) {
        expect(varTypes.length).toBeGreaterThan(0);

        nonEmptyVarType = _.find(varTypes, function(varType){
          return varType.getText();
        });

        expect(nonEmptyVarType.getText()).toBeTruthy();
      });
    });

    it('should expand eda row on click', function() {
      var featureIndex = 0;
      eda.expandFeature(featureIndex);
      expect(eda.isGraphDisplayed(featureIndex)).toBe(true);
    });

    it('should run models after selecting target, metric and mode', function() {
      eda.setTargetFeature('IsBadBuy');
      eda.setRankingMetric();
      eda.selectSemiAutoModel();
      eda.runModels();
      expect(eda.isTargetVariableDisplayed()).toBe(false);
    });
  });

   describe('Models View', function(){
    var main = new Main();
    var models = new Models();
    var queue = new Queue();
    var topModelInfo = {};

    it('should have at least one leaderboard item', function() {
      browser.ignoreSynchronization = true;
      main.navigateToModels();

      browser.driver.wait(function(){
        return models.isPresent();
      }, 20000);

      models.getLeaderboard().then(function(items){
        expect(items.length).toBeGreaterThan(0);
      });

      var topModel = models.getModelAt(0);

      topModel.getAttribute('id').then(function(attrId){
        topModelInfo.lid = attrId;
      });

      topModel.getText().then(function(text){
        topModelInfo.title = text;
      });
    });

    it('should submit a new model with increased sample size', function(){
      var newSampleSize = 50;
      models.submitSampleSize(topModelInfo.lid, newSampleSize);

      browser.driver.wait(function(){
        return queue.exists(topModelInfo.title, newSampleSize);
      }, 5000);
    });
  });

  describe('Queue', function(){
    var queue = new Queue();

    it('should contain in-process q items', function(){
      expect(queue.areInprocessItemsDisplayed()).toBe(true);
    });

    it('should have queued items', function(){
      expect(queue.areQueuedItemsDisplayed()).toBe(true);
    });

    it('should display cpu resources', function(){

      browser.driver.wait(function(){
        return queue.areCPUResourceGraphsPresent();
      }, 5000);
    });

    it('should display memory resources', function(){
      browser.driver.wait(function(){
        return queue.areMemoryResourceGraphsPresent();
      }, 5000);
    });

    it('should change number of wokers', function(){
      queue.getWorkerCount().then(function(workerCount){
        expect(workerCount).toBeGreaterThan(0);
      });

      var workerCount;
      var turnOffAllWorkers =  function(){
        queue.decreaseWorker();
        queue.getWorkerCount().then(function(count){
          if(workerCount){
            expect(workerCount).not.toEqual(count);
          }

          workerCount = count;

          if(workerCount > 0){
            turnOffAllWorkers();
          }
        });
      };

      turnOffAllWorkers();

      queue.getWorkerCount().then(function(workerCount){
        expect(parseInt(workerCount)).toEqual(0);
      });
    });

  });
});
