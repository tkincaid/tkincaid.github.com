var system = require('system');
var x = require('casper').selectXPath;
var Login = require('./objects/login');
var Join = require('./objects/join');
var Project = require('./objects/project');
var EDA = require('./objects/eda');
var Main = require('./objects/main');
var Models = require('./objects/models');
var Queue = require('./objects/queue');
var RStudio = require('./objects/rstudio');
var IPython = require('./objects/ipython');
var Tasks = require('./objects/tasks');
var Insights = require('./objects/insights');

var USERNAME;
var PASSWORD;
var INVITE_CODE;
var UI_TEST_URL;
var SCREENSHOT_PATH;
var VERBOSE;
var PROJECT_NAME;
var PROJECT_NAMES;
var SKIP_JOIN;
var DATASETS;
var configurationFile = system.env.MMAPP_UI_TEST_CONFIG;
var config =  configurationFile ? require(configurationFile) : {};

USERNAME = config.username || system.env.MMAPP_UI_TEST_USERNAME;
PASSWORD = config.password || system.env.MMAPP_UI_TEST_PASSWORD;
PROJECT_NAME = config.projectName || system.env.MMAPP_UI_SCREENSHOTS_PROJECT_NAME;
PROJECT_NAMES = config.projectNames || [PROJECT_NAME];
DATASETS = config.datasets;
INVITE_CODE = config.inviteCode || system.env.MMAPP_UI_TEST_INVITE_CODE;
VERBOSE = system.env.MMAPP_UI_VERBOSE;
UI_TEST_URL = system.env.MMAPP_UI_TEST_URL;
SCREENSHOT_PATH = system.env.MMAPP_UI_SCREENSHOTS_PATH;

var eda = new EDA();
var queue = new Queue(x);
var models = new Models(x);

if(VERBOSE){
  casper.on('remote.message', function(msg) {
    this.echo(msg);
  });
}

var join = new Join();
var login = new Login(UI_TEST_URL);
var main = new Main(UI_TEST_URL);

var captureAndDie = function(name, msg){
  var path = SCREENSHOT_PATH + '/' + name;
  casper.capture(path);
  main.hardLogout();
  casper.die(msg, 1);
};

var captureAndWarn = function(name, msg){
  var path = SCREENSHOT_PATH + '/' + name;
  casper.capture(path);
  casper.warn(msg);
};

var capture = function(name){
  var path = SCREENSHOT_PATH + '/' + name;
  casper.capture(path);
};

casper.options.waitTimeout = 10000;
casper.options.onWaitTimeout = function() {
  captureAndDie('timeout.png', 'Unhandled timeout');
};

if(INVITE_CODE){
  var joinUrl = join.getUrl(UI_TEST_URL, USERNAME, INVITE_CODE);
  casper.echo('Joining with '+ USERNAME + ' into: ' + UI_TEST_URL);
  casper.start(joinUrl);
}
else{
  casper.echo('Using '+ USERNAME + ' to log into: ' + UI_TEST_URL);
  casper.start(UI_TEST_URL);
}


// desktop-standard
var viewport = {width: 1280, height: 1024};
casper.viewport(viewport.width, viewport.height);

if(INVITE_CODE){
  casper.test.begin('Join', 2, function(test) {

    casper.waitForSelector(join.selectManualSignup, function(){
      join.clickManualSignupButton();
    });

    casper.waitForSelector(join.signupForm, function(){
      test.assert(join.isDisplayed(), 'Join form is displayed');
      join.signup('FirstName', 'LastName', PASSWORD);
    });

    var signUpMsg = 'Submitted signup form';
    casper.waitWhileVisible(login.loginForm, function(){
      test.assert(true, signUpMsg);
    }, function timeout(){
      test.fail(signUpMsg);
    });

    casper.run(function() {
      test.done();
    });
  });
}
else
{
  casper.test.begin('Login', 1, function(test) {

    casper.waitForSelector(login.loginForm, function(){
      test.assert(login.isDisplayed(), 'Login form is displayed');
      login.setUsername(USERNAME);
      login.setPassword(PASSWORD);
      login.login();
    });

    casper.run(function() {
      test.done();
    });
  });
}
