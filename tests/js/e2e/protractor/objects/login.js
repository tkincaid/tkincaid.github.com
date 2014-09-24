var LoginPage = function() {

  this.username = by.model('user.email');
  this.password = by.model('user.password');
  this.loginButton = by.id('login-button');
  this.loginForm = by.id('login_form');

  this.setUsername = function(name) {
    element(this.username).sendKeys(name);
  };

  this.setPassword = function(password) {
    element(this.password).sendKeys(password);
  };

  this.login = function(){
    element(this.loginButton).click();
  };

  this.isPresent = function(){
    element(this.loginForm).isPresent();
  };

  this.get = function(url) {
    browser.get(url);
  };
};

module.exports = LoginPage;