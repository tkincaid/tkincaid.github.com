/*
This object interacts with a page that does not run angular.
For this reason, the underlying selenium webdriver is used instead
*/

var Join = function(){
  this.email = by.id('email');
  this.password = by.id('password');
  this.passwordConfirmation = by.id('password-confirmation');
  this.TermsOfService = by.id('toscb');

  this.get = function(baseUrl, username, inviteCode){
    this.baseUrl = baseUrl;
    var joinUrl = baseUrl + '/join?email=' + username + '&code=' + inviteCode;
    console.log('Joining at: ' + joinUrl);
    browser.driver.get(joinUrl);
  };

  this.signup = function(password){
    browser.driver.findElement(this.password).sendKeys(password);
    browser.driver.findElement(this.passwordConfirmation).sendKeys(password);

    var tosCheckbox = browser.driver.findElement(this.TermsOfService);
    tosCheckbox.click();
    tosCheckbox.submit();
  };

  this.logout = function(){
    browser.driver.get(this.baseUrl + '/account/logout');
  };
};

module.exports = Join;