var Join = function(){
  this.email = '#email';
  this.termsOfService = '#toscb';
  this.register = 'button.login-button';
  this.signupForm = 'form[name="signupForm"]';
  this.password = '#password';
  this.firstName = '#first-name';
  this.lastName = '#last-name';
  this.passwordConfirmation = '#password-confirmation';
  this.selectManualSignup = '.login-button';

  this.getUrl = function(baseUrl, username, inviteCode){
    this.baseUrl = baseUrl;
    var joinUrl = baseUrl + '/join?email=' + username + '&code=' + inviteCode;
    casper.echo('Joining at: ' + joinUrl);
    return joinUrl;
  };

  this.isDisplayed = function(){
    return casper.visible(this.signupForm);
  };

  this.clickManualSignupButton = function(){
    casper.click(this.selectManualSignup);
  };

  this.signup = function(firstName, lastName, password){
    casper.sendKeys(this.firstName, firstName);
    casper.sendKeys(this.lastName, lastName);
    casper.sendKeys(this.password, password);
    casper.sendKeys(this.passwordConfirmation, password);
    casper.click(this.register);
  };
};

module.exports = Join;