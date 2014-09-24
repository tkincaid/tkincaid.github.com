var LoginPage = function(baseUrl) {

  this.loginForm = 'form[name="login"]';
  this.username = 'input[name="username"]';
  this.password = 'input[name="userpassword"]';
  this.logginButton = '#login-button';
  this.errorMessage = '.error-msg.user-email';
  this.forgotPasswordEmail = '#forgot_username';
  this.forgotPasswordForm = '#login-forgot';
  this.forgotPasswordLink = '.ui-forgot-password-link';
  this.resetPasswordSubmitBtn = '#forgot_btn';
  this.resetPasswordCancel = '#forgot_cancel';



  this.setUsername = function(username) {
    casper.sendKeys(this.username, username);
  };

  this.setPassword = function(password) {
    casper.sendKeys(this.password, password);
  };

  this.setForgotPasswordEmail = function(email){
    casper.sendKeys(this.forgotPasswordEmail, email);
  }; 

  this.emptyForgotPasswordEmail = function(){
    casper.sendKeys(this.forgotPasswordEmail, '', {reset:true});
  };


  this.login = function(password) {
    casper.click(this.logginButton);
  };

  this.forgotPassword = function(){
    casper.click(this.forgotPasswordLink);
  };

  this.submitForgotPassword = function(){
    casper.click('#forgot_btn');
  };

  this.cancelForgotPassword = function(){
    casper.click(this.resetPasswordCancel);
  };

  this.isDisplayed = function() {
    return casper.visible(this.loginForm);
  };

  this.isErrorDisplayed = function(){
    return casper.visible(this.errorMessage);
  };

  this.isForgotPasswordDisplayed = function(){
    return casper.visible(this.forgotPasswordForm);
  };
};

module.exports = LoginPage;