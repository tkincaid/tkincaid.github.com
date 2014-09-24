casper.test.begin('Login/logout', 11, function(test) {
  casper.waitForSelector(main.userMenu);

  casper.then(function(){
    main.logout();
    test.assert(true, 'Select logout in user menu');
  });

  casper.waitForSelector(login.loginForm, function(){
    test.assert(login.isDisplayed(), 'Login form is displayed');   
  });

  casper.then(function() {
    //access reset password page
    login.forgotPassword();
  });

  //ensure it is displayed
  casper.waitForSelector(login.forgotPasswordForm, function(){
    test.assert(login.isForgotPasswordDisplayed(), 'Forgot Password form Showing');
  });

  //input initial invalid email
  casper.then(function(){
    login.setForgotPasswordEmail("asdf");
    casper.wait(10);
  });
  //error should not show until they fail a submit
  casper.then(function(){
    test.assert(!login.isErrorDisplayed(), "Error not shown until submit is clicked");
  });

  //submit an empty field
  casper.then(function(){
    login.emptyForgotPasswordEmail();
    login.submitForgotPassword();
  });

  //check correct error message
  casper.waitForSelector(login.errorMessage, function(){
    test.assert(login.isErrorDisplayed(), "Error is shown");
    test.assertSelectorHasText(login.errorMessage, "Email is required", "Email required text shown on empty field");
  });
  //input invalid email
  casper.then(function(){
    login.setForgotPasswordEmail("a");
    casper.wait(10);
  });
  //check new error message
  casper.then(function() {
    test.assertSelectorHasText(login.errorMessage, "Please enter a valid Email address");
  });

  //check valid email removing error
  casper.then(function(){
    login.setForgotPasswordEmail("sdf@asdf.com");
    casper.wait(10);
    test.assertNotVisible(login.errorMessage, "Error message was correctly removed");
  });

  //check error showing one last time
  casper.then(function(){
    login.emptyForgotPasswordEmail();
    casper.wait(10); 
    test.assertSelectorHasText(login.errorMessage, "Email is required", "Email required text shown on empty field");
  });
  //leave forgot password page
  casper.then(function(){
    login.cancelForgotPassword();
  });

  casper.waitForSelector(login.loginForm, function(){
    test.assert(login.isDisplayed(), 'Login form is displayed');   
  });

  casper.then(function() {
    login.setUsername(USERNAME);
    login.setPassword(PASSWORD);
    login.login();
  });

  casper.waitForSelector(eda.view, function(){
    test.assert(eda.isDisplayed(), 'Login succeeded, EDA is displayed');
    main.logout();
  });

  casper.run(function() {
    test.done();
  });
});