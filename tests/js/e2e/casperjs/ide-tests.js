var IdeTests = function(){
  casper.test.begin('RStudio', 4, function(test){
    main.selectRStudioTab();
    var rstudio = new RStudio();
    casper.waitForSelector(rstudio.view, function() {
      test.assert(rstudio.isDisplayed(), 'RStudio visible');
      test.assert(rstudio.isFormHidden(), 'RStudio form hidden');
    });

    casper.withFrame(rstudio.frame, function() {
      // HACK: using appropriate test object because it contains the state
      // number of asserts: 2
      rstudio.frameLoading(test);
    });

    casper.run(function() {
      test.done();
    });
  });

  casper.test.begin('IPython', 4, function(test){
    main.selectIPythonTab();
    var ipython = new IPython();
    casper.waitForSelector(ipython.view, function() {
      test.assert(ipython.isDisplayed(), 'IPython visible');
      test.assert(ipython.isFormHidden(), 'IPython form hidden');
    });

    casper.withFrame(ipython.frame, function() {
      // HACK: using appropriate test object because it contains the state
      // number of asserts: 2
      ipython.frameLoading(test);
    });

    casper.run(function() {
      test.done();
    });
  });
};
module.exports = IdeTests;