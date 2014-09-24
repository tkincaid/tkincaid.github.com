var InsightsTests = function(){

  casper.test.begin('Insights', 4, function(test){
    main.selectInsightsTab();

    var insights = new Insights();

    casper.waitForSelector(insights.view, function(){
      test.assert(true, 'Insights View loaded');
    });

    var importanceMsg = 'Importance chart preview displayed';
    casper.waitForSelector(insights.importanceChartPreview,
      function then(){
      test.assert(true, importanceMsg);
    }, function timeout(){
      capture('insights-no-chart-preview.png', importanceMsg);
      test.fail(importanceMsg);
    }, 30000);

    var fullImportanceMsg = 'Full Importance chart displayed';
    casper.then(function(){
      insights.expandImportanceChart();
      casper.waitForSelector(insights.importanceChartFull, function(){
        test.assert(insights.isImportanceChartDisplayed(), fullImportanceMsg);
      }, function timeout(){
        capture('insights-full-importance-chart.png', fullImportanceMsg);
        test.fail(fullImportanceMsg);
      });
    });

    var insightsGridMsg = 'Back to Insights grid';
    casper.then(function(){
      insights.backToMenu();
      casper.waitForSelector(insights.insightsGrid, function(){
        test.assert(insights.isDisplayed(), insightsGridMsg);
      }, function timeout(){
        capture('back-to-insights-grid.png', insightsGridMsg);
        test.fail(insightsGridMsg);
      });
    });

    casper.run(function() {
      test.done();
    });
  });
};
module.exports = InsightsTests;
