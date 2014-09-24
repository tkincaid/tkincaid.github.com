var Insights = function(){

  this.view = '#insights-view';
  this.insightsGrid = '#insights_grid';
  this.importanceChartPreview = '#importance-chart-preview svg';
  this.importanceChartPreviewButton = '.ui-importance-chart-preview';
  this.importanceChartFull = '#importance-chart-full svg';
  this.backToChartGrid = '.backtomenu-button';


  this.isDisplayed = function(){
    return casper.visible(this.insightsGrid);
  };

  this.expandImportanceChart = function(){
    casper.click(this.importanceChartPreviewButton);
  };

  this.isImportanceChartDisplayed = function(){
    return casper.visible(this.importanceChartFull);
  };

  this.backToMenu = function(){
    return casper.click(this.backToChartGrid);
  };
};

module.exports = Insights;
