var EDATests = function(){
  var isTargetSelected;

  this.features = function(){
    casper.test.begin('EDA Features', 4, function(test) {

      casper.waitForSelector(eda.view, function() {
        test.assert(eda.isDisplayed(), 'EDA is displayed');
      });

      var msg = 'Vartypes are present';
      casper.waitForSelector(eda.varTypes, function() {
        var varTypes = eda.getVarTypes();
        test.assert(varTypes.length > 0, msg);
      }, function timeout(){
          capture('no-var-types.png');
          test.fail(msg);
      });

      var rowCount;
      var rowIndex = 0;
      var graphMsg = 'EDA rows are expanded on click and graph displays';
      var edaRowsMsg = 'EDA rows displayed';

      casper.then(function(){
        casper.waitForSelector(eda.edaRowsLoaded, function() {
          rowCount = eda.getEdaRows().length;
          test.assert(rowCount > 0, edaRowsMsg + ': ' + rowCount);
        }, function timeout(){
            capture('no-eda-rows.png');
            test.fail(edaRowsMsg);
        }, 30000);
      });

      casper.waitFor(function(){
        if(!rowCount){
          return false;
        }
        // The row can't expand if the plot has not been received
        // Iterate on columns since some of them may not have EDA info
        // such as ref id columns
        try {
          eda.expandFeatureAt(rowIndex % rowCount);
          return eda.isGraphDisplayed();
        }
        catch(e){ }
        finally {
            rowIndex ++;
        }
      }, function then(){
        test.assert(true, graphMsg);
      }, function timeout(){
        capture('eda-in-row-graph.png');
        test.fail(graphMsg);
      }, 30000);

      casper.run(function() {
        test.done();
      });
    });
  };

  this.selectTarget = function(targetVariable){
    casper.test.begin('EDA Target Variable', 2, function(test) {

      casper.then(function(){
         isTargetSelected = eda.isTargetSelected();
      });

      casper.then(function(){
        if(isTargetSelected){
          test.assert(!eda.isTargetVariableDisplayed(), 'Select target variable is not displayed');
        }
        else{

          // Click back to see target variable input
          casper.then(function(){
             casper.click(eda.returnToAim);
          });

          casper.waitForSelector(eda.targetVariable, function() {
            test.assert(eda.isTargetVariableDisplayed(), 'Select target variable is displayed');
          }, function timeout(){
            capture('no-target-input.png');
            test.fail('unable to set target');
          },10000);

          casper.then(function(){
            eda.setTargetFeature(targetVariable);
          });

        }
      });

      casper.then(function(){
        if(isTargetSelected){
          test.skip(1, 'Target already selected: Skipping target chart test');
        }
        else{
          test.assert(eda.isTargetHistogramDisplayed(), 'Chart shows when you select a target');
        }
      });

      casper.run(function() {
        test.done();
      });
    });
  };


  this.runModels = function(){
    casper.test.begin('EDA Push Start', 1, function(test) {
      casper.then(function(){

        if(isTargetSelected){
          test.skip(1, 'Target already selected: Skipping run models test');
        }
        else{

          eda.setRankingMetric();
          eda.selectSemiAutoModel();

          // 6/1/14 Step no longer required but may come back
          // eda.runModels();

          casper.waitWhileSelector(eda.targetVariable, function(){
            test.assertNot(eda.isTargetVariableDisplayed(), 'Selected target, metric and started models (target variable is hidden)');
          });
        }

      });

      casper.run(function() {
        test.done();
      });
    });
  };
};

module.exports = EDATests;