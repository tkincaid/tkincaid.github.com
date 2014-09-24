var LeaderboardTests = function(){
  var defaultSampleSize = '64';
  var defaultBlender = {
    label: 'Average Blend',
    title: 'AVG Blender'
  };
  var predictions;
  var lastPrediction;
  var currentMetric;
  // Sorting direction by metric (true means higher is better)
  var metrics = {
    AUC: true,
    LogLoss: false
  };

  var scoreTypes = {
    SINGLE: 0,
    CV: 1,
    HOLDOUT : 2
  };

  var randomIntFromInterval = function(min,max){
      return Math.floor(Math.random()*(max-min+1)+min);
  };

  var testLeaderboardSorting = function(test, scoreType){

    var verifySorting = function(metric){
      var modelScores;

      switch (scoreType){
      case scoreTypes.SINGLE:
        modelScores = models.getScores();
        break;
      case scoreTypes.CV:
        modelScores = models.getAvgScores();
        break;
      case scoreTypes.HOLDOUT:
        modelScores = models.getHoldoutScores();
        break;
      }

      var firstScore = modelScores.shift();
      var lastScore = modelScores.pop();
      var testMsg = 'Sorting works correctly for: ' + metric;
      var sortingWorks;
      if(metrics[metric]){
        sortingWorks = firstScore >= lastScore;
      }
      else{
        sortingWorks = firstScore <= lastScore;
      }

      if(!sortingWorks){
        captureAndWarn('models-sorting-' + metric + '.png',
          'Sorting faield for ' + metric + ' with ['+ modelScores.join() +
          ']. First:'+ firstScore +'  Last: ' + lastScore );
      }
      test.assert(sortingWorks, testMsg);
    };

    var changeMetric = function(metric){
      casper.then(function(){
        models.openMetricMenu();
      });

      casper.waitForSelector(models.metricMenu, function(){
        models.setMetric(metric);
      });

      casper.then(function(){
        var actualMetric = models.getMetric();
        casper.echo('Actual metric: ' + actualMetric);
        test.assert(metric === actualMetric, 'Change metric to: ' + metric);
        currentMetric = metric;
      });
    };

    casper.then(function(){
      changeMetric('AUC');
    });

    casper.then(function(){
      verifySorting(currentMetric);
    });

    casper.then(function(){
      changeMetric('LogLoss');
    });

    casper.then(function(){
      verifySorting(currentMetric);
    });
  };


  var check_top_model = function(test) {
    var topModelId;
    var modelTitle;

    main.selectModelsTab();

    casper.waitForSelector(models.allModels, function() {
      test.assert(true, 'Leaderboard is displayed');
    }, function timeout(){
      capture('leaderboard-view.png');
      test.fail('Leaderboard view did not load on time');
    });

    var msgTopModel = 'At least one model finished successfully';
    casper.waitForSelector(models.finishedModels, function() {
      topModelId = models.getTopModelId();
      modelTitle = models.getModelTitle(topModelId);
      test.assertTruthy(topModelId, msgTopModel);
    }, function timeout(){
      capture('no-top-model.png');
      test.fail(msgTopModel);
    });

    casper.then(function(){
      models.expandModel(topModelId);
    });


    casper.waitForSelector(models.blueprint, function(){
      //TODO: Test more tabs based on the model type, e.g., grid search, ROC curve
      test.assert(models.isBlueprintDisplayed(topModelId), 'Blueprint displayed');
    });

    casper.waitForSelector(models.modelInfoTab, function(){
      models.selectModelInfoTab();
      test.assert(models.isModelInfoDisplayed(topModelId), 'Model info displayed');

      models.selectLiftChartTab();
      test.assert(models.isLiftChartDisplayed(topModelId), 'Lift chart displayed');

      models.selectLiftTableTab();
      test.assert(models.isLiftTableDisplayed(topModelId), 'Lift table displayed');

      models.selectPredictTab(topModelId);
      test.assert(true, 'Predict displayed');
    });

    casper.run(function() {
      test.done();
    });
  };

  this.just_top_model_ui = function(){
    casper.test.begin('Leaderboard: Top model', 7, check_top_model);
  };

  this.basic = function(){
    casper.test.begin('Leaderboard: Top model', 7, check_top_model);
    casper.test.begin('Leaderboard: Run models', 3, function(test){
      var topModelId;
      var modelTitle;
      var newSampleSize = randomIntFromInterval(50, 63);

      casper.then(function(){
        topModelId = models.getTopModelId();
        modelTitle = models.getModelTitle(topModelId);
      });

      // Run at different sample size
      casper.then(function(){
        newSampleSize = String(newSampleSize);
        casper.echo('Running top model '+ modelTitle + ' with new sample size: ' + newSampleSize);
        models.submitSampleSize(topModelId, newSampleSize);
      });
      casper.then(function(){
        casper.waitForSelector(queue.selector(modelTitle, newSampleSize), function (){
          test.assert(true, 'Sample size run: Model "' + modelTitle + '" (' + newSampleSize + ') in queue');
        }, function timeout(){
          captureAndWarn('run-model-new-sample-size.png', 'Not found in queue');
          test.fail('Sample size run failed');
        });
      });

      // Run on new featurelist
      casper.then(function(){
        models.openFeatureListMenu(topModelId);
      });
      //Wait for the menu animation
      casper.waitForSelector(models.featureListMenu, function(){
        var featureList = 'Raw Features';
        casper.echo('Running top model '+ modelTitle + ' with feature list: ' + featureList);
        models.runFeatureList(featureList);
      });
      casper.then(function(){
        casper.waitForSelector(queue.selector(modelTitle, newSampleSize), function (){
          test.assert(true, 'Feature list run: Model "' + modelTitle + '" (' + newSampleSize + ') in queue');
        }, function timeout(){
          captureAndWarn('run-model-feature-list.png', 'Not found in queue');
          test.fail('Feature list run failed');
        });
      });

      casper.waitFor(function(){
        var leaderboard = models.getLeaderboard();
        return leaderboard.length >=2;
      });

      var selectedModels = [];
      var selectedIds;

      casper.then(function(){
        selectedIds = models.selectModelsByIndex([0,1]);
      });

      casper.then(function(){
        for (var i = 0; i < selectedIds.length; i++) {
          var id = selectedIds[i];
          var title = models.getModelTitle(id);
          selectedModels.push({
            id : id,
            title : title
          });
          casper.echo('Model '+ i +' selected: ' + title + '('+ id +')');
        }
      });

      casper.then(function(){
        models.openLeaderboardMenu();
      });

      //Wait for the menu animation
      casper.waitForSelector(models.leaderboardMenu, function(){
        models.openRunBatchForm();
      });

      casper.then(function(){
        newSampleSize = randomIntFromInterval(50, 63);
        newSampleSize = String(newSampleSize);
        casper.echo('Running models in batch with sample size: ' + newSampleSize);
      });

      casper.waitForSelector(models.batchSampleSizeInput, function(){
        models.runSelectedModels(newSampleSize);
      }, function timeout(){
        captureAndDie('leaderboard-batch-run.png', 'Could not submit batch run');
      });

      casper.then(function(){
        casper.waitForSelector(queue.multipleSelector(selectedModels, newSampleSize), function (){
          test.assert(true, 'Batch Run: models with sample size ' + newSampleSize + ' found in queue');
        },function (){
          capture('run-model-in-batch.png');
          test.fail('Batch Run: Models not found in queue');
        });
      });

      casper.run(function() {
        test.done();
      });
    });
  };

  this.advanced = function(){
    // Helper functions that facilitate code reuse between top model and
    // blender predictions

    var downloadPrediction =  function(test, modelId, index){
        var downloadMsg = 'Downloaded Prediction: ' + index;
        var predictionUrl = models.downloadPrediction(modelId, index);
        casper.waitForResource(function testResource(resource) {
            return resource.url.match(predictionUrl);
        }, function then(){
          test.assert(true, downloadMsg);
        }, function timeout(){
          capture('download-prediction.png');
          test.fail(downloadMsg);
        });
      };

    var predictOnNewDataSet = function(test, modelId){
      casper.then(function(){
        models.computePrediction(modelId, lastPrediction);
      });

      casper.waitFor(function(){
        return models.isPredictionInProcess(modelId, lastPrediction);
      }, function then(){
        test.assert(true,'New prediction is now in process');
      });

      casper.waitFor(function(){
        return models.isPredictionReady(modelId, lastPrediction);
      }, function then(){

      }, function timeout(){
        var msg = 'Prediction job for model '+ modelId +' did not finish on time (index '+ lastPrediction +')';
        capture('prediction-not-ready.png',msg);
        test.fail(msg);
      }, 30000);

      casper.then(function(){
        downloadPrediction(test, modelId, lastPrediction);
        test.assert(true, 'New prediction downloaded successfully');
      });
    };

    casper.test.begin('Leaderboard: Learning curve', 1, function(test){

      casper.then(function(){
        main.selectModelsTab();
      });

      casper.waitForSelector(models.allModels, function(){
        models.selectLearningCurveTab();
      });

      casper.waitForSelector(models.learningCurveGraph, function(){
        test.assert(true, 'Learning curve is displayed');
      }, function timeout(){
        capture('learning-curve.png');
        test.fail('Learning curve did not display on time');
      });

      casper.then(function(){
        models.selectLeaderboardTab();
      });

      casper.run(function() {
        test.done();
      });
    });

    casper.test.begin('Leaderboard: Top Model predictions', 7, function(test){
      var topModelId;

      casper.waitForSelector(models.allModels, function(){
        topModelId = models.getTopModelId();
        test.assertTruthy(topModelId, 'Top model selected');
      }, function timeout(){
        captureAndWarn('leaderboard.png', 'Leaderboard did not load on time');
      });

      casper.waitFor(function(){
        models.expandModel(topModelId);
        return models.isRowExpanded();
      }, function then(){
      }, function onTimeout(){
        captureAndWarn('lb-top-model-expand.png', 'Top model row did not expand');
      });

      casper.waitForSelector(models.blueprint, function(){
        models.selectPredictTab(topModelId);
      }, function timeout(){
        captureAndWarn('lb-top-model-blueprint.png', 'Top model blueprint did not load on time');
      });

      casper.then(function(){
        downloadPrediction(test, topModelId, 0);
      });

      casper.then(function(){
        predictions = models.getPredictions(topModelId);
        models.addDataset(topModelId);
      });

      casper.waitForSelector(models.uploadUrl, function(){
        models.uploadNewDataset(models.newDataset);
        lastPrediction = predictions.length;
      }, function timeout(){
        captureAndWarn('lb-top-model-new-dataset.png', 'Upload form did not load on time');
      });

      casper.then(function(){
        test.assert(lastPrediction > 0, 'New prediction uploaded');
      });

      var newUploadMsg = 'New prediction is ready to compute';
      casper.waitFor(function test(){
        return models.isReadyToComputePrediction(topModelId, lastPrediction);
      }, function  then(){
          test.assert(true, newUploadMsg);
      },function timeout(){
        capture('pred-new-upload-failed.png');
        test.fail(newUploadMsg);
      }, 10000);

      casper.then(function(){
        predictOnNewDataSet(test, topModelId);
      });

      casper.then(function(){
        //Close
        models.expandModel(topModelId);
      });

      casper.run(function() {
        test.done();
      });
    });

    casper.test.begin('Leaderboard: Run Avg CV from Batch', 2, function(test){

      casper.then(function(){
        queue.increaseWorkers(queue.MAX_WORKERS);
      });

      casper.waitFor(function(){
        var leaderboard = models.getLeaderboard();
        return leaderboard.length >=2;
      });

      var selectedModels = [];
      var selectedIds;

      casper.then(function(){
        selectedIds = models.selectModelsByIndex([0,1]);
      });

      casper.then(function(){
        for (var i = 0; i < selectedIds.length; i++) {
          var id = selectedIds[i];
          var title = models.getModelTitle(id);
          selectedModels.push({
            id : id,
            title : title
          });
          casper.echo('Model '+ i +' selected: ' + title + '('+ id +')');
        }
      });

      casper.then(function(){
        models.openLeaderboardMenu();
      });

      casper.waitForSelector(models.leaderboardMenu, function(){
        models.openRunBatchForm();
      });

      casper.echo('Running models in batch ( Avg CV ): ');

      var newSampleSize = 30;
      casper.waitForSelector(models.batchSampleSizeInput, function(){
        casper.fill('form#runTaskForm', {'max_reps':'all','input':newSampleSize}, false);
        casper.click(models.batchSubmit);
      }, function timeout(){
        captureAndDie('leaderboard-batch-run.png', 'Could not submit batch run');
      });

      casper.then(function(){
        casper.waitForSelector(queue.multipleSelector(selectedModels, newSampleSize), function (){
          test.assert(true, 'Batch Run: models with sample size ' + newSampleSize + ' found in queue');
        },function (){
          capture('run-model-in-batch.png');
          test.fail('Batch Run: Models not found in queue');
        });
      });

      casper.waitForSelector(queue.inProcess, function(){
        var queueitems = ( queue.inProgressCount() + queue.inQueuedCount() );
        casper.echo('Waiting for '+queueitems+' Models to Finish');

        casper.waitWhileVisible(queue.inProcess,function(){
          test.assertNot(casper.exists(queue.inProcess), 'Nothing is in-progress - Continue');
        },function timeout(){
          var msg = 'Timeout: waiting for in-progress items';
          capture('q-waiting-for-in-progress.png');
          test.fail(msg);
        }, 180000);
      });

      casper.run(function() {
        test.done();
      });

    });

    casper.test.begin('Leaderboard: Run Blender & Prediction', 6, function(test){
      var blenderId;

      casper.then(function(){
        queue.increaseWorkers(queue.MAX_WORKERS);
      });

      var modelIndexes = [0, 1];

      casper.then(function(){
        models.selectModelsByIndex(modelIndexes);
      });

      casper.then(function(){
        models.openLeaderboardMenu();
      });

      casper.wait(500, function(){
        models.blendSelectedModels(defaultBlender.label);
        test.assert(true, 'Blender submitted');
      });

      casper.then(function(){
        casper.waitForSelector(queue.selector(defaultBlender.title, defaultSampleSize), function (){
          test.assert(true, 'Blender "' + defaultBlender.title + '" (' + defaultSampleSize + '%) in queue');
        },  function timeout(){
          captureAndWarn('run-blender.png', 'Not found in queue');
          test.fail('Blender failed');
        });
      });

      casper.echo('Submitted blender, now waiting for it to complete (120 second timeout)');
      var blenderFinishedMsg = 'Blender model completed';
      casper.waitFor(function(){
        blenderId = models.getModelIdByTitle(defaultBlender.title);
        return blenderId;
      },function then(){
        test.assert(true, blenderFinishedMsg);
      },function timeout(){
        capture('run-blender.png');
        test.fail(blenderFinishedMsg);
      }, 120000);

      casper.then(function(){
        blenderId = models.getModelIdByTitle(defaultBlender.title);
        models.expandModel(blenderId);
        casper.waitForSelector(models.blueprint, function(){
          models.selectPredictTab(blenderId);
        });
      });

      casper.then(function(){
        predictOnNewDataSet(test, blenderId);
      });

      casper.then(function(){
        queue.decreaseWorkers(0);
      });

      casper.run(function() {
        test.done();
      });

    });

    casper.test.begin('Leaderboard: Run 1 of 5 Sorting', 4, function(test){

      casper.then(function(){
        testLeaderboardSorting(test, scoreTypes.SINGLE);
      });

      casper.run(function() {
        test.done();
      });
    });

    casper.test.begin('Leaderboard: CV Sorting', 4, function(test){

      casper.then(function(){
        models.sortByAvgScore();
      });

      casper.then(function(){
        testLeaderboardSorting(test, scoreTypes.CV);
      });

      casper.run(function() {
        test.done();
      });
    });

    this.removeItems = function(){
      casper.test.begin('Leaderboard: Remove models', 1, function(test){
        var selectedIds;

        casper.then(function(){
          selectedIds = models.selectModelsByIndex([2,3]);
        });

        casper.then(function(){
          models.openLeaderboardMenu();
        });

        //Wait for the menu animation
        casper.wait(500, function(){
          models.removeSelected();
        });

        //Wait for the menu animation
        casper.waitFor(function(){
          return !models.modelsExist(selectedIds);
        }, function onThen(){
          test.assert(true, 'Models removed ('+ selectedIds.length +')');
        }, function onTimeout(){
          capture('remove-model.png');
          test.fail('Models could not be removed: ' + selectedIds.join(','));
        });

        casper.run(function(){
          test.done();
        });
      });
    };
  };

  this.holdoutSorting = function(){

    casper.test.begin('Leaderboard: Holdout Sorting', 4, function(test){

      casper.then(function(){
        main.selectModelsTab();
      });

      casper.waitForSelector(models.holdoutSort, function(){
        models.sortByHoldout();
      });

      casper.then(function(){
        testLeaderboardSorting(test, scoreTypes.HOLDOUT);
      });

      casper.run(function() {
        test.done();
      });

    });
  };
};
module.exports = LeaderboardTests;
