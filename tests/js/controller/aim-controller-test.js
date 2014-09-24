define(
  [
    'angular-mocks',
    'datarobot',
    'lodash',
    'DatasetServiceResponse',
    'ProjectServiceResponse',
    'js/controller/eda-controller.min'
  ],
  function(angularMocks, datarobot, _, DatasetServiceResponse, ProjectServiceResponse){

    describe('AimController', function(){

      beforeEach(module('datarobot'));

      // This is needed since the controller does not store this data in the scope, only in the service
      var scope, dsService, projectService, tally, assert;

      beforeEach(inject(function($controller, $rootScope, DatasetService, ProjectService){
        scope = $rootScope.$new();
        $controller('AimController', { $scope: scope});
        dsService = DatasetService;
        projectService = ProjectService;
        dsService.metricsList = DatasetServiceResponse.getMetrics();
      }));

      var setTargetFeature = function(key){
        return _.find(DatasetServiceResponse.aimFeaturesReady(), {name:key});
      };

      var defaultAdvancedFeature = {
        'cv_method' : {
          class : "options-random",
          enabled : true,
          name : "Random",
          overview : "",
          restrict : "none",
          template : "cv-method-random.html",
          value : "RandomCV",
        },
        'errors' : {},
        'holdout_pct':20,
        'randomseed':0,
        'reps':5,
        'time_holdout_pct':20,
        'time_training_pct':64,
        'time_validation_pct':16,
        'training_pct':64,
        'userValidationMethod':null,
        'validationMethod':"CV",
        'validation_pct':16
      };

      describe('Permissions Validation', function(){
        it('should be false if project or permissions are not defined', function(){
          projectService.project = null;
          expect(scope.getPermissions('DOES_NOT_MATTER')).toBeFalsy();
        });

        it('should be false if project or permissions are not defined', function(){
          projectService.project = null;
          expect(scope.getPermissions('DOES_NOT_MATTER')).toBeFalsy();
        });

        it('should be able to run models only if user has permissions', function(){
          projectService.project = null;
          expect(scope.canRunModels()).toBeFalsy();
        });

        it('should run models only if the user has permissions', function(){

          projectService.project = null;
          scope.runModels(1, 'does-not-matter-metric');
          expect(scope.targetError).toMatch('You do not have permission');

          // fails for another reason, not permissions
          projectService.project = {
            permissions:{
              CAN_SET_TARGET : true
            }
          };
          scope.runModels(1, 'does-not-matter-metric');
          expect(scope.targetError).not.toMatch('You do not have permission');
        });
      });

      describe('Target variable validation', function(){

        it('isValidTarget() should return false if the target variable has not been selected yet', function(){
          scope.targetFeature = null;
          expect(scope.isValidTarget()).toBe(false);
          expect(scope.targetError).toBeFalsy();
        });

        it('should return false if the EDA has not arrived yet', function(){
          // setting target to 'IsBadBuy' before eda is ready
          scope.targetFeature = DatasetServiceResponse.aimFeaturesNotReady()[1];
          expect(scope.isValidTarget()).toBe(false);
          expect(scope.targetError).toMatch('wait for the EDA');
        });

        it('should not allow low info variables to be selected as the target', function(){
          scope.targetFeature = setTargetFeature('IsOnlineSale');
          expect(scope.isValidTarget()).toBe(false);
          expect(scope.targetError).toMatch('few values');
        });

        it('should not allow variables with high cardinality', function(){
          scope.targetFeature = setTargetFeature('high_cardinality');
          expect(scope.isValidTarget()).toBe(false);
          expect(scope.targetError).toMatch('too many classes');
        });

        it('should not allow an empty feature to be set as target', function(){
          scope.targetFeature = setTargetFeature('empty');
          expect(scope.isValidTarget()).toBe(false);
          expect(scope.targetError).not.toMatch('empty');
        });

        it('should not allow variables with low_info', function(){
          scope.targetFeature = setTargetFeature('few_values');
          expect(scope.isValidTarget()).toBe(false);
          expect(scope.targetError).toMatch('cannot be used');
          expect(scope.targetError).not.toMatch('few values|too many classes');
        });

      });

      describe('Advanced Options Input Validation: balance percentage levels for TVH', function(){

        var setTestDefaults = function(){
          scope.advancedFeature = JSON.parse( JSON.stringify( defaultAdvancedFeature ) );
          scope.advancedFeature.validationMethod = "TVH";
        };

        it('adjusting holdout_pct should balance the percentages subtracting from training_pct first', function(){
          setTestDefaults();
          // user changes holdout pct to 40,
          //   training_pct should = 44
          //   validation_pct should remain at 16
          scope.advancedFeature.holdout_pct = 44;
          scope.balancePercentages('hpct');
          expect(scope.advancedFeature.training_pct).toBe(40);
          expect(scope.advancedFeature.validation_pct).toBe(16);
          tally = scope.advancedFeature.validation_pct + scope.advancedFeature.training_pct + scope.advancedFeature.holdout_pct;
          expect(tally).toBe(100);
        });

        it('adjusting holdout_pct should balance the percentages subtracting from training_pct first until it is a min of 1', function(){
          setTestDefaults();
          // user changes holdout pct to 98,
          //   training_pct should equal 1
          //   validation_pct should equal 1
          scope.advancedFeature.holdout_pct = 98;
          scope.balancePercentages('hpct');
          expect(scope.advancedFeature.training_pct).toBe(1);
          expect(scope.advancedFeature.validation_pct).toBe(1);
          tally = scope.advancedFeature.validation_pct + scope.advancedFeature.training_pct + scope.advancedFeature.holdout_pct;
          expect(tally).toBe(100);
        });

        it('adjusting holdout_pct to 0 should set the training size to 60 when the user sets vpct to 40 first ', function(){
          setTestDefaults();
          // user changes validation_pct to 40,
          //   training_pct should equal 40
          //   holdout_pct should remain at 20
          scope.advancedFeature.validation_pct = 40;
          scope.balancePercentages('vpct');
          expect(scope.advancedFeature.training_pct).toBe(40);
          expect(scope.advancedFeature.holdout_pct).toBe(20);
          tally = scope.advancedFeature.validation_pct + scope.advancedFeature.training_pct + scope.advancedFeature.holdout_pct;
          expect(tally).toBe(100);

          // user then changes holdout pct to 0,
          //   validation_pct should remain at 40
          //   training_pct should readjust to 60
          scope.advancedFeature.holdout_pct = 0;
          scope.balancePercentages('hpct');
          expect(scope.advancedFeature.training_pct).toBe(60);
          expect(scope.advancedFeature.validation_pct).toBe(40);
          tally = scope.advancedFeature.validation_pct + scope.advancedFeature.training_pct + scope.advancedFeature.holdout_pct;
          expect(tally).toBe(100);
        });

      });

      describe('Advanced Options Input Validation: Holdout percentage :', function(){

        var setTestDefaults = function(){
          scope.advancedFeature = JSON.parse( JSON.stringify( defaultAdvancedFeature ) );
          scope.advancedFeature.validationMethod = "TVH";
        };

        it('Strings are not allowed', function(){
          setTestDefaults();
          scope.advancedFeature.holdout_pct = 'lol';
          scope.validate_holdout_pct();
          expect(scope.advancedFeature.errors.holdout_pct).toBe(true);
        });

        it('Floats are not allowed', function(){
          setTestDefaults();
          scope.advancedFeature.holdout_pct = 5.5;
          scope.validate_holdout_pct();
          expect(scope.advancedFeature.errors.holdout_pct).toBe(true);
        });

        it('100% is not allowed', function(){
          setTestDefaults();
          scope.advancedFeature.holdout_pct = 100;
          scope.validate_holdout_pct();
          expect(scope.advancedFeature.errors.holdout_pct).toBe(true);
        });

        it('When valid, re-balance Validation/Training pct (when specified)', function(){
          setTestDefaults();
          scope.advancedFeature.holdout_pct = 44;
          // passing true will trigger the call to balancePercentages()
          scope.validate_holdout_pct(true);
          expect(scope.advancedFeature.errors.holdout_pct).toBeFalsy();
          expect(scope.advancedFeature.training_pct).toBe(40);
          expect(scope.advancedFeature.validation_pct).toBe(16);
          tally = scope.advancedFeature.validation_pct + scope.advancedFeature.training_pct + scope.advancedFeature.holdout_pct;
          expect(tally).toBe(100);
        });
      });

      describe('Advanced Options Input Validation: Validation percentage :', function(){

        var setTestDefaults = function(){
          scope.advancedFeature = JSON.parse( JSON.stringify( defaultAdvancedFeature ) );
          scope.advancedFeature.validationMethod = "TVH";
        };

        it('Strings are not allowed', function(){
          setTestDefaults();
          scope.advancedFeature.validation_pct = 'lol';
          scope.validate_validation_pct();
          expect(scope.advancedFeature.errors.validation_pct).toBe(true);
        });

        it('Floats are not allowed', function(){
          setTestDefaults();
          scope.advancedFeature.validation_pct = 5.5;
          scope.validate_validation_pct();
          expect(scope.advancedFeature.errors.validation_pct).toBe(true);
        });

        it('100% is not allowed', function(){
          setTestDefaults();
          scope.advancedFeature.validation_pct = 100;
          scope.validate_validation_pct();
          expect(scope.advancedFeature.errors.validation_pct).toBe(true);
        });

        it('0% is not allowed', function(){
          setTestDefaults();
          scope.advancedFeature.validation_pct = 0;
          scope.validate_validation_pct();
          expect(scope.advancedFeature.errors.validation_pct).toBe(true);
        });

        it('When valid, re-balance Holdout/Training pct (when specified)', function(){
          setTestDefaults();
          scope.advancedFeature.validation_pct = 60;
          // passing true will trigger the call to balancePercentages()
          scope.validate_validation_pct(true);
          expect(scope.advancedFeature.errors.validation_pct).toBeFalsy();
          expect(scope.advancedFeature.training_pct).toBe(20);
          expect(scope.advancedFeature.holdout_pct).toBe(20);
          tally = scope.advancedFeature.validation_pct + scope.advancedFeature.training_pct + scope.advancedFeature.holdout_pct;
          expect(tally).toBe(100);
        });
      });

      describe('Advanced Options Input Validation: DateCV | Datetime_col :', function(){
        // Value must be a valid column in the feature list
        // Value must be Numeric or Date
        // Value must not be the same as the target feature
        var setTestDefaults = function(){
          // Mock Project and Initial Aim Parameters
          projectService.project = ProjectServiceResponse.newProjectStageAim();
          scope.features         = DatasetServiceResponse.aimFeaturesReady();
          scope.advancedFeature  = JSON.parse( JSON.stringify( defaultAdvancedFeature ) );
        };

        it('Value must be in the featurelist', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "wanker";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBe(true);
        });

        // Ensure expected values do not return an error
        it('TypeOf Date(D) is Valid', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "PurchDate";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBeFalsy();
        });

        it('TypeOf Numeric(N) is Valid', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "BYRNO";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBeFalsy();
        });

        it('TypeOf Time(T) is Valid', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "TestTime";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBeFalsy();
        });

        // Ensure bad types return an error

        it('TypeOf Cat(C) is Invalid', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "Make";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBe(true);
          expect(scope.advancedFeature.errorMsg.datetime_col).toBeTruthy();
        });

        it('TypeOf Text(X) is Invalid', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "TestText";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBe(true);
          expect(scope.advancedFeature.errorMsg.datetime_col).toBeTruthy();
        });

        it('TypeOf Percent(P) is Invalid', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "TestPercent";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBe(true);
          expect(scope.advancedFeature.errorMsg.datetime_col).toBeTruthy();
        });

        it('TypeOf Percent(P) is Invalid', function(){
          setTestDefaults();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.advancedFeature.datetime_col = "TestLength";
          scope.validate_datetime_col();
          expect(scope.advancedFeature.errors.datetime_col).toBe(true);
          expect(scope.advancedFeature.errorMsg.datetime_col).toBeTruthy();
        });

      });

      describe('Advanced Options CV Type Menu : Stratified CV :', function(){
        var setTestDefaults = function(){
          scope.advancedFeature = JSON.parse( JSON.stringify( defaultAdvancedFeature ) );
        };

        it('Should be available for classification problems', function(){
          // currently defined as target features that have 2 unique values
          setTestDefaults();
          // set target to 'IsBadBuy' where unique = 2
          // aim-controller watches $scope.selections.targetFeatureName and when changed
          // we set scope.targetFeature and run $scope.setAvailableCVMethods();
          scope.targetFeature = setTargetFeature('IsBadBuy');
          scope.setAvailableCVMethods();
          assert = _.find(scope.availableCVMethods, {value:'StratifiedCV'});
          expect(assert.value).toBe('StratifiedCV');
        });

        it('Should NOT be available for regression problems', function(){
          // currently defined as target features whose unique values are > 2
          setTestDefaults();
          // set target to 'VehYear' where unique = 9
          // aim-controller watches $scope.selections.targetFeatureName and when changed
          // we set scope.targetFeature and run $scope.setAvailableCVMethods();
          scope.targetFeature = setTargetFeature('VehYear');
          scope.setAvailableCVMethods();
          assert = _.find(scope.availableCVMethods, {value:'StratifiedCV'});
          expect(assert).toBeFalsy();
        });

      });

      describe('Advanced Options: Start Project using RandomCV & Cross Validation :', function(){

        var setTestDefaults = function(){
          // Mock Project
          projectService.project = ProjectServiceResponse.newProjectStageAim();
          // set target to 'IsBadBuy'
          scope.targetFeature = setTargetFeature('IsBadBuy');
          // set advanced options defaults
          scope.advancedFeature = JSON.parse( JSON.stringify( defaultAdvancedFeature ) );
        };

        it('Cannot submit target when advanced options are invalid', function(){
          setTestDefaults();
          // set invalid values for project
          scope.advancedFeature.holdout_pct = 99;
          // press start button
          assert = scope.runModels(1,scope.targetFeature.selectedMetric);
          expect(scope.advancedFeature.errors).toBeTruthy();
          expect(scope.submittingTarget).toBeFalsy();
          expect(assert).toBe(false);
        });

        it('Verify Post Data', function(){
          setTestDefaults();
          // set valid values for project
          scope.advancedFeature.reps=10;
          // press start button
          assert = scope.runModels(1,scope.targetFeature.selectedMetric);
          expect(scope.advancedFeature.errors).toEqual({});
          expect(scope.submittingTarget).toBe(true);

          // Run assertions on the outbound post data
          // Do not directly compare the objects as payload may change, however
          // the following keys are mission critical

          expect(assert.metric).toEqual('RMSE');
          expect(assert.target).toEqual('IsBadBuy');
          expect(assert.mode).toEqual(1);
          expect(assert.cv_method).toEqual('RandomCV');
          expect(assert.holdout_pct).toEqual(20);
          expect(assert.validation_type).toEqual('CV');
          expect(assert.reps).toEqual(10);
          expect(assert.validation_pct).toBeFalsy();
        });

      });

      describe('Advanced Options: Start Project using RandomCV & Train/Validate/Holdout Split :', function(){

        var setTestDefaults = function(){
          // Mock Project
          projectService.project = ProjectServiceResponse.newProjectStageAim();
          // set target to 'IsBadBuy'
          scope.targetFeature = setTargetFeature('IsBadBuy');
          // set advanced options defaults
          scope.advancedFeature = JSON.parse( JSON.stringify( defaultAdvancedFeature ) );
        };

        it('Cannot submit target when advanced options are invalid', function(){
          setTestDefaults();
          // Set CV Method
          scope.advancedFeature.validationMethod = "TVH";
          // set invalid values for project
          scope.advancedFeature.validation_pct = 100;
          // press start button
          assert = scope.runModels(1,scope.targetFeature.selectedMetric);
          expect(scope.advancedFeature.errors).toBeTruthy();
          expect(scope.submittingTarget).toBeFalsy();
          expect(assert).toBe(false);
        });

        it('Verify Post Data', function(){
          setTestDefaults();

          // set valid values for project
          scope.advancedFeature.validationMethod = "TVH";
          scope.advancedFeature.validation_pct = 40;
          scope.advancedFeature.holdout_pct=10;

          // press start button
          assert = scope.runModels(1,scope.targetFeature.selectedMetric);
          expect(scope.advancedFeature.errors).toEqual({});
          expect(scope.submittingTarget).toBe(true);

          // Run assertions on the outbound post data
          // Do not directly compare the objects as payload may change, however
          // the following keys are mission critical

          expect(assert.metric).toEqual('RMSE');
          expect(assert.target).toEqual('IsBadBuy');
          expect(assert.mode).toEqual(1);
          expect(assert.cv_method).toEqual('RandomCV');
          expect(assert.holdout_pct).toEqual(10);
          expect(assert.validation_pct).toEqual(40);
          expect(assert.validation_type).toEqual('TVH');
          expect(assert.reps).toBeFalsy();
        });
      });

    });
});
