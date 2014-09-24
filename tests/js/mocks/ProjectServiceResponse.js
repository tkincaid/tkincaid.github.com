define("ProjectServiceResponse", function(){
  return {
    get : function(){
      // get('/listprojects')
      return [{
        "project_name": "Untitled Project",
        "users": [{"username": "test@datarobot.com","_id": "53710ced6656dc7dc809740e","roles": ["OWNER"]}],
        "holdout_unlocked": false,
        "metric": "LogLoss",
        "active": 1,
        "stage": "modeling",
        "target": {"type": "Binary","name": "Sex","size": 8424.0},
        "created": 1405460868.906125,
        "partition": {
          "cv_method": "RandomCV",
          "recommender_item_id": null,
          "recommender_user_id": null,
          "total_size": 10530,
          "folds": 5,
          "is_recommender": false,
          "holdout_pct": 20,
          "reps": 5,
          "ui_validation_type": "CV"
        },
        "holdout_pct": 20,
        "version": 1.1,
        "mode": 0,
        "originalName": "census_salary_small.csv",
        "permissions":{"CAN_SET_TARGET":true,"CAN_SET_WORKERS":true,"CAN_RUN_AUTOPILOT":true,"CAN_UPDATE_PROJECT":true,"CAN_DELETE_PROJECT":true,"CAN_VIEW":true,"CAN_CLONE_PROJECT":true,"CAN_EDIT_FEATURE_LISTS":true,"CAN_DELETE_QUEUE_ITEM":true,"CAN_LAUNCH_RSTUDIO":true,"CAN_MANAGE_MODELS":true,"CAN_MANAGE_USER_ACCOUNTS":true,"SHOW_AUTOPILOT":true,"CAN_UNLOCK_HOLDOUT":true},
        "_id": "53c5a184a6844e4451f1e8b5"
      },{
        "project_name": "kickcars",
        "users": [{"username": "test@datarobot.com","_id": "53710ced6656dc7dc809740e","roles": ["OWNER"]}],
        "holdout_unlocked": false,
        "metric": "LogLoss",
        "active": 1,
        "stage": "modeling",
        "target": {"type": "Binary","name": "IsBadBuy","size": 176.0},
        "created": 1405522056.780268,
        "partition": {
          "cv_method": "StratifiedCV",
          "recommender_item_id": null,
          "recommender_user_id": null,
          "total_size": 200,
          "folds": 5,
          "is_recommender": false,
          "holdout_pct": 10,
          "reps": 10,
          "ui_validation_type": "CV"
        },
        "holdout_pct": 10,
        "version": 1.1,
        "mode": 1,
        "originalName": "kickcars-sample-200.csv",
        "permissions":{"CAN_SET_TARGET":true,"CAN_SET_WORKERS":true,"CAN_RUN_AUTOPILOT":true,"CAN_UPDATE_PROJECT":true,"CAN_DELETE_PROJECT":true,"CAN_VIEW":true,"CAN_CLONE_PROJECT":true,"CAN_EDIT_FEATURE_LISTS":true,"CAN_DELETE_QUEUE_ITEM":true,"CAN_LAUNCH_RSTUDIO":true,"CAN_MANAGE_MODELS":true,"CAN_MANAGE_USER_ACCOUNTS":true,"SHOW_AUTOPILOT":true,"CAN_UNLOCK_HOLDOUT":true},
        "_id": "53c69088a6844e2b479794bd"
      }];
    },

    newProjectStageAim: function(){
      return {
        "pid":"53c6ce65a6844e2b479796c0",
        "id":"53c6ce65a6844e2b479796c0",
        "project_name":"Untitled Project",
        "users":[{"username":"test@datarobot.com","_id":"53710ced6656dc7dc809740e","roles":["OWNER"]}],
        "holdout_unlocked":false,
        "metric":null,
        "active":1,
        "stage":"aim",
        "target":null,
        "created":"2014-07-16",
        "partition":null,
        "holdout_pct":20,
        "version":1.1,
        "mode":1,
        "originalName":"kickcars-sample-200.csv",
        "permissions":{"CAN_SET_TARGET":true,"CAN_SET_WORKERS":true,"CAN_RUN_AUTOPILOT":true,"CAN_UPDATE_PROJECT":true,"CAN_DELETE_PROJECT":true,"CAN_VIEW":true,"CAN_CLONE_PROJECT":true,"CAN_EDIT_FEATURE_LISTS":true,"CAN_DELETE_QUEUE_ITEM":true,"CAN_LAUNCH_RSTUDIO":true,"CAN_MANAGE_MODELS":true,"CAN_MANAGE_USER_ACCOUNTS":true,"SHOW_AUTOPILOT":true,"CAN_UNLOCK_HOLDOUT":true},
        "_id":"53c6ce65a6844e2b479796c0",
        "partitioning":{
          "baseTrainingSize":64,
          "baseHoldoutSize":20,
          "baseValidationSize":16,
          "maxNewSampleSize":80,
          "holdoutUnlocked":false,
          "uvt":"CV"
        },
        "$$hashKey":"036",
        "session":{
          "pid":"53c6ce65a6844e2b479796c0",
          "token":"gC3P3vQjzUlsmA==",
          "userid":"53710ced6656dc7dc809740e"
        }
      };
    }

  };
});