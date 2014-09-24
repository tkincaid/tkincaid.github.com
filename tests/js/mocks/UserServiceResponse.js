define("UserServiceResponse", function(){
  return {
    getRegistered : function(){
      // get('/account/profile')
      return {
        "username": "test@datarobot.com",
        "first_name": "test",
        "last_name": "test",
        "guest": 0,
        "userhash": "3a7d9a250373d61e7078bd4cfcc163ba",
        "invitecode": "code",
        "activated": 1,
        "max_workers": 10,
        "registered_on": 1399917884.88008,
        "account_permissions": {
          "CAN_MANAGE_INSTANCES": true
        },
        "permissions": {
          "CAN_MANAGE_APP_USERS": true,
          "GRAYBOX_ENABLED": true,
          "CAN_MANAGE_INSTANCES": true
        },
        "id": "53710ced6656dc7dc809740e",
        "statusCode": "0"
      };
    },

  };
});