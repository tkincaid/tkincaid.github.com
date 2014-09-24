define(
  function config(){
    return {
      "logging": 1,
      "socketio": {
        "host": "",
        "port": "8011"
      },
      "runmode": 0,
      "upload": {
        "host": "",
        "port": "8022",
        "path": "upload_data"
      },
      "version":"0.1.1",
      "loginnotice" : "<p>Use of this computer system, authorized or unauthorized, constitutes consent to monitoring of this system for authorized purposes. Unauthorized use may subject you to criminal prosecution. Evidence of unauthorized use collected during monitoring may be used for administrative, criminal, or other adverse action.</p><p>Copyright &copy; DataRobot Inc. 2013</p>"
    };
  }
);