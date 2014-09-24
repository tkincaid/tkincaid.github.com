#!/bin/env jasmine-node --verbose
/*******************************************************************
 *
 *      Modeling Machine Queue Server Tests: projects.js
 *
 *      Author: David Lapointe
 *
 *      Copyright (C) 2014 DataRobot, Inc.
*******************************************************************/

var jasmine = require('jasmine-node'),
  _ = require('lodash');

var users = require('../../MMQueue/users');

describe('User', function() {
  it('should not add duplicate sockets', function() {
    var userid = '5387feb72c5cea0afbd23339'
    var socket = {
      id:'cSC9aQwnzd6ZW4aodGOH'
    }

    var user = users.add(userid, socket);
    expect(user.sockets.length).toEqual(1);


    var sameSocket = {
      id:'cSC9aQwnzd6ZW4aodGOH'
    }

    var user = users.add(userid, socket);
    expect(user.sockets.length).toEqual(1);
  });
});

