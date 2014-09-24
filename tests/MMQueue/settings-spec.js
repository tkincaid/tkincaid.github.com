#!/bin/env jasmine-node --verbose
/*******************************************************************
 *
 *      Modeling Machine Queue Server Tests: settings.js
 *
 *      Author: David Lapointe
 *
 *      Copyright (C) 2014 DataRobot, Inc.
*******************************************************************/

var jasmine = require('jasmine-node'),
  _ = require('lodash'),
  events = require('events');

require('./mocks/zmq');

var settings = require('../../MMQueue/settings'),
  projects = require('../../MMQueue/projects'),
  utils = require('../../MMQueue/utils');

var pidCounter = 10000;
var dummySocket = function(pid) {
  this.pid = ''+(pid || pidCounter++);
};

describe('Settings', function() {
  it('should request the Queue upon client connection', function() {
    var client1 = new dummySocket();

    spyOn(utils.http, 'get').andCallFake(function(url, cb) {
      cb({status: 'OK', queue: {1: true}, backlog: {2: true, 3: true}});
    });
    projects.add(client1.pid);
    expect(utils.http.get).toHaveBeenCalled();
    var project = projects.get(client1.pid);
    expect(_.keys(project.queue).length).toBe(1);
    expect(_.keys(project.backlog).length).toBe(2);
  });
});
