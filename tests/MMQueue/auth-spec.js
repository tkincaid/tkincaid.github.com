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
  _ = require('lodash'),
  events = require('events');

var auth = require('../../MMQueue/auth');

var pidCounter = 10000;
var dummySocket = function(pid) {
  this.pid = pid || pidCounter++;
};
dummySocket.prototype = new events.EventEmitter();

describe('Auth', function() {
  var socket,
    testCount,
    errorCount;
  beforeEach(function() {
    testCount = 0, errorCount = 0;
    socket = new dummySocket();
    socket.disconnect = jasmine.createSpy('disconnect');
    socket.on('test', function() {
      testCount += 1;
    });
    socket.on('error', function() {
      errorCount += 1;
    });
  });

  it('should stop listening on error', function() {
    runs(function() {
      socket.emit('test');
    }, 100);
    runs(function() {
      expect(errorCount).toBe(0);
      expect(testCount).toBe(1);
    });

    runs(function() {
      auth.error(socket);
      socket.emit('test');
    }, 100);
    runs(function() {
      expect(testCount).toBe(1);
      expect(errorCount).toBe(1);
    });
  });

  it('should disconnect on error', function() {
    auth.error(socket);
    expect(socket.disconnect).toHaveBeenCalled();
  });

  xit('should not authenticate an unauthorized user', function() {

  });
});

