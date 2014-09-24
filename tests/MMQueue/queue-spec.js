#!/bin/env jasmine-node --verbose
/*******************************************************************
 *
 *      Modeling Machine Queue Server Tests: queue.js
 *
 *      Author: David Lapointe
 *
 *      Copyright (C) 2014 DataRobot, Inc.
*******************************************************************/

var jasmine = require('jasmine-node'),
  _ = require('lodash'),
  events = require('events');

require('./mocks/zmq');

var queue = require('../../MMQueue/queue'),
  projects = require('../../MMQueue/projects'),
  utils = require('../../MMQueue/utils');

clearInterval(queue.pulser);

var pidCounter = 10000;
var dummySocket = function(pid) {
  this.pid = ''+(pid || pidCounter++);
};

describe('Queue', function() {
  it('should not request autopilot twice within two seconds', function(done) {
    var client1 = new dummySocket();

    projects.add(client1.pid);
    var project = projects.get(client1.pid);
    project.remaining.next_steps = 2;
    spyOn(utils.http, 'post').andCallFake(function() {
      project.servicing = false;
      project.next_next_steps = (+new Date()) + 2000;
      project.persist('next_steps', false);
      project.remaining.next_steps -= 1;
    });
    queue.doNextSteps(project);
    expect(project.remaining.next_steps).toBe(1);
    queue.doNextSteps(project);
    expect(project.remaining.next_steps).toBe(1);
    setTimeout(function() {
      expect(project.remaining.next_steps).toBe(1);
      queue.doNextSteps(project);
      expect(project.remaining.next_steps).toBe(0);
      done();
    }, 2000);
  }, 3000);
});
