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

var projects = require('../../MMQueue/projects');
var config = require('../../config/queue_config');

var pidCounter = 10000;
var dummySocket = function(pid) {
  this.pid = pid || pidCounter++;
};

var activeProjects = projects.activeProjects;

describe('Project tests', function() {
  it('should generate unique clients', function() {
    var client1 = new dummySocket(),
      client2 = new dummySocket();
    expect(client1.pid).not.toBe(client2.pid);
    expect(client1).not.toEqual(client2);
  });
});

describe('Projects', function() {
  it('should be able to emit events', function(done) {
    projects.on('test-bind', function(val) {
      expect(val).toBe(123);
      done();
    });
    projects.emit('test-bind', 123);
  }, 100);

  it('should keep a list of clients', function() {
    var pid = pidCounter++;
    expect(projects.get(pid)).toBeUndefined();
    projects.add(pid);
    expect(projects.get(pid)).not.toBeUndefined();
  });

  it('should not add duplicate projects', function() {
    var pid = pidCounter++;

    var count = _.values(activeProjects).length;
    projects.add(pid);
    expect(_.values(activeProjects).length).toBe(count+1);
    projects.add(pid);
    expect(_.values(activeProjects).length).toBe(count+1);
  });
});

describe('Project', function() {
  it('should keep track of its pid', function() {
    var pid = pidCounter++;

    projects.add(pid);
    var project = projects.get(pid);
    expect(project.pid).toBe(pid);
  });

  it('should not add duplicate clients', function() {
    var pid = pidCounter++;

    projects.add(pid);
    var project = projects.get(pid);

    expect(project.clients).toEqual([]);

    var user = {
      userid: '5387feb72c5cea0afbd23339'
    }

    project.addClient(user)
    expect(project.clients.length).toEqual(1);

    var sameUser = {
      userid: '5387feb72c5cea0afbd23339'
    }

    project.addClient(sameUser)
    expect(project.clients.length).toEqual(1);
  });

  it('should keep an up-to-date list of clients', function() {
    var client1 = new dummySocket();

    projects.add(client1.pid);
    var project = projects.get(client1.pid);
    expect(project.clients.length).toBe(0);
    project.addClient(client1);
    expect(project.clients.length).toBe(1);
    project.removeClient(client1);
    expect(project.clients.length).toBe(0);
  });

  it('should not delete clients it does not have', function() {
    var client1 = new dummySocket(),
      client2 = new dummySocket();

    projects.add(client1.pid);
    var project = projects.get(client1.pid);
    expect(project.clients.length).toBe(0);
    project.addClient(client1);
    expect(project.clients.length).toBe(1);
    project.removeClient(client2);
    expect(project.clients.length).toBe(1);
  });

  it('should be cleaned up by default', function() {
    var pid = pidCounter++;

    projects.add(pid);
    var project = projects.get(pid);
    expect(project.shouldCleanup()).toBeTruthy();
  });

  it('should not be cleaned up with clients still connected', function() {
    var client1 = new dummySocket(),
      client2 = new dummySocket(client1.pid);

    projects.add(client1.pid);
    var project = projects.get(client1.pid);
    expect(project.shouldCleanup()).toBeTruthy();
    project.addClient(client1);
    expect(project.shouldCleanup()).toBeFalsy();
    project.addClient(client2);
    expect(project.shouldCleanup()).toBeFalsy();
    project.removeClient(client1);
    expect(project.shouldCleanup()).toBeFalsy();
    project.removeClient(client2);
    expect(project.shouldCleanup()).toBeTruthy();
  });

  it('should delete falsy persistence keys', function() {
    var client1 = new dummySocket();

    projects.add(client1.pid);
    var project = projects.get(client1.pid);
    expect(_.keys(project.persistFlags).length).toBe(0);
    project.persist('test', true);
    expect(_.keys(project.persistFlags).length).toBe(1);
    project.persist('test', false);
    expect(_.keys(project.persistFlags).length).toBe(0);

  });

  it('should be not be cleaned with persistence set', function() {
    var client1 = new dummySocket();

    projects.add(client1.pid);
    var project = projects.get(client1.pid);
    expect(project.shouldCleanup()).toBeTruthy();
    project.persist('test', false);
    expect(project.shouldCleanup()).toBeTruthy();
    project.persist('test', true);
    expect(project.shouldCleanup()).toBeFalsy();
    project.persist('test2', true);
    project.persist('test3', false);
    expect(project.shouldCleanup()).toBeFalsy();
  });

  it('should be not be cleaned with persistence set with different types', function() {
    var client1 = new dummySocket();

    projects.add(client1.pid);
    var project = projects.get(client1.pid);
    expect(project.shouldCleanup()).toBeTruthy();
    project.persist('test', undefined);
    expect(project.shouldCleanup()).toBeTruthy();
    project.persist('test', null);
    expect(project.shouldCleanup()).toBeTruthy();
    project.persist('test', 0);
    expect(project.shouldCleanup()).toBeTruthy();
    project.persist('test', true);
    expect(project.shouldCleanup()).toBeFalsy();
    project.persist('test', 5);
    expect(project.shouldCleanup()).toBeFalsy();
    project.persist('test', "yes");
    expect(project.shouldCleanup()).toBeFalsy();
  });

  it('should be removed if it can be cleaned up', function() {
    var pid = pidCounter++;

    projects.add(pid);
    var project = projects.get(pid);
    project.persist('test', true);
    expect(project.shouldCleanup()).toBeFalsy();
    project.cleanup();
    expect(projects.get(pid)).not.toBeUndefined();
    project.persist('test', false);
    expect(project.shouldCleanup()).toBeTruthy();
    project.cleanup();
    expect(projects.get(pid)).toBeUndefined();
  });
});
