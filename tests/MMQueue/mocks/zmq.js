/*******************************************************************
 *
 *      Modeling Machine Queue Server Test Mocks: zmq
 *
 *      Author: David Lapointe
 *
 *      Copyright (C) 2014 DataRobot, Inc.
*******************************************************************/

var zmq = require('zmq');

zmq.socket = function() {
  return {
    connect: function() {},
    on: function() {},
    subscribe: function() {},
    unsubscribe: function() {}
  }
};
