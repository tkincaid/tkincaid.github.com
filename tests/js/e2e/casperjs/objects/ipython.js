var ide = require('./abstract/ide'),
  _ = require('lodash');

module.exports = function() {
  return _.extend(_.cloneDeep(ide()), {
    view: '#python-view',
    form: '#ipythonform',
    frame: 'ipython_frame'
  });
};