var ide = require('./abstract/ide'),
  _ = require('lodash');

module.exports = function() {
  return _.extend(_.cloneDeep(ide()), {
    view: '#rstudio-view',
    form: '#rstudioform',
    frame: 'rstudio_frame',
    frame_loading: 'body.linux'
  });
};