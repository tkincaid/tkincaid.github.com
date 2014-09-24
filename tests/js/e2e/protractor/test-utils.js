var fs = require('fs');
var utils = {};

utils.writeScreenShot = function(data, filename) {
  var stream = fs.createWriteStream(filename);

  stream.write(new Buffer(data, 'base64'));
  stream.end();
};

module.exports = utils;