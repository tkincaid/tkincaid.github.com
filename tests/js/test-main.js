var tests = [];
for (var file in window.__karma__.files) {
  if (window.__karma__.files.hasOwnProperty(file)) {
    if (/.*-test\.js$/.test(file)) {
      tests.push(file);
    }
  }
}

//remove CDN
requirejsConfig.paths.angular.splice(0,1);
requirejsConfig.paths.jquery.splice(0,1);

requirejsConfig.baseUrl = '/base/static';
requirejsConfig.paths.config = '../tests/testdata/config';

requirejsConfig.paths.DatasetServiceResponse = '../tests/js/mocks/DatasetServiceResponse';
requirejsConfig.paths.ProjectServiceResponse = '../tests/js/mocks/ProjectServiceResponse';

requirejsConfig.urlArgs = ''; // karma-requirejs automatically appends a hash to bust cache
requirejsConfig.deps = tests;
requirejsConfig.callback = window.__karma__.start;

requirejs.config(requirejsConfig);