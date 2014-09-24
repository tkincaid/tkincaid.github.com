define(
  [
    'angular-mocks',
    'datarobot',
    'js/service/utility-service.min'
  ],
  function () {
    describe("UtilityService", function () {

      beforeEach(module('datarobot'));

      var utilityService;
      beforeEach(inject(function (UtilityService) {
        utilityService = UtilityService;
      }));

      describe("Format date", function () {
        it('should return a human-readable datetime string', function () {
          expect(utilityService.formatDate(1388984400)).toBe('2014-01-06');
        });

        it('should return a human-readable datetime string containing ms', function () {
          expect(utilityService.formatDate(1388169086.2292311)).toBe('2013-12-27');
        });
      });
    });
  }
);