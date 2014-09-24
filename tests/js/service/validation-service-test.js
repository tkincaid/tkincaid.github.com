define(
  [
    'angular-mocks',
    'datarobot',
    'js/service/validation-service.min'
  ],
  function() {
    describe('ValidationService', function() {

      beforeEach(module('datarobot'));

      var validationService;
      beforeEach(inject(function (ValidationService) {
        validationService = ValidationService;
      }));

      describe('.url(http://iamaurl)', function() {
        it('should return false', function() {
          expect(validationService.url('http://iamaurl')).toBe(false);
        });
      });

      describe('.url(https://s3.amazonaws.com/temp-data-files/kickcars.rawdata.csv)', function() {
        it('should return true', function() {
          expect(validationService.url('https://s3.amazonaws.com/temp-data-files/kickcars.rawdata.csv')).toBe(true);
        });
      });

      describe('.url(httpx://lolo.com)', function() {
        it('should return false', function() {
          expect(validationService.url('httpx://lolo.com')).toBe(false);
        });
      });

      describe('.url(https://lol!!.com)', function() {
        it('should return false', function() {
          expect(validationService.url('https://lol!!.com')).toBe(false);
        });
      });

      describe('https://usr:psw@s3.amazonaws.com/testlio/project/260/test-data/normal-size.csv', function() {
        it('should return true', function() {
          expect(validationService.url('https://usr:psw@s3.amazonaws.com/testlio/project/260/test-data/normal-size.csv')).toBe(true);
        });
      });

      describe('https://usr:@s3.amazonaws.com', function() {
        it('should return false', function() {
          expect(validationService.url('https://usr:@s3.amazonaws.com')).toBe(false);
        });
      });

      describe('email', function(){
        it('Should validate email addresses', function(){
          // check bad email
          var email = "blahblah";
          verify = validationService.email(email);
          expect( verify ).toBe(false);
          // check good email
          email = "user@datarobot.com";
          verify = validationService.email(email);
          expect( verify ).toBe(true);
        });
      });

    });
  }
);