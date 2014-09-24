define(
  [
    'angular-mocks',
    'datarobot',
    'js/controller/header-controller.min'
  ],
  function(){

    describe('HeaderController', function(){

      beforeEach(module('datarobot'));

      var mockHttpBackend, scope, appConstants;
      beforeEach(inject(function(_$httpBackend_, $controller, $rootScope, constants){
        mockHttpBackend = _$httpBackend_;
        scope = $rootScope.$new();
        $controller('HeaderController', { $scope: scope });
        appConstants = constants;
      }));

      describe('Display user menu based on run mode and user status', function(){
        /* IsPublicMode  statusCode  userMenuType
          n             -1  private
          n              0  registered
          n              1  ?
          y              1  guest
          y             -1  public
          y              0  registered
        */

        var assert = function(sc, IsPublicMode, result){
          var registeredUserResponse = { username: 'user@datarobot.com', statusCode: sc };
          appConstants.isPublicMode = IsPublicMode;
          mockHttpBackend
            .expectGET('/account/profile')
            .respond(registeredUserResponse);
          scope.displayUserMenu();
          mockHttpBackend.flush();
          expect(scope.userMenu).toBe(result);
        };

        it('Menu should display as public when sc = -1 (no user) and isPublicMode = true ', function(){
          assert('-1', true, 'public');
        });

        it('Menu should display as registered when sc = 0 and isPublicMode = true ', function(){
          assert('0', true, 'registered');
        });

        it('Menu should display as guest when sc = 1 and isPublicMode = true ', function(){
          assert('1', true, 'guest');
        });

        it('Menu should display as public when sc = -1 (no user) and isPublicMode = false ', function(){
          assert('-1', false, 'private');
        });

        it('Menu should display as registered when sc = 0 and isPublicMode = false ', function(){
          assert('0', false, 'registered');
        });

        // guest menu is not applicable in privte mode

      });

      afterEach(function () {
          mockHttpBackend.verifyNoOutstandingExpectation();
          mockHttpBackend.verifyNoOutstandingRequest();
      });
    });
  }
);