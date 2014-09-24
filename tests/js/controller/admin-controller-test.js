define(
  [
    'angular-mocks',
    'datarobot',
    'js/controller/admin-controller.min',
    'js/service/admin-service.min'
  ],
  function(){

    describe('AdminController', function(){

      beforeEach(module('datarobot'));

      var mockHttpBackend, scope, appConstants;
      beforeEach(inject(function(_$httpBackend_, $controller, $rootScope, AdminService){
        mockHttpBackend = _$httpBackend_;
        scope = $rootScope.$new();
        $controller('AdminController', { $scope: scope });
      }));

      describe('Saving permissions', function(){
        it('Should add available permissions to the user object as unset', function(){
          var permissions = {
              'PERMISSION_1' : true,
              'PERMISSION_2' : false
          };

          var allPermissions = ['PERMISSION_1', 'PERMISSION_2', 'PERMISSION_3'];

          scope.addUnsetPermissions(permissions, allPermissions);

          expect(Object.keys(permissions).length).toEqual(3);
          expect(permissions.PERMISSION_3).toBeNull();
        });

        it('Should remove unset permissions', function(){
          var user = {
              permissions : {
                'PERMISSION_1' : true,
                'PERMISSION_2' : false,
                'PERMISSION_3' : null
            }
          };

          var permissions = scope.getOnlySetPermissions(user.permissions);

          // Didn't touch the original object
          expect(Object.keys(user.permissions).length).toEqual(3);
          // The return value does not contain permissions with null (unset) values
          expect(Object.keys(permissions).length).toEqual(2);
          expect(permissions.PERMISSION_1).toBe(true);
          expect(permissions.PERMISSION_2).toBe(false);
        });
      });

      afterEach(function () {
          mockHttpBackend.verifyNoOutstandingExpectation();
          mockHttpBackend.verifyNoOutstandingRequest();
      });
    });
  }
);