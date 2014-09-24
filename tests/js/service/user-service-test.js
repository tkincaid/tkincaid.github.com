define(
  [
    'js/model/user.min',
    'angular-mocks',
    'datarobot',
    'js/service/user-service.min'
  ],
  function(User) {
    describe('UserService', function() {

      beforeEach(module('datarobot'));

      var userService, mockBackend;
      beforeEach(inject(function (UserService, $httpBackend) {
        userService = UserService;
        mockBackend = $httpBackend;
      }));

      describe('Server fetch', function() {
        it('should return a username', function() {
          var user = {
              username: 'CHILIFRESH14',
              statusCode: 1
          };

          mockBackend
            .expectGET('/account/profile')
            .respond(user);

          userService.get().then(function(response) {
            expect(response.username).toBeTruthy();
            expect(response.username).toEqual((new User(user)).username);
          });

          mockBackend.flush();
        });
      });

      describe('Validate Login', function() {

        it('should test if the user has filled in the login form', function(){
          var user = { email: '', password: '' };
          // test for no input
          var verify = userService.validateLogin(user);
          expect(verify).toBe(false);

          //verify error message was set
          expect(user.emailErrorMsg).not.toBeUndefined();
          // expect(user.errorMessage).not.toBeUndefined();
        });

        it('should test if the user forgot the email address and we get only 1 error message', function(){
          var user = {email:'',password:'valid_password'};
          // test for no input
          var verify = userService.validateLogin(user);
          expect(verify).toBe(false);

          //verify error message was set
          expect(user.emailErrorMsg).not.toBeUndefined();
          expect(user.errorMessage).toBeUndefined();
        });

        it('should test a valid login -before it hits server', function(){
          var user = {email:'test@test.com',password:'valid_password'};
          // test for success
          var verify = userService.validateLogin(user);
          expect(verify).toBe(true);

          //verify error message are not set
          expect(user.emailErrorMsg).toBeUndefined();
          expect(user.errorMessage).toBeUndefined();
        });

      });

      describe('Save profile', function(){
        it('should update the in-memory copy', function(){

          mockBackend
            .expectPOST('/account/profile')
            .respond({});


          userService.user = {'first_name' : 'test-name'};
          updated_user = {'first_name' : 'new-name'};

          userService.save(updated_user).then(function(){
            expect(userService.user.first_name).toEqual('new-name');
          });

          mockBackend.flush();
        });

        it('should not update the in-memory copy if it is not existent', function(){

          mockBackend
            .expectPOST('/account/profile')
            .respond({});


          userService.user = null;
          updated_user = {'first_name' : 'new-name'};

          userService.save(updated_user).then(function(){
            expect(userService.user).toEqual(null);
          });

          mockBackend.flush();
        });
      });

      describe('Change password', function(){
        it('should indicate success with no error message', function(){

          mockBackend.expectPOST('/account/password').respond({
            message : 'OK'
          });

          userService.changePassword({}).then(function(response){
            expect(response.success).toBe(true);
            expect(response.msg).toBe(null);
          });

          mockBackend.flush();
        });

        it('should indicate failure along with an error message from the server', function(){

          mockBackend.expectPOST('/account/password').respond({
            message : 'FAILED!'
          });

          userService.changePassword({}).then(function(response){
            expect(response.success).toBe(false);
            expect(response.msg).toBe('FAILED!');
          });

          mockBackend.flush();
        });
      });

//      describe('.setUserInfo()', function() {
//        it('should fetch transformations even if saved transformations are available', function() {
//          userService.transfomrations = [
//            { name: 'log(x)' }
//          ];
//
//          var transformations = [
//            { name: 'x^2' }
//          ];
//
//          mockBackend
//            .expectGET('http://localhost:3000/transformations')
//            .respond(transformations);
//
//          userService.get().then(function(response) {
//            expect(response).toBe(transformations);
//          });
//        });
//      });

    });
  }
);