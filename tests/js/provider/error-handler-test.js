define(['angular-mocks', 'datarobot', 'jquery', 'js/providers/error-handler.min'],
  function(angularMocks, datarobot, $) {
    describe('ErrorHandler', function() {

      beforeEach(module('datarobot'));

      var rootScope, $scope;
      beforeEach(inject(function($rootScope) {
        $scope = $rootScope.$new();

      }));

      it('should send a predetermined request on angular error', function() {
        runs(function() {
          spyOn($, 'ajax');
          try{
            // Throw an angular error
            $scope.$apply(function() { $scope.$apply(); });
          } catch(e) { }
        });

        waitsFor(function() {
          return $.ajax.callCount > 0;
        }, 500);

        runs(function() {
          expect($.ajax).toHaveBeenCalled();
          var data = JSON.parse($.ajax.calls[0].args[0].data)[0];
          expect(data.cause).toBe('angular');
          expect(data.errorMessage).toMatch(/Error: \[\$rootScope:inprog\].*apply/);
        });
      });

      it('should not report error messages if they are matched by "ignore" rules', function() {

        spyOn($, 'ajax');

        $scope.$apply(function() {
          throw 'rstudio/50_214_119_44664/auth-public-key 502';
        });

        expect($.ajax).not.toHaveBeenCalled();
      });

    });
  }
);