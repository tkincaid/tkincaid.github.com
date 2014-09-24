define(
  [
    'angular-mocks',
    'datarobot',
    'js/service/transformation-service.min'
  ],
  function() {
    describe('TransformationService', function() {

      beforeEach(module('datarobot'));

      var transformationService, mockBackend;
      beforeEach(inject(function (TransformationService, $httpBackend) {
        transformationService = TransformationService;
        mockBackend = $httpBackend;
      }));

      it('should return saved transformations if they are available', function() {
        transformationService.transfomrations = [
          { name: 'log(x)' }
        ];

        transformationService.get().then(function(response) {
          expect(response).toBe(transformationService.transfomrations);
        });
      });

      it('should fetch transformations even if saved transformations are available', function() {
        transformationService.transformations = [
          { name: 'log(x)' }
        ];

        var fakeResponse = {
          data: [
            { name: 'x^2' }
          ]
        };

        mockBackend
          .expectGET('/transformations')
          .respond(fakeResponse);

        transformationService.get(true).then(function(response) {
          //Add custom transformation
          // fakeResponse.data.push(transformationService.PYTHON_USER_TRANSFORMATION);
          // fakeResponse.data.push(transformationService.R_USER_TRANSFORMATION);
          expect(response).toEqual(fakeResponse.data);
        });

        mockBackend.flush();
      });
    });
  }
);