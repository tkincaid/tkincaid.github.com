define(
  [
    'angular-mocks',
    'datarobot',
    'js/model/data-column.min',
  ],
  function(angularMocks, datarobot, FeatureModel) {
    describe('Feature Model (data column)', function() {

      beforeEach(module('datarobot'));

      describe('Feature', function() {
        it('should indictate whether the feature has info', function() {
          var item = {profile: {info: undefined}};
          var feature = new FeatureModel(item);
          expect(feature.hasInfo()).toBe(false);

          item = {profile: {}};
          feature = new FeatureModel(item);
          expect(feature.hasInfo()).toBe(false);

          item = {};
          feature = new FeatureModel(item);
          expect(feature.hasInfo()).toBe(false);


          item = {profile: {info: 123}};
          feature = new FeatureModel(item);
          expect(feature.hasInfo()).toBe(true);
        });

        it('should indictate whether the feature is text', function() {
          var item = {types: {text: false}};
          var feature = new FeatureModel(item);
          expect(feature.isText()).toBe(false);

          item = {types: {}};
          feature = new FeatureModel(item);
          expect(feature.isText()).toBe(false);

          item = {};
          feature = new FeatureModel(item);
          expect(feature.isText()).toBe(false);


          item = {types: {text: true}};
          feature = new FeatureModel(item);
          expect(feature.isText()).toBe(true);
        });
      });
    });
  }
);
