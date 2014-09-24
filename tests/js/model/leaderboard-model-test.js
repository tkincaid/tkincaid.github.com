define(
  [
    'angular-mocks',
    'datarobot',
    'js/model/leaderboard-model.min',
  ],
  function(angularMocks, datarobot, LbModel ) {
    describe('LbModel', function() {

      var _LbModel;

      beforeEach(module('datarobot'));
      beforeEach(inject(function(LbModel) {
        _LbModel = LbModel;
      }));

      // describe('ScoreStatus', function() {
      //   it('should be "In Progress" if no metric scores available', function() {
      //     item = {test: {metrics: ['ametricname'], ametricname: [null]}};
      //     lbItem = new _LbModel(item);

      //     expect(lbItem.ScoreStatus).toBe('In Progress');
      //   });

      //   it('should be "Errored" if no_finish is "error"', function() {
      //     item = {test: {metrics: ['ametricname'], ametricname: [null]},
      //             no_finish: 'Errored'};
      //     lbItem = new _LbModel(item);

      //     expect(lbItem.ScoreStatus).toBe('Errored');
      //   });

      // });
    });
  }
);
