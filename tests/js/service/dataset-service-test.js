define(
[
  'lodash',
  'js/model/feature-list.min',
  'js/model/data-column.min',
  'angular-mocks',
  'datarobot',
  'js/service/dataset-service.min',
  'js/service/project-service.min',
  'js/service/user-service.min'
],
function(_, FeatureList, DataColumn) {
  describe('DatasetService', function() {

    beforeEach(module('datarobot'));

    var datasetService, projectService, mockBackend;

    var userServiceMock = { username: 'username' };

    module(function($provide) {
      $provide.value('UserService', userServiceMock);
    });

    beforeEach(inject(function (DatasetService, ProjectService, constants, strings, $httpBackend) {
      datasetService = DatasetService;
      projectService = ProjectService;
      projectService.project = { pid: 'ABC123', filename: 'kickcars-training-sample.csv' };
      mockBackend = $httpBackend;
      constants.xlsAndCompression = true;
      constants.maxUploadSize = 104857600;
      constants.allowedFileArchive = {tgz:1,zip:1,tar:1};
      constants.allowedFileCompression = {tgz:1,gz:1,bz2:1};
      constants.allowedFileExtensions = {xlsx:1,csv:1,xls:1};
    }));

    var serverFeatureLists = [
      {'files': ['bc01fcd7-a683-429a-9e76-7d4172c065a6'],
        'name': 'universe',
        'created': '2014-02-03 20:12:03.699000',
        'pid': '52eff812637aba4a5d1ca5be',
        'controls': {'bc01fcd7-a683-429a-9e76-7d4172c065a6': {'type': 'csv',
        'encoding': 'ASCII'},
        'dirty': false},
        'varTypeString': 'NNNNNNNNCNCCCCCCTCCTTCTTCCCTTNTCCTTTTTTTTCCTNTTCTCTCC',
        'shape': [10000, 53],
        'originalName': 'universe',
        '_id': '52eff813637aba4a5d1ca5bf',
        'columns': [
        {'transform_id': 0,'transform_args': [],'name': 'SalesID', 'raw_variable_index': 0},
        {'transform_id': 0, 'transform_args': [], 'name': 'SalePrice', 'raw_variable_index': 1},
        {'transform_id': 0, 'transform_args': [], 'name': 'MachineID', 'raw_variable_index': 2},
        {'transform_id': 0, 'transform_args': [], 'name': 'ModelID', 'raw_variable_index': 3},
        {'transform_id': 0, 'transform_args': [], 'name': 'datasource', 'raw_variable_index': 4},
        {'transform_id': 0, 'transform_args': [], 'name': 'auctioneerID', 'raw_variable_index': 5},
        {'transform_id': 0, 'transform_args': [], 'name': 'YearMade', 'raw_variable_index': 6},
        {'transform_id': 0, 'transform_args': [], 'name': 'MachineHoursCurrentMeter', 'raw_variable_index': 7},
        {'transform_id': 0, 'transform_args': [], 'name': 'UsageBand', 'raw_variable_index': 8},
        {'transform_id': 0, 'transform_args': [], 'name': 'saledate', 'raw_variable_index': 9},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelDesc', 'raw_variable_index': 10},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiBaseModel', 'raw_variable_index': 11},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiSecondaryDesc', 'raw_variable_index': 12},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelSeries', 'raw_variable_index': 13},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelDescriptor', 'raw_variable_index': 14},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductSize', 'raw_variable_index': 15},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiProductClassDesc', 'raw_variable_index': 16},
        {'transform_id': 0, 'transform_args': [], 'name': 'state', 'raw_variable_index': 17},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductGroup', 'raw_variable_index': 18},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductGroupDesc', 'raw_variable_index': 19},
        {'transform_id': 0, 'transform_args': [], 'name': 'Drive_System', 'raw_variable_index': 20},
        {'transform_id': 0, 'transform_args': [], 'name': 'Enclosure', 'raw_variable_index': 21},
        {'transform_id': 0, 'transform_args': [], 'name': 'Forks', 'raw_variable_index': 22},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pad_Type', 'raw_variable_index': 23},
        {'transform_id': 0, 'transform_args': [], 'name': 'Ride_Control', 'raw_variable_index': 24},
        {'transform_id': 0, 'transform_args': [], 'name': 'Stick', 'raw_variable_index': 25},
        {'transform_id': 0, 'transform_args': [], 'name': 'Transmission', 'raw_variable_index': 26},
        {'transform_id': 0, 'transform_args': [], 'name': 'Turbocharged', 'raw_variable_index': 27},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Extension', 'raw_variable_index': 28},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Width', 'raw_variable_index': 29},
        {'transform_id': 0, 'transform_args': [], 'name': 'Enclosure_Type', 'raw_variable_index': 30},
        {'transform_id': 0, 'transform_args': [], 'name': 'Engine_Horsepower', 'raw_variable_index': 31},
        {'transform_id': 0, 'transform_args': [], 'name': 'Hydraulics', 'raw_variable_index': 32},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pushblock', 'raw_variable_index': 33},
        {'transform_id': 0, 'transform_args': [], 'name': 'Ripper', 'raw_variable_index': 34},
        {'transform_id': 0, 'transform_args': [], 'name': 'Scarifier', 'raw_variable_index': 35},
        {'transform_id': 0, 'transform_args': [], 'name': 'Tip_Control', 'raw_variable_index': 36},
        {'transform_id': 0, 'transform_args': [], 'name': 'Tire_Size', 'raw_variable_index': 37},
        {'transform_id': 0, 'transform_args': [], 'name': 'Coupler', 'raw_variable_index': 38},
        {'transform_id': 0, 'transform_args': [], 'name': 'Coupler_System', 'raw_variable_index': 39},
        {'transform_id': 0, 'transform_args': [], 'name': 'Grouser_Tracks', 'raw_variable_index': 40},
        {'transform_id': 0, 'transform_args': [], 'name': 'Hydraulics_Flow', 'raw_variable_index': 41},
        {'transform_id': 0, 'transform_args': [], 'name': 'Track_Type', 'raw_variable_index': 42},
        {'transform_id': 0, 'transform_args': [], 'name': 'Undercarriage_Pad_Width', 'raw_variable_index': 43},
        {'transform_id': 0, 'transform_args': [], 'name': 'Stick_Length', 'raw_variable_index': 44},
        {'transform_id': 0, 'transform_args': [], 'name': 'Thumb', 'raw_variable_index': 45},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pattern_Changer', 'raw_variable_index': 46},
        {'transform_id': 0, 'transform_args': [], 'name': 'Grouser_Type', 'raw_variable_index': 47},
        {'transform_id': 0, 'transform_args': [], 'name': 'Backhoe_Mounting', 'raw_variable_index': 48},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Type', 'raw_variable_index': 49},
        {'transform_id': 0, 'transform_args': [], 'name': 'Travel_Controls', 'raw_variable_index': 50},
        {'transform_id': 0, 'transform_args': [], 'name': 'Differential_Type', 'raw_variable_index': 51},
        {'transform_id': 0, 'transform_args': [], 'name': 'Steering_Controls', 'raw_variable_index': 52}]},
      {'files': ['bc01fcd7-a683-429a-9e76-7d4172c065a6'],
        'name': 'Raw Features',
        'created': '2014-02-03 20:12:03.700000',
        'pid': '52eff812637aba4a5d1ca5be',
        'controls': {'bc01fcd7-a683-429a-9e76-7d4172c065a6': {'type': 'csv',
        'encoding': 'ASCII'},
        'dirty': false},
        'varTypeString': 'NNNNNNNNCNCCCCCCTCCTTCTTCCCTTNTCCTTTTTTTTCCTNTTCTCTCC',
        'shape': [10000, 53],
        'originalName': 'original',
        '_id': '52eff813637aba4a5d1ca5c0',
        'columns': [
        {'transform_id': 0, 'transform_args': [], 'name': 'SalesID', 'raw_variable_index': 0},
        {'transform_id': 0, 'transform_args': [], 'name': 'SalePrice', 'raw_variable_index': 1},
        {'transform_id': 0, 'transform_args': [], 'name': 'MachineID', 'raw_variable_index': 2},
        {'transform_id': 0, 'transform_args': [], 'name': 'ModelID', 'raw_variable_index': 3},
        {'transform_id': 0, 'transform_args': [], 'name': 'datasource', 'raw_variable_index': 4},
        {'transform_id': 0, 'transform_args': [], 'name': 'auctioneerID', 'raw_variable_index': 5},
        {'transform_id': 0, 'transform_args': [], 'name': 'YearMade', 'raw_variable_index': 6},
        {'transform_id': 0, 'transform_args': [], 'name': 'MachineHoursCurrentMeter', 'raw_variable_index': 7},
        {'transform_id': 0, 'transform_args': [], 'name': 'UsageBand', 'raw_variable_index': 8},
        {'transform_id': 0, 'transform_args': [], 'name': 'saledate', 'raw_variable_index': 9},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelDesc', 'raw_variable_index': 10},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiBaseModel', 'raw_variable_index': 11},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiSecondaryDesc', 'raw_variable_index': 12},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelSeries', 'raw_variable_index': 13},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelDescriptor', 'raw_variable_index': 14},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductSize', 'raw_variable_index': 15},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiProductClassDesc', 'raw_variable_index': 16},
        {'transform_id': 0, 'transform_args': [], 'name': 'state', 'raw_variable_index': 17},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductGroup', 'raw_variable_index': 18},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductGroupDesc', 'raw_variable_index': 19},
        {'transform_id': 0, 'transform_args': [], 'name': 'Drive_System', 'raw_variable_index': 20},
        {'transform_id': 0, 'transform_args': [], 'name': 'Enclosure', 'raw_variable_index': 21},
        {'transform_id': 0, 'transform_args': [], 'name': 'Forks', 'raw_variable_index': 22},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pad_Type', 'raw_variable_index': 23},
        {'transform_id': 0, 'transform_args': [], 'name': 'Ride_Control', 'raw_variable_index': 24},
        {'transform_id': 0, 'transform_args': [], 'name': 'Stick', 'raw_variable_index': 25},
        {'transform_id': 0, 'transform_args': [], 'name': 'Transmission', 'raw_variable_index': 26},
        {'transform_id': 0, 'transform_args': [], 'name': 'Turbocharged', 'raw_variable_index': 27},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Extension', 'raw_variable_index': 28},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Width', 'raw_variable_index': 29},
        {'transform_id': 0, 'transform_args': [], 'name': 'Enclosure_Type', 'raw_variable_index': 30},
        {'transform_id': 0, 'transform_args': [], 'name': 'Engine_Horsepower', 'raw_variable_index': 31},
        {'transform_id': 0, 'transform_args': [], 'name': 'Hydraulics', 'raw_variable_index': 32},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pushblock', 'raw_variable_index': 33},
        {'transform_id': 0, 'transform_args': [], 'name': 'Ripper', 'raw_variable_index': 34},
        {'transform_id': 0, 'transform_args': [], 'name': 'Scarifier', 'raw_variable_index': 35},
        {'transform_id': 0, 'transform_args': [], 'name': 'Tip_Control', 'raw_variable_index': 36},
        {'transform_id': 0, 'transform_args': [], 'name': 'Tire_Size', 'raw_variable_index': 37},
        {'transform_id': 0, 'transform_args': [], 'name': 'Coupler', 'raw_variable_index': 38},
        {'transform_id': 0, 'transform_args': [], 'name': 'Coupler_System', 'raw_variable_index': 39},
        {'transform_id': 0, 'transform_args': [], 'name': 'Grouser_Tracks', 'raw_variable_index': 40},
        {'transform_id': 0, 'transform_args': [], 'name': 'Hydraulics_Flow', 'raw_variable_index': 41},
        {'transform_id': 0, 'transform_args': [], 'name': 'Track_Type', 'raw_variable_index': 42},
        {'transform_id': 0, 'transform_args': [], 'name': 'Undercarriage_Pad_Width', 'raw_variable_index': 43},
        {'transform_id': 0, 'transform_args': [], 'name': 'Stick_Length', 'raw_variable_index': 44},
        {'transform_id': 0, 'transform_args': [], 'name': 'Thumb', 'raw_variable_index': 45},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pattern_Changer', 'raw_variable_index': 46},
        {'transform_id': 0, 'transform_args': [], 'name': 'Grouser_Type', 'raw_variable_index': 47},
        {'transform_id': 0, 'transform_args': [], 'name': 'Backhoe_Mounting', 'raw_variable_index': 48},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Type', 'raw_variable_index': 49},
        {'transform_id': 0, 'transform_args': [], 'name': 'Travel_Controls', 'raw_variable_index': 50},
        {'transform_id': 0, 'transform_args': [], 'name': 'Differential_Type', 'raw_variable_index': 51},
        {'transform_id': 0, 'transform_args': [], 'name': 'Steering_Controls', 'raw_variable_index': 52}]},
      {'files': ['bc01fcd7-a683-429a-9e76-7d4172c065a6'],
        'name': 'Informative Features',
        'created': '2014-02-03 20:12:21.137000',
        'pid': '52eff812637aba4a5d1ca5be',
        'controls': {'bc01fcd7-a683-429a-9e76-7d4172c065a6': {'type': 'csv',
        'encoding': 'ASCII'},
        'dirty': false},
        'varTypeString': 'NNCCTTCTNNNCCCTCNNCCTTTCTTNTTTCTTCNCCCTTTTCCCCNTTCTC',
        'shape': [10000, 52],
        'originalName': 'Informative Features',
        '_id': '52eff825637aba4ae51ca5bd',
        'columns': [
        {'transform_id': 0, 'transform_args': [], 'name': 'SalesID', 'raw_variable_index': 0},
        {'transform_id': 0, 'transform_args': [], 'name': 'datasource', 'raw_variable_index': 4},
        {'transform_id': 0, 'transform_args': [], 'name': 'UsageBand', 'raw_variable_index': 8},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiSecondaryDesc', 'raw_variable_index': 12},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiProductClassDesc', 'raw_variable_index': 16},
        {'transform_id': 0, 'transform_args': [], 'name': 'Drive_System', 'raw_variable_index': 20},
        {'transform_id': 0, 'transform_args': [], 'name': 'Ride_Control', 'raw_variable_index': 24},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Extension', 'raw_variable_index': 28},
        {'transform_id': 0, 'transform_args': [], 'name': 'SalePrice', 'raw_variable_index': 1},
        {'transform_id': 0, 'transform_args': [], 'name': 'MachineID', 'raw_variable_index': 2},
        {'transform_id': 0, 'transform_args': [], 'name': 'YearMade', 'raw_variable_index': 6},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelDesc', 'raw_variable_index': 10},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelDescriptor', 'raw_variable_index': 14},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductGroup', 'raw_variable_index': 18},
        {'transform_id': 0, 'transform_args': [], 'name': 'Forks', 'raw_variable_index': 22},
        {'transform_id': 0, 'transform_args': [], 'name': 'Transmission', 'raw_variable_index': 26},
        {'transform_id': 0, 'transform_args': [], 'name': 'ModelID', 'raw_variable_index': 3},
        {'transform_id': 0, 'transform_args': [], 'name': 'MachineHoursCurrentMeter', 'raw_variable_index': 7},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiBaseModel', 'raw_variable_index': 11},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductSize', 'raw_variable_index': 15},
        {'transform_id': 0, 'transform_args': [], 'name': 'ProductGroupDesc', 'raw_variable_index': 19},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pad_Type', 'raw_variable_index': 23},
        {'transform_id': 0, 'transform_args': [], 'name': 'Turbocharged', 'raw_variable_index': 27},
        {'transform_id': 0, 'transform_args': [], 'name': 'Hydraulics', 'raw_variable_index': 32},
        {'transform_id': 0, 'transform_args': [], 'name': 'Tip_Control', 'raw_variable_index': 36},
        {'transform_id': 0, 'transform_args': [], 'name': 'Grouser_Tracks', 'raw_variable_index': 40},
        {'transform_id': 0, 'transform_args': [], 'name': 'Stick_Length', 'raw_variable_index': 44},
        {'transform_id': 0, 'transform_args': [], 'name': 'Enclosure_Type', 'raw_variable_index': 30},
        {'transform_id': 0, 'transform_args': [], 'name': 'Ripper', 'raw_variable_index': 34},
        {'transform_id': 0, 'transform_args': [], 'name': 'Coupler', 'raw_variable_index': 38},
        {'transform_id': 0, 'transform_args': [], 'name': 'Engine_Horsepower', 'raw_variable_index': 31},
        {'transform_id': 0, 'transform_args': [], 'name': 'Scarifier', 'raw_variable_index': 35},
        {'transform_id': 0, 'transform_args': [], 'name': 'Backhoe_Mounting', 'raw_variable_index': 48},
        {'transform_id': 0, 'transform_args': [], 'name': 'Steering_Controls', 'raw_variable_index': 52},
        {'transform_id': 0, 'transform_args': [], 'name': 'saledate', 'raw_variable_index': 9},
        {'transform_id': 0, 'transform_args': [], 'name': 'fiModelSeries', 'raw_variable_index': 13},
        {'transform_id': 0, 'transform_args': [], 'name': 'state', 'raw_variable_index': 17},
        {'transform_id': 0, 'transform_args': [], 'name': 'Track_Type', 'raw_variable_index': 42},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pattern_Changer', 'raw_variable_index': 46},
        {'transform_id': 0, 'transform_args': [], 'name': 'Travel_Controls', 'raw_variable_index': 50},
        {'transform_id': 0, 'transform_args': [], 'name': 'Coupler_System', 'raw_variable_index': 39},
        {'transform_id': 0, 'transform_args': [], 'name': 'Undercarriage_Pad_Width', 'raw_variable_index': 43},
        {'transform_id': 0, 'transform_args': [], 'name': 'Grouser_Type', 'raw_variable_index': 47},
        {'transform_id': 0, 'transform_args': [], 'name': 'Differential_Type', 'raw_variable_index': 51},
        {'transform_id': 0, 'transform_args': [], 'name': 'Enclosure', 'raw_variable_index': 21},
        {'transform_id': 0, 'transform_args': [], 'name': 'Stick', 'raw_variable_index': 25},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Width', 'raw_variable_index': 29},
        {'transform_id': 0, 'transform_args': [], 'name': 'Pushblock', 'raw_variable_index': 33},
        {'transform_id': 0, 'transform_args': [], 'name': 'Tire_Size', 'raw_variable_index': 37},
        {'transform_id': 0, 'transform_args': [], 'name': 'Hydraulics_Flow', 'raw_variable_index': 41},
        {'transform_id': 0, 'transform_args': [], 'name': 'Thumb', 'raw_variable_index': 45},
        {'transform_id': 0, 'transform_args': [], 'name': 'Blade_Type', 'raw_variable_index': 49}]}
    ];

    var serverFeatures = [
        {'profile': {'plot': [], 'name': 'WarrantyCost', 'miss_count': 0.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'WarrantyCost', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [88, 0, 1263.53, 694.0322636701231, 462.0, 1131.0, 6519.0], 'raw_variable_index': 33, 'transform_id': 0, 'id': 'WarrantyCost', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': true}},
        {'profile': {'plot': [], 'name': 'PRIMEUNIT', 'plot2': [], 'y': null, 'type': 'C', 'type_label': 'Categorical'}, 'name': 'PRIMEUNIT', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [1, 184], 'raw_variable_index': 26, 'transform_id': 0, 'id': 'PRIMEUNIT', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': false}},
        {'profile': {'plot': [], 'name': 'Make', 'plot2': [], 'y': null, 'type': 'C', 'type_label': 'Categorical'}, 'name': 'Make', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [19, 0], 'raw_variable_index': 6, 'transform_id': 0, 'id': 'Make', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': false}},
        {'profile': {'plot': [], 'name': 'IsBadBuy', 'miss_count': 5.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'IsBadBuy', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [2, 5, 0.3128205128205128, 0.4648353478469372, 0.0, 0.0, 1.0], 'raw_variable_index': 1, 'transform_id': 0, 'id': 'IsBadBuy', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': false}},
        {'profile': {'plot': [], 'name': 'MMRAcquisitionAuctionCleanPrice', 'miss_count': 0.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'MMRAcquisitionAuctionCleanPrice', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [194, 0, 6823.38, 2551.671978335995, 0.0, 6682.0, 14504.0], 'raw_variable_index': 19, 'transform_id': 0, 'id': 'MMRAcquisitionAuctionCleanPrice', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': true}},
        {'profile': {'plot': [], 'name': 'WheelType', 'plot2': [], 'y': null, 'type': 'C', 'type_label': 'Categorical'}, 'name': 'WheelType', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [3, 20], 'raw_variable_index': 13, 'transform_id': 0, 'id': 'WheelType', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': false}},
        {'profile': {'plot': [], 'name': 'MMRCurrentRetailAveragePrice', 'miss_count': 1.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'MMRCurrentRetailAveragePrice', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [197, 1, 8638.934673366834, 2896.9293933149784, 0.0, 8620.0, 16028.0], 'raw_variable_index': 24, 'transform_id': 0, 'id': 'MMRCurrentRetailAveragePrice', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': true}}, {'name': 'IsOnlineSale', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': true}, 'summary': [1, 0], 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': false}, 'transform_id': 0, 'id': 'IsOnlineSale', 'raw_variable_index': 32},
        {'profile': {'plot': [], 'name': 'RefId', 'miss_count': 0.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'RefId', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [200, 0, 244.46, 144.64824219412154, 1.0, 245.5, 481.0], 'raw_variable_index': 0, 'transform_id': 0, 'id': 'RefId', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': true}},
        {'profile': {'plot': [], 'name': 'BYRNO', 'miss_count': 0.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'BYRNO', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [8, 0, 14927.17, 6864.407599316036, 5546.0, 19619.0, 21973.0], 'raw_variable_index': 28, 'transform_id': 0, 'id': 'BYRNO', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': true}},
        {'profile': {'plot': [], 'name': 'MMRAcquisitionRetailAveragePrice', 'miss_count': 0.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'MMRAcquisitionRetailAveragePrice', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [197, 0, 8679.245, 2850.270970980668, 0.0, 8465.5, 16545.0], 'raw_variable_index': 20, 'transform_id': 0, 'id': 'MMRAcquisitionRetailAveragePrice', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': true}},
        {'profile': {'plot': [], 'name': 'VNZIP1', 'miss_count': 0.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'VNZIP1', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [2, 0, 31601.05, 4815.73626002615, 20166.0, 33619.0, 33619.0], 'raw_variable_index': 29, 'transform_id': 0, 'id': 'VNZIP1', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': false}},
        {'profile': {'plot': [], 'name': 'VehOdo', 'miss_count': 0.0, 'y': null, 'type_label': 'Numeric', 'plot2': [], 'type': 'N', 'miss_ymean': null}, 'name': 'VehOdo', 'transform_args': [], 'low_info': {'high_cardinality': false, 'duplicate': false, 'empty': false, 'few_values': false}, 'metric_options': [], 'summary': [198, 0, 70718.535, 13466.454464259137, 32671.0, 71482.5, 98358.0], 'raw_variable_index': 14, 'transform_id': 0, 'id': 'VehOdo', 'types': {'text': false, 'currency': false, 'length': false, 'date': false, 'percentage': false, 'nastring': true}}
    ];

    describe('File upload', function() {

        it('should allow csv extensions', function() {
            var file  = [{ name: 'is-valid.csv',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow gzipped csvs', function() {
            var file  = [{ name: 'is-valid.csv.gz',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow bzip2 csvs', function() {
            var file  = [{ name: 'is-valid.csv.bz2',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow tar.bz2 csvs', function() {
            var file  = [{ name: 'is-valid.csv.tar.bz2',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow tar.gz csvs', function() {
            var file  = [{ name: 'is-valid.csv.tar.gz',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow .tgz csvs', function() {
            var file  = [{ name: 'is-valid.csv.tgz',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should assume no-filetype compressed is okay', function() {
            var file = [{ name: 'is-valid.gz',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should accept csv zip', function() {
            var file = [{ name: 'is-valid.csv.zip',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should accept zip without other extension', function() {
            var file = [{ name: 'is-valid.zip',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should detect invalid extensions', function() {
            var file  = [{ name: 'is-not-valid.grue',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).not.toBe(false);
        });

        it('should allow excel 2003 and before files', function() {
            var file  = [{ name: 'is-valid.xls',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow excel 2007+ files', function() {
            var file  = [{ name: 'is-valid.xlsx',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow zipped excel files', function() {
            var file  = [{ name: 'is-valid.xlsx.zip',size:19804572 }];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow files with no extensions', function() {
            var file  = [{ name: 'ok-file',size:19804572}];
            var isValid = datasetService.isFileInValid(file);
            expect(isValid).toBe(false);
        });

        it('should allow arbitrary number of prefixes', function() {
          var file = [{name: 'kickcars.chas.a-eda.chewbacca.csv',size:19804572}];
          var isValid = datasetService.isFileInValid(file);
          expect(isValid).toBe(false);
        });
    });

    describe('Getting features from the server', function() {

      it('should return saved features if they are available', function() {
        datasetService.features = [
          { filename: 'kickcars-training-sample.csv' }
        ];

        datasetService.get().then(function(response) {
          expect(response).toBe(datasetService.features);
        });
      });


      it('should add new features on server fetch', function(){
        // existing eda with 2 rows
        datasetService.features = [
          { name: 'feature_1' },
          { name: 'feature_2' }
        ];

        // full fetch returns existing rows plus 2 more
        var serverData =  { data: [
          { name: 'feature_1' },
          { name: 'feature_2' },
          { name: 'feature_3' },
          { name: 'feature_4' }
        ]};

        var expected = [
          { name: 'feature_1' },
          { name: 'feature_2' },
          { name: 'feature_3' },
          { name: 'feature_4' }
        ];

        mockBackend
          .expectGET('/eda/names/' + projectService.project.pid)
          .respond(serverData);

        mockBackend
          .expectGET('/eda/profile/' + projectService.project.pid)
          .respond(serverData);

        mockBackend
          .expectGET('/eda/graphs/' + projectService.project.pid)
          .respond(serverData);

        datasetService.get(projectService.project.pid).then(function(response) {
          expect(response.length).toEqual(expected.length);
        });

        mockBackend.flush();
      });

      it('Should update existing elements based on column name', function(){
        // existing eda with 2 rows + 1
        // the extra column proves the server fetch updates the local copy and does NOT overwrite
        datasetService.features = [
          { name: 'WarrantyCost' },
          { name: 'PRIMEUNIT' },
          { name: 'LOCAL_RANDOM_FEATURE', property : 'RANDOM_VALUE' }
        ];

        datasetService.features =_.map(datasetService.features, DataColumn);

        var serverData = { data: serverFeatures};

        mockBackend
          .expectGET('/eda/names/' + projectService.project.pid)
          .respond(serverData);

        mockBackend
          .expectGET('/eda/profile/' + projectService.project.pid)
          .respond(serverData);

        mockBackend
          .expectGET('/eda/graphs/' + projectService.project.pid)
          .respond(serverData);

        datasetService.get(projectService.project.pid).then(function(response) {
            expect(response.length).toEqual(serverFeatures.length + 1);
            var warrantyCostFeature = _.where(response, {name : 'WarrantyCost'})[0];
            expect(warrantyCostFeature.summary).not.toBeUndefined();

            var localFeature = _.where(response, {name : 'LOCAL_RANDOM_FEATURE'})[0];
            expect(localFeature.property).not.toBeUndefined();
        });

        mockBackend.flush();
      });

      it('Should update the in-memory feature list without overriding existing values', function(){
        datasetService.features = [
          { name: 'FEATURE_NAME', profile : {type:'X', y:'SalesPrice'} }
        ];

        var serverFeatures = [
          { name: 'FEATURE_NAME', profile : {info: 0.37846 }}
        ];

        datasetService.updateFeatures(serverFeatures);

        expect(datasetService.features[0].profile.info).not.toBeUndefined();
        expect(datasetService.features[0].profile.type).not.toBeUndefined();
        expect(datasetService.features[0].profile.y).not.toBeUndefined();
      });

      it('Should exclude empty column names', function(){
        datasetService.features = [];

        var serverFeatures = [
            {name: ''}
        ];

        datasetService.updateFeatures(serverFeatures);

        expect(_.size(datasetService.features)).toEqual(0);


        serverFeatures = [
            {name: '  '}
        ];

        datasetService.updateFeatures(serverFeatures);

        expect(_.size(datasetService.features)).toEqual(0);


        serverFeatures = [
            {low_info: 'does-not-have-name'}
        ];

        datasetService.updateFeatures(serverFeatures);

        expect(_.size(datasetService.features)).toEqual(0);
      });
    });

    describe('Feature lists', function() {
      it('should return saved feature lists if they are available', function() {
        datasetService.featureLists = [
          { name: 'DebtRatio' }
        ];

        datasetService.getFeatureLists().then(function(response) {
          expect(response).toBe(datasetService.featureLists);
        });
      });

      it('should fetch feature lists even if saved feature lists are available', function() {
        datasetService.featureLists = [
          { name: 'MonthlyIncome' },
          { name: 'age' }
        ];

        var mockedResponse = {
          data: [
            { name: 'NumberOfDependents' }
          ]
        };

        var featureLists = [
            new FeatureList({ name: 'NumberOfDependents' })
        ];

        mockBackend
          .expectGET('/project/' + projectService.project.pid + '/dataset')
          .respond(mockedResponse);

        datasetService.getFeatureLists(true).then(function(response) {
          expect(response[0].name).toEqual(featureLists[0].name);
        });

        mockBackend.flush();
      });

      it('should send a POST request to the server and add the feature list to the service', function() {
        datasetService.featureLists = [
          new FeatureList({ _id: '123', name: 'Feature List 1', columns: ['BYRNO', 'VehicleAge'] })
        ];

        expect(datasetService.featureLists[0].id).toBe('123');

        var newFeatureList = { id: 456, name: 'Feature List 2', columns: ['VehOdo', 'WarrantyCost'] };

        var newFeatureLists = datasetService.featureLists;
        newFeatureLists.push(new FeatureList(newFeatureList));

        mockBackend
          .expectPOST('/project/' + projectService.project.pid + '/dataset')
          .respond(201, '456');

        datasetService.addFeatureList(newFeatureList).then(function(response) {
          expect(response.id).toEqual(new FeatureList(newFeatureList).id);
        });

        mockBackend.flush();
      });
    });

    describe('Feature lists socket updates', function(){
      it('should add server data to the in-memory of feature lists', function(){
        datasetService.featureLists = [
          {name : 'UserFeatureList'}
        ];

        var serverData = [
          {name: 'List1'},
          {name: 'List2'}
        ];

        datasetService.updateFeatureLists(serverData);
        expect(datasetService.featureLists.length).toEqual(serverData.length + 1);

        var originalFeatureList = _.find(datasetService.featureLists, {name : 'UserFeatureList'});
        expect(originalFeatureList).not.toBeUndefined();
      });

      it('Should update the in-memory copy with socket updates', function(){
        datasetService.featureLists = [
          {name : 'universe', columns : []}
        ];

        var originalFeatureList = datasetService.featureLists[0];

        // No columns, no id
        expect(originalFeatureList.columns.length).toEqual(0);
        expect(originalFeatureList.id).toBeUndefined();

        datasetService.updateFeatureLists(serverFeatureLists);

        expect(datasetService.featureLists.length).toEqual(serverFeatureLists.length);

        // We got columns and ID
        expect(originalFeatureList.columns.length).toBeGreaterThan(0);
        expect(originalFeatureList.id).not.toBeUndefined();
      });

      it('Should update ids without overriding columns', function(){
        datasetService.featureLists = [
          {name : 'universe', columns : [
            {raw_variable_index : 1, name: 'Testcolumn1'}
          ]}
        ];

        var originalFeatureList = datasetService.featureLists[0];

        datasetService.updateFeatureLists([{id: 1, name: 'universe'}]);

        expect(datasetService.featureLists.length).toEqual(1);

        // We got ID and columns were not overwritten
        expect(originalFeatureList.columns.length).toEqual(1);
      });


    });

  });
}
);
