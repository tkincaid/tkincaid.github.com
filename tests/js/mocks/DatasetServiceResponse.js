define("DatasetServiceResponse", function(){
  return {
    getMetrics : function(){
      return [
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Rate@Top5%",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"AMS@15%tsh",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted Gini",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"R Squared",
           "recommend":false
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":false,
           "short_name":"Weighted RMSLE",
           "recommend":false
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"AMS@opt_tsh",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted Normalized RMSE",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":false,
           "short_name":"Weighted MAPE",
           "recommend":false
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Rate@Top10%",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted Normalized MAD",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":false,
           "short_name":"Weighted Gamma Deviance",
           "recommend":false
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Gini",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"MAD",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"RMSE",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted MAD",
           "recommend":true
        },
        {
           "default":false,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Coldstart RMSE",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"LogLoss",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Tweedie Deviance",
           "recommend":false
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Gini Norm",
           "recommend":true
        },
        {
           "default":false,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Coldstart MAD",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted LogLoss",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"RMSLE",
           "recommend":false
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"R Squared 20/80",
           "recommend":false
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"AUC",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Poisson Deviance",
           "recommend":false
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted RMSE",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted AUC",
           "recommend":true
        },
        {
           "default":false,
           "weighted":false,
           "weight+rec":false,
           "short_name":"NDCG",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"MAPE",
           "recommend":false
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":false,
           "short_name":"Weighted Poisson Deviance",
           "recommend":false
        },
        {
           "default":false,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Normalized MAD",
           "recommend":true
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":false,
           "short_name":"Weighted R Squared",
           "recommend":false
        },
        {
           "default":false,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Normalized RMSE",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Gamma Deviance",
           "recommend":false
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":true,
           "short_name":"Weighted Gini Norm",
           "recommend":true
        },
        {
           "default":true,
           "weighted":false,
           "weight+rec":false,
           "short_name":"Ians Metric",
           "recommend":false
        },
        {
           "default":false,
           "weighted":true,
           "weight+rec":false,
           "short_name":"Weighted Tweedie Deviance",
           "recommend":false
        }
      ];
    },

    metricOptions : function(){
      return {
        "all":[
          {"short_name":"Gini Norm"},
          {"short_name":"Weighted Gini Norm"},
          {"short_name":"R Squared"},
          {"short_name":"Weighted RMSLE"},
          {"short_name":"Weighted MAPE"},
          {"short_name":"Weighted Gamma Deviance"},
          {"short_name":"RMSE"},
          {"short_name":"Weighted MAD"},
          {"short_name":"Tweedie Deviance"},
          {"short_name":"RMSLE"},
          {"short_name":"Weighted Tweedie Deviance"},
          {"short_name":"Weighted RMSE"},
          {"short_name":"MAPE"},
          {"short_name":"Weighted Poisson Deviance"},
          {"short_name":"Weighted R Squared"},
          {"short_name":"Gamma Deviance"},
          {"short_name":"MAD"},
          {"short_name":"Poisson Deviance"}
        ],
        "recommended":{
          "default":{"short_name":"RMSE"},
          "weighted":{"short_name":"Weighted RMSE"},
          "recommender":{"short_name":"RMSE"},
          "weight+rec":{"short_name":"Weighted RMSE"}
        }
      };
    },

    aimFeaturesNotReady : function(){
      // DatasetService.features when we have names and no summary -
      // We cannot submit to aim at this point
      return [
        {"enabled":false,"expanded":false,"type":0,"name":"RefId","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"IsBadBuy","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"PurchDate","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Auction","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"VehYear","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"VehicleAge","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Make","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Model","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Trim","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"SubModel","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Color","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Transmission","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"WheelTypeID","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"WheelType","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"VehOdo","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Nationality","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"Size","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"TopThreeAmericanName","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitionAuctionAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitionAuctionCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitionRetailAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitonRetailCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentAuctionAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentAuctionCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentRetailAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentRetailCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"PRIMEUNIT","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"AUCGUART","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"BYRNO","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"VNZIP1","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"VNST","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"VehBCost","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"IsOnlineSale","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"WarrantyCost","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"high_cardinality","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"empty","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"few_values","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"TestTime","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"TestText","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"TestPercent","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
        {"enabled":false,"expanded":false,"type":0,"name":"TestLength","transform_id":0,"tab":"histogram","feature_lists":[],"profile":{}},
      ];
    },

    aimFeaturesReady : function(){
      // DatasetService.features ready for aim -
      return [
        {"enabled":false,"expanded":false,"type":0,"name":"RefId","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"RefId","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[1,3],[9.275862068965518,3],[17.551724137931036,3],[25.827586206896555,5],[34.10344827586207,4],[42.37931034482759,5],[50.65517241379311,3],[58.931034482758626,4],[67.20689655172414,3],[75.48275862068967,5],[83.75862068965517,3],[92.0344827586207,4],[100.31034482758622,2],[108.58620689655173,3],[116.86206896551725,3],[125.13793103448276,3],[133.41379310344828,4],[141.6896551724138,4],[149.96551724137933,4],[158.24137931034483,3],[166.51724137931035,2],[174.79310344827587,4],[183.0689655172414,4],[191.34482758620692,3],[199.62068965517244,2],[207.89655172413794,3],[216.17241379310346,3],[224.44827586206898,4],[232.7241379310345,3],[241.00000000000003,2],[249.27586206896552,3],[257.55172413793105,3],[265.82758620689657,4],[274.1034482758621,4],[282.3793103448276,3],[290.65517241379314,3],[298.93103448275866,4],[307.2068965517242,2],[315.48275862068965,4],[323.7586206896552,4],[332.0344827586207,3],[340.3103448275862,3],[348.58620689655174,3],[356.86206896551727,3],[365.1379310344828,4],[373.4137931034483,2],[381.68965517241384,3],[389.96551724137936,3],[398.2413793103449,3],[406.51724137931035,2],[414.7931034482759,5],[423.0689655172414,3],[431.3448275862069,5],[439.62068965517244,5],[447.89655172413796,4],[456.1724137931035,5],[464.448275862069,4],[472.72413793103453,5]],"plot2":[["121",1],["124",1],["127",1],["130",1],["133",1],["134",1],["136",1],["138",1],["139",1],["142",1],["144",1],["145",1],["148",1],["151",1],["154",1],["156",1],["157",1],["160",1],["163",1],["166",1],["169",1],["172",1],["175",1],["178",1],["181",1],["182",1],["184",1],["185",1],["187",1],["190",1],["193",1],["196",1],["199",1],["202",1],["205",1],["208",1],["211",1],["214",1],["217",1],["220",1],["223",1],["226",1],["229",1],["231",1],["232",1],["235",1],["238",1],["480",1],["481",1],["=All Other=",151]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[200,0,244.46,144.28616842927113,1,245.5,481],"raw_variable_index":0,"id":"RefId","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":200,"missing":0,"mean":244.46,"sd":144.28616842927113,"min":1,"median":245.5,"max":481},

        {"enabled":false,"expanded":false,"type":0,"name":"IsBadBuy","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"IsBadBuy","miss_count":5,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[0,134],[1,61]],"plot2":[["0.0",134],["1.0",61],["==Missing==",5]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[2,5,0.3128205128205128,0.4636419303505932,0,0,1],"raw_variable_index":1,"id":"IsBadBuy","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":2,"missing":5,"mean":0.3128205128205128,"sd":0.4636419303505932,"min":0,"median":0,"max":1},

        {"enabled":false,"expanded":false,"type":0,"name":"PurchDate","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"PurchDate","miss_count":0,"y":null,"type_label":"Date","type":"D","miss_ymean":null,"plot":[["03/31/2009",2],["04/11/2009",0],["04/21/2009",0],["05/02/2009",0],["05/13/2009",0],["05/23/2009",0],["06/03/2009",0],["06/14/2009",0],["06/24/2009",0],["07/05/2009",0],["07/16/2009",0],["07/26/2009",0],["08/06/2009",0],["08/17/2009",24],["08/27/2009",4],["09/07/2009",0],["09/18/2009",0],["09/28/2009",0],["10/09/2009",0],["10/19/2009",0],["10/30/2009",0],["11/10/2009",0],["11/20/2009",0],["12/01/2009",4],["12/12/2009",7],["12/22/2009",5],["01/02/2010",7],["01/13/2010",8],["01/23/2010",12],["02/03/2010",4],["02/14/2010",7],["02/24/2010",9],["03/07/2010",8],["03/18/2010",4],["03/28/2010",12],["04/08/2010",5],["04/19/2010",4],["04/29/2010",8],["05/10/2010",3],["05/20/2010",2],["05/31/2010",1],["06/11/2010",0],["06/21/2010",0],["07/02/2010",0],["07/13/2010",0],["07/23/2010",0],["08/03/2010",0],["08/14/2010",0],["08/24/2010",0],["09/04/2010",0],["09/15/2010",0],["09/25/2010",7],["10/06/2010",3],["10/17/2010",13],["10/27/2010",3],["11/07/2010",9],["11/18/2010",9],["11/28/2010",8],["12/09/2010",1],["12/20/2010",7]],"plot2":[["02/18/2009",2],["08/19/2009",10],["08/26/2009",14],["09/02/2009",4],["12/07/2009",4],["12/14/2009",4],["12/21/2009",3],["12/28/2009",5],["01/04/2010",2],["01/11/2010",5],["01/18/2010",8],["01/25/2010",9],["02/01/2010",3],["02/08/2010",4],["02/22/2010",5],["03/01/2010",9],["03/08/2010",5],["03/15/2010",3],["03/22/2010",4],["03/29/2010",7],["04/05/2010",5],["04/12/2010",3],["04/26/2010",4],["05/03/2010",4],["05/10/2010",4],["05/17/2010",3],["09/27/2010",2],["10/04/2010",5],["10/11/2010",3],["10/18/2010",4],["10/25/2010",9],["11/01/2010",3],["11/08/2010",5],["11/15/2010",4],["11/22/2010",9],["11/29/2010",4],["12/06/2010",4],["12/20/2010",7],["=All Other=",8]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[43,0,"04/13/2010",158.72407810017359,"02/18/2009","03/15/2010","12/20/2010"],"raw_variable_index":2,"id":"PurchDate","types":{"category":false,"conversion":"%m/%d/%Y","text":false,"numeric":true,"currency":false,"length":false,"date":true,"percentage":false,"nastring":false},"unique":43,"missing":0,"mean":"04/13/2010","sd":158.72407810017359,"min":"02/18/2009","median":"03/15/2010","max":"12/20/2010"},

        {"enabled":false,"expanded":false,"type":0,"name":"Auction","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":null,"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"empty":false,"few_values":true},
          "summary":[1,0],"raw_variable_index":3,"id":"Auction","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":1,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"VehYear","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"VehYear","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[2001,8],[2002,12],[2003,23],[2004,43],[2005,39],[2006,39],[2007,24],[2008,12]],"plot2":[["2001",8],["2002",12],["2003",23],["2004",43],["2005",39],["2006",39],["2007",24],["2008",9],["2009",3]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[9,0,2004.845,1.775098588885209,2001,2005,2009],"raw_variable_index":4,"id":"VehYear","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":9,"missing":0,"mean":2004.845,"sd":1.775098588885209,"min":2001,"median":2005,"max":2009},

        {"enabled":false,"expanded":false,"type":0,"name":"VehicleAge","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"VehicleAge","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[1,3],[2,16],[3,23],[4,43],[5,39],[6,39],[7,23],[8,7],[9,7]],"plot2":[["1",3],["2",16],["3",23],["4",43],["5",39],["6",39],["7",23],["8",7],["9",7]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[9,0,4.925,1.777463079785344,1,5,9],"raw_variable_index":5,"id":"VehicleAge","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":9,"missing":0,"mean":4.925,"sd":1.777463079785344,"min":1,"median":5,"max":9},

        {"enabled":false,"expanded":false,"type":0,"name":"Make","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"Make","type_label":"Categorical","plot":[["=All Other=",7],["BUICK",3],["CHEVROLET",45],["CHRYSLER",19],["DODGE",28],["FORD",35],["HYUNDAI",12],["JEEP",5],["KIA",6],["MAZDA",7],["MERCURY",3],["NISSAN",5],["PONTIAC",11],["SATURN",10],["TOYOTA",4]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[19,0],"raw_variable_index":6,"id":"Make","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":19,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"Model","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":null,"synched":true,"transform_args":[],"low_info":{"high_cardinality":true,"duplicate":false,"empty":false},
          "summary":[106,0],"raw_variable_index":7,"id":"Model","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":106,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"Trim","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"Trim","type_label":"Categorical","plot":[["150",1],["2",5],["==Missing==",7],["=All Other=",9],["Adv",2],["Bas",49],["CE",3],["CXL",1],["EX",5],["Edd",2],["GL",3],["GLS",9],["GS",2],["GT",1],["GXP",1],["L20",2],["LS",22],["LT",7],["LX",5],["Lar",1],["Lim",1],["SE",18],["SEL",3],["SLT",1],["SR5",1],["ST",5],["SXT",8],["Spo",5],["Tou",6],["XLS",2],["XLT",3],["ZTS",1],["ZX3",2],["i",4],["s",3]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[42,7],"raw_variable_index":8,"id":"Trim","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":42,"missing":7,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"SubModel","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"SubModel","type_label":"Categorical","plot":[["2D CONVERTIBLE TOURING",1],["2D COUPE",9],["2D COUPE LS",2],["2D COUPE ZX3",2],["2D SUV 4.0L SPORT",1],["4D CUV 3.0L SEL",1],["4D SEDAN",52],["4D SEDAN 2.7L",1],["4D SEDAN CE",3],["4D SEDAN EX",2],["4D SEDAN GLS",2],["4D SEDAN GS",2],["4D SEDAN GXP",1],["4D SEDAN I",2],["4D SEDAN L200",2],["4D SEDAN LEVEL 1",1],["4D SEDAN LEVEL 2",5],["4D SEDAN LS",5],["4D SEDAN LT",1],["4D SEDAN LT 3.5L",2],["4D SEDAN LT 3.9L",1],["4D SEDAN LTZ",1],["4D SEDAN LX",2],["4D SEDAN SE",6],["4D SEDAN SXT",2],["4D SEDAN ZTS",1],["4D SPORT TOURER",2],["4D SPORT TOURING",2],["4D SPORT UTILITY EX",2],["4D SPORT UTILITY I",2],["4D SPORT UTILITY SPORT EDITION",1],["4D SUV",2],["4D SUV 2.2L LS",3],["4D SUV 4.2L",2],["4D SUV 4.2L LS",4],["4D SUV 4.6L XLT",2],["4D SUV 4.7L ADVENTURER",2],["4D SUV 5.4L EDDIE BAUER",2],["4D SUV LS",2],["4D WAGON",4],["4D WAGON LAREDO",1],["=All Other=",45],["EXT CAB 4.8L",2],["MAZDA3 4D I SPORT",1],["MINIVAN 3.3L",2],["MINIVAN 3.8L",1],["PASSENGER 3.5L LS",1],["PASSENGER 3.8L LX",2],["PASSENGER 3.9L LX",2],["PASSENGER EXT 3.5L",1]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[94,0],"raw_variable_index":9,"id":"SubModel","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":94,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"Color","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"Color","type_label":"Categorical","plot":[["=All Other=",7],["BEIGE",4],["BLACK",17],["BLUE",26],["GOLD",20],["GREEN",13],["GREY",22],["RED",26],["SILVER",40],["WHITE",25]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[12,0],"raw_variable_index":10,"id":"Color","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":12,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"Transmission","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"Transmission","type_label":"Categorical","plot":[["AUTO",186],["MANUAL",14]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[2,0],"raw_variable_index":11,"id":"Transmission","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":2,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"WheelTypeID","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"WheelTypeID","miss_count":20,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[1,96],[2,81],[3,3]],"plot2":[["1.0",96],["2.0",81],["3.0",3],["==Missing==",20]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[3,20,1.4833333333333334,0.5320296566504114,1,1,3],"raw_variable_index":12,"id":"WheelTypeID","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":3,"missing":20,"mean":1.4833333333333334,"sd":0.5320296566504114,"min":1,"median":1,"max":3},

        {"enabled":false,"expanded":false,"type":0,"name":"WheelType","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"WheelType","type_label":"Categorical","plot":[["==Missing==",20],["Alloy",96],["Covers",81],["Special",3]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[3,20],"raw_variable_index":13,"id":"WheelType","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":3,"missing":20,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"VehOdo","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"VehOdo","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[39911.31827586207,4],[40901.94,2],[41892.56172413794,0],[42883.183448275864,0],[43873.8051724138,0],[44864.426896551726,0],[45855.04862068966,1],[46845.67034482759,3],[47836.29206896552,0],[48826.91379310345,1],[49817.535517241384,4],[50808.15724137931,2],[51798.778965517246,2],[52789.40068965517,3],[53780.02241379311,2],[54770.644137931034,1],[55761.26586206897,5],[56751.887586206896,2],[57742.50931034483,2],[58733.131034482765,6],[59723.75275862069,8],[60714.37448275862,4],[61704.996206896554,4],[62695.61793103449,7],[63686.239655172416,7],[64676.86137931034,7],[65667.48310344828,7],[66658.10482758621,2],[67648.72655172413,5],[68639.34827586207,2],[69629.97,4],[70620.59172413794,4],[71611.21344827587,3],[72601.83517241379,2],[73592.45689655172,11],[74583.07862068966,4],[75573.70034482758,6],[76564.32206896553,2],[77554.94379310345,5],[78545.56551724138,9],[79536.18724137932,5],[80526.80896551724,5],[81517.43068965517,7],[82508.0524137931,5],[83498.67413793103,2],[84489.29586206898,6],[85479.9175862069,5],[86470.53931034483,3],[87461.16103448276,2],[88451.78275862068,2],[89442.40448275862,4],[90433.02620689655,0],[91423.64793103447,1],[92414.26965517242,2],[93404.89137931034,1],[94395.51310344828,1],[95386.13482758621,2],[96376.75655172413,1],[97367.37827586207,2],[98358,1]],"plot2":[["46237",1],["49408",1],["49893",1],["50385",1],["56039",1],["58024",1],["59072",1],["59287",1],["59391",1],["59634",1],["60060",1],["60559",1],["61081",1],["61184",1],["63079",2],["64139",1],["64677",1],["65711",1],["65795",1],["66433",1],["66708",1],["67744",1],["67795",1],["71423",1],["71542",1],["73870",1],["73963",1],["75513",2],["78593",1],["78992",1],["79015",1],["79081",1],["79092",1],["79576",1],["80120",1],["80531",1],["81646",1],["82164",1],["82617",1],["83081",1],["83135",1],["87008",1],["87775",1],["89240",1],["89750",1],["89769",1],["91888",1],["92865",1],["95443",1],["=All Other=",149]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[198,0,70718.535,13432.746139891688,32671,71482.5,98358],"raw_variable_index":14,"id":"VehOdo","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":198,"missing":0,"mean":70718.535,"sd":13432.746139891688,"min":32671,"median":71482.5,"max":98358},

        {"enabled":false,"expanded":false,"type":0,"name":"Nationality","transform_id":0,"tab":"histogram","feature_lists":[],
        "profile":{"y":null,"type":"C","name":"Nationality","type_label":"Categorical","plot":[["AMERICAN",161],["OTHER",1],["OTHER ASIAN",28],["TOP LINE ASIAN",10]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[4,0],"raw_variable_index":15,"id":"Nationality","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":4,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"Size","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"Size","type_label":"Categorical","plot":[["=All Other=",7],["COMPACT",26],["LARGE",20],["LARGE SUV",7],["LARGE TRUCK",5],["MEDIUM",78],["MEDIUM SUV",21],["SMALL SUV",9],["SPECIALTY",7],["SPORTS",6],["VAN",14]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[12,0],"raw_variable_index":16,"id":"Size","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":12,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"TopThreeAmericanName","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"TopThreeAmericanName","type_label":"Categorical","plot":[["CHRYSLER",52],["FORD",38],["GM",71],["OTHER",39]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[4,0],"raw_variable_index":17,"id":"TopThreeAmericanName","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":4,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitionAuctionAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRAcquisitionAuctionAveragePrice","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[455.76724137931035,1],[632.5,0],[809.2327586206897,0],[985.9655172413793,0],[1162.6982758620688,0],[1339.4310344827586,0],[1516.1637931034484,1],[1692.896551724138,3],[1869.6293103448274,3],[2046.3620689655172,2],[2223.094827586207,4],[2399.8275862068967,2],[2576.560344827586,7],[2753.293103448276,1],[2930.0258620689656,6],[3106.758620689655,5],[3283.4913793103447,4],[3460.2241379310344,3],[3636.956896551724,7],[3813.689655172414,4],[3990.4224137931033,5],[4167.1551724137935,9],[4343.887931034483,1],[4520.620689655172,6],[4697.353448275862,7],[4874.086206896552,6],[5050.818965517241,3],[5227.551724137931,4],[5404.2844827586205,9],[5581.01724137931,7],[5757.75,5],[5934.482758620689,4],[6111.2155172413795,6],[6287.948275862069,6],[6464.681034482758,3],[6641.413793103448,4],[6818.146551724138,7],[6994.879310344828,6],[7171.612068965517,6],[7348.3448275862065,4],[7525.077586206897,6],[7701.810344827586,2],[7878.543103448275,3],[8055.275862068966,4],[8232.008620689656,0],[8408.741379310344,0],[8585.474137931034,1],[8762.206896551725,5],[8938.939655172413,2],[9115.672413793103,1],[9292.405172413793,0],[9469.137931034482,1],[9645.870689655172,1],[9822.603448275862,6],[9999.33620689655,1],[10176.068965517241,1],[10352.801724137931,0],[10529.53448275862,0],[10706.26724137931,0],[10883,5]],"plot2":[["10883",2],["1708",1],["1893",1],["1915",1],["2136",1],["2236",1],["2747",1],["3242",1],["3442",1],["3443",1],["3501",1],["3666",1],["3667",1],["3678",1],["4083",1],["4196",2],["4225",1],["4245",1],["4690",1],["4704",1],["5462",1],["5498",1],["5524",2],["5566",1],["5610",1],["5703",1],["5736",1],["5998",1],["6216",1],["6217",1],["6261",1],["6269",1],["6487",1],["6608",1],["6770",1],["6789",1],["6867",2],["6976",2],["7020",1],["7033",1],["7182",2],["7533",1],["7548",1],["8015",1],["8028",1],["8811",1],["8817",1],["9846",1],["9900",1],["=All Other=",145]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[194,0,5629.07,2315.779196102254,0,5511,12693],"raw_variable_index":18,"id":"MMRAcquisitionAuctionAveragePrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":194,"missing":0,"mean":5629.07,"sd":2315.779196102254,"min":0,"median":5511,"max":12693},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitionAuctionCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRAcquisitionAuctionCleanPrice","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[1210.3103448275863,1],[1396,0],[1581.6896551724137,0],[1767.3793103448274,0],[1953.0689655172414,0],[2138.758620689655,0],[2324.448275862069,3],[2510.137931034483,2],[2695.8275862068967,2],[2881.5172413793102,3],[3067.2068965517237,3],[3252.8965517241377,2],[3438.5862068965516,4],[3624.2758620689656,3],[3809.965517241379,4],[3995.655172413793,6],[4181.3448275862065,3],[4367.0344827586205,4],[4552.724137931034,5],[4738.4137931034475,7],[4924.103448275862,4],[5109.793103448275,4],[5295.482758620689,7],[5481.172413793103,7],[5666.862068965517,1],[5852.551724137931,6],[6038.241379310344,7],[6223.931034482758,3],[6409.620689655172,3],[6595.310344827586,9],[6781,4],[6966.689655172413,6],[7152.379310344827,7],[7338.068965517241,6],[7523.758620689655,5],[7709.448275862069,2],[7895.137931034482,10],[8080.827586206896,2],[8266.51724137931,5],[8452.206896551725,6],[8637.896551724138,3],[8823.58620689655,3],[9009.275862068964,6],[9194.965517241379,1],[9380.655172413793,1],[9566.344827586207,2],[9752.03448275862,3],[9937.724137931034,0],[10123.413793103447,2],[10309.103448275862,2],[10494.793103448275,4],[10680.482758620688,3],[10866.172413793103,1],[11051.862068965516,2],[11237.551724137931,1],[11423.241379310344,3],[11608.931034482757,0],[11794.620689655172,1],[11980.310344827585,1],[12166,5]],"plot2":[["10486",1],["10563",1],["10569",1],["10815",1],["11097",1],["12057",1],["12166",2],["12950",1],["14504",1],["2423",1],["2669",1],["2675",1],["3158",1],["3491",1],["3564",1],["3786",1],["3939",1],["4419",1],["4868",1],["5053",2],["5244",1],["5928",1],["5990",1],["5998",1],["6001",1],["6068",1],["6216",1],["6610",1],["6643",2],["6768",1],["6876",2],["7027",1],["7176",2],["7204",2],["7238",1],["7271",1],["7495",1],["7767",1],["8021",1],["8057",1],["8063",1],["8067",1],["8296",1],["8314",1],["8799",1],["9026",1],["9075",1],["9078",1],["9851",1],["=All Other=",145]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[194,0,6823.38,2545.2848044177695,0,6682,14504],"raw_variable_index":19,"id":"MMRAcquisitionAuctionCleanPrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":194,"missing":0,"mean":6823.38,"sd":2545.2848044177695,"min":0,"median":6682,"max":14504},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitionRetailAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRAcquisitionRetailAveragePrice","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[2433.4612068965516,3],[2639.75,0],[2846.0387931034484,0],[3052.3275862068967,1],[3258.616379310345,1],[3464.905172413793,0],[3671.1939655172414,1],[3877.4827586206898,1],[4083.7715517241377,2],[4290.060344827586,2],[4496.349137931034,3],[4702.637931034483,4],[4908.926724137931,1],[5115.2155172413795,3],[5321.504310344828,4],[5527.793103448275,5],[5734.081896551725,2],[5940.370689655172,3],[6146.659482758621,5],[6352.948275862069,7],[6559.237068965517,5],[6765.525862068966,3],[6971.814655172414,8],[7178.103448275862,5],[7384.392241379311,4],[7590.681034482759,7],[7796.969827586207,6],[8003.258620689656,6],[8209.547413793103,5],[8415.83620689655,10],[8622.125,1],[8828.41379310345,2],[9034.702586206897,6],[9240.991379310344,7],[9447.280172413793,2],[9653.568965517243,5],[9859.85775862069,7],[10066.146551724138,3],[10272.435344827587,7],[10478.724137931034,6],[10685.012931034482,4],[10891.301724137931,3],[11097.59051724138,7],[11303.879310344828,2],[11510.168103448275,4],[11716.456896551725,2],[11922.745689655172,2],[12129.034482758621,3],[12335.323275862069,1],[12541.612068965518,2],[12747.900862068966,3],[12954.189655172415,2],[13160.478448275862,0],[13366.767241379312,1],[13573.056034482759,3],[13779.344827586207,0],[13985.633620689656,1],[14191.922413793103,2],[14398.211206896553,1],[14604.5,4]],"plot2":[["10044",1],["10055",1],["10072",1],["10090",1],["10323",1],["10572",1],["10611",1],["10836",1],["10865",1],["11299",1],["11300",1],["11354",1],["11607",2],["11636",1],["11715",1],["12758",1],["13671",1],["13679",1],["13687",1],["14036",1],["16310",1],["3673",1],["4281",1],["5209",1],["5470",1],["5731",1],["6222",1],["6240",1],["6466",2],["6502",1],["6726",1],["6986",1],["6993",1],["7003",1],["7038",1],["7137",2],["7234",1],["7462",1],["7488",1],["7778",1],["7812",1],["7832",1],["8035",1],["8066",1],["8517",1],["8540",1],["9323",1],["9803",1],["9925",1],["=All Other=",148]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[197,0,8679.245,2843.1363641188586,0,8465.5,16545],"raw_variable_index":20,"id":"MMRAcquisitionRetailAveragePrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":197,"missing":0,"mean":8679.245,"sd":2843.1363641188586,"min":0,"median":8465.5,"max":16545},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRAcquisitonRetailCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRAcquisitonRetailCleanPrice","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[3713.8943103448278,3],[3928.75,1],[4143.605689655173,1],[4358.4613793103445,0],[4573.317068965517,0],[4788.17275862069,2],[5003.028448275862,0],[5217.884137931034,2],[5432.739827586207,1],[5647.59551724138,5],[5862.451206896552,4],[6077.306896551724,2],[6292.162586206897,2],[6507.018275862069,0],[6721.873965517241,5],[6936.729655172414,5],[7151.585344827587,2],[7366.441034482759,6],[7581.296724137931,6],[7796.152413793104,3],[8011.0081034482755,8],[8225.863793103448,5],[8440.71948275862,3],[8655.575172413794,7],[8870.430862068966,2],[9085.286551724137,6],[9300.142241379312,9],[9514.997931034482,7],[9729.853620689655,3],[9944.709310344828,4],[10159.565,2],[10374.420689655173,10],[10589.276379310344,8],[10804.132068965519,3],[11018.98775862069,4],[11233.843448275862,7],[11448.699137931035,3],[11663.554827586207,4],[11878.41051724138,4],[12093.266206896551,3],[12308.121896551726,5],[12522.977586206896,4],[12737.833275862069,7],[12952.688965517242,0],[13167.544655172414,3],[13382.400344827587,0],[13597.25603448276,6],[13812.111724137932,5],[14026.967413793103,0],[14241.823103448276,1],[14456.678793103449,2],[14671.534482758621,3],[14886.390172413794,2],[15101.245862068967,1],[15316.10155172414,0],[15530.95724137931,2],[15745.812931034483,1],[15960.668620689656,1],[16175.524310344828,1],[16390.38,4]],"plot2":[["10006",1],["10108",1],["10570",1],["10634",2],["10838",1],["11079",1],["11343",1],["11384",1],["11538",1],["11805",1],["11866",1],["11910",1],["12389",1],["12611",1],["12628",1],["12875",1],["12883",1],["13240",1],["13654",1],["13656",1],["13928",1],["13988",1],["14681",1],["15253",1],["15739",1],["15991",1],["5314",1],["5378",1],["5707",1],["5951",1],["6172",1],["6472",1],["6751",1],["7274",1],["7515",1],["7674",2],["7747",1],["8010",1],["8089",2],["8270",1],["8814",1],["8820",1],["8825",1],["9312",1],["9314",1],["9557",1],["9579",1],["9586",1],["9822",1],["=All Other=",148]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[197,0,10073.24,3016.961647485761,0,9965.5,18175],"raw_variable_index":21,"id":"MMRAcquisitonRetailCleanPrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":197,"missing":0,"mean":10073.24,"sd":3016.961647485761,"min":0,"median":9965.5,"max":18175},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentAuctionAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRCurrentAuctionAveragePrice","miss_count":1,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[86.06551724137935,2],[271,0],[455.9344827586207,0],[640.8689655172413,0],[825.8034482758619,0],[1010.7379310344826,0],[1195.6724137931033,0],[1380.6068965517238,0],[1565.5413793103446,2],[1750.4758620689652,3],[1935.4103448275857,1],[2120.3448275862065,1],[2305.279310344827,3],[2490.2137931034476,4],[2675.148275862068,5],[2860.082758620689,11],[3045.01724137931,5],[3229.9517241379303,3],[3414.886206896551,7],[3599.8206896551715,6],[3784.7551724137925,3],[3969.689655172413,4],[4154.624137931034,4],[4339.558620689654,6],[4524.493103448275,2],[4709.427586206895,6],[4894.362068965516,3],[5079.296551724136,7],[5264.231034482757,6],[5449.165517241378,7],[5634.0999999999985,8],[5819.03448275862,6],[6003.96896551724,7],[6188.903448275861,6],[6373.837931034482,4],[6558.772413793102,7],[6743.706896551723,4],[6928.641379310343,6],[7113.575862068964,7],[7298.510344827585,5],[7483.444827586205,5],[7668.379310344826,3],[7853.313793103446,1],[8038.248275862067,3],[8223.18275862069,4],[8408.117241379308,0],[8593.05172413793,2],[8777.98620689655,1],[8962.92068965517,2],[9147.85517241379,3],[9332.789655172412,1],[9517.724137931033,0],[9702.658620689654,0],[9887.593103448273,0],[10072.527586206894,4],[10257.462068965515,2],[10442.396551724136,1],[10627.331034482757,0],[10812.265517241376,2],[10997.199999999997,4]],"plot2":[["0.0",2],["10310.0",1],["10563.0",1],["11386.0",1],["11871.0",1],["1626.0",1],["2128.0",1],["2422.0",1],["2659.0",1],["2685.0",1],["2747.0",1],["2895.0",1],["2897.0",1],["2902.0",1],["2908.0",1],["2960.0",2],["3094.0",1],["3160.0",1],["3242.0",2],["3405.0",1],["3426.0",1],["3464.0",1],["3671.0",1],["3676.0",1],["3718.0",1],["4009.0",1],["4267.0",2],["4446.0",1],["4725.0",1],["4974.0",1],["5090.0",1],["5199.0",1],["5451.0",1],["5456.0",1],["5500.0",1],["5733.0",1],["5747.0",1],["5960.0",1],["6001.0",1],["6263.0",1],["6491.0",1],["7016.0",1],["7033.0",1],["7219.0",1],["7651.0",1],["8055.0",1],["8320.0",1],["8379.0",1],["9316.0",1],["==Missing==",1],["=All Other=",146]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[195,1,5630.743718592965,2374.1823495241892,0,5526,12290],"raw_variable_index":22,"id":"MMRCurrentAuctionAveragePrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":195,"missing":1,"mean":5630.743718592965,"sd":2374.1823495241892,"min":0,"median":5526,"max":12290},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentAuctionCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRCurrentAuctionCleanPrice","miss_count":1,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[812.9468965517242,2],[1012,0],[1211.0531034482758,0],[1410.1062068965516,0],[1609.1593103448276,0],[1808.2124137931032,0],[2007.2655172413793,0],[2206.3186206896553,1],[2405.371724137931,1],[2604.4248275862064,3],[2803.4779310344825,2],[3002.5310344827585,1],[3201.584137931034,5],[3400.63724137931,2],[3599.6903448275857,6],[3798.7434482758617,3],[3997.7965517241373,11],[4196.849655172413,4],[4395.902758620689,4],[4594.955862068965,6],[4794.008965517241,3],[4993.062068965517,5],[5192.115172413793,3],[5391.168275862068,6],[5590.221379310344,5],[5789.27448275862,5],[5988.327586206896,5],[6187.380689655171,3],[6386.433793103448,4],[6585.4868965517235,10],[6784.539999999999,6],[6983.593103448275,8],[7182.646206896551,3],[7381.699310344827,4],[7580.752413793102,9],[7779.805517241379,8],[7978.858620689654,3],[8177.91172413793,6],[8376.964827586206,7],[8576.017931034483,3],[8775.071034482757,6],[8974.124137931034,3],[9173.17724137931,4],[9372.230344827585,2],[9571.28344827586,2],[9770.336551724136,3],[9969.389655172412,1],[10168.442758620688,1],[10367.495862068965,1],[10566.54896551724,2],[10765.602068965516,0],[10964.655172413792,1],[11163.708275862067,2],[11362.761379310343,0],[11561.814482758618,2],[11760.867586206896,3],[11959.920689655171,1],[12158.973793103447,2],[12358.026896551723,2],[12557.079999999998,4]],"plot2":[["0.0",2],["10419.0",1],["10988.0",1],["11611.0",1],["12184.0",1],["12403.0",1],["13466.0",1],["2690.0",1],["3700.0",1],["3701.0",1],["3702.0",1],["3721.0",1],["4218.0",1],["4782.0",1],["4823.0",1],["5190.0",1],["5324.0",1],["5718.0",1],["5737.0",1],["5783.0",1],["5796.0",1],["6336.0",1],["6341.0",1],["6732.0",1],["6794.0",1],["6830.0",1],["6837.0",1],["6851.0",1],["7051.0",1],["7359.0",1],["7386.0",1],["7595.0",1],["7649.0",1],["7813.0",1],["7866.0",1],["7876.0",1],["7923.0",1],["7933.0",1],["8040.0",1],["8412.0",1],["8441.0",1],["8835.0",1],["8837.0",1],["8897.0",1],["8934.0",1],["9174.0",1],["9344.0",1],["9370.0",1],["9388.0",1],["==Missing==",1],["=All Other=",149]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[198,1,6858.954773869346,2640.1741439756224,0,6774,14671],"raw_variable_index":23,"id":"MMRCurrentAuctionCleanPrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":198,"missing":1,"mean":6858.954773869346,"sd":2640.1741439756224,"min":0,"median":6774,"max":14671},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentRetailAveragePrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRCurrentRetailAveragePrice","miss_count":1,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[2441.0010344827588,3],[2650.5,0],[2859.9989655172412,0],[3069.4979310344825,1],[3278.9968965517237,1],[3488.495862068965,2],[3697.994827586206,0],[3907.493793103448,0],[4116.992758620689,1],[4326.49172413793,4],[4535.990689655171,2],[4745.489655172412,1],[4954.9886206896535,3],[5164.487586206896,3],[5373.986551724137,6],[5583.485517241378,3],[5792.984482758619,4],[6002.483448275861,6],[6211.982413793102,5],[6421.481379310343,8],[6630.980344827584,7],[6840.4793103448255,6],[7049.978275862067,3],[7259.477241379308,4],[7468.976206896549,7],[7678.475172413791,2],[7887.974137931033,5],[8097.473103448274,4],[8306.972068965515,2],[8516.471034482756,12],[8725.969999999998,5],[8935.468965517239,3],[9144.96793103448,2],[9354.466896551721,11],[9563.965862068962,5],[9773.464827586204,8],[9982.963793103445,6],[10192.462758620686,2],[10401.961724137927,3],[10611.460689655169,8],[10820.95965517241,4],[11030.458620689651,5],[11239.957586206892,3],[11449.456551724134,4],[11658.955517241375,1],[11868.454482758616,0],[12077.953448275857,1],[12287.452413793098,4],[12496.95137931034,1],[12706.450344827583,0],[12915.949310344824,1],[13125.448275862065,3],[13334.947241379306,2],[13544.446206896548,1],[13753.945172413789,2],[13963.44413793103,2],[14172.943103448271,1],[14382.442068965513,1],[14591.941034482754,1],[14801.439999999995,4]],"plot2":[["0.0",2],["10148.0",1],["10323.0",1],["10354.0",1],["10728.0",1],["10797.0",1],["10866.0",1],["11135.0",1],["11136.0",1],["11330.0",1],["11340.0",1],["11393.0",1],["11597.0",1],["11643.0",1],["12372.0",1],["12382.0",1],["13394.0",1],["13446.0",1],["14209.0",1],["15959.0",1],["2373.0",1],["3673.0",1],["4957.0",1],["5224.0",1],["5498.0",1],["5832.0",1],["5997.0",1],["6132.0",1],["6222.0",1],["6230.0",1],["6517.0",1],["6526.0",1],["6602.0",2],["6774.0",1],["6911.0",1],["7028.0",1],["7512.0",1],["8523.0",1],["8540.0",1],["8546.0",1],["8548.0",1],["8576.0",1],["8850.0",1],["9545.0",1],["9605.0",1],["9610.0",1],["9792.0",1],["9808.0",1],["9969.0",1],["==Missing==",1],["=All Other=",148]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[197,1,8638.934673366834,2889.6415090818473,0,8620,16028],"raw_variable_index":24,"id":"MMRCurrentRetailAveragePrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":197,"missing":1,"mean":8638.934673366834,"sd":2889.6415090818473,"min":0,"median":8620,"max":16028},

        {"enabled":false,"expanded":false,"type":0,"name":"MMRCurrentRetailCleanPrice","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"MMRCurrentRetailCleanPrice","miss_count":1,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[3339.44,3],[3563,0],[3786.56,0],[4010.12,1],[4233.679999999999,0],[4457.24,2],[4680.799999999999,0],[4904.36,1],[5127.919999999999,1],[5351.48,5],[5575.039999999999,1],[5798.5999999999985,3],[6022.159999999999,5],[6245.719999999999,1],[6469.279999999999,2],[6692.839999999998,7],[6916.399999999999,3],[7139.959999999999,3],[7363.519999999999,5],[7587.079999999998,6],[7810.6399999999985,8],[8034.199999999998,4],[8257.759999999998,4],[8481.319999999998,5],[8704.879999999997,3],[8928.439999999999,6],[9151.999999999998,8],[9375.559999999998,3],[9599.119999999999,8],[9822.679999999997,9],[10046.239999999998,4],[10269.799999999997,2],[10493.359999999997,7],[10716.919999999998,4],[10940.479999999996,4],[11164.039999999997,8],[11387.599999999997,5],[11611.159999999996,2],[11834.719999999998,5],[12058.279999999997,7],[12281.839999999997,7],[12505.399999999996,5],[12728.959999999997,1],[12952.519999999997,1],[13176.079999999996,6],[13399.639999999996,2],[13623.199999999997,1],[13846.759999999997,2],[14070.319999999996,0],[14293.879999999996,1],[14517.439999999995,3],[14740.999999999996,2],[14964.559999999996,0],[15188.119999999995,1],[15411.679999999995,1],[15635.239999999996,1],[15858.799999999996,5],[16082.359999999995,0],[16305.919999999995,1],[16529.479999999996,4]],"plot2":[["0.0",2],["10567.0",1],["10570.0",1],["10587.0",1],["10815.0",1],["11054.0",1],["11204.0",1],["11326.0",1],["11356.0",1],["11404.0",1],["11454.0",1],["12052.0",1],["12122.0",2],["12152.0",1],["12399.0",1],["12409.0",1],["12522.0",2],["12573.0",1],["12984.0",1],["13398.0",1],["13893.0",1],["14575.0",1],["14657.0",1],["15980.0",1],["17274.0",1],["18815.0",1],["4460.0",1],["4497.0",1],["5742.0",1],["5825.0",1],["5987.0",1],["6051.0",1],["6209.0",1],["6773.0",1],["7512.0",1],["7724.0",1],["7766.0",1],["7778.0",1],["7961.0",1],["8060.0",1],["8115.0",1],["8562.0",1],["8563.0",1],["8771.0",1],["8800.0",1],["9035.0",1],["9225.0",1],["9288.0",1],["9329.0",1],["==Missing==",1],["=All Other=",147]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[196,1,10001.809045226131,3124.8342948981226,0,9919,18815],"raw_variable_index":25,"id":"MMRCurrentRetailCleanPrice","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":196,"missing":1,"mean":10001.809045226131,"sd":3124.8342948981226,"min":0,"median":9919,"max":18815},

        {"enabled":false,"expanded":false,"type":0,"name":"PRIMEUNIT","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"PRIMEUNIT","type_label":"Categorical","plot":[["==Missing==",184],["NO",16]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[1,184],"raw_variable_index":26,"id":"PRIMEUNIT","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":1,"missing":184,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"AUCGUART","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"AUCGUART","type_label":"Categorical","plot":[["==Missing==",184],["GREEN",16]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[1,184],"raw_variable_index":27,"id":"AUCGUART","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":1,"missing":184,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"BYRNO","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"BYRNO","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[5546,69],[5829.224137931034,0],[6112.448275862069,0],[6395.672413793103,0],[6678.896551724138,0],[6962.120689655172,0],[7245.3448275862065,0],[7528.568965517241,0],[7811.793103448275,0],[8095.01724137931,0],[8378.241379310344,0],[8661.465517241379,0],[8944.689655172413,0],[9227.91379310345,0],[9511.137931034482,0],[9794.362068965518,0],[10077.58620689655,0],[10360.810344827587,0],[10644.03448275862,0],[10927.258620689656,0],[11210.482758620688,0],[11493.706896551725,0],[11776.931034482757,0],[12060.155172413793,0],[12343.379310344828,0],[12626.603448275862,0],[12909.827586206897,0],[13193.051724137931,0],[13476.275862068966,0],[13759.5,0],[14042.724137931034,0],[14325.948275862069,0],[14609.172413793103,0],[14892.396551724138,0],[15175.620689655172,0],[15458.844827586207,0],[15742.068965517241,0],[16025.293103448275,0],[16308.51724137931,0],[16591.741379310344,0],[16874.965517241377,0],[17158.189655172413,0],[17441.41379310345,10],[17724.637931034482,0],[18007.862068965514,0],[18291.08620689655,0],[18574.310344827587,0],[18857.53448275862,1],[19140.758620689656,0],[19423.98275862069,82],[19707.206896551725,0],[19990.43103448276,0],[20273.655172413793,0],[20556.879310344826,18],[20840.103448275862,15],[21123.3275862069,0],[21406.55172413793,0],[21689.775862068964,0],[21973,5]],"plot2":[["17675",10],["19064",1],["19619",25],["19638",57],["20740",18],["20928",15],["21973",5],["5546",69]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[8,0,14927.17,6847.225075247637,5546,19619,21973],"raw_variable_index":28,"id":"BYRNO","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":8,"missing":0,"mean":14927.17,"sd":6847.225075247637,"min":5546,"median":19619,"max":21973},

        {"enabled":false,"expanded":false,"type":0,"name":"VNZIP1","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"VNZIP1","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[20166,30],[20397.94827586207,0],[20629.896551724138,0],[20861.844827586207,0],[21093.793103448275,0],[21325.741379310344,0],[21557.689655172413,0],[21789.637931034482,0],[22021.58620689655,0],[22253.53448275862,0],[22485.48275862069,0],[22717.431034482757,0],[22949.379310344826,0],[23181.327586206895,0],[23413.275862068964,0],[23645.224137931036,0],[23877.1724137931,0],[24109.120689655174,0],[24341.06896551724,0],[24573.01724137931,0],[24804.965517241377,0],[25036.91379310345,0],[25268.862068965518,0],[25500.810344827587,0],[25732.758620689656,0],[25964.706896551725,0],[26196.655172413793,0],[26428.603448275862,0],[26660.55172413793,0],[26892.5,0],[27124.44827586207,0],[27356.396551724138,0],[27588.344827586207,0],[27820.293103448275,0],[28052.241379310344,0],[28284.189655172413,0],[28516.137931034482,0],[28748.08620689655,0],[28980.03448275862,0],[29211.98275862069,0],[29443.931034482757,0],[29675.879310344826,0],[29907.8275862069,0],[30139.775862068964,0],[30371.724137931036,0],[30603.6724137931,0],[30835.620689655174,0],[31067.56896551724,0],[31299.51724137931,0],[31531.465517241377,0],[31763.41379310345,0],[31995.362068965514,0],[32227.310344827587,0],[32459.258620689652,0],[32691.206896551725,0],[32923.15517241379,0],[33155.10344827586,0],[33387.05172413793,0],[33619,170]],"plot2":[["20166",30],["33619",170]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[2,0,31601.05,4803.681832459348,20166,33619,33619],"raw_variable_index":29,"id":"VNZIP1","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":2,"missing":0,"mean":31601.05,"sd":4803.681832459348,"min":20166,"median":33619,"max":33619},

        {"enabled":false,"expanded":false,"type":0,"name":"VNST","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"y":null,"type":"C","name":"VNST","type_label":"Categorical","plot":[["FL",170],["VA",30]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[2,0],"raw_variable_index":30,"id":"VNST","types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":2,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"VehBCost","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"VehBCost","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[2592.155172413793,1],[2722.5,2],[2852.844827586207,0],[2983.189655172414,1],[3113.5344827586205,0],[3243.8793103448274,1],[3374.2241379310344,0],[3504.5689655172414,2],[3634.9137931034484,1],[3765.258620689655,1],[3895.6034482758623,2],[4025.948275862069,6],[4156.293103448275,4],[4286.637931034483,3],[4416.982758620689,2],[4547.327586206897,9],[4677.672413793103,0],[4808.01724137931,3],[4938.362068965517,3],[5068.706896551725,6],[5199.051724137931,8],[5329.396551724138,3],[5459.741379310344,5],[5590.086206896552,9],[5720.431034482759,3],[5850.775862068966,8],[5981.120689655172,8],[6111.465517241379,4],[6241.810344827586,3],[6372.1551724137935,5],[6502.5,7],[6632.8448275862065,4],[6763.189655172413,1],[6893.5344827586205,9],[7023.879310344827,4],[7154.224137931034,5],[7284.568965517241,3],[7414.913793103448,6],[7545.258620689655,3],[7675.603448275861,7],[7805.948275862069,5],[7936.293103448275,3],[8066.637931034483,2],[8196.982758620688,6],[8327.327586206895,1],[8457.672413793103,5],[8588.01724137931,4],[8718.362068965518,4],[8848.706896551725,1],[8979.051724137931,5],[9109.396551724138,1],[9239.741379310344,0],[9370.08620689655,2],[9500.431034482757,1],[9630.775862068966,1],[9761.120689655172,0],[9891.465517241379,1],[10021.810344827587,2],[10152.155172413793,1],[10282.5,3]],"plot2":[["10315",1],["2830",1],["3600",2],["4100",4],["4140",2],["4200",3],["4500",2],["4600",7],["4900",3],["5000",2],["5100",4],["5200",2],["5275",2],["5300",2],["5400",2],["5500",4],["5600",5],["5650",1],["5700",3],["5800",2],["5900",6],["6100",7],["6150",1],["6200",3],["6300",2],["6400",2],["6500",3],["6600",7],["6685",2],["6700",2],["6900",3],["6985",2],["7000",3],["7100",2],["7250",3],["7300",3],["7500",4],["7600",3],["7700",3],["7800",2],["7900",3],["8000",2],["8200",1],["8300",5],["8500",4],["8700",2],["8800",4],["9000",2],["9500",2],["=All Other=",58]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[107,0,6510.475,1724.819164253169,2230,6450,10600],"raw_variable_index":31,"id":"VehBCost","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":107,"missing":0,"mean":6510.475,"sd":1724.819164253169,"min":2230,"median":6450,"max":10600},

        {"enabled":false,"expanded":false,"type":0,"name":"IsOnlineSale","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":null,"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"empty":false,"few_values":true},"summary":[1,0],"raw_variable_index":32,"id":"IsOnlineSale","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":1,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        {"enabled":false,"expanded":false,"type":0,"name":"WarrantyCost","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"WarrantyCost","miss_count":0,"y":null,"type_label":"Numeric","type":"N","miss_ymean":null,"plot":[[462,3],[496.3251724137931,3],[530.6503448275862,4],[564.9755172413793,6],[599.3006896551724,7],[633.6258620689655,0],[667.9510344827587,6],[702.2762068965518,8],[736.6013793103449,5],[770.926551724138,4],[805.2517241379311,4],[839.5768965517243,4],[873.9020689655174,1],[908.2272413793105,13],[942.5524137931036,3],[976.8775862068967,6],[1011.2027586206898,11],[1045.527931034483,0],[1079.853103448276,7],[1114.178275862069,6],[1148.5034482758622,6],[1182.8286206896555,11],[1217.1537931034486,8],[1251.4789655172417,2],[1285.8041379310348,3],[1320.129310344828,1],[1354.454482758621,4],[1388.779655172414,8],[1423.1048275862072,3],[1457.4300000000003,2],[1491.7551724137934,8],[1526.0803448275865,2],[1560.4055172413796,0],[1594.7306896551727,1],[1629.0558620689658,4],[1663.3810344827589,1],[1697.706206896552,3],[1732.031379310345,0],[1766.3565517241382,2],[1800.6817241379313,1],[1835.0068965517244,0],[1869.3320689655177,0],[1903.6572413793108,2],[1937.982413793104,0],[1972.307586206897,6],[2006.63275862069,0],[2040.9579310344832,3],[2075.283103448276,0],[2109.6082758620696,0],[2143.9334482758622,4],[2178.258620689656,0],[2212.5837931034484,0],[2246.908965517242,3],[2281.2341379310346,2],[2315.559310344828,2],[2349.884482758621,2],[2384.2096551724144,0],[2418.5348275862075,1],[2452.8600000000006,4]],"plot2":[["1003",2],["1020",7],["1038",3],["1086",1],["1103",2],["1113",4],["1118",4],["1155",5],["1215",9],["1220",2],["1243",4],["1272",2],["1320",2],["1373",4],["1389",6],["1411",2],["1455",3],["1500",2],["1503",4],["1543",2],["1633",2],["1641",2],["1703",3],["1923",2],["1974",4],["2003",2],["2063",2],["2152",3],["2274",3],["2282",2],["2322",2],["2351",2],["462",2],["505",3],["533",4],["569",2],["594",2],["630",5],["671",4],["723",2],["728",6],["754",5],["803",3],["825",2],["853",3],["920",12],["975",3],["983",2],["986",2],["=All Other=",39]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "metric_options":this.metricOptions(),
          "summary":[88,0,1263.53,692.2950087209931,462,1131,6519],"raw_variable_index":33,"id":"WarrantyCost","types":{"category":false,"conversion":true,"text":false,"numeric":true,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":88,"missing":0,"mean":1263.53,"sd":692.2950087209931,"min":462,"median":1131,"max":6519},

        // [Too many values]
        {"enabled":false,"expanded":false,"type":0,"name":"high_cardinality","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{},"synched":true,"transform_args":[],"low_info":{"high_cardinality":true,"duplicate":false,"empty":false},
          "summary":[70,0],"raw_variable_index":6,"id":6,"types":{"category":true,"conversion":"","text":false,"numeric":false,"currency":false,"length":false,"date":false,"percentage":false,"nastring":false},"unique":70,"missing":0,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        // [empty]
        {"enabled":false,"expanded":false,"type":0,"name":"empty","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{},"synched":true,"transform_args":[],"low_info":{"duplicate":false,"empty":true},
          "summary":[0,200],"raw_variable_index":1,"id":1,"types":{},"unique":0,"missing":200,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        // [few values]
        {"enabled":false,"expanded":false,"type":0,"name":"few_values","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{},"synched":true,"transform_args":[],"low_info":{},
          "summary":[0,200],"raw_variable_index":1,"id":1,"types":{},"unique":0,"missing":200,"mean":null,"sd":null,"min":null,"median":null,"max":null},

        // Data Type = Time
        {"enabled":false,"expanded":false,"type":0,"name":"TestTime","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"TestTime","miss_count":0,"y":null,"type_label":"Time","type":"T","miss_ymean":null,"plot":[[462,3],[496.3251724137931,3]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "summary":[43,0,"04/13/2010",158.72407810017359,"02/18/2009","03/15/2010","12/20/2010"],"raw_variable_index":2,"id":"PurchTime","types":{"category":false,"conversion":"%m/%d/%Y","text":false,"numeric":true,"currency":false,"length":false,"date":true,"percentage":false,"nastring":false},"unique":43,"missing":0,"mean":"04/13/2010","sd":158.72407810017359,"min":"02/18/2009","median":"03/15/2010","max":"12/20/2010"},

        // Data Type = Text
        {"enabled":false,"expanded":false,"type":0,"name":"TestText","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"TestText","miss_count":0,"y":null,"type_label":"Text","type":"X","miss_ymean":null,"plot":[[462,3],[496.3251724137931,3]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "summary":[43,0,"04/13/2010",158.72407810017359,"02/18/2009","03/15/2010","12/20/2010"],"raw_variable_index":2,"id":"PurchTime","types":{"category":false,"conversion":"%m/%d/%Y","text":false,"numeric":true,"currency":false,"length":false,"date":true,"percentage":false,"nastring":false},"unique":43,"missing":0,"mean":"04/13/2010","sd":158.72407810017359,"min":"02/18/2009","median":"03/15/2010","max":"12/20/2010"},

        // Data Type = Percent
        {"enabled":false,"expanded":false,"type":0,"name":"TestPercent","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"TestPercent","miss_count":0,"y":null,"type_label":"Percent","type":"P","miss_ymean":null,"plot":[[462,3],[496.3251724137931,3]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "summary":[43,0,"04/13/2010",158.72407810017359,"02/18/2009","03/15/2010","12/20/2010"],"raw_variable_index":2,"id":"PurchTime","types":{"category":false,"conversion":"%m/%d/%Y","text":false,"numeric":true,"currency":false,"length":false,"date":true,"percentage":false,"nastring":false},"unique":43,"missing":0,"mean":"04/13/2010","sd":158.72407810017359,"min":"02/18/2009","median":"03/15/2010","max":"12/20/2010"},

        // Data Type = Length
        {"enabled":false,"expanded":false,"type":0,"name":"TestLength","transform_id":0,"tab":"histogram","feature_lists":[],
          "profile":{"name":"TestLength","miss_count":0,"y":null,"type_label":"Length","type":"L","miss_ymean":null,"plot":[[462,3],[496.3251724137931,3]]},"synched":true,"transform_args":[],"low_info":{"high_cardinality":false,"duplicate":false,"few_values":false,"empty":false,"ref_id":false},
          "summary":[43,0,"04/13/2010",158.72407810017359,"02/18/2009","03/15/2010","12/20/2010"],"raw_variable_index":2,"id":"PurchTime","types":{"category":false,"conversion":"%m/%d/%Y","text":false,"numeric":true,"currency":false,"length":false,"date":true,"percentage":false,"nastring":false},"unique":43,"missing":0,"mean":"04/13/2010","sd":158.72407810017359,"min":"02/18/2009","median":"03/15/2010","max":"12/20/2010"},

      ];
    },

    fetch : function(){
      return [{
        'profile': {
          'info': 0.0051058323477858325,
          'plot': [],
          'name': 'WarrantyCost',
          'miss_count': 0.0,
          'y': 'IsBadBuy',
          'plot2': [],
          'type': 'N',
          'miss_ymean': 0.0
        },
        'name': 'WarrantyCost',
        'transform_args': [],
        'low_info': {
          'high_cardinality': false,
          'duplicate': false,
          'few_values': false
        },
        'metric_options': this.metricOptions(),
        'summary': [236, 0, 1305.733846769354, 619.8178313987596, 462.0, 1215.0, 6519.0],
        'transform_id': 0,
        'id': 'WarrantyCost',
        'raw_variable_index': 33
      },{
        'profile': {
          'y': null,
          'plot': [],
          'info': 0.004496194348175162,
          'type': 'C',
          'name': 'PRIMEUNIT'
        },
        'name': 'PRIMEUNIT',
        'transform_args': [],
        'low_info': {
          'high_cardinality': false,
          'duplicate': false,
          'ACE': true,
          'few_values': false
        },
        'metric_options': this.metricOptions(),
        'summary': [2, 9437],
        'transform_id': 0,
        'id': 'PRIMEUNIT',
        'raw_variable_index': 26
      },{
        'profile': {
          'y': 'IsBadBuy',
          'plot': [],
          'info': 0.007342835733232215,
          'type': 'C',
          'name': 'Make'
        },
        'name': 'Make',
        'transform_args': [],
        'low_info': {
          'high_cardinality': false,
          'duplicate': false,
          'few_values': false
        },
        'metric_options': this.metricOptions(),
        'summary': [30, 0],
        'transform_id': 0,
        'id': 'Make',
        'raw_variable_index': 6
      },{
        'profile': {
          'info': 1,
          'plot': [],
          'name': 'IsBadBuy',
          'miss_count': 0.0,
          'y': 'IsBadBuy',
          'plot2': [],
          'type': 'N',
          'miss_ymean': 0.0
        },
        'name': 'IsBadBuy',
        'transform_args': [],
        'low_info': {
          'high_cardinality': false,
          'duplicate': false,
          'few_values': false
        },
        'metric_options': this.metricOptions(),
        'summary': [2, 0, 0.11602320464092819, 0.3202687619572318, 0.0, 0.0, 1.0],
        'transform_id': 0,
        'id': 'IsBadBuy',
        'raw_variable_index': 1
      }];
    },


  };
});