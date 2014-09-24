
from bson.objectid import ObjectId

old_worker_report = {u'features': [u'Missing Values Imputed'], u'qid': u'1', u'task_version': {u'NI': u'0.1', u'GLMB': u'0.1'}, u'roc': {u'(0,-1)': [[0.0, 0.07692307692307693, 0.9990012426055844], [0.0, 0.15384615384615385, 0.9989974050405092], [0.0, 0.23076923076923078, 0.99804098935017], [0.0, 0.3076923076923077, 0.9979829514940015], [0.0, 0.38461538461538464, 0.991992078494643], [0.0, 0.46153846153846156, 0.8337616418837093], [0.045454545454545456, 0.46153846153846156, 0.712806862910356], [0.045454545454545456, 0.5384615384615384, 0.5545768668267775], [0.09090909090909091, 0.5384615384615384, 0.5340388905562765], [0.13636363636363635, 0.5384615384615384, 0.5032889302133705], [0.18181818181818182, 0.5384615384615384, 0.4662299761591789], [0.22727272727272727, 0.5384615384615384, 0.43493931279034515], [0.2727272727272727, 0.5384615384615384, 0.4215643414328379], [0.2727272727272727, 0.6153846153846154, 0.4038403384738926], [0.3181818181818182, 0.6153846153846154, 0.3835753919940193], [0.36363636363636365, 0.6153846153846154, 0.32840656202145957], [0.4090909090909091, 0.6153846153846154, 0.31544918348224416], [0.4090909090909091, 0.6923076923076923, 0.2983317995881547], [0.4090909090909091, 0.7692307692307693, 0.29683700670482066], [0.45454545454545453, 0.7692307692307693, 0.2866670981394291], [0.5, 0.7692307692307693, 0.27893908252434324], [0.5454545454545454, 0.7692307692307693, 0.27216550883348933], [0.5909090909090909, 0.7692307692307693, 0.26154652106452886], [0.6363636363636364, 0.7692307692307693, 0.25148121245698], [0.6818181818181818, 0.7692307692307693, 0.21871508480897958], [0.7272727272727273, 0.7692307692307693, 0.16648444708228385], [0.7727272727272727, 0.7692307692307693, 0.140469494415484], [0.8181818181818182, 0.7692307692307693, 0.11297423954518451], [0.8636363636363636, 0.7692307692307693, 0.09141114074650543], [0.9090909090909091, 0.7692307692307693, 0.08841345221451419], [0.9090909090909091, 0.8461538461538461, 0.08416371120354428], [0.9090909090909091, 0.9230769230769231, 0.05945785455470924], [0.9545454545454546, 0.9230769230769231, 0.016871531724260223], [1.0, 0.9230769230769231, 0.007364821559915612], [1.0, 1.0, 0.0009999921689384458], [1, 1, 0]]}, u'part_size': [[u'1', 125, 35]], u'max_reps': 1, u'samplesize': 160.0, u'ec2': {u'reps=1': {u'spot_price': 0, u'on_demand_price': 0, u'availability_zone': u'None', u'instance_size': u'None', u'CPU_count': 4, u'instance_type': u'local', u'workers': 2}}, u'dataset_id': u'53165b0c3e0fd17de1f6f122', u'partition_stats': {u'(0, -1)': {u'train_size': 125, u'test_size': 35, u'time_real': u'0.01399'}}, u'uid': ObjectId('53165b0b57fc7b01bafddfac'), u'resource_summary': {u'reps=1': {u'total_cpu_time': 0.0799999999999983, u'total_noncached_clock_time': 0.0295560359954834, u'cpu_usage': 1.1154952576935497, u'bp_time': 0.333420991897583, u'cached': u'true', u'max_noncached_ram': 0, u'bp_cost': 0.0, u'noncached_cost': 0.0, u'rate': 0.013666666666666666, u'cost': 0.0, u'total_noncached_cpu_time': 0.02999999999999936, u'total_clock_time': 0.0717170238494873, u'max_ram': 0, u'noncached_cpu_usage': 1.0150210943234677}}, u'training_dataset_id': u'53165b0c3e0fd17de1f6f122', u'icons': [0], u'total_size': 160, u'parts': [[u'1', 13]], u'test': {u'metrics': [u'LogLoss', u'AUC', u'Ians Metric', u'Gini', u'Gini Norm', u'Rate@Top10%', u'Rate@Top5%'], u'AUC': [0.6958], u'Gini': [0.12308], u'Gini Norm': [0.39161], u'labels': [u'(0,-1)'], u'LogLoss': [0.7008], u'Rate@Top5%': [1.0], u'Ians Metric': [0.29222], u'Rate@Top10%': [1.0]}, u'insights': u'NA', u'parts_label': [u'partition', u'NonZeroCoefficients'], u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'GLMB'], u'P']}, u'hash': u'5b6b8ad6e3613e1309139bb51b8d2aa2e401f277', u'blueprint_id': u'd4c06a5c23cf1d917019720bceba32c8', u'pid': ObjectId('53165b0b3e0fd17de1f6f121'), u'originalName': None, u'holdout': {u'metrics': [u'LogLoss', u'AUC', u'Ians Metric', u'Gini', u'Gini Norm', u'Rate@Top10%', u'Rate@Top5%'], u'AUC': 0.79028, u'Gini': 0.16691, u'Gini Norm': 0.58056, u'LogLoss': 0.81124, u'Rate@Top5%': 1.0, u'Ians Metric': 0.35839, u'Rate@Top10%': 0.75}, u'lift': {u'(0,-1)': {u'pred': [0.008364800954006222, 0.0763293171672343, 0.17257529941137828, 0.0914107064206044, 0.2534422147394357, 0.3852027274842422, 0.2514856526100858, 0.5337075739080746, 0.5656044527927884, 0.29684103379890187, 0.6137711292850032, 0.7119797712246692, 0.8253983773054319, 0.4349457622499108, 0.9695213679341377, 1.0886268147206897, 0.7128043270051895, 1.825759695137307, 1.9960079104943067, 1.9980178269438178], u'rows': [2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0], u'act': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0]}}, u'bp': 1, u'task_parameters': u'{"NI": {"threshold": "50", "fullset": "True"}, "GLMB": {"GridSearch: stratified": "True", "GridSearch: algorithm": "Tom", "GridSearch: max_iterations": "15", "GridSearch: validation_pct": "None", "tweedie_log": "True", "GridSearch: CV_folds": "5", "Censoring: right_censoring": "None", "p": "1.5", "GridSearch: metric": "None", "Censoring: left_censoring": "None", "distribution": "Bernoulli", "GridSearch: step": "10", "GridSearch: random_state": "1234"}}', u'task_cnt': 2, u'max_folds': 0, u'metablueprint': [u'Metablueprint', u'8.6'], u'vertex_cache_hits': 0, u'samplepct': 100, u'training_dataset_name': u'Raw Features', u'finish_time': {u'reps=1': 1393974031.397255}, u'vertices': {u'1': {u'output_shape': [[160, 13]]}, u'2': {u'output_shape': [[160, 1]]}}, u'time_real': [[u'1', u'0.01399']], u's': 0, u'task_info': {u'reps=1': [[{u'fit max RAM': 0, u'fit CPU pct': 0, u'fit CPU time': 0.019999999999999574, u'transform max RAM': 0, u'fit clock time': 0.01318502426147461, u'fit avg RAM': 0, u'cached': True, u'ytrans': None, u'transform avg RAM': 0, u'xtrans': None, u'transform clock time': 0.019706010818481445, u'version': u'0.1', u'fit total RAM': 8060366848L, u'task_name': u'NI', u'transform CPU time': 0.019999999999999574, u'transform total RAM': 8060366848L, u'transform CPU pct': 0, u'arguments': None}], [{u'fit max RAM': 0, u'fit CPU pct': 0, u'fit CPU time': 0.02999999999999936, u'predict CPU pct': 0, u'fit clock time': 0.028975963592529297, u'task_name': u'GLMB', u'fit avg RAM': 0, u'cached': True, u'ytrans': None, u'fit total RAM': 8060366848L, u'predict CPU time': 0.009999999999999787, u'xtrans': None, u'predict clock time': 0.009850025177001953, u'version': u'0.1', u'predict avg RAM': 0, u'arguments': None, u'predict max RAM': 0, u'predict total RAM': 8060366848L}]]}, u'time': {u'finish_time': {u'reps=1': 1393974031.397258}, u'total_time': {u'reps=1': 0.333420991897583}, u'start_time': {u'reps=1': 1393974031.063838}}, u'model_type': u'Generalized Linear Model (Bernoulli Distribution)', u'vertex_cnt': 2, u'_id': ObjectId('53165b0f3e0fd17de1f6f124'), u'blend': 0, u'reference_model': True, u'extras': {u'(0, -1)': {u'coefficients': [[u'NumberOfTime60_89DaysPastDueNotWorse', 3.597567], [u'NumberOfTimes90DaysLate', 2.06655], [u'NumberOfDependents', 0.052613], [u'NumberOfDependents-mi', -1.383201], [u'MonthlyIncome', -9.7e-05], [u'MonthlyIncome-mi', -0.659862], [u'NumberOfTime30_59DaysPastDueNotWorse', 0.49996], [u'NumberRealEstateLoansOrLines', 0.337828], [u'NumberOfOpenCreditLinesAndLoans', 0.07979], [u'age', -0.041883], [u'RevolvingUtilizationOfUnsecuredLines', 0.003535], [u'DebtRatio', -0.001087], [u'Unnamed: 0', 0.000488]]}}}

old_worker_pred = {u'lid': ObjectId('53165b0f3e0fd17de1f6f124'), u'actual': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], u'partition': [2.0, 2.0, 3.0, 1.0, 3.0, 0.0, 4.0, 3.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 3.0, 2.0, 0.0, 4.0, 0.0, 3.0, 4.0, 1.0, 1.0, 1.0, 2.0, 1.0, 4.0, 2.0, 4.0, 4.0, 2.0, 4.0, 0.0, 1.0, 3.0, 4.0, 1.0, 3.0, 2.0, 4.0, 3.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 4.0, 1.0, 0.0, 2.0, 1.0, 3.0, 4.0, 2.0, 3.0, 0.0, 4.0, 2.0, 3.0, 1.0, 4.0, 0.0, 0.0, 3.0, 2.0, 1.0, 1.0, 3.0, 4.0, 4.0, 2.0, 1.0, 4.0, 2.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 1.0, 3.0, 1.0, 3.0, 0.0, 1.0, 4.0, 3.0, 1.0, 0.0, 2.0, 3.0, 3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 4.0, 1.0, 0.0, 4.0, 1.0, 0.0, 3.0, 1.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0, 2.0, 4.0, 1.0, 0.0, 3.0, 3.0, 4.0], u'pid': ObjectId('53165b0b3e0fd17de1f6f121'), u'predicted-0': [0.15348507331153932, 0.4510028399692696, 0.022057931070142447, 0.981300223890865, 0.02144771233847453, 0.21871657474651, 0.2610824735284512, 0.3406554275741259, 0.20820972090118645, 0.32800732657326076, 0.1912804236065384, 0.4488464660627799, 0.0841633200989491, 0.24885364428543522, 0.1477364809571224, 0.20475476561777692, 0.27216360757203906, 0.829588924317122, 0.5340440766421121, 0.5769988410252115, 0.08900501437836514, 0.30960804349810533, 0.999, 0.3493728783565292, 0.7544793976619331, 0.011517190700256048, 0.39491035061970614, 0.0920958769507189, 0.8173913537886924, 0.8060879143104994, 0.14043206532975236, 0.3097974084281291, 0.712811212211938, 0.13074471883853472, 0.0845027373569417, 0.3687966858987532, 0.20218209122733824, 0.26629220648456164, 0.4042393939981008, 0.642591051891456, 0.43606168215301205, 0.3679755179126107, 0.1339323704953344, 0.9979772769916608, 0.9940828233843635, 0.28148189891289077, 0.770282130026849, 0.32840480144478323, 0.27538502236068924, 0.24019553044472758, 0.014326704640008622, 0.4153878411377252, 0.08350880275764824, 0.49892661913636616, 0.44673551148555724, 0.4003830134862957, 0.852468207972148, 0.27750495457730645, 0.11297410266426605, 0.03729730097521527, 0.010812512331138807, 0.00589687735082934, 0.001, 0.9585936586420427, 0.033413654071866805, 0.29683924896932606, 0.5141149768548914, 0.029835413722402414, 0.08407658480436232, 0.9020262474118317, 0.1495870508470204, 0.007364856137440843, 0.2866686133819479, 0.47470951507681997, 0.19877184020311367, 0.8712239366730677, 0.34832195688519985, 0.04437295113505663, 0.2080724243899444, 0.14952611910306748, 0.06250685719203676, 0.999, 0.8353929706371038, 0.684884504374901, 0.2746510697290559, 0.999, 0.2983289321467828, 0.05945818118853827, 0.999, 0.4038388116977538, 0.7684832997080859, 0.4221840330699043, 0.020312453436045, 0.27660321199197985, 0.9980377324276396, 0.801292532054723, 0.999, 0.19563601888460067, 0.4146675132845172, 0.994732505663175, 0.14046907110730641, 0.09864848907340094, 0.18635529589119737, 0.409481806821463, 0.769323557223457, 0.27893704624890314, 0.018270166578319105, 0.6640299437930843, 0.13855358507504142, 0.2531358948953584, 0.9946050921718852, 0.26154435571000684, 0.41881882384361446, 0.9366999230024878, 0.19656412194099282, 0.8831550609535495, 0.001, 0.022993804308176755, 0.5689576210876572, 0.2582633473373478, 0.16648601162360147, 0.1620713048856581, 0.9591263990558293, 0.0884127208814, 0.3154465951460641, 0.5545783311404697, 0.0914113241228575, 0.35248135083331383, 0.8337585584797964, 0.9389429596880987, 0.04961256230405046, 0.09564217609875303, 0.421565003212808, 0.715795301266547, 0.22321848046555265, 0.2514831688559742, 0.22829317489410456, 0.008685888424604223, 0.4349414942714794, 0.999, 0.1735695917941189, 0.048842559602951516, 0.46623180640062856, 0.38357488416964947, 0.04844372088673749, 0.9919986016586476, 0.016871454576749736, 0.9387974766164507, 0.21218877613032675, 0.5803989865315063, 0.0047396777083512955, 0.15186950060139315, 0.27244908583816063, 0.04734809766824736, 0.001, 0.4305938272870561, 0.5032842014657585, 0.5507363138567495, 0.20407698440495237, 0.9290140352674673], u'row_index': [169, 7, 123, 164, 159, 93, 60, 131, 73, 162, 57, 147, 148, 69, 149, 125, 46, 151, 173, 76, 59, 51, 170, 89, 153, 97, 175, 85, 0, 114, 112, 122, 132, 142, 94, 189, 185, 197, 130, 21, 96, 154, 105, 54, 135, 45, 72, 161, 22, 141, 134, 71, 56, 26, 152, 150, 77, 65, 20, 16, 48, 198, 95, 178, 194, 12, 118, 37, 92, 133, 128, 33, 87, 104, 52, 40, 157, 167, 53, 177, 19, 120, 127, 63, 146, 64, 196, 50, 174, 81, 143, 145, 186, 43, 90, 193, 86, 3, 68, 172, 91, 129, 58, 115, 100, 18, 80, 199, 49, 39, 195, 62, 6, 28, 79, 176, 34, 103, 137, 70, 181, 8, 101, 32, 102, 156, 61, 23, 75, 187, 24, 117, 165, 140, 13, 144, 36, 88, 182, 106, 138, 35, 136, 160, 113, 188, 66, 191, 109, 183, 166, 67, 11, 41, 15, 82, 168, 99, 30, 121], u'dataset_id': u'53165e4d3e0fd17e4bf63b33', u'_id': ObjectId('53165e5057fc7b01bafddfba'), u'newdata': u'NO'}