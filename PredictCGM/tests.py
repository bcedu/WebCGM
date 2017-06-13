from CGMTester import *

namefile = 'all_data_junts_EDU_Ideal.csv'
# namefile = 'all_data_junts_EDU_Sim.csv'

from datetime import datetime
now = datetime.now()
tester = CGMTester(namefile)

# params_svm = [[0.5, 'rbf', 10, 0.1, 15, 10, 1]] # v1
params_svm = [[0.5, 'poly', 10, 0.1, 5, 3, 1], [0.5, 'linear', 10, 0.1, 5, 3, 1]] # v2
# params_svm = [[4.5, 'rbf', 10, 0.1, 5, 3, 1]] # v3
# params_svm = [[0.5, 'rbf', 100, 0.1, 15, 10, 1]] # v4
# params_knn = [[0.5, 5, 15, 10, 1]] # v1
params_knn = [[0.5, 5, 5, 3, 1]] # v2
# params_knn = [[4.5, 5, 5, 3, 1]] # v3
# params_knn = [[3.0, 5, 5, 10, 1]] # v4
svm_predicters = ["SVMPredicter"]
knn_predicters = ["KNNPredicter"]
apt_predicters = ["ApatsSVMPredicter"]
## EM1
models = [("SequenceModels", "WordsModels", "SimpleModels")]
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="SVM_EM1_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM1_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM1_")
## EM2
models = [("SequenceModels", "WordsModels", "LmFitModels", "SimpleModels")]
tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="K_SVM_EM2_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM2_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM2_")
## EM3
models = [("SequenceModels", "LmFitWordsModels", "SimpleModels")]
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="SVM_EM3_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM3_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM3_")
## EM4
models = [("SequenceModels", "WordsModels", "NormalizedModels", "SimpleModels")]
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="T_SVM_EM4_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM4_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM4_")
## EM5
models = [("SequenceModels", "WordsModels", "LmFitModels", "NormalizedModels", "SimpleModels")]
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="SVM_EM5_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM5_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM5_")
## EM6
models = [("SequenceModels", "LmFitWordsModels", "NormalizedModels", "SimpleModels")]
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="SVM_EM6_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM6_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM6_")
"""
models = [
    ("SequenceModels", "WordsModels", "SimpleModels"),
    ("SequenceModels", "WordsModels", "NormalizedModels", "SimpleModels")
]

# PARAMS:   periode;    kernel;     c;      gamma;      vocabulari;     seq.len;    atr_weight
params = [
    [       0.5,        'rbf',       10,     0.1,          5,               3,          1],
    [       0.5,        'rbf',       10,     0.1,          5,              10,          1],
    [       0.5,        'rbf',       10,     0.1,         15,               3,          1],
    [       0.5,        'rbf',       10,     0.1,         15,              10,          1],
    [       0.5,        'rbf',       10,     0.1,         50,               3,          1],
    [       0.5,        'rbf',       10,     0.1,         50,              10,          1],
    [       0.5,        'rbf',       10,     1.0,          5,               3,          1],
    [       0.5,        'rbf',       10,     1.0,          5,              10,          1],
    [       0.5,        'rbf',       10,     1.0,         15,               3,          1],
    [       0.5,        'rbf',       10,     1.0,         15,              10,          1],
    [       0.5,        'rbf',       10,     1.0,         50,               3,          1],
    [       0.5,        'rbf',       10,     1.0,         50,              10,          1],
    [       0.5,        'rbf',      100,     0.1,          5,               3,          1],
    [       0.5,        'rbf',      100,     0.1,          5,              10,          1],
    [       0.5,        'rbf',      100,     0.1,         15,               3,          1],
    [       0.5,        'rbf',      100,     0.1,         15,              10,          1],
    [       0.5,        'rbf',      100,     0.1,         50,               3,          1],
    [       0.5,        'rbf',      100,     0.1,         50,              10,          1],
    [       0.5,        'rbf',      100,     1.0,          5,               3,          1],
    [       0.5,        'rbf',      100,     1.0,          5,              10,          1],
    [       0.5,        'rbf',      100,     1.0,         15,               3,          1],
    [       0.5,        'rbf',      100,     1.0,         15,              10,          1],
    [       0.5,        'rbf',      100,     1.0,         50,               3,          1],
    [       0.5,        'rbf',      100,     1.0,         50,              10,          1],
    [       4.5,        'rbf',       10,     0.1,          5,               3,          1],
    [       4.5,        'rbf',       10,     0.1,          5,              10,          1],
    [       4.5,        'rbf',       10,     0.1,         15,               3,          1],
    [       4.5,        'rbf',       10,     0.1,         15,              10,          1],
    [       4.5,        'rbf',       10,     0.1,         50,               3,          1],
    [       4.5,        'rbf',       10,     0.1,         50,              10,          1],
    [       4.5,        'rbf',       10,     1.0,          5,               3,          1],
    [       4.5,        'rbf',       10,     1.0,          5,              10,          1],
    [       4.5,        'rbf',       10,     1.0,         15,               3,          1],
    [       4.5,        'rbf',       10,     1.0,         15,              10,          1],
    [       4.5,        'rbf',       10,     1.0,         50,               3,          1],
    [       4.5,        'rbf',       10,     1.0,         50,              10,          1],
    [       4.5,        'rbf',      100,     0.1,          5,               3,          1],
    [       4.5,        'rbf',      100,     0.1,          5,              10,          1],
    [       4.5,        'rbf',      100,     0.1,         15,               3,          1],
    [       4.5,        'rbf',      100,     0.1,         15,              10,          1],
    [       4.5,        'rbf',      100,     0.1,         50,               3,          1],
    [       4.5,        'rbf',      100,     0.1,         50,              10,          1],
    [       4.5,        'rbf',      100,     1.0,          5,               3,          1],
    [       4.5,        'rbf',      100,     1.0,          5,              10,          1],
    [       4.5,        'rbf',      100,     1.0,         15,               3,          1],
    [       4.5,        'rbf',      100,     1.0,         15,              10,          1],
    [       4.5,        'rbf',      100,     1.0,         50,               3,          1],
    [       4.5,        'rbf',      100,     1.0,         50,              10,          1],
    [       0.5,        'rbf',       10,     0.1,          5,               3,          100],
    [       0.5,        'rbf',       10,     0.1,          5,              10,          100],
    [       0.5,        'rbf',       10,     0.1,         15,               3,          100],
    [       0.5,        'rbf',       10,     0.1,         15,              10,          100],
    [       0.5,        'rbf',       10,     0.1,         50,               3,          100],
    [       0.5,        'rbf',       10,     0.1,         50,              10,          100],
    [       0.5,        'rbf',       10,     1.0,          5,               3,          100],
    [       0.5,        'rbf',       10,     1.0,          5,              10,          100],
    [       0.5,        'rbf',       10,     1.0,         15,               3,          100],
    [       0.5,        'rbf',       10,     1.0,         15,              10,          100],
    [       0.5,        'rbf',       10,     1.0,         50,               3,          100],
    [       0.5,        'rbf',       10,     1.0,         50,              10,          100],
    [       0.5,        'rbf',      100,     0.1,          5,               3,          100],
    [       0.5,        'rbf',      100,     0.1,          5,              10,          100],
    [       0.5,        'rbf',      100,     0.1,         15,               3,          100],
    [       0.5,        'rbf',      100,     0.1,         15,              10,          100],
    [       0.5,        'rbf',      100,     0.1,         50,               3,          100],
    [       0.5,        'rbf',      100,     0.1,         50,              10,          100],
    [       0.5,        'rbf',      100,     1.0,          5,               3,          100],
    [       0.5,        'rbf',      100,     1.0,          5,              10,          100],
    [       0.5,        'rbf',      100,     1.0,         15,               3,          100],
    [       0.5,        'rbf',      100,     1.0,         15,              10,          100],
    [       0.5,        'rbf',      100,     1.0,         50,               3,          100],
    [       0.5,        'rbf',      100,     1.0,         50,              10,          100],
    [       4.5,        'rbf',       10,     0.1,          5,               3,          100],
    [       4.5,        'rbf',       10,     0.1,          5,              10,          100],
    [       4.5,        'rbf',       10,     0.1,         15,               3,          100],
    [       4.5,        'rbf',       10,     0.1,         15,              10,          100],
    [       4.5,        'rbf',       10,     0.1,         50,               3,          100],
    [       4.5,        'rbf',       10,     0.1,         50,              10,          100],
    [       4.5,        'rbf',       10,     1.0,          5,               3,          100],
    [       4.5,        'rbf',       10,     1.0,          5,              10,          100],
    [       4.5,        'rbf',       10,     1.0,         15,               3,          100],
    [       4.5,        'rbf',       10,     1.0,         15,              10,          100],
    [       4.5,        'rbf',       10,     1.0,         50,               3,          100],
    [       4.5,        'rbf',       10,     1.0,         50,              10,          100],
    [       4.5,        'rbf',      100,     0.1,          5,               3,          100],
    [       4.5,        'rbf',      100,     0.1,          5,              10,          100],
    [       4.5,        'rbf',      100,     0.1,         15,               3,          100],
    [       4.5,        'rbf',      100,     0.1,         15,              10,          100],
    [       4.5,        'rbf',      100,     0.1,         50,               3,          100],
    [       4.5,        'rbf',      100,     0.1,         50,              10,          100],
    [       4.5,        'rbf',      100,     1.0,          5,               3,          100],
    [       4.5,        'rbf',      100,     1.0,          5,              10,          100],
    [       4.5,        'rbf',      100,     1.0,         15,               3,          100],
    [       4.5,        'rbf',      100,     1.0,         15,              10,          100],
    [       4.5,        'rbf',      100,     1.0,         50,               3,          100],
    [       4.5,        'rbf',      100,     1.0,         50,              10,          100],
]
# PARAMS:   periode;    k;      vocabulari;     seq.len;    atr_weight
kparams = [
    [       0.2,        5,          5,              3,          1],
    [       0.2,        5,          5,              3,        100],
    [       0.2,        5,          5,             12,          1],
    [       0.2,        5,          5,             12,        100],
    [       0.2,        5,         20,              3,          1],
    [       0.2,        5,         20,              3,        100],
    [       0.2,        5,         20,             12,          1],
    [       0.2,        5,         20,             12,        100],
    [       3.0,        5,          5,              3,          1],
    [       3.0,        5,          5,              3,        100],
    [       3.0,        5,          5,             12,          1],
    [       3.0,        5,          5,             12,        100],
    [       3.0,        5,         20,              3,          1],
    [       3.0,        5,         20,              3,        100],
    [       3.0,        5,         20,             12,          1],
    [       3.0,        5,         20,             12,        100],
]
predicters = ["SVMPredicter"]
tester.test(params=params, predict_obj_names=predicters, model_generator_names=models, nfolds=5, prefix="FIN_SVM")
predicters = ["ApatsSVMPredicter"]
tester.test(params=params, predict_obj_names=predicters, model_generator_names=models, nfolds=5, prefix="FIN_ApatsSVM")
kpredicters = ["KNNPredicter"]
tester.test(params=kparams, predict_obj_names=kpredicters, model_generator_names=models, nfolds=5, prefix="FIN_KNN")


models = [
    ("SequenceModels", "WordsModels", "LmFitModels", "SimpleModels"),
    ("SequenceModels", "WordsModels", "LmFitModels", "NormalizedModels", "SimpleModels"),
]
predicters = ["SVMPredicter"]
tester.test(params=params, predict_obj_names=predicters, model_generator_names=models, nfolds=5, prefix="FIN_WLM_SVM")
predicters = ["ApatsSVMPredicter"]
tester.test(params=params, predict_obj_names=predicters, model_generator_names=models, nfolds=5, prefix="FIN_WLM_ApatsSVM")
kpredicters = ["KNNPredicter"]
tester.test(params=kparams, predict_obj_names=kpredicters, model_generator_names=models, nfolds=5, prefix="FIN_WLM_KNN")


models = [
    ("SequenceModels", "LmFitWordsModels", "SimpleModels"),
    ("SequenceModels", "LmFitWordsModels", "NormalizedModels", "SimpleModels"),
]
predicters = ["SVMPredicter"]
tester.test(params=params, predict_obj_names=predicters, model_generator_names=models, nfolds=5, prefix="FIN_LM_SVM")
predicters = ["ApatsSVMPredicter"]
tester.test(params=params, predict_obj_names=predicters, model_generator_names=models, nfolds=5, prefix="FIN_LM_ApatsSVM")
kpredicters = ["KNNPredicter"]
tester.test(params=kparams, predict_obj_names=kpredicters, model_generator_names=models, nfolds=5, prefix="FIN_LM_KNN")
"""
print(datetime.now() - now)
