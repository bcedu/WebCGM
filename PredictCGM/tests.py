from CGMTester import *

# namefile = 'all_data_junts_EDU_Ideal.csv'
namefile = 'all_data_junts_EDU_Sim.csv'

from datetime import datetime
now = datetime.now()
tester = CGMTester(namefile)

# params_svm = [[0.5, 'rbf', 10, 0.1, 15, 10, 1]] # v1
params_svm = [
    [0.5, 'rbf', 10, 0.1, 5, 3, 1],
    [0.5, 'poly', 10, 0.1, 5, 3, 1],
    [0.5, 'linear', 10, 0.1, 5, 3, 1]] # v2
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
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="K_SVM_EM2_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM2_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM2_")
## EM3
models = [("SequenceModels", "LmFitWordsModels", "SimpleModels")]
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="SVM_EM3_")
# tester.test(params=params_knn, predict_obj_names=knn_predicters, model_generator_names=models, nfolds=5, prefix="KNN_EM3_")
# tester.test(params=params_svm, predict_obj_names=apt_predicters, model_generator_names=models, nfolds=5, prefix="APT_EM3_")
## EM4
models = [("SequenceModels", "WordsModels", "NormalizedModels", "SimpleModels")]
# tester.test(params=params_svm, predict_obj_names=svm_predicters, model_generator_names=models, nfolds=5, prefix="SVM_EM4_")
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
print(datetime.now() - now)
