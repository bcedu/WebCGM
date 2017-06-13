from CGMLoader import *
import CGMPredict
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score, recall_score, auc, roc_curve
import signal
from sklearn.preprocessing import label_binarize
import random
import math


class CGMTester(object):

    def __init__(self, file, new_db=True):
        self.loader = Loader(new_db=new_db)
        if new_db:
            self.loader.load_data_from_csv(file)
            self.loader.generate_fdps(normalized=True)
        self.best = None
        if not os.path.exists("results"):
            os.mkdir("results")

    @staticmethod
    def signal_handler(signum, frame):
        raise Exception("Timed out!")

    def test(self, params, predict_obj_names=["SVMPredicter"],
             model_generator_names=[("SequenceModels","WordsModels","SimpleModels")],
             nfolds=10, prefix=""):
        self.prefix = prefix
        import CGMLoader as models_module
        for model_gen_tuple in model_generator_names:
            for predict_obj_name in predict_obj_names:
                predict_obj = getattr(CGMPredict, predict_obj_name)
                model_filename = ""
                model_obj = None
                for gen in reversed(model_gen_tuple):
                    if gen not in ["SequenceModels","WordsModels","SimpleModels"]:
                        model_filename += "{0}_".format(gen)
                    new_model = getattr(models_module, gen)
                    if not model_obj:
                        model_obj = new_model(loader=self.loader)
                    else:
                        model_obj = new_model(model_obj)
                for para_list in params:
                    timeout = False
                    some_ok = False
                    second_fail = False
                    res = []
                    filename = predict_obj_name + "_" + model_filename + \
                               str(para_list)[1:-1].replace(",","_").replace(" ","").replace("'","")
                    filename += "_results"
                    if self.loader.round_val != para_list[0]:
                        self.loader.round_val = para_list[0]
                        self.loader.clean_data()
                        self.loader.generate_fdps(normalized=True)
                        try:self.loader.cursor.execute("DROP TABLE LmFitModels;")
                        except: pass
                    self.loader.set_generator(model_obj)
                    try:
                        predicter = predict_obj(self.loader, *para_list[1:])
                    except Exception as e:
                        print(e)
                    sessions_list = self.loader.get_all_sessions(for_train=True)
                    folds = self._get_stratified_kfolds(k=nfolds, sessions=sessions_list)
                    i = 0
                    for (train_ids, test_ids) in folds:
                        i += 1
                        signal.signal(signal.SIGALRM, self.signal_handler)
                        signal.alarm(60*60)  # 15 min
                        try:
                            info_pred = predicter.predict(test_ids, train_ids)
                        except Exception as msg:
                            print("ERROR in {0} fold {1}: {2}.".format(filename, i, msg))
                            timeout = True
                            if not second_fail:
                                second_fail = True
                                continue
                            else:
                                break

                        y_real = self.loader.get_models_label(test_ids)
                        res.append((info_pred, y_real))

                        some_ok = True
                        print("{0}: {1} fold DONE".format(filename, i))

                    if timeout and not some_ok:
                        print("ERROR in {0}. No folds success.".format(filename))
                        new = open('results/no_results_{0}.txt'.format(filename), 'w')
                        new.close()
                        break

                    self.write_metrics(res, filename)
                    print("Tested: {0} with {1} and {2}.".format(predict_obj_name, model_filename, para_list))
        if self.best:
            print("Best: {0}, with acuacity: {1}".format(self.best[0], self.best[1]))

    def _get_stratified_kfolds(self, k, sessions):
        folds = []
        skf = StratifiedKFold(n_splits=k, random_state=False, shuffle=True)
        X = sessions
        y = pd.Series(self.loader.get_session_labels(X))
        X = pd.Series(X)
        for train, test in skf.split(X, y):
            y_train, y_test = y.iloc[train], y.iloc[test]
            # y_train, y_test = self.retallar_dataset(y_train, y_test, 1, 0, -1)
            X_train = X.iloc[y_train.index.values].tolist()
            X_test = X.iloc[y_test.index.values].tolist()
            folds.append((X_train, X_test))
        return folds

    def write_metrics(self, fold_info_list, filename):
        new = open('results/{0}'.format(filename), 'w')
        # Get together the info of all folds
        #               y_pred  y_bin   y_scor  w_pred  w_scor  y_real
        total_info= [   [],     [],     [],     [],     [],     []     ]
        for fold_info, y_real in fold_info_list:
            total_info[0] += fold_info[0]
            total_info[1] += list(fold_info[4])
            total_info[2] += list(fold_info[1])
            total_info[3] += fold_info[2]
            total_info[4] += fold_info[3]
            total_info[5] += y_real
            # write fold info
            self.write_fold_metrics(new, fold_info[0], list(fold_info[4]), list(fold_info[1]), y_real)

        # write global info
        y_pred = total_info[0]
        y_bin = total_info[1]
        y_scor = total_info[2]
        y_real = total_info[5]
        ac = self.write_fold_metrics(new, y_pred, y_bin, y_scor, y_real)
        new.close()
        import os
        new_name = "{0}{1}_{2}.txt".format(self.prefix, ac, filename)
        os.rename('results/{0}'.format(filename), 'results/{0}'.format(new_name))
        print("New file: {0}".format(new_name))

    def write_fold_metrics(self, nfile, y_pred, y_bin, y_scor, y_real):
        # Construct binari labels information
        y_pred_hipo = []
        y_real_hipo = []
        y_scor_hipo = []

        y_pred_hipe = []
        y_real_hipe = []
        y_scor_hipe = []

        y_pred_norm = []
        y_real_norm = []
        y_scor_norm = []

        y_pred_total = []
        y_real_total = []

        not_classified = 0

        for i in range(len(y_bin)):
            # Get label predicted from binary predictions
            label = None
            for l in range(-1, 2):
                if y_bin[i][l + 1] == 1:
                    label = l
                    break

            # Add to no classified if no label found and go to next iteration
            if label is None:
                not_classified +=1
                continue

            # Add labels to total lists
            y_pred_total.append(label)
            y_real_total.append(y_real[i])

            # Add labels to hipo lists
            target = '-1'
            self.add_binari_label_to_list(target, label, y_pred_hipo)
            self.add_binari_label_to_list(target, y_real[i], y_real_hipo)
            y_scor_hipo.append(y_scor[i][0])

            # Add labels to norm lists
            target = '0'
            self.add_binari_label_to_list(target, label, y_pred_norm)
            self.add_binari_label_to_list(target, y_real[i], y_real_norm)
            y_scor_norm.append(y_scor[i][1])

            # Add labels to hipe lists
            target = '1'
            self.add_binari_label_to_list(target, label, y_pred_hipe)
            self.add_binari_label_to_list(target, y_real[i], y_real_hipe)
            y_scor_hipe.append(y_scor[i][2])

        # Write metrics for binari labels ant total labels
        self.write_metrics_from_lists(nfile, y_pred_hipo, y_real_hipo, y_scor_hipo, "HYPOGLUCEMIA")
        self.write_metrics_from_lists(nfile, y_pred_norm, y_real_norm, y_scor_norm, "NORMALUCEMIA")
        self.write_metrics_from_lists(nfile, y_pred_hipe, y_real_hipe, y_scor_hipe, "HYPERGLUCEMIA")
        ac = self.write_metrics_from_lists(nfile, y_pred_total, y_real_total, None, "TOTAL")
        if math.isnan(ac) or not_classified > len(y_real)/2:
            ac = -1
        nfile.write("\nNO CLASSIFIED: {0}/{1}\n".format(not_classified, len(y_real)))
        ac2 = self.write_metrics_from_lists(nfile, y_pred, y_real, None, "Non Confidence Predictions")
        return max([round(ac, 4), round(ac2, 4)])

    @staticmethod
    def write_metrics_from_lists(new_file, y_pred, y_real, y_score, name):
        metrics = "\nAcuracity: {0}.\nPrecision: {1}.\nF1: {2}.\nRecall: {3}.\n----;\n"

        new_file.write("*---\n")
        new_file.write("{0} Classifier\n".format(name))

        cm = confusion_matrix(y_real, y_pred)
        text = str(cm).replace(' ', ',').replace(',,', ',').replace('[,', '[') + "\n"
        new_file.write(text)

        if y_score:
            try:
                fpr, tpr, _ = roc_curve(y_real, y_score)
                aucroc = auc(fpr, tpr)
                new_file.write("AUC: {0}".format(aucroc))
            except Exception as e:
                print(e)
                pass

        if y_score:
            acuracity = accuracy_score(y_real, y_pred)
            precision = precision_score(y_real, y_pred)
            f1 = f1_score(y_real, y_pred)
            recall = recall_score(y_real, y_pred)
        else:
            acuracity = accuracy_score(y_real, y_pred, normalize=True)
            precision = precision_score(y_real, y_pred, average='micro')
            f1 = f1_score(y_real, y_pred, average='micro')
            recall = recall_score(y_real, y_pred, average='micro')

        new_file.write(metrics.format(acuracity, precision, f1, recall))
        return acuracity

    @staticmethod
    def calc_roc_aucs(y_real, y_score):
        y_test = label_binarize(y_real, classes=[-1, 0, 1])
        aucs = []
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
            aucs.append(auc(fpr, tpr))
        return aucs

    @staticmethod
    def add_binari_label_to_list(target_label, label_predicted, plist):
        if str(label_predicted) == target_label:
            return plist.append(1)
        else:
            return plist.append(0)

    @staticmethod
    def calc_aucs(cm):
        tp_0 = cm[0][0]
        fn_0 = cm[0][1] + cm[0][2]
        fp_0 = cm[1][0] + cm[2][0]
        tn_0 = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]

        tp_1 = cm[1][1]
        fn_1 = cm[1][0] + cm[1][2]
        fp_1 = cm[0][1] + cm[2][1]
        tn_1 = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]

        tp_2 = cm[2][2]
        fn_2 = cm[2][0] + cm[2][1]
        fp_2 = cm[0][2] + cm[1][2]
        tn_2 = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]

        # TPR = TP/(TP+FN)
        tpr_0 = tp_0 / (tp_0 + fn_0)
        tpr_1 = tp_1 / (tp_1 + fn_1)
        tpr_2 = tp_2 / (tp_2 + fn_2)

        # FPR = FP/(FP+TN)
        fnr_0 = fp_0 / (fp_0 + tn_0)
        fnr_1 = fp_1 / (fp_1 + tn_1)
        fnr_2 = fp_2 / (fp_2 + tn_2)

        auc0 = auc(fnr_0, tpr_0)
        auc1 = auc(fnr_1, tpr_1)
        auc2 = auc(fnr_2, tpr_2)
        return auc0, auc1, auc2

    def retallar_dataset(self, y_train, y_test, class2, class1, class0):
        train_2_indices = np.array(np.where(y_train.values == class2)).tolist()[0]
        train_1_indices = np.array(np.where(y_train.values == class1)).tolist()[0]
        train_0_indices = np.array(np.where(y_train.values == class0)).tolist()[0]

        ordered_indices = sorted([train_2_indices, train_1_indices, train_0_indices], key=len)
        shortest = ordered_indices[0]
        shortest_len = len(shortest)
        for lindices in ordered_indices[1:]:
            random_train_selection = random.sample(lindices, shortest_len)
            shortest.extend(random_train_selection)
        random.shuffle(shortest)

        y_train_copy = y_train.iloc[shortest]

        test_2_indices = np.array((np.where(y_test.values[:] == class2))).tolist()[0]
        test_1_indices = np.array((np.where(y_test.values[:] == class1))).tolist()[0]
        test_0_indices = np.array((np.where(y_test.values[:] == class0))).tolist()[0]

        ordered_indices = sorted([test_2_indices, test_1_indices, test_0_indices], key=len)
        shortest = ordered_indices[0]
        shortest_len = len(shortest)
        for lindices in ordered_indices[1:]:
            random_train_selection = random.sample(lindices, shortest_len)
            shortest.extend(random_train_selection)
        random.shuffle(shortest)

        y_test_copy = y_test.iloc[shortest]

        return y_train_copy, y_test_copy
