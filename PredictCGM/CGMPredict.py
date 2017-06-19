from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import numpy as np


class Predicter(object):

    def __init__(self, loader, words_num=20, seq_len=3, weights=1):
        """
        :param loader:  object of the type Loader. It contains all the data about CGM and
                        is able to generate the models with which the classifier will work.
        :param words_num:   number of words of the vocabulary generated
        :param seq_len: number of sessions of the sequences generated
        """
        self.loader = loader
        self.loader.generate_models({'k': words_num, 'ns': seq_len})
        self.weights = weights

    def predict(self, session_ids, train_sessions=None):
        """
        :param session_ids:  ids of the session to predict
        :param train_index: ids of the data used as train to built the predict_obj.
                            If no train_index is given, it will be used all the data exept the ones of the 'session_ids'
        :return: labels predicted
        """
        if not train_sessions:
            train_sessions = self.loader.get_all_sessions(for_train=True)
            train_sessions = [x for x in train_sessions if x not in session_ids]
        res = self._predict(train_sessions, session_ids)
        return res

    def _predict(self, train_index, model_indexs):
        """
        :param model_indexs:  ids of the session to predict
        :param train_index: ids of the data used as train to built the predict_obj.
        :return labels predicted
        """
        """To be implemented by childs"""
        return None

    def get_similar_models(self, session_id, n=5):
        """
        :param session_id: id of a session
        :param n: number of sequences 
        :return: ids of the n sessions most similar to session_id
        """
        neigh = NearestNeighbors(n_neighbors=n)

        train_sessions = self.loader.get_all_sessions(for_train=True)
        train_sessions = [x for x in train_sessions if x != session_id]
        models = self.loader.get_models(train_sessions, {'to_test': True})
        models_data = models.iloc[:, 1:-1]

        data_pred = self.loader.get_models([session_id], {'to_test': False})
        data_pred = data_pred .iloc[:, 1:-1]

        neigh.fit(models_data)
        res = neigh.kneighbors(data_pred)

        session_res = []
        for i in range(len(res)):
            session_index = res[1][0][i]
            session_name = models.iloc[session_index, 0]

            session_conf = res[0][0][i]

            session_res.append((session_name, session_conf))

        return session_res


class KNNPredicter(Predicter):

    def __init__(self, loader, k=5, words_num=20, seq_len=3, weights=1):
        """
        :param loader:  object of the type Loader. It contains all the data about CGM and
                        is able to generate the models with which the classifier will work.
        :param k:   number of neighbours used in the algortihm of KNN
        :param words_num:   number of words of the vocabulary generated
        :param seq_len: number of sessions of the sequences generated
        """
        super(KNNPredicter, self).__init__(loader, words_num, seq_len, weights)
        self._K = k

    def _predict(self, train_index, test_index):
        """
        :param train_index: list with the index of the models data used in the algorithm of SVM.
        :param test_index:  list with the index to predict.
        :return:    tuple with (label_prediction, label_score, word_prediction, word_score, bin_predictions) of test_index made by: 
                        - predict Label (hiper/hipo/normal) with KNN (using all trainindex)
                        - predict word with KNN using only hiper/hipo/normal trainindex depending on the label predicted
        """
        # models to predict
        data_pred = self.loader.get_models(test_index, {'weight': self.weights})
        data_pred = data_pred.iloc[:, 1:-1]

        # Get models with words
        wdata = self.loader.get_models(train_index, {'weight': self.weights})

        # Get models hipo/hiper/norm labels
        ldata_labels = self.loader.get_labels_of_words(wdata.iloc[:, -1].tolist())

        # Predict label
        knn = OneVsRestClassifier(KNN(n_neighbors=self._K))
        y = label_binarize(ldata_labels, classes=[-1, 0, 1])
        knn.fit(wdata.iloc[:, 1:-1], y)
        bin_labels = knn.predict(data_pred)
        score_label = knn.predict_proba(data_pred)

        # Predict word using only vocabulary of label predicted
        res_word = []
        res_label = []
        score_word = []
        predicters = {}
        for i in range(len(bin_labels)):
            label = None
            for l in range(-1, 2):
                if bin_labels[i][l+1] == 1:
                    label = l
                    break
            if not label:
                maxscore = -1000
                for s in range(-1, 2):
                    score = score_label[i][s+1]
                    if score >= maxscore:
                        label = s
                        maxscore = score
            res_label.append(label)
            # Work only with sessions of the label
            sessions = [wdata.iloc[z, 0] for z in range(len(wdata)) if ldata_labels[z] == label]

            models = wdata[wdata.id.isin(sessions)]
            models_labels = models.iloc[:, -1]
            models = models.iloc[:, 1:-1]

            if not predicters.get(str(label), False):
                knn = KNN(n_neighbors=self._K)
                knn.fit(models, models_labels)
                predicters[str(label)] = knn

            knn = predicters.get(str(label))
            res_w = knn.predict(data_pred)
            score_w = knn.predict_proba(data_pred)

            res_word.append(res_w)
            score_word.append(score_w)

        return (res_label, score_label, res_word, score_word, bin_labels)


class SVMPredicter(Predicter):
    def __init__(self, loader=None, kernel='rbf', c=10, gamma=0.1, words_num=5, seq_len=3, wheights=1):
        """
        :param loader:  object of the type Loader. It contains all the data about CGM and
                        is able to generate the models with which the classifier will work.
        :param kernel: 'rbf', 'linear', 'poly', 'sigmoid'. Kernel of the algortihm SVM
        :param words_num:   number of words of the vocabulary generated
        :param seq_len: number of sessions of the sequences generated
        """
        super().__init__(loader, words_num, seq_len, wheights)
        self._kernel = kernel
        self._C = c
        self._gamma = gamma

    def _predict(self, train_index, test_index):
        """
        :param train_index: list with the index of the models data used in the algorithm of SVM.
        :param test_index:  list with the index to predict.
        :return:    tuple with (label_prediction, label_score, word_prediction, word_score, bin_predictions) of test_index made by: 
                        - predict Label (hiper/hipo/normal) with SVM (using all trainindex)
                        - predict word with SVM using only hiper/hipo/normal trainindex depending on the label predicted
        """
        # models to predict
        data_pred = self.loader.get_models(test_index, {'weight': self.weights})
        data_pred = data_pred.iloc[:, 1:-1]

        # Get models with words
        wdata = self.loader.get_models(train_index, {'weight': self.weights})

        # Get models hipo/hiper/norm labels
        ldata_labels = self.loader.get_labels_of_words(wdata.iloc[:, -1].tolist())

        # Predict label
        svm = OneVsRestClassifier(SVC(kernel=self._kernel, C=self._C, gamma=self._gamma))
        y = label_binarize(ldata_labels, classes=[-1, 0, 1])
        svm.fit(wdata.iloc[:, 1:-1], y)
        bin_labels = svm.predict(data_pred)
        score_label = svm.decision_function(data_pred)

        # Predict word using only vocabulary of label predicted
        res_word = []
        res_label = []
        score_word = []
        predicters = {}
        for i in range(len(bin_labels)):
            label = None
            for l in range(-1, 2):
                if bin_labels[i][l + 1] == 1:
                    label = l
                    break
            if not label:
                maxscore = -1000
                for s in range(-1, 2):
                    score = score_label[i][s + 1]
                    if score >= maxscore:
                        label = s
                        maxscore = score
            res_label.append(label)
            # Work only with sessions of the label
            sessions = [wdata.iloc[z, 0] for z in range(len(wdata)) if ldata_labels[z] == label]

            models = wdata[wdata.id.isin(sessions)]
            models_labels = models.iloc[:, -1]
            models = models.iloc[:, 1:-1]

            if not predicters.get(str(label), False):
                svm = SVC(kernel=self._kernel, C=self._C, gamma=self._gamma)
                svm.fit(models, models_labels)
                predicters[str(label)] = svm

            svm = predicters.get(str(label))
            res_w = svm.predict(data_pred)
            score_w = svm.decision_function(data_pred)

            res_word.append(res_w)
            score_word.append(score_w)

        return (res_label, score_label, res_word, score_word, bin_labels)


class ApatsSVMPredicter(SVMPredicter):

    def _predict(self, train_index, test_index):
        """
        :param train_index: list with the index of the models data used in the algorithm of SVM.
        :param test_index:  list with the index to predict.
        :return:    tuple with (label_prediction, label_score, word_prediction, word_score, bin_predictions) of test_index made by:
                        - predict Label (hiper/hipo/normal) with SVM (using all trainindex of the same meal type)
                        - predict word with SVM using only hiper/hipo/normal trainindex depending on the label predicted
        """
        # models to predict
        data_pred = self.loader.get_models(test_index, {'weight': self.weights})
        data_pred = data_pred.iloc[:, 1:-1]

        # Get models with words
        wdata = self.loader.get_models(train_index, {'weight': self.weights})

        bin_labels = []
        score_label = []

        # Get Possible meal labels
        meal_types = list(wdata.iloc[:, 3].unique())
        for d in data_pred.iloc[:, 2].unique():
            if d not in meal_types:
                meal_types.append(d)

        # split models by type of meal
        for etiqueta_apat in meal_types:
            mdata_pred = data_pred.ix[data_pred.iloc[:, 2] == etiqueta_apat]
            mwdata = wdata.ix[wdata.iloc[:, 3] == etiqueta_apat]

            # Get models hipo/hiper/norm labels
            ldata_labels = self.loader.get_labels_of_words(mwdata.iloc[:, -1].tolist())

            # Predict label
            svm = OneVsRestClassifier(SVC(kernel=self._kernel, C=self._C, gamma=self._gamma))
            y = label_binarize(ldata_labels, classes=[-1, 0, 1])
            svm.fit(mwdata.iloc[:, 1:-1], y)
            mbin_labels = svm.predict(mdata_pred)
            mscore_label = svm.decision_function(mdata_pred)
            if len(bin_labels):
                bin_labels = np.concatenate((bin_labels, mbin_labels), axis=0)
                score_label = np.concatenate((score_label, mscore_label), axis=0)
            else:
                bin_labels = mbin_labels
                score_label = mscore_label

        ldata_labels = self.loader.get_labels_of_words(wdata.iloc[:, -1].tolist())
        # Predict word using only vocabulary of label predicted
        res_word = []
        res_label = []
        score_word = []
        predicters = {}
        for i in range(len(bin_labels)):
            label = None
            for l in range(-1, 2):
                if bin_labels[i][l + 1] == 1:
                    label = l
                    break
            if not label:
                maxscore = -1000
                for s in range(-1, 2):
                    score = score_label[i][s + 1]
                    if score >= maxscore:
                        label = s
                        maxscore = score
            res_label.append(label)
            # Work only with sessions of the label
            # sessions = [wdata.iloc[z, 0] for z in range(len(wdata)) if ldata_labels[z] == label]
            #
            # models = wdata[wdata.id.isin(sessions)]
            # models_labels = models.iloc[:, -1]
            # models = models.iloc[:, 1:-1]
            #
            # if not predicters.get(str(label), False):
            #     svm = SVC(kernel=self._kernel, C=self._C, gamma=self._gamma)
            #     svm.fit(models, models_labels)
            #     predicters[str(label)] = svm
            #
            # svm = predicters.get(str(label))
            # res_w = svm.predict(data_pred)
            # score_w = svm.decision_function(data_pred)
            #
            # res_word.append(res_w)
            # score_word.append(score_w)

        return (res_label, score_label, res_word, score_word, bin_labels)


class SVM_KNN_Predicter(SVMPredicter):

    def _predict(self, train_index, test_index):
        """
        :param train_index: list with the index of the models data used in the algorithm of SVM.
        :param test_index:  list with the index to predict.
        :return:    tuple with (label_prediction, label_score, word_prediction, word_score, bin_predictions) of test_index made by: 
                        - predict Label (hiper/hipo/normal) with SVM (using all trainindex)
                        - predict word with KNN using only hiper/hipo/normal trainindex depending on the label predicted
        """
        # models to predict
        data_pred = self.loader.get_models(test_index, {'weight': self.weights})
        data_pred = data_pred.iloc[:, 1:-1]

        # Get models with words
        wdata = self.loader.get_models(train_index, {'weight': self.weights})

        # Get models hipo/hiper/norm labels
        ldata_labels = self.loader.get_labels_of_words(wdata.iloc[:, -1].tolist())

        # Predict label
        svm = OneVsRestClassifier(SVC(kernel=self._kernel, C=self._C, gamma=self._gamma))
        y = label_binarize(ldata_labels, classes=[-1, 0, 1])
        svm.fit(wdata.iloc[:, 1:-1], y)
        bin_labels = svm.predict(data_pred)
        score_label = svm.decision_function(data_pred)

        # Predict word using only vocabulary of label predicted
        res_word = []
        res_label = []
        score_word = []
        predicters = {}
        for i in range(len(bin_labels)):
            label = None
            for l in range(-1, 2):
                if bin_labels[i][l + 1] == 1:
                    label = l
                    break
            if not label:
                maxscore = -1000
                for s in range(-1, 2):
                    score = score_label[i][s + 1]
                    if score >= maxscore:
                        label = s
                        maxscore = score
            res_label.append(label)
            # Work only with sessions of the label
            sessions = [wdata.iloc[z, 0] for z in range(len(wdata)) if ldata_labels[z] == label]

            models = wdata[wdata.id.isin(sessions)]
            models_labels = models.iloc[:, -1]
            models = models.iloc[:, 1:-1]

            if not predicters.get(str(label), False):
                knn = KNN(n_neighbors=5)
                knn.fit(models, models_labels)
                predicters[str(label)] = knn

            knn = predicters.get(str(label))
            res_w = knn.predict(data_pred)
            score_w = knn.predict_proba(data_pred)

            res_word.append(res_w)
            score_word.append(score_w)

        return (res_label, score_label, res_word, score_word, bin_labels)
