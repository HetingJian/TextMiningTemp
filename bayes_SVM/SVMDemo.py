import os
import datetime
import time
import re
import sys
import pickle
import jieba
from imp import reload

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets.base import Bunch
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm


class SVM_model():
    import sys
    def __init__(self):
        self.bb = sys.path[1] + '/'
        self.bunch_relative_path =' '
        self.fearture_count = 5000
        self.abs_bunch_path = " "
        self.train_size = 0.3
    # def set_feature_count (self,n):
    #     self.fearture_count =  n
    #
    # def set_abs_bunch_path(self,str1):
    #     self.fearture_count =  str1

    def read_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            with open(path, 'rb') as f:
                return f.read()

    def writebunchobj(self, path, bunchobj):
        with open(path, "wb") as file_obj:
            pickle.dump(bunchobj, file_obj)

    def readbunchobj(self, path):
        with open(path, "rb") as file_obj:
            bunch = pickle.load(file_obj)
        return bunch

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)
        # print(pandas.DataFrame(metrics.confusion_matrix(label,predictions)))
        return predictions

    def Vectorize_classify_process (self, case = "count"):
        begintime = time.time()
        loadbunch = self.readbunchobj(self.abs_bunch_path)
        train_size = self.train_size
        trainDF = pd.DataFrame()
        train_text_list = loadbunch.contents
        label_list = loadbunch.label_num
        trainDF['text'] = train_text_list
        trainDF['title'] = label_list
        # model_selection.train_test_spilt 对数据进行分割,train_size,feature_count 在模型Init里定义
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['title'], train_size=train_size)
        count_vect = CountVectorizer(max_features=self.fearture_count)
        count_vect.fit(trainDF['text'])  # Learn a vocabulary dictionary of all tokens in the raw documents.

        xtrain_count = count_vect.transform(train_x)  # 拟合模型，并返回文本矩阵
        xvalid_count = count_vect.transform(valid_x)
        print("文本向量化用时：" + str(time.time()-begintime))
        begintime = time.time()
        print("SVM start:")
        # clf = svm.SVC(C=1.0, cache_size=800, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
        #               gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,
        #               tol=0.001, verbose=False)
        pre_mat = self.train_model(svm.LinearSVC(verbose=True), xtrain_count, train_y, xvalid_count)
        # print(pre_mat)
        print("SVM结束了,用时： " + str(time.time()-begintime))
        confusion_matrix = metrics.confusion_matrix(valid_y, pre_mat)
        accur = metrics.accuracy_score(valid_y, pre_mat)
        print("\n" + " confusion_matrix ")
        # print(confusion_matrix)
        # print(accur)

        print(pd.DataFrame(confusion_matrix, columns=loadbunch.target_name, index=loadbunch.target_name))
        print("\n" + "正确率为：")
        print(accur)
        print("\n" + "report : ")
        print(metrics.classification_report(valid_y, pre_mat, target_names=loadbunch.target_name))

if __name__ == '__main__':
    bunch_path = "../bunch_dataset/dataset_bunch.dat"  # 该字符串为分好词的bunch绝对文件路径
    #载入结束
    SVM_model = SVM_model()
    SVM_model.abs_bunch_path = bunch_path
    SVM_model.fearture_count = 400000  # 该属性限制词袋维数
    SVM_model.Vectorize_classify_process()


