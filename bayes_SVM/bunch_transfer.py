import os
import datetime
import sys
import pickle

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

# 将分词后的文件转换成bunch格式
class bunch_generate:
    # 初始化
    def __init__(self):
        self.data_path = "../cleared_dataset"
        self.result_path = "../bunch_dataset/"
        self.bunch_path = self.result_path + "dataset_bunch.dat"
        self.bunch_data = None
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)  # 如果保存文件夹路径不存在，创建文件夹
    # 文件读取
    def read_file(self, path):
        try:
            f = open(path, encoding='utf-8')  # 打开文件
            return f.read()
        except:
            f = open(path, 'rb')
            return f.read()
        #
        # with open(path, 'r', encoding='gbk') as f:
        #     return f.read()
        #bunch文件写入
    def writebunchobj(self ,path, bunchobj):
        with open(path, "wb") as file_obj:
            pickle.dump(bunchobj, file_obj)
    # 将txt文件读取转换为bunch对象
    def gen_bunch(self):
        begintime = datetime.datetime.now()
        fileDocs = os.listdir(self.data_path)  # 取出的为标签列表['体育', '娱乐',... '股票']
        bunch = Bunch(target_name=[], label=[], filenames=[], contents=[], filefullnames=[], label_num=[])
        # target_name是标签列表，label是标签，filenames是文件名，contents是内容，filefullnames是文件路径，label_num是标签对应的代码
        bunch.target_name.extend(fileDocs)
        # print(fileDocs)
        # 获取每个目录下所有的文件
        for mydir in fileDocs:
            class_path = self.data_path + "/" + mydir # 拼出分类子目录的路径
            # print(class_path)
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for filename in file_list:  # 遍历类别目录下文件
                if not filename.endswith("txt"):
                    continue
                fullname = class_path +"/" +  filename  # 拼出文件名全路径 ../cleared_dataset/体育/0.txt
                bunch.label_num.append(fileDocs.index(mydir))  # label : 股票
                bunch.label.append(mydir)  # label : 股票
                bunch.contents.append(self.read_file(fullname))  # 读取文件内容
                bunch.filenames.append(mydir + "/" + filename)  # 股票/644677.txt
                bunch.filefullnames.append(fullname)
        self.writebunchobj(self.result_path + "dataset_bunch.dat", bunch)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print("训练bunch:contents长度、label长度:", len(bunch.contents), len(bunch.label))
        print("训练数据保存完成,所花费时间为", span.seconds)
    # 将缓存文件转换为bunch对象
    def read_tmp_transfer_bunch(self, bunch_file_path):
        begintime = datetime.datetime.now()
        with open(bunch_file_path, 'rb') as bunch_object:
             self.bunch_data = pickle.load(bunch_object)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print("读取数据完成,所花费时间为", span.seconds)
    # 测试并解析bunch对象
    def print_bunch_file(self):
        if(self.bunch_data == None):
            return "bunch数据未读入"
        print("标签有：")
        print(len(self.bunch_data.label))
        print("名称有：")
        print(len(self.bunch_data.filenames))

if __name__ == '__main__':
    bunch = bunch_generate()
    # bunch.gen_bunch() # 生成一次即可
    bunch.read_tmp_transfer_bunch(bunch.bunch_path)
    bunch.print_bunch_file()