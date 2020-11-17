import datetime

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import os
import jieba
import re


#目前所做的工作：
# 一、数据预处理
# 文件读取
# 文件分词保存
# 文本清洗一阶段去除标点和停止词
# 文本清洗二阶段去除低频词，降维
# label向量转换为int型
# 37分训练和测试数据集，存入DF中
# 二、特征提取
# 特征一：计数器特征
# 特征二：tfidf特征
# 特征三：word embeding特征
#
# 三、模型训练
# 模型一：朴素贝叶斯
# 模型二：逻辑斯谛回归
# 模型三：SVM
# 模型四：Bagging Model
# 模型五：Boosting Model
# Main: 数据转换，单一文本文件jieba分词保存， 默认用空格分隔
def read_file_to_spilt(full_filename):
    str1 = ""
    f = open(full_filename, encoding='utf-8')  # 打开文件
    line1 = f.readlines()
    for i in line1:
        seg_list = jieba.cut(i)  # 默认是精确模式
        a = " ".join(seg_list)  #分词后用“、”间隔
        str1 = str1 + a  #文档内容换行后与上一行拼接
    f.close()  #关闭文件
    # print(str1)
    return str1

# Main:数据清洗，去除停止词 返回去除停止词的文本
def clear_bad_character(str):                  #删除【逗号、多余的空格、句号】后的字符串
    #数据清洗1：
    # \xa0是不间断空白符  \u3000是全角的空白符
    # str.strip('\r\n').replace(u'\u3000', u' ').replace(u'\xa0', u' ')
    str = str.strip("").strip('\r\n').strip(u'\u3000').strip(u'\xa0').strip(u'\n\u3000')
    #数据清洗2：
    str_split_list = str.split(" ")
    newtext = ""
    f = open("../stop_words_cn.txt", encoding='utf-8-sig')
    stopword_list = f.read().splitlines()
    f.close()
    for textword in str_split_list:
        flag = True
        textword = textword.strip("\n")
        if(len(textword)==1 or textword == "" or textword=="\u3000" or textword=="\xa0" or textword=="\n\u3000" or textword =="\n\xa0"):
            continue
        for stopword in stopword_list:
            if(textword == stopword):
                flag = False
                break;
        if(flag):
            newtext += textword
            newtext += " "
    # print(newtext)
    newtext.rstrip(" ")
    return newtext

# Main:把某一label的文档关键词倒排索引打印, 并建立低频词表
# 注意spilt后需要清洗
def get_high_frequency_word(textlist):
    dic = {}
    for text in textlist:
        splittest = text.split(" ")
        for word in splittest:
            if (word =="") : # spilt 后需要清洗==============================================
                continue
            if dic:
                flag = False
                for dicword in dic:
                    if (word == dicword):
                        flag = True
                        dic[word] += 1
                        break
                if (flag == False):
                    dic[word] = 1
            else:
                dic[word] = 1
    sorted_dict = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_dict) # 倒序排列词频 打印-------------------------------------
    # print(len(sorted_dict))
    # print("---------------------------------------------")
    # low_frequency_word =[]
    high_frequency_word = []
    count = 0
    for item in sorted_dict:
        if(item[1] > 20):
            count+=1
            high_frequency_word.append(item[0])
    count *= 0.7
    count = numpy.math.ceil(count)
    return high_frequency_word[:count]

# 文本大清洗第二阶段: 去除文本中低频词
def remain_high_frequency_word_for_text(text, high_frequency_word_list):
    newtext = ""
    wordlist = text.split(" ")
    for word in wordlist:
        if(word==""):
            continue
        flag = False
        for high_fre_word in high_frequency_word_list:
            if(word == high_fre_word):
                flag = True
                break
        if(flag):
            newtext += word
            newtext += " "
    newtext.rstrip(" ")
    return newtext

#写入文件
def write_cleared_file(text, label, filename):
    filepath = "../cleared_dataset/"+label +"/"+filename
    if os.path.exists(filepath):
        os.remove(filepath)
    # 以写的方式打开文件，如果文件不存在，就会自动创建
    file_write_obj = open(filepath, 'w')
    file_write_obj.write(text)

# 通过文件夹路径读取文件，生成文本和label列表
def readallfile_to_text_label(path):
    subfiles = os.listdir(path)           # 得到文件夹下的所有文件名称v 此处为0，1，2....txt
    same_label_wordlist = []
    filename_list = []
    for file in subfiles:               #遍历文件夹，建立总词库
    # #     if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        filepath = path + "/"+file  # file 就是100.txt
        text = read_file_to_spilt(filepath)            # 数据转换,读取文件内容并分词
        cle_text = clear_bad_character(text)            # 数据清洗 一阶段
        write_cleared_file(cle_text, "文化", file)
    #     same_label_wordlist.append(cle_text)            # 一阶段清洗后的文本添加至同label列表，为了二阶段清洗
    #     filename_list.append(file)
    # high_frequency_word_list = get_high_frequency_word(same_label_wordlist)             # 倒排索引+ 建立高频词库
    # count = 0
    # for text in same_label_wordlist:
    #     cleared_text = remain_high_frequency_word_for_text(text, high_frequency_word_list)
    #     filename = filename_list[count]
    #     # 清洗过后的文本保存写入
    #     write_cleared_file(cleared_text, "体育", filename)
    #     count+=1

# Main:---------------------------------------------
alltextlist = []  # 存放所有的文本
alllabels = []  # 存放所有的label
path = "../dataset/data/文化" # 数据集相对路径
readallfile_to_text_label(path)







