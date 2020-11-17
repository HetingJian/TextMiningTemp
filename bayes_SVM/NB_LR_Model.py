import datetime

import sys
sys.path.append('~/textdata/TextMiningByDannis')
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import os
import jieba
import re
from bayes_SVM import bunch_transfer


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

# 将模型训练结果写入txt文件
def write_result(result_list, path):
    if os.path.exists(path):
        os.remove(path)
    # 以写的方式打开文件，如果文件不存在，就会自动创建
    file_write_obj = open(path, 'w')
    for var in result_list:
        file_write_obj.writelines(var)
        file_write_obj.write('\n')

# Main: 数据转换，单一文本文件jieba分词保存， 默认用空格分隔
# def read_file_to_spilt(full_filename):
# #     str1 = ""
# #     f = open(full_filename, encoding='utf-8')  # 打开文件
# #     line1 = f.readlines()
# #     for i in line1:
# #         seg_list = jieba.cut(i)  # 默认是精确模式
# #         a = " ".join(seg_list)  #分词后用“、”间隔
# #         str1 = str1 + a  #文档内容换行后与上一行拼接
# #     f.close()  #关闭文件
# #     # print(str1)
# #     return str1

# Main:数据清洗，去除停止词 返回去除停止词的文本
# def clear_bad_character(str):                  #删除【逗号、多余的空格、句号】后的字符串
#     #数据清洗1：
#     # \xa0是不间断空白符  \u3000是全角的空白符
#     # str.strip('\r\n').replace(u'\u3000', u' ').replace(u'\xa0', u' ')
#     str = str.strip("").strip('\r\n').strip(u'\u3000').strip(u'\xa0').strip(u'\n\u3000')
#     #数据清洗2：
#     str_split_list = str.split(" ")
#     newtext = ""
#     f = open("../stop_words_cn.txt", encoding='utf-8-sig')
#     stopword_list = f.read().splitlines()
#     f.close()
#     for textword in str_split_list:
#         flag = True
#         textword = textword.strip("\n")
#         if(len(textword)==1 or textword == "" or textword=="\u3000" or textword=="\xa0" or textword=="\n\u3000" or textword =="\n\xa0"):
#             continue
#         for stopword in stopword_list:
#             if(textword == stopword):
#                 flag = False
#                 break;
#         if(flag):
#             newtext += textword
#             newtext += " "
#     # print(newtext)
#     newtext.rstrip(" ")
#     return newtext


# Main:把某一label的文档关键词倒排索引打印, 并建立低频词表
# 注意spilt后需要清洗
# def get_low_frequency_word(textlist):
#     dic = {}
#     for text in textlist:
#         splittest = text.split(" ")
#         for word in splittest:
#             if (word =="") : # spilt 后需要清洗==============================================
#                 continue
#             if dic:
#                 flag = False
#                 for dicword in dic:
#                     if (word == dicword):
#                         flag = True
#                         dic[word] += 1
#                         break
#                 if (flag == False):
#                     dic[word] = 1
#             else:
#                 dic[word] = 1
#     sorted_dict = sorted(dic.items(), key=lambda x: x[1], reverse=True)
#     write_result(sorted_dict, "word_frequency.txt")
#     # print(sorted_dict) # 倒序排列词频 打印-------------------------------------
#     # print(len(sorted_dict))
#     # print("---------------------------------------------")
#     low_frequency_word =[]
#     for item in sorted_dict:
#         if(item[1] < 20):
#             low_frequency_word.append(item[0])
#     return low_frequency_word

# 文本大清洗第二阶段: 去除文本中低频词
# def clear_low_frequency_word_for_text(text, low_frequency_word_list):
#     newtext = ""
#     wordlist = text.split(" ")
#     for word in wordlist:
#         if(word==""):
#             continue
#         flag = False
#         for low_fre_word in low_frequency_word_list:
#             if(word == low_fre_word):
#                 flag = True
#                 break
#         if(flag == False):
#             newtext += word
#             newtext += " "
#     newtext.rstrip(" ")
#     return newtext

# 通过文件夹路径读取文件，生成文本和label列表
# def readallfile_to_text_label(path):
#     files = os.listdir(path)# 得到文件夹下的所有label名称，files元素【体育，娱乐，游戏.....】
#     filename_list = [] # 存放14个label文件夹路径,元素【E:\Workplace..\体育，E:\Workplace..\娱乐，E:\Workplace..\游戏,.....】
#     for j in files:
#         filename_list.append(path + "/" + j)
#     for i in filename_list:
#         same_label_wordlist = []            # 建立同一label的文本列表
#         s1 = i
#         pathsplit = i.split("/")            # 路径拆分
#         label = pathsplit[len(pathsplit)-1] # 获取label,同一目录文件下具有相同的label
#         subfiles = os.listdir(s1)           # 得到文件夹下的所有文件名称v 此处为0，1，2....txt
#         for file in subfiles:               #遍历文件夹，建立总词库
#         # #     if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
#             filename1 = s1 + "/"+file
#             text = read_file_to_spilt(filename1)            # 数据转换
#             cle_text = clear_bad_character(text)            # 数据清洗 一阶段
#             same_label_wordlist.append(cle_text)            # 一阶段清洗后的文本添加至同label列表，为了二阶段清洗
#         clear_test_list = []  # TEST: 文本清洗二阶段文本列表
#         low_frequency_word_list = get_low_frequency_word(same_label_wordlist)             # 倒排索引+ 建立低频词库
#         for text in same_label_wordlist:
#             cleared_text = clear_low_frequency_word_for_text(text, low_frequency_word_list)
#             clear_test_list.append(cleared_text) # TEST:添加二阶段清洗文本到列表
#             alltextlist.append(cleared_text)  # 添加清洗过后的文本到总文本列表里
#             alllabels.append(label)
#         get_low_frequency_word(clear_test_list)  # TEST: 打印二阶段清洗效果

begintime = datetime.datetime.now()

# Main:---------------------------------------------
# alltextlist = []  # 存放所有的文本
# alllabels = []  # 存放所有的label
# path = "../dataset/data" # 数据集相对路径
# readallfile_to_text_label(path)

#读入分词好的数据
def read_bunch_data():
    bunch = bunch_transfer.bunch_generate()
    bunch.read_tmp_transfer_bunch(bunch.bunch_path)
    data = bunch.bunch_data
    return data

bunchdata = read_bunch_data()
print(len(bunchdata.contents))


# 将数据放置DF中
trainDF = pandas.DataFrame()
trainDF['text'] = bunchdata.contents
trainDF['label'] = bunchdata.label

#分训练验证集合 7500 训练集合， 2500 测试集 x是文本， y是label
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], train_size=0.3)

# 编码为变量 把label的种类 映射成向量，分别用0,1,2,3,4,5，等等去映射种类
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
vaild_y = encoder.fit_transform(valid_y)

# print(trainDF['text'])
# 二：特征工程
# 2.1 计数向量作为特征
# 创建一个向量计数器对象
count_vect = CountVectorizer(max_df=1.0, min_df=1,max_features=300000)
count_vect.fit(trainDF['text']) # Learn a vocabulary dictionary of all tokens in the raw documents.
print(count_vect.get_feature_names())
print(len(count_vect.get_feature_names()))
# 使用向量计数器对象转换训练集和验证集
xtrain_count = count_vect.transform(train_x)  # 拟合模型，并返回文本矩阵
xvalid_count = count_vect.transform(valid_x)
# print('xtrain_count')
# print(xtrain_count)
# print('xvalid_count')
# print(xvalid_count)

# 2.2 TF-IDF向量作为特征
# 词语级tf-idf
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# # ngram 级tf-idf
# tfidf_vect_ngram = TfidfVectorizer()
# tfidf_vect_ngram.fit(trainDF['text'])
# xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
# xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)
#
# # 词性级tf-idf
# tfidf_vect_ngram_chars = TfidfVectorizer()
# tfidf_vect_ngram_chars.fit(trainDF['text'])
# xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
# xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

# # 2.3 词嵌入
# # 加载预先训练好的词嵌入向量
# embeddings_index = {}
# for i, line in enumerate(open('../dataset/wiki-news-300d-1M.vec', encoding='UTF-8')):
#     values = line.split()
#     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
#
# # 创建一个分词器
# token = text.Tokenizer()
# token.fit_on_texts(trainDF['text'])
# word_index = token.word_index
#
# # 将文本转换为分词序列，并填充它们保证得到相同长度的向量
# train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
# valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
#
# # 创建分词嵌入映射
# embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
# if embedding_vector is not None:
#     embedding_matrix[i] = embedding_vector
#
# # 2.4 基于文本/NLP的特征
# trainDF['char_count'] = trainDF['text'].apply(len)
# trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
# trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
# trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
# trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
# trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
#
# trainDF['char_count'] = trainDF['text'].apply(len)
# trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
# trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
# trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
# trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
# trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
# pos_family = {
#     'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
#     'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
#     'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
#     'adj': ['JJ', 'JJR', 'JJS'],
#     'adv': ['RB', 'RBR', 'RBS', 'WRB']
# }
#
#
# # 检查和获得特定句子中的单词的词性标签数量
# def check_pos_tag(x, flag):
#     cnt = 0
#     try:
#         wiki = textblob.TextBlob(x)
#         for tup in wiki.tags:
#             ppo = list(tup)[1]
#         if ppo in pos_family[flag]:
#             cnt += 1
#     except:
#         pass
#     return cnt
#
# trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
# trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
# trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
# trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
# trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))



# 2.5 主题模型作为特征
# 训练主题模型
# lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
# X_topics = lda_model.fit_transform(xtrain_count)
# topic_word = lda_model.components_
# vocab = count_vect.get_feature_names()
#
# # 可视化主题模型
# n_top_words = 10
# topic_summaries = []
# for i, topic_dist in enumerate(topic_word):
#     topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words + 1):-1]
#     topic_summaries.append(' '.join(topic_words))
#
# # 三、建模---------------------
#
# # 朴素贝叶斯分类器
# # 线性分类器
# # 支持向量机（SVM）
# # Bagging Models
# # Boosting Models
# # 浅层神经网络
# # 深层神经网络
# # 卷积神经网络（CNN）
# # LSTM
# # GRU
# # 双向RNN
# # 循环卷积神经网络（RCNN）
# # 其它深层神经网络的变种
#
result_list = []
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, vaild_y), metrics.classification_report(vaild_y,predictions), metrics.confusion_matrix(vaild_y,predictions)
#


# # 3.1 朴素贝叶斯
#
# 特征为计数向量的朴素贝叶斯
accuracy,classification_report,confusion_matrix = train_model(naive_bayes.MultinomialNB(alpha=0.001), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy)
bayes_cv_result = "NB, Count Vectors: " + str(accuracy)
result_list.append(bayes_cv_result)

#混淆矩阵和F1
print(classification_report)
print(confusion_matrix)
#
# # print("未去除停用词的CountVectorizer提取的特征学习模型准确率：", mnb_count.score(x_count_test, y_test))
# # print("更加详细的评估指标:\n", classification_report(mnb_count_y_predict, y_test))
# # print("去除停用词的CountVectorizer提取的特征学习模型准确率：", mnb_count_stop.score(x_count_stop_test, y_test))
# # print("更加详细的评估指标:\n", classification_report(mnb_count_stop_y_predict, y_test))
#
# 特征为词语级别TF-IDF向量的朴素贝叶斯
accuracy,classification_report,confusion_matrix  = train_model(naive_bayes.MultinomialNB(alpha=0.001), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy)
bayes_tfidf_result = "NB, WordLevel TF-IDF: " + str(accuracy)
result_list.append(bayes_tfidf_result)

print(classification_report)
print(confusion_matrix)
# 特征为多个词语级别TF-IDF向量的朴素贝叶斯
# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
# print("NB, N-Gram Vectors: ", accuracy)

# 特征为词性级别TF-IDF向量的朴素贝叶斯
# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
# print("NB, CharLevel Vectors: ", accuracy)
# # # 输出结果
# # NB, Count
# # Vectors: 0.7004
# # NB, WordLevel
# # TF - IDF: 0.7024
# # NB, N - Gram
# # Vectors: 0.5344
# # NB, CharLevel
# # Vectors: 0.6872
#
#
#
# Linear Classifier on Count Vectors
accuracy,classification_report,confusion_matrix  = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)
LR_cv_result = "LR, Count Vectors: " + str(accuracy)
result_list.append(LR_cv_result)

print(classification_report)
print(confusion_matrix)
#
# # 特征为词语级别TF-IDF向量的线性分类器
# accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
# print("LR, WordLevel TF-IDF: ", accuracy)
# LR_tfidf_result = "LR, WordLevel TF-IDF: " + str(accuracy)
# result_list.append(LR_tfidf_result)
# 特征为多个词语级别TF-IDF向量的线性分类器
# accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
# print("LR, N-Gram Vectors: ", accuracy)
#
# # 特征为词性级别TF-IDF向量的线性分类器
# accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
# print("LR, CharLevel Vectors: ", accuracy)
# # # 输出结果
# # LR, Count
# # Vectors: 0.7048
# # LR, WordLevel
# # TF - IDF: 0.7056
# # LR, N - Gram
# # Vectors: 0.4896
# # LR, CharLevel
# # Vectors: 0.7012
#
#
#
# accuracy = train_model(svm.SVC(C=30.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
#                   gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
#                   tol=0.001, verbose=False), xtrain_count, train_y, xvalid_count)
# print("SVM, Count Vectors: ", accuracy)
# SVM_cv_result = "SVM, Count Vectors: " + str(accuracy)
# result_list.append(SVM_cv_result)
# accuracy = train_model(svm.SVC(C=30.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
#                   gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
#                   tol=0.001, verbose=False), xtrain_tfidf, train_y, xvalid_tfidf)
# print("SVM, WordLevel TF-IDF: ", accuracy)
# #特征为多个词语级别TF-IDF向量的SVM
# accuracy = train_model(svm.SVC(C=30.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
#                   gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
#                   tol=0.001, verbose=False), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
# print("SVM, N-Gram Vectors: ", accuracy)
# # #输出结果
# # # SVM, N-Gram Vectors:  0.5296
# #
# #
# # # 3.4 Bagging Model
# # #特征为计数向量的RF
# accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
# print ("RF, Count Vectors: ", accuracy)
# #
# # #特征为词语级别TF-IDF向量的RF
# accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
# print ("RF, WordLevel TF-IDF: ", accuracy)
# # #输出结果
# # # RF, Count Vectors: 0.6972
# # # RF, WordLevel TF-IDF: 0.6988
# #
# # # 3.5 Boosting Model
# # #特征为计数向量的Xgboost
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
# print("Xgb, Count Vectors: ", accuracy)
# #
# # #特征为词语级别TF-IDF向量的Xgboost
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
# print ("Xgb, WordLevel TF-IDF: ", accuracy)
# #
# # #特征为词性级别TF-IDF向量的Xgboost
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
# print ("Xgb, CharLevel Vectors: ", accuracy)
# #输出结果
# # Xgb, Count Vectors: 0.6324
# # Xgb, WordLevel TF-IDF: 0.6364
# # Xgb, CharLevel Vectors: 0.6548

endtime = datetime.datetime.now()
span = endtime - begintime
cost_of_train = "训练时间:" + str(span.min)
result_list.append(cost_of_train)

write_result(result_list, "result.txt")


