# -*- coding: utf-8 -*-
"""Training models batch by batch to decrease memory usage"""

import os
import sys
import json
import time
import io
import shutil
import tensorflow as tf
#from tensorflow.keras.layers import Dense, LSTM, Dropout
#from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam
from gensim.models import word2vec
import numpy as np
import pandas as pd
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, classification_report


def pre_word(data_path="", save_path="", size=None, min_count=None, workers=None):
    """Proessing words to generate w2v model"""
    print("start process word...")
    sentences = word2vec.LineSentence(data_path + 'test.txt')
    model = word2vec.Word2Vec(sentences, size=size,
                              min_count=min_count, workers=workers)
    model.save(save_path + 'w2v.model')


def pre_train_data(data_path="", save_path="", input_vec_size=None):
    """Processing training data using w2v model to generate npy data package"""
    elem_zero = np.array([0] * input_vec_size)
    print("loading w2v.model...")
    w2v_model = word2vec.Word2Vec.load(save_path + 'w2v.model')
    print("completed w2v.model loading")
    train_list = []
    train_label_list = []
    # data preprocessing
    print("read data...")
    train_file = io.open(data_path + 'train.txt', encoding='utf-8')
    train_arr = train_file.readlines()
    train_file.close()
    train_label_file = io.open(
        data_path + 'train_label.txt', encoding='utf-8')
    train_label_arr = train_label_file.readlines()
    train_label_file.close()
    test_file = io.open(data_path + 'test.txt', encoding='utf-8')
    test_arr = test_file.readlines()
    test_file.close()
    print("process train label ...")
    for x in train_label_arr:
        train_label_list.append(int(x))
    print("save train label in npy...")
    train_label = np.array(train_label_list)
    np.save(save_path + "train_label.npy", train_label)

    g_max_col_count = 0
    # find max g_max_col_count
    for x in train_arr:
        x = x.strip().split()
        if len(x) > g_max_col_count:
            g_max_col_count = len(x)
    for x in test_arr:
        x = x.strip().split()
        if len(x) > g_max_col_count:
            g_max_col_count = len(x)

    print("process train data ...")
    for x in train_arr:
        x = x.strip().split()
        if len(x) > g_max_col_count:
            g_max_col_count = len(x)
        x_list = []
        for y in x:
            if y in w2v_model.wv.vocab:
                x_list.append(w2v_model.wv[y])
            else:
                x_list.append(elem_zero)
        train_list.append(x_list)

    for x in train_list:
        if len(x) < g_max_col_count:
            for _ in range(g_max_col_count - len(x)):
                x.append(elem_zero)
    print("save train data in npy ...")
    train_data = np.array(train_list)
    np.save(save_path + "train_data.npy", train_data)


def pre_test_data(test_arr=None, test_label_arr=None, w2v_model=None,
                  input_vec_size=None, g_max_col_count=None):
    """Processing test data using w2v model to generate npy data package"""
    elem_zero = np.array([0] * input_vec_size)
    test_list = []
    test_label_list = []
    print("process test data ...")
    for x in test_arr:
        x = x.strip().split()
        if len(x) > g_max_col_count:
            g_max_col_count = len(x)
        x_list = []
        for y in x:
            if y in w2v_model.wv.vocab:
                x_list.append(w2v_model.wv[y])
            else:
                x_list.append(elem_zero)
        test_list.append(x_list)
    for x in test_list:
        if len(x) < g_max_col_count:
            for _ in range(g_max_col_count - len(x)):
                x.append(elem_zero)
    print("process test label ...")
    for x in test_label_arr:
        test_label_list.append(int(x))
    return np.array(test_list), np.array(test_label_list)


def main():
    """Main entry"""
    # initing
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        print("No config")
    config_file_ptr = io.open(config_file, "r")
    settings = json.loads(config_file_ptr.read())
    data_path = settings["data_path"]
    work_path = settings["work_path"]
    g_input_vec_size = settings["g_input_vec_size"]

    word_min_count = settings["word_min_count"]
    word_workers = settings["word_workers"]

    lstm_units = settings["lstm_units"]
    # dense_units = settings["dense_units"]
    model_epochs = settings["model_epochs"]
    save_model_name = settings["model_name"]
    model_loss_name = settings["model_loss"]
    drop_ratio = settings["drop_ratio"]
    learning_rate = settings["learning_rate"]

    train_batch_size = settings["train_batch_size"]
    train_verbose = settings["train_verbose"]
    test_batch_size = settings["test_batch_size"]
    test_verbose = settings["test_verbose"]

    # whether to retatin the w2v.model;1-yes,
    retrain_flag = settings["retrain_flag"]
    reading_test_lines = settings["reading_test_lines"]
    test_threshold = settings["test_threshold"]
    config_file_ptr.close()

    # require to retrain
    if retrain_flag == 1:
        pre_word(data_path=data_path, save_path=work_path, size=g_input_vec_size,
                 min_count=word_min_count, workers=word_workers)
        pre_train_data(data_path=data_path, save_path=work_path,
                       input_vec_size=g_input_vec_size)
    # not require
    else:
        all_files = os.listdir(work_path)
        if ("train_data.npy" in all_files) and ("train_label.npy" in all_files):
            pass
        else:
            if "w2v.model" in all_files:
                pre_train_data(data_path=data_path, save_path=work_path,
                               input_vec_size=g_input_vec_size)
            else:
                pre_word(data_path=data_path, save_path=work_path, size=g_input_vec_size,
                         min_count=word_min_count, workers=word_workers)
                pre_train_data(data_path=data_path, save_path=work_path,
                               input_vec_size=g_input_vec_size)

    train_data = np.load(work_path + "train_data.npy", allow_pickle=True)
    train_label_data = np.load(
        work_path + "train_label.npy", allow_pickle=True)
    train_label = to_categorical(train_label_data)
    g_max_col_count = train_data.shape[1]
    last_dense_units = train_label.shape[1]

    # create directory for saving the result
    result_dir = work_path + str(int(time.time()))
    os.makedirs(result_dir)

    # save the config
    shutil.copy(sys.argv[1], result_dir)
    # save the source code
    shutil.copy(sys.argv[0], result_dir)
    # build the model
    #model = Sequential()
    #model.add(Dropout(drop_ratio))
    #model.add(LSTM(units=lstm_units))
    #model.add(Dropout(drop_ratio))
    # model.add(Dense(units = dense_units, kernel_regularizer = l2(0.01), activation = 'softmax'))
    #model.add(Dense(units=last_dense_units, activation='softmax'))
    model = tf.keras.Sequential([
        #tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(drop_ratio),
        #tf.keras.layers.Dense(1, activation='sigmoid')
        tf.keras.layers.Dense(last_dense_units, activation='softmax')
    ])
    adam = Adam(lr=learning_rate)
    # compile
    model.compile(loss=model_loss_name, optimizer=adam,
                  metrics=['accuracy'])

    # train
    model.fit(train_data, train_label, epochs=model_epochs,
              batch_size=train_batch_size, verbose=train_verbose)
    # save model---model's structure,weights,Training configuration and Optimizer status
    print("saving the train model ...")
    model.save(result_dir + "/" + save_model_name)
    print("train model is saved")
    # evaluation
    model.summary()

    # test data
    ftest_data = io.open(data_path + "test.txt", "r")
    test_lines = ftest_data.readlines()
    ftest_data.close()
    ftest_label = io.open(data_path + "test_label.txt", "r")
    test_label_lines = ftest_label.readlines()
    ftest_label.close()
    f_result = io.open(result_dir + "/result.txt", "a+")
    f_result.write("****************************************\n")
    w2v_model = word2vec.Word2Vec.load(work_path + 'w2v.model')
    continue_test_flag = True
    line_num = 0
    test_truth = []
    test_pred = []
    while continue_test_flag:
        test_arr = test_lines[line_num *
                              reading_test_lines:(line_num+1)*reading_test_lines]
        test_label_arr = test_label_lines[line_num *
                                          reading_test_lines:(line_num+1)*reading_test_lines]
        test_data, test_pre_label = pre_test_data(test_arr=test_arr,
                                                  test_label_arr=test_label_arr,
                                                  w2v_model=w2v_model,
                                                  input_vec_size=g_input_vec_size,
                                                  g_max_col_count=g_max_col_count)
        test_truth.extend(test_pre_label)
        temp_pred = model.predict_classes(test_data)
        test_pred.extend(temp_pred)
        line_num += 1
        score = accuracy_score(test_pre_label, temp_pred)
        if score < test_threshold:
            continue_test_flag = False
        if len(test_arr) < reading_test_lines:
            print("all test complete!")
            continue_test_flag = False
    acc = accuracy_score(test_truth, test_pred)
    precision = precision_score(test_truth, test_pred, average='macro')
    recall = recall_score(test_truth, test_pred, average='macro')
    f1 = f1_score(test_truth, test_pred, average='macro')
    f_result.write('accuracy:'+ str(acc)+'\n')
    f_result.write('precision:'+ str(precision)+'\n')
    f_result.write('recall:'+ str(recall)+'\n')
    f_result.write('f1:'+ str(f1)+'\n')
    f_result.close()

    report_test = classification_report(test_truth, test_pred)
    print(report_test)
    with open(result_dir+'report_test.txt', 'w') as f:
        f.write(report_test)

    train_pred = model.predict_classes(train_data)
    report_train = classification_report(train_label_data, train_pred)
    print(report_train)
    with open(result_dir+'report_train.txt', 'w') as f:
        f.write(report_train)

main()
