# -*- coding: utf-8 -*-
"""test with saved models"""

import os
import sys
import json
import time
import io
import shutil
#import math
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import to_categorical
from gensim.models import word2vec


def pre_test_data(test_arr=None, test_label_arr=None, w2v_model=None,
                  input_vec_size=None, g_max_col_count=None):
    '''test function'''
    elem_zero = np.array([0] * input_vec_size)
    test_list = []
    test_label_list = []
    print("process test ...")
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
    '''main entry'''
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

    save_model_name = settings["model_name"]
    test_batch_size = settings["test_batch_size"]
    test_verbose = settings["test_verbose"]

    # whether to retatin the w2v.model;1-yes,
    #retrain_flag = settings["retrain_flag"]
    reading_test_lines = settings["reading_test_lines"]
    test_threshold = settings["test_threshold"]
    config_file_ptr.close()

    # create directory for saving the result
    result_dir = work_path + str(int(time.time()))
    os.makedirs(result_dir)

    # save the config
    shutil.copy(sys.argv[1], result_dir)
    # save the source code
    shutil.copy(sys.argv[0], result_dir)

    print("loading train and train_label ...")
    train_data = np.load(work_path + "train_data.npy")
    train_label_data = np.load(work_path + "train_label.npy")
    train_label = tf.keras.utils.to_categorical(train_label_data)
    g_max_col_count = train_data.shape[1]
    last_dense_units = train_label.shape[1]
    train_data = []
    train_label_data = []

    print("loading word2vecModel ...")
    w2v_model = word2vec.Word2Vec.load(work_path + 'w2v.model')
    print("loading model ...")
    model = tf.keras.models.load_model(work_path + save_model_name)

    shutil.copy(work_path + save_model_name, result_dir)

    # test data
    print("reading data ...")
    ftest_data = io.open(data_path + "test.txt", "r")
    test_lines = ftest_data.readlines()
    ftest_data.close()
    ftest_label = io.open(data_path + "test_label.txt", "r")
    test_label_lines = ftest_label.readlines()
    ftest_label.close()

    f_result = io.open(result_dir + "/result.txt", "a+")
    f_result.write("****************************************\n")

    continue_test_flag = True
    line_num = 0
    print("start test ...")
    while continue_test_flag:
        test_arr = test_lines[line_num *
                              reading_test_lines:(line_num + 1) * reading_test_lines]
        test_label_arr = test_label_lines[line_num *
                                          reading_test_lines:(line_num + 1) * reading_test_lines]
        test_data, test_pre_label = pre_test_data(test_arr=test_arr,
                                                  test_label_arr=test_label_arr,
                                                  w2v_model=w2v_model,
                                                  input_vec_size=g_input_vec_size,
                                                  g_max_col_count=g_max_col_count)
        print("preparation phase 1 finished.")
        test_label = tf.keras.utils.to_categorical(
            test_pre_label, last_dense_units)
        print("preparation phase 2 finished.")
        score = model.evaluate(test_data, test_label,
                               batch_size=test_batch_size, verbose=test_verbose)
        print(score)
        f_result.write(str(score) + "\n")
        line_num += 1
        if score[1] < test_threshold:
            continue_test_flag = False
        if len(test_arr) < reading_test_lines:
            print("all test completed!")
            continue_test_flag = False

    f_result.close()


main()
