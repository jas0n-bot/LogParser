import os
import sys
import json
import time
import codecs
import shutil
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from gensim.models import word2vec
import numpy as np


def pre_word(data_path="", save_path="", size=None, min_count=None, workers=None):

    print("start process word...")
    sentences = word2vec.LineSentence(data_path + 'test.txt')
    model = word2vec.Word2Vec(sentences, size=size,
                              min_count=min_count, workers=workers)
    model.save(save_path + 'w2v.model')


def pre_data(data_path="", save_path="", input_vec_size=None):

    print("start process data...")
    elem_zero = [0] * input_vec_size
    word2vecModel = word2vec.Word2Vec.load(save_path + 'w2v.model')
    train_list = []
    train_label_list = []
    test_list = []
    test_label_list = []

    # data preprocessing
    train_file = codecs.open(data_path + 'train.txt', encoding='utf-8')
    train_arr = train_file.readlines()
    train_file.close()
    train_label_file = codecs.open(
        data_path + 'train_label.txt', encoding='utf-8')
    train_label_arr = train_label_file.readlines()
    train_label_file.close()
    test_file = codecs.open(data_path + 'test.txt', encoding='utf-8')
    test_arr = test_file.readlines()
    test_file.close()
    test_label_file = codecs.open(
        data_path + 'test_label.txt', encoding='utf-8')
    test_label_arr = test_label_file.readlines()
    test_label_file.close()

    g_max_col_count = 0

    for x in train_arr:
        x = x.strip().split()
        if len(x) > g_max_col_count:
            g_max_col_count = len(x)
        x_list = []
        for y in x:
            if y in word2vecModel.wv.vocab:
                x_list.append(word2vecModel[y])
            else:
                x_list.append(elem_zero)
        train_list.append(x_list)
    for x in train_label_arr:
        train_label_list.append(int(x))

    for x in test_arr:
        x = x.strip().split()
        if len(x) > g_max_col_count:
            g_max_col_count = len(x)
        x_list = []
        for y in x:
            if y in word2vecModel.wv.vocab:
                x_list.append(word2vecModel[y])
            else:
                x_list.append(elem_zero)
        test_list.append(x_list)
    for x in test_label_arr:
        test_label_list.append(int(x))

    for x in train_list:
        if len(x) < g_max_col_count:
            for _ in range(g_max_col_count - len(x)):
                x.append(elem_zero)
    for x in test_list:
        if len(x) < g_max_col_count:
            for _ in range(g_max_col_count - len(x)):
                x.append(elem_zero)

    train_data = np.array(train_list)
    train_label = np.array(train_label_list)

    test_data = np.array(test_list)
    test_label = np.array(test_label_list)

    np.savez(save_path + "data.npz", train_data=train_data,
             train_label=train_label, test_data=test_data, test_label=test_label)


if __name__ == '__main__':
    # initing
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        print("No config")
        path = ''
    fp = open(config_file, "r")
    settings = json.loads(fp.read())
    g_input_vec_size = settings["g_input_vec_size"]

    word_min_count = settings["word_min_count"]
    word_workers = settings["word_workers"]

    lstm_units = settings["lstm_units"]
    #dense_units = settings["dense_units"]
    model_epochs = settings["model_epochs"]
    save_model_name = settings["model_name"]
    model_loss_name = settings["model_loss"]
    drop_ratio = settings["drop_ratio"]
    learning_rate = settings["learning_rate"]

    data_path = settings["data_path"]
    work_path = settings["work_path"]

    train_batch_size = settings["train_batch_size"]
    train_verbose = settings["train_verbose"]
    test_batch_size = settings["test_batch_size"]
    test_verbose = settings["test_verbose"]

    # whether to retatin the w2v.model;1-yes,
    retrain_flag = settings["retrain_flag"]
    fp.close()

    #require to retrain
    if retrain_flag == 1:
        pre_word(data_path=data_path, save_path=work_path,
                 size=g_input_vec_size, min_count=word_min_count, workers=word_workers)
        pre_data(data_path=data_path, save_path=work_path,
                 input_vec_size=g_input_vec_size)
    #not require
    else:
        all_files = os.listdir(work_path)
        if "data.npz" in all_files:
            pass
        else:
            if "w2v.model" in all_files:
                pre_data(data_path=data_path, save_path=work_path,
                         input_vec_size=g_input_vec_size)
            else:
                pre_word(data_path=data_path, save_path=work_path,
                         size=g_input_vec_size, min_count=word_min_count, workers=word_workers)
                pre_data(data_path=data_path, save_path=work_path,
                         input_vec_size=g_input_vec_size)
    train_test = np.load(work_path + "data.npz")

    train_data_shape = train_test["train_data"].shape
    train_data_label_shape = train_test["train_label"].shape
    test_data_shape = train_test["test_data"].shape
    test_data_label_shape = train_test["test_label"].shape

    print(train_data_shape)
    print(train_data_label_shape)
    print(test_data_shape)
    print(test_data_label_shape)

    # how much cols we have in the test matrix
    g_max_col_count = test_data_shape[1]

    # TEST:
    dense_units = g_max_col_count

    train_label = to_categorical(train_test["train_label"])
    test_label = to_categorical(train_test["test_label"])

    #create directory for saving the result
    result_dir = work_path + str(int(time.time()))
    os.makedirs(result_dir)

    # save the config
    shutil.copy(sys.argv[1], result_dir)
    # save the source code
    shutil.copy(sys.argv[0], result_dir)

    # build the model
    model = Sequential()
    model.add(Dropout(0.3, input_shape=(g_max_col_count, g_input_vec_size)))
    model.add(LSTM(units=lstm_units, kernel_initializer=initializers.glorot_normal(
    ), input_shape=(g_max_col_count, g_input_vec_size)))
    model.add(Dropout(0.3))
    model.add(Dense(units=dense_units, kernel_regularizer=l2(
        0.01), activation='softmax'))

    adam = Adam(lr=learning_rate)

    # compile
    model.compile(loss=model_loss_name, optimizer=adam, metrics=['accuracy'])

    # train
    model.fit(train_test["train_data"], train_label, epochs=model_epochs,
              batch_size=train_batch_size, verbose=train_verbose)

    # save model---model's structure,weights,Training configuration and Optimizer status
    print("saving the train model ...")
    model.save(result_dir + "/" + save_model_name)
    print("train model is saved")
    # evaluation
    model.summary()

    fw = open(result_dir+"/result.txt", "a+")
    fw.write(str(train_data_shape) + "\n")
    fw.write(str(train_data_label_shape) + "\n")
    fw.write(str(test_data_shape) + "\n")
    fw.write(str(test_data_label_shape) + "\n")

    score = model.evaluate(
        train_test["test_data"], test_label, batch_size=test_batch_size, verbose=test_verbose)
    print(score)

    fw.write("****************************************\n")
    fw.write(str(score) + "\n")
    fw.close()
