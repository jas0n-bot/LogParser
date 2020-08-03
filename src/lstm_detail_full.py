# -*- coding: utf-8 -*-

import os,sys,json
import time,codecs
import shutil,datetime
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2, l1_l2
from gensim.models import word2vec
import numpy as np

def pre_word( data_path=None, save_path=None, size=None, min_count=None, workers=None):

    print("start process word...")
    sentences = word2vec.LineSentence(data_path + 'test.txt')
    model = word2vec.Word2Vec(sentences, size = size, min_count = min_count, workers = workers)
    model.save(save_path + 'w2v.model')

def pre_train_data( data_path=None, save_path=None, input_vec_size=None):

    elem_zero = [0] * input_vec_size
    print("loading w2v.model...")
    word2vecModel = word2vec.Word2Vec.load(save_path + 'w2v.model')
    print("completed w2v.model loading")
    train_list = []
    train_label_list = []
    # data preprocessing
    print("read data...")
    train_file = codecs.open(data_path + 'train.txt', encoding='utf-8')
    train_arr = train_file.readlines()
    train_file.close()
    train_label_file = codecs.open(data_path + 'train_label.txt', encoding='utf-8')
    train_label_arr = train_label_file.readlines()
    train_label_file.close()
    test_file = codecs.open(data_path + 'test.txt', encoding='utf-8')
    test_arr = test_file.readlines()
    test_file.close()
    print("process train label ...")
    for x in train_label_arr:
        train_label_list.append(int(x))
    print("save train label in npy...")
    train_label = np.array(train_label_list)
    np.save(save_path + "train_label.npy", train_label)
    all_number = len(test_arr)
    g_max_col_count = 0
    # max g_max_col_count
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
        x_list = []
        for y in x:
            if y in word2vecModel.wv.vocab:
                x_list.append(word2vecModel[y])
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
    return all_number

def pre_test_data(content=None, word2vecModel=None, input_vec_size=None, g_max_col_count=None):

    elem_zero = [0] * input_vec_size
    test_list = []
    x = content.strip().split()
    x_list = []
    for y in x:
        if y in word2vecModel.wv.vocab:
            x_list.append(word2vecModel[y])
        else:
            x_list.append(elem_zero)
    test_list.append(x_list)
    for x in test_list:
        if len(x) < g_max_col_count:
            for _ in range(g_max_col_count - len(x)):
                x.append(elem_zero)
    test_data = np.array(test_list)

    return test_data

if __name__ == '__main__':
    # initing
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        print("No config")
        # path = ''
    fp = open(config_file, "r")
    settings = json.loads(fp.read())
    g_input_vec_size = settings["g_input_vec_size"]

    word_min_count = settings["word_min_count"]
    word_workers = settings["word_workers"]

    lstm_units = settings["lstm_units"]
    # last_dense_units = settings["dense_units"]
    model_epochs = settings["model_epochs"]
    save_model_name = settings["model_name"]
    model_loss_name = settings["model_loss"]
    model_optimizer_name = settings["model_optimizer"]

    data_path = settings["data_path"]
    work_path = settings["work_path"]

    train_batch_size = settings["train_batch_size"]
    # train_verbose = settings["train_verbose"]
    test_batch_size = settings["test_batch_size"]
    # test_verbose = settings["test_verbose"]

    retrain_flag = settings["retrain_flag"]  # whether to retatin the w2v.model;1-yes,
    fp.close()

    # require to retrain
    if retrain_flag == 1:
        pre_word(data_path=data_path, save_path=work_path, size=g_input_vec_size, min_count=word_min_count,
                 workers=word_workers)
        all_items = pre_train_data(data_path=data_path, save_path=work_path, input_vec_size=g_input_vec_size)
    # not require
    else:
        all_files = os.listdir(work_path)
        if "train_data.npy" in all_files:
            ff = codecs.open(data_path + 'test.txt', encoding='utf-8')
            all_items = len(ff.readlines())
            ff.close()
        else:
            if "w2v.model" in all_files:
                all_items = pre_train_data(data_path=data_path, save_path=work_path, input_vec_size=g_input_vec_size)
            else:
                pre_word(data_path=data_path, save_path=work_path, size=g_input_vec_size, min_count=word_min_count,
                         workers=word_workers)
                all_items = pre_train_data(data_path=data_path, save_path=work_path, input_vec_size=g_input_vec_size)

    train_data = np.load(work_path + "train_data.npy")
    train_label_data = np.load(work_path + "train_label.npy")
    train_label = to_categorical(train_label_data)
    g_max_col_count = train_data.shape[1]
    last_dense_units = train_label.shape[1]

    # create directory for saving the result
    result_dir = work_path + str(int(time.time()))
    os.makedirs(result_dir)

    # build the model
    model = Sequential()
    model.add(Dropout(0.3, input_shape=(g_max_col_count, g_input_vec_size)))
    model.add(LSTM(units=lstm_units, input_shape=(g_max_col_count, g_input_vec_size)))
    model.add(Dropout(0.3))
    model.add(Dense(units=last_dense_units, kernel_regularizer=l2(0.01), activation='softmax'))
    # compile
    model.compile(loss=model_loss_name, optimizer=model_optimizer_name, metrics=['accuracy'])
    # train
    model.fit(train_data, train_label, epochs=model_epochs, batch_size=train_batch_size, verbose=1)
    # save model---model's structure,weights,Training configuration and Optimizer status
    print("saving the train model ...")
    model.save(result_dir + "/" + save_model_name)
    print("train model is saved")
    # evaluation
    model.summary()

    #test
    print("starting test ...")
    fp = open( data_path + "test.txt", "r" )
    fw = open( data_path + "test_label.txt", "r" )
    f_error = open(result_dir + "/" + "test_error.txt", "a+")
    # f_error = open(result_dir + "/" + "test_error.txt", "a+")
    f_result = open(result_dir + "/" + "result.txt", "a+")
    f_test  = open(result_dir + "/" + "test_process.txt", "a+") 
    word2vecModel = word2vec.Word2Vec.load(work_path + 'w2v.model')
    data_line = fp.readline()
    label_line = fw.readline()
    test_all = 0 
    test_error = 0 
    test_correct = 0 
    print("all data: " + str(all_items))
    start_time = datetime.datetime.now()
    while data_line:
        test_data = pre_test_data( content=data_line, word2vecModel=word2vecModel, input_vec_size=g_input_vec_size,
                                g_max_col_count=g_max_col_count )
        test_label_list = []
        test_label_list.append(int(label_line))
        test_label = to_categorical(test_label_list, last_dense_units)
        score = model.evaluate(test_data, test_label, batch_size=test_batch_size, verbose=0)
        f_test.write(str(score).strip() + "\n")
        test_all += 1
        if score[1] == 0:
            test_error += 1
            # f_error.write(data_line.strip() + "\n")
            f_error.write(str(label_line).strip() + "\n")
        else:
            test_correct += 1
        data_line = fp.readline()
        label_line = fw.readline()
        percent = 1.0 * test_all / all_items * 100
        print('complete percent:' + str(percent) + '%')
        sys.stdout.write("\r")

    print(test_correct/float(test_all))
    f_result.write("the number of all data:  %s"%test_all + "\n")
    f_result.write("the number of correct:  %s"%test_correct + "\n")
    f_result.write("the number of error:  %s"%test_error + "\n")
    f_result.write("the result :" + str(test_correct/float(test_all)))
    fp.close()
    fw.close()
    f_test.close()
    f_result.close()
    f_error.close()
    end_time = datetime.datetime.now()
    print("uesd time: " + str((end_time - start_time).seconds))
    # save the config
    shutil.copy(sys.argv[1], result_dir)
    # save the source code
    shutil.copy(sys.argv[0], result_dir)








