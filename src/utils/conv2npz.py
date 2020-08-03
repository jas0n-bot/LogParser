"""Convert train data and test data to npz format use w2v model"""
import sys
import codecs
import numpy as np
from gensim.models import word2vec


def main():
    """Main entry"""
    g_vec_size = 100

    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = ''

    elem_zero = np.array([0] * g_vec_size)
    w2v_model = word2vec.Word2Vec.load(path + 'w2v.model')

    train_list = []
    train_label_list = []
    test_list = []
    test_label_list = []

    # data preprocessing
    train_file = codecs.open(path + 'train.txt', encoding='utf-8')
    train_arr = train_file.readlines()
    train_file.close()
    train_label_file = codecs.open(path + 'train_label.txt', encoding='utf-8')
    train_label_arr = train_label_file.readlines()
    train_label_file.close()
    test_file = codecs.open(path + 'test.txt', encoding='utf-8')
    test_arr = test_file.readlines()
    test_file.close()
    test_label_file = codecs.open(path + 'test_label.txt', encoding='utf-8')
    test_label_arr = test_label_file.readlines()
    test_label_file.close()

    g_max_col_count = 0

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
    for x in train_label_arr:
        train_label_list.append(int(x))

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

    np.savez(path + "data.npz", train_data=train_data,
             train_label=train_label, test_data=test_data, test_label=test_label)


main()
