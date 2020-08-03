# List of source files and details

|File name|processing version|how to run|description
|---|---|---|---
|lstm_batch.py|batch|python lstm_batch.py config.json|simple LSTM network
|lstm_detail_full.py|batch|python lstm_detail_full.py config.json|function as the same as lstm_batch.py but record every fault records
|lstm_split_full.py|batch|python lstm_split_full.py config.json|function as the same as lstm_batch.py but design to process big data by processing test data up in one batch with maximum size of **reading_test_lines**
|lstm_split_test.py|batch|python lstm_split_test.py config.json|perform just the test part of *lstm_split_full.py*, requires **w2v.model**, trained **model**, **train_data.npy**, **train_label.npy** exists in **work_path**
