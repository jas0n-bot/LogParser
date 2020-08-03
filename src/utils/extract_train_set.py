"""Script to generate train data from test data"""
import sys
import io
import logging

def main():
    """Main entry"""
    # path and max amount retrieve from each class
    if len(sys.argv) == 3:
        path = sys.argv[1]
        max_amount = int(sys.argv[2])
    else:
        path = ''
        max_amount = 200
    logger = logging.getLogger()
    logger.info(path)
    logger.info(max_amount)

    train_file = io.open(path + 'train.txt', 'w+', encoding='utf-8')
    train_label_file = io.open(path + 'train_label.txt', 'w+', encoding='utf-8')
    test_file = io.open(path + 'test.txt', encoding='utf-8')
    test_label_file = io.open(path + 'test_label.txt', encoding='utf-8')

    item_count = {}
    for test_line in test_file:
        test_line = test_line.strip()
        test_label_line = test_label_file.readline().strip()
        if not item_count.get(test_label_line):
            item_count[test_label_line] = 1
        else:
            item_count[test_label_line] = item_count[test_label_line] + 1
        if item_count[test_label_line] <= max_amount:
            train_file.write(test_line + '\n')
            train_label_file.write(test_label_line + '\n')

    train_file.close()
    train_label_file.close()
    test_file.close()
    test_label_file.close()

main()
