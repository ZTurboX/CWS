import re
from Feature import Feature


def read_file(fileName):
    '''
    读取文件内容
    '''
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l for l in lines if l != '\n' and len(l) > 0]
    return lines


def write_file(filter_file, corpus):
    '''
    写入文件
    '''
    with open(filter_file, 'a', encoding='utf-8') as f:
        for data in corpus:
            f.write(data+"\n")


def prepare_data(data, filter_file):
    '''
    去除语料中的标签
    '''
    corpus = []
    for line in data:
        line.strip('\n')
        split_line = line.split()
        filter_arr = []
        data = ''
        for arr in split_line:
            index = arr.find(']')
            if index > 0:
                arr = arr[:index+1]
            ret = re.sub("/[a-zA-Z]+", "", arr)
            filter_arr.append(ret)
        filter_arr = filter_arr[1:]
        data = ' '.join(filter_arr)
        corpus.append(data)
    write_file(filter_file, corpus)
    print("过滤文件成功")


if __name__ == "__main__":
    train_file = "./data/train.txt"
    test_file = "./data/test.txt"
    filter_train_file = "./data/filter_train.txt"
    filter_test_file = "./data/filter_test.txt"
    # data=read_file(train_file)
    data = read_file(test_file)
    # prepare_data(data,filter_train_file)
    prepare_data(data, filter_test_file)
