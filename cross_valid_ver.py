import sklearn
import numpy as np
from sklearn.model_selection import KFold

import logging
import os.path
import time
import sys

from Feature import Feature
from Decoder import Decoder
import prepare_data
from evaluating_utils import evaluate


def train_avg(iterations, train_data, beam_size, cv_epoch_num):
    import logging
    # data = prepare_data.read_file(train_file)
    data = train_data
    feature = Feature()
    decoder = Decoder(beam_size, feature.get_score)
    n = 0
    for t in range(iterations):
        count = 0
        data_size = len(data)

        for line in data:
            n += 1
            y = line.split()
            z = decoder.beamSearch(line)
            if z != y:
                feature.update_avgWeight(y, z, n, t, data_size)

            train_seg = ' '.join(z)

            count += 1
            if count % 1000 == 0:
                print("iter %d , finish %.2f%%" %
                      (t, (count / data_size) * 100))
                logging.info("iter %d , finish %.2f%%" %
                             (t, (count / data_size) * 100))

        model_file = open('cv-' + str(cv_epoch_num) + "-model_result/model-" +
                            str(t) + "_beam-size-" + str(beam_size) + '.pkl', 'wb')
        feature.save_model(model_file)
        model_file.close()
        print("segment with model-%d finish" % t)
        logging.info("segment with model-%d finish" % t)
        print("iteration %d finish" % t)
        logging.info("iteration %d finish" % t)

    feature.last_update(iterations, data_size)
    feature.cal_avg_weight(iterations, data_size)
    avg_model = open(
        'cv-' + str(cv_epoch_num) + "-model_result/avg-model_beam-size-" + str(beam_size) + '.pkl', 'wb')
    feature.save_model(avg_model)
    avg_model.close()
    print("segment with avg-model finish")
    logging.info("segment with avg-model finish")


def test_avg(iterations, test_data, beam_size, cv_epoch_num):
    import logging
    # data = prepare_data.read_file(test_file)
    data = test_data
    feature = Feature()
    decoder = Decoder(beam_size, feature.get_score)

    count = 0
    data_size = len(data)

    model_file = open(
        'cv-' + str(cv_epoch_num) + '-model_result/avg-model_beam-size-' + str(beam_size) + '.pkl', 'rb')
    feature.load_model(model_file)
    model_file.close()
    for line in data:
        z = decoder.beamSearch(line)
        seg_data = ' '.join(z)
        seg_data_file = 'cv-' + str(cv_epoch_num) + '-test_seg_data/avg-test-seg-data' + \
            '_beam-size-' + str(beam_size) + '.txt'
        with open(seg_data_file, 'a') as f:
            f.write(seg_data + '\n')
        count += 1
        if count % 1000 == 0:
            print("segment with avg-model, finish %.2f%%" %
                  ((count / data_size) * 100))
            logging.info("segment with avg-model, finish %.2f%%" %
                         ((count / data_size) * 100))
    f.close()
    print("segment with avg model finish")


if __name__ == '__main__':

    logging.basicConfig(filename='5-fold-cv-process.log',
                        format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info("This is a info log.")

    kf = KFold(n_splits=5, random_state=10, shuffle=True)

    train_file = 'data/fusion_data.txt'
    raw_data = prepare_data.read_file(train_file)
    cv_epoch_num = 1

    beam_size=16

    # generate gold for test_data in every cross-valid
    for train_id, test_id in kf.split(raw_data):
        logging.info("cross-valid epoch: %d", cv_epoch_num)
        print("cross-valid epoch:", cv_epoch_num)

        train_id_str = str(train_id[0:3]) + '...' + str(train_id[-3:])
        test_id_str = str(test_id[0:3]) + '...' + str(test_id[-3:])
        logging.info("TRAIN: %s, TEST: %s", train_id_str, test_id_str)
        print("TRAIN:", train_id, "TEST:", test_id)

        train_data, test_data = np.array(
            raw_data)[train_id], np.array(raw_data)[test_id]

        gold_test_filepath = "data/cv-" + str(cv_epoch_num) + "-filter_test.txt"
        prepare_data.prepare_data(test_data, gold_test_filepath)

        logging.info("len of raw_data: %d, train_data: %d, test_data: %d",
                    len(raw_data), len(train_data), len(test_data))
        print("len of raw_data:", len(raw_data), "train_data:",
            len(train_data), "test_data:", len(test_data))
        logging.info(
            "-------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------")
        cv_epoch_num += 1
        # break


    # train and generate test_seg_data
    cv_epoch_num = 1
    for train_id, test_id in kf.split(raw_data):
        print("Start cross-validation.")
        logging.info("Start cross-validation.")
        logging.info("cross-valid epoch: %d", cv_epoch_num)
        print("cross-valid epoch:", cv_epoch_num)
        
        train_data, test_data = np.array(raw_data)[train_id], np.array(raw_data)[test_id]

        train_avg(iterations=3, train_data=train_data, beam_size=beam_size, cv_epoch_num=cv_epoch_num)
        print("-------------------------------------------------------------------------------------------------")
        test_avg(iterations=3, test_data=test_data, beam_size=beam_size, cv_epoch_num=cv_epoch_num)
        print("-------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------")

        cv_epoch_num += 1
        # print(len(raw_data), len(train_data), len(test_data))
        # break

    # evaluate
    cv_iter = 1
    for cv_iter in range(1, 6):
        test_filepath = 'cv-' + str(cv_iter) + '-test_seg_data/avg-test-seg-data' + \
                '_beam-size-' + str(beam_size) + '.txt'
        gold_test_filepath = "data/cv-" + str(cv_iter) + "-filter_test.txt"
        word_precision, word_recall, word_fmeasure = evaluate(test_filepath, gold_test_filepath)
        
        logging.info("cv-%d-eval:", cv_iter)
        logging.info("precision: %.3f" % word_precision)
        logging.info("recall: %.3f" % word_recall)
        logging.info("F1: %.3f" % word_fmeasure)
        logging.info("-------------------------------------------------------------")

        cv_iter += 1
