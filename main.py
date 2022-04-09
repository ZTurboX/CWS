import prepare_data

from Feature import Feature
from Decoder import Decoder
import operator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', type=str,
                    default='train_avg', help='train/train_avg/test')
parser.add_argument('--beam', '-b', type=int, default=16, help='beam size')
parser.add_argument('--iter', '-n', type=int, default=10, help='iterations')
args = parser.parse_args()
args.mode = 'test_avg' # test_avg
mode = args.mode
iter = args.iter
beam_size = args.beam


def train(iterations, train_file, beam_size):
    data = prepare_data.read_file(train_file)
    feature = Feature()
    decoder = Decoder(beam_size, feature.get_score)

    for t in range(iterations):
        count = 0
        data_size = len(data)

        for line in data:
            y = line.split()
            z = decoder.beamSearch(line)
            if z != y:
                feature.update_weight(y, z)

            train_seg = ' '.join(z)
            seg_data_file = 'train_seg_data/train-seg-data_ model-' + \
                str(t) + '.txt'
            with open(seg_data_file, 'a') as f:
                f.write(train_seg + '\n')

            count += 1
            if count % 1000 == 0:
                print("iter %d , finish %.2f%%" % (t, (count/data_size)*100))

        model_file = open("model_result/model-" + 
                            str(t)+"_beam-size-"+str(beam_size)+'.pkl', 'wb')
        feature.save_model(model_file)

        model_file.close()
        f.close()
        print("segment with model-%d finish" % t)
        print("iteration %d finish" % t)


def train_avg(iterations, train_file, beam_size):
    data = prepare_data.read_file(train_file)
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
                print("iter %d , finish %.2f%%" % \
                        (t, (count / data_size) * 100))

        model_file = open("model_result/model-" + \
                            str(t) + "_beam-size-" + str(beam_size) + '.pkl', 'wb')
        feature.save_model(model_file)
        model_file.close()
        print("segment with model-%d finish" % t)
        print("iteration %d finish" % t)

    feature.last_update(iterations, data_size)
    feature.cal_avg_weight(iterations, data_size)
    avg_model = open(
        "model_result/avg-model_beam-size-" + str(beam_size) + '.pkl', 'wb')
    feature.save_model(avg_model)
    avg_model.close()
    print("segment with avg-model finish")


def test(iterations, test_file, beam_size, mode):
    data = prepare_data.read_file(test_file)
    feature = Feature()
    decoder = Decoder(beam_size, feature.get_score)

    for t in range(iterations):
        count = 0
        data_size = len(data)

        model_file = open('model_result/model-' +
                            str(t)+'_beam-size-'+str(beam_size)+'.pkl', 'rb')

        feature.load_model(model_file)
        model_file.close()
        for line in data:
            z = decoder.beamSearch(line)
            seg_data = ' '.join(z)
            seg_data_file = 'test_seg_data/test-seg-data_model-' + \
                str(t)+'_beam-size-'+str(beam_size)+'.txt'
            with open(seg_data_file, 'a') as f:
                f.write(seg_data+'\n')
            count += 1
            if count % 1000 == 0:
                print("segment with model-%d , finish %.2f%%" %
                      (t, (count / data_size) * 100))
        f.close()
        print("segment with model-%d finish" % t)


def test_avg(iterations, test_file, beam_size):
    data = prepare_data.read_file(test_file)
    feature = Feature()
    decoder = Decoder(beam_size, feature.get_score)

    count = 0
    data_size = len(data)

    model_file = open(
        'model_result/avg-model_beam-size-' + str(beam_size) + '.pkl', 'rb')
    feature.load_model(model_file)
    model_file.close()
    for line in data:
        z = decoder.beamSearch(line)
        seg_data = ' '.join(z)
        seg_data_file = 'test_seg_data/avg-test-seg-data' + \
            '_beam-size-' + str(beam_size) + '.txt'
        with open(seg_data_file, 'a') as f:
            f.write(seg_data + '\n')
        count += 1
        if count % 1000 == 0:
            print("segment with avg-model, finish %.2f%%" %
                  ((count / data_size) * 100))
    f.close()
    print("segment with avg model finish")


if __name__ == '__main__':

    print("The mode at this moment is:", mode)

    train_file = 'data/filter_train.txt'
    test_file = 'data/filter_test.txt'

    if mode == 'train':
        train(iter, train_file, beam_size)
    elif mode == 'train_avg':
        train_avg(iter, train_file, beam_size)
    elif mode == 'test':
        test(iter, test_file, beam_size, mode)
    elif mode == 'test_avg':
        test_avg(iter, test_file, beam_size)
